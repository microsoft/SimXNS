"""
Using the T5-base model released at https://github.com/castorini/docTTTTTquery to generate queries for each document of beir.
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import json
import logging
import sys
import os
import argparse
from tqdm import tqdm
pyfile_path = os.path.abspath(__file__) 
pyfile_dir = os.path.dirname(os.path.dirname(pyfile_path)) # equals to the path '../'
sys.path.append(pyfile_dir)

from utils.util import set_seed, is_first_worker
logger = logging.getLogger("__main__")

class DocDataset(Dataset):
    """
    load the passage data
    """
    def __init__(self, data, tokenizer, max_seq_length =512):
        self.data = data 
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        psg_id, text, title = self.data[index]
        ctx_token_ids = self.tokenizer.encode(text.strip(), max_length=self.max_seq_length, truncation=True) 
        return ctx_token_ids, psg_id
    
    def collate_fn(self, samples):
        d_list = []
        psg_id_list = []
        for sample in samples:
            d_list.append(sample[0])
            psg_id_list.append(sample[1])

        encoder_input_list = [torch.tensor(e, dtype=torch.long) for e in d_list]
        # create encoder_inputs, attention_mask, decoder_inputs, decoder_labels
        _mask = pad_sequence(encoder_input_list, batch_first=True, padding_value=-100)

        attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)
        encoder_inputs = pad_sequence(encoder_input_list, batch_first=True, padding_value=0)

        return {'generator': {'input_ids': encoder_inputs, 
                              'attention_mask': attention_mask},
                'psg_id_list': psg_id_list, 
                }

def set_env(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = torch.distributed.get_world_size()
    args.device = device
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
    else:
        args.world_size = 1
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1)
                   )

    # Set seed
    set_seed(args)

    # Create output directory if needed
    if is_first_worker() and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.local_rank != -1:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will create the output dir.


    basic_format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    formatter = logging.Formatter(basic_format)
    log_path = os.path.join(args.output_dir, 'log.txt')
    # sh = logging.StreamHandler()
    handler = logging.FileHandler(log_path, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.addHandler(sh)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    print(logger)
    logger.info(args)

def load_data(corpus_path):
    corpus = []
    with open(corpus_path, 'r', encoding="utf-8") as fr:
        for line in fr:
            d = json.loads(line.strip())
            corpus.append([d['_id'], d['text'], d['title']])
    print(f'The {corpus_path} corpus has {len(corpus)} documents.')
    return corpus


def generate(model, tokenizer, test_dataloader, args, passage_id_list):
    """
    Generate queries with the well-trained model.
    """
    _id = 0
    if args.local_rank==-1:
        output_filename = os.path.join(args.output_dir, f"single.txt")
    else:
        output_filename = os.path.join(args.output_dir, f"gpu_{args.local_rank}.txt")
    
    generated_psg_id_set = set()
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as fr:
            for i, line in enumerate(fr):
                passage = json.loads(line.strip())
                generated_psg_id_set.add(passage['passage_id'])
        fw = open(output_filename, 'a', encoding='utf-8')
    else:
        fw = open(output_filename, 'w', encoding='utf-8')

    def _check(psg_id_list, generated_psg_id_set):
        """
        whether generated_psg_id_set contains all index in psg_id_list 
        """
        if generated_psg_id_set is None:
            return False
        for psg_id in psg_id_list:
            if psg_id not in generated_psg_id_set:
                return False
        return True


    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Generate", disable=args.local_rank not in [-1, 0]):
            
            example_list = []
            batch_generator = batch['generator']
            psg_id_list = batch['psg_id_list']

            if _check(psg_id_list, generated_psg_id_set): # all has generated in last time
                continue
            
            f = model.module if hasattr(model, "module") else model
            outputs = f.generate(input_ids=batch_generator['input_ids'].to(args.device), 
                                 attention_mask=batch_generator['attention_mask'].to(args.device),
                                 max_length=64,
                                 do_sample=True,
                                 top_k=10,
                                 num_return_sequences=args.num_query
                                )
            generated_example = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            assert len(generated_example)%args.num_query==0
            tmp_list = []
            for i, example in enumerate(generated_example):
                tmp_list.append(example)
                if len(tmp_list) == args.num_query:
                    # tmp_list = sorted(tmp_list)
                    example_list.append(tmp_list)
                    tmp_list = []
                    _id += 1
            for psg_id, example in zip(psg_id_list, example_list):
                if psg_id in generated_psg_id_set: # has generated in last time
                    continue
                output_data = {'passage_id': str(psg_id), 'generated_queries': example}
                line = json.dumps(output_data)
                fw.write(line+'\n')
            fw.flush()
    fw.close()
    if args.world_size>1: 
        # merge the data from all gpus
        torch.distributed.barrier()
        if args.local_rank==0:
            merge_output_data = {}
            for i in range(args.world_size):
                output_filename = os.path.join(args.output_dir, f"gpu_{i}.txt")
                print(f'Load data from {output_filename}')
                
                with open(output_filename, 'r', encoding='utf-8') as fr:
                    for line in fr:
                        passage = json.loads(line.strip())
                        if passage['passage_id'] not in merge_output_data:
                            merge_output_data[passage['passage_id']] = passage['generated_queries']

    else:
        # single gpu
        merge_output_data = {}
        print(f'Load data from {output_filename}')
        with open(output_filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                passage = json.loads(line.strip())
                if passage['passage_id'] not in merge_output_data:
                    merge_output_data[passage['passage_id']] = passage['generated_queries']

    if args.local_rank in [-1, 0]: 
        output_filename = os.path.join(args.output_dir,  f"query.tsv")
        logger.info(f'Output data to {output_filename}')
        with open(output_filename, 'w', encoding='utf-8') as fw:
            for passage_id in passage_id_list:
                content = '\t'.join([passage_id] + merge_output_data[passage_id] )
                fw.write(content+'\n')
        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--num_query", type=int, default=10, help="Number of queries generated for each document.")
    parser.add_argument("--dataset_dir", type=str, default = './beir/datasets', help='the dir of beir datasets')
    parser.add_argument("--dataset", type=str, default = 'nfcorpus', help='the name of the dataset')
    parser.add_argument("--per_device_eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for predition.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--start", type=int, default=-1, help="Start index of passages.")
    parser.add_argument("--end", type=int, default=-1, help="End index of passages.")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.dataset_dir, args.dataset, 'generated_query')

    set_env(args)
    logger.info("args:\n%s", '\n'.join([f'    {arg}={getattr(args, arg)}'  for arg in vars(args)]))
    # load model
    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
    model.to(args.device)


    passages = load_data(os.path.join(args.dataset_dir, args.dataset, 'corpus.jsonl'))
    passage_id_list = [p[0] for p in passages]

    if args.start>=0 and args.end>0:
        passages = passages[args.start:args.end]
    if args.local_rank!=-1:
        shard_size = len(passages) // args.world_size
        start_idx = args.local_rank * shard_size
        end_idx = start_idx + shard_size
        if args.local_rank == args.world_size - 1:
            end_idx = len(passages)
        passages_piece = passages[start_idx:end_idx]
        del passages
        logger.info(f'Query generation for {len(passages_piece)} passages from idx {start_idx} to {end_idx}')
    else:
        passages_piece = passages
    test_dataset = DocDataset(passages_piece, tokenizer)
    model.eval()
    test_sampler = SequentialSampler(test_dataset) 
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                 collate_fn=test_dataset.collate_fn,
                                 batch_size=args.per_device_eval_batch_size, 
                                 num_workers=5)
    generate(model, tokenizer, test_dataloader, args, passage_id_list)

