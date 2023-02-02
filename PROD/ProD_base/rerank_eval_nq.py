from os.path import join
import sys

sys.path += ['../']
import argparse
import glob
import json
import logging
import os
import random
import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
#  
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from model.models import BiEncoderNllLoss, BiBertEncoder, HFBertEncoder, Reranker
from utils.lamb import Lamb
import random
from transformers import (
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
# from utils.MARCO_until import convert_to_unicode

logger = logging.getLogger(__name__)
from utils.util import (
    StreamingDataset,
    EmbeddingCache,
    set_seed,
    is_first_worker,
)
from utils.dpr_utils import (
    load_states_from_checkpoint,
    get_model_obj,
    CheckpointState,
    get_optimizer,
    all_gather_list
)


def get_bert_reranker_components(args, **kwargs):
    encoder = HFBertEncoder.init_encoder(
        args,
    )
    hidden_size = encoder.config.hidden_size
    reranker = Reranker(encoder, hidden_size)

    return reranker


def sum_main(x, opt):
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x


from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import pickle
from collections import namedtuple
# from utils.msmarco_eval import compute_metrics_from_files
def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question
import collections
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])

class Wiki_reranker_infer_Dataset(Dataset):
    def __init__(self, file_path,tokenizer, trainer_id=0, trainer_num=1,max_seq_length = 256):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = self._read_example(file_path, trainer_id, trainer_num)
        self.max_seq_length = max_seq_length
    def _read_example(self, input_file, trainer_id=0, trainer_num=1):
        """Reads a tab separated value file."""
        examples_split = []
        with open(input_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
            for i, example in tqdm(enumerate(data)):
                if i % trainer_num == trainer_id:
                    examples_split.append(example)
        return examples_split

    def __getitem__(self, index):
        json_sample = self.data[index]

        query = normalize_question(json_sample["question"])
        def create_passage(ctx: dict):
            return BiEncoderPassage( ctx["text"],
                ctx["title"],
            )
        ctxs = [create_passage(ctx) for ctx in json_sample['ctxs']]
        ctx_hit = [ctx['hit'] == 'True' for ctx in json_sample['ctxs']]
        ctx_token_ids = [self.tokenizer.encode(ctx.title, text_pair=ctx.text.strip(), add_special_tokens=True,
                                        max_length=self.max_seq_length,truncation=True,
                                        pad_to_max_length=False) for ctx in ctxs]
        
        question_token_ids = self.tokenizer.encode(query)

        def remove_special_token(token_id):
            if token_id[-1] == self.tokenizer.sep_token_id:
                return token_id[1:-1]
            else:
                return token_id[1:]
        c_e_token_ids = [question_token_ids + remove_special_token(ctx_token_id) for ctx_token_id in ctx_token_ids]

        return c_e_token_ids,ctx_hit

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_collate_fn(cls, args):
        def create_biencoder_input2(features):
            doc_per_question = len(features[0][1])
            q_num, d_num = len(features), doc_per_question

            

            c_e_input_list = []
            for index, feature in enumerate(features):
                c_e_input_list.extend(feature[0])
            max_c_e_len = max([len(d) for d in c_e_input_list])
            c_e_list = [c_e+[0]*(max_c_e_len-len(c_e)) for c_e in c_e_input_list]
            ctx_tensor_out = torch.LongTensor(c_e_list)
            ctx_tensor_out = ctx_tensor_out.reshape(q_num, d_num,-1)

            hit_list = [feature[1] for feature in features]
            return {'teacher': [ctx_tensor_out, (ctx_tensor_out!= 0).long(), hit_list]}

        return create_biencoder_input2

from utils.dpr_utils import all_gather_list
def evaluate_dev(args, model, tokenizer, passage=None):
    dev_dataset = Wiki_reranker_infer_Dataset(args.data_path,tokenizer,trainer_id=args.local_rank,
                                               trainer_num=args.world_size)
    dev_sample = RandomSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sample,
                                collate_fn=Wiki_reranker_infer_Dataset.get_collate_fn(args),
                                batch_size=args.per_gpu_train_batch_size, num_workers=3, shuffle=False)
    model.eval()
    total_d_ids_ranker_list = []
    total_new_hit_list = []
    total_origin_hit_list =[]
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dev_dataloader)):
            batch_teacher = batch['teacher']
            # q_ids, d_ids_lists = batch_teacher[:2]
            inputs_teacher = {"input_ids": batch_teacher[0].long().to(args.device),
                              "attention_mask": batch_teacher[1].long().to(args.device)}
                
            output_teacher = model(**inputs_teacher)
            binary_logits, relevance_logits, _ = output_teacher
            topk_sim, topk_index = relevance_logits.topk(100)
            new_d_ids_ranker_list = topk_index.tolist()
            hit_list = batch_teacher[2]
            for index,hitl in enumerate(hit_list):
                new_hitl = []
                for ij in new_d_ids_ranker_list[index]:
                    new_hitl.append(hitl[ij])
                total_new_hit_list.append(new_hitl)
                total_origin_hit_list.append(hitl)
    pickle_path = os.path.join(args.output_dir, str(dist.get_rank()) + 'temp.pick')
    with open(pickle_path, 'wb') as handle:
        pickle.dump([total_new_hit_list, total_origin_hit_list], handle, protocol=4)
    dist.barrier()
    qids_to_ranked_candidate_passages ={}
    if is_first_worker():
        new_scores ,origin_scores = [],[]
        for i in tqdm(range(args.world_size)):  # TODO: dynamically find the max instead of HardCode
            pickle_path = os.path.join(args.output_dir, str(i) + 'temp.pick')
            with open(pickle_path, 'rb') as handle:
                item = pickle.load(handle)
                new_scores.extend(item[0])
                origin_scores.extend(item[1])

        logger.info('Per question validation results len=%d', len(new_scores))
        top_k_hits = [0] * len(origin_scores[0])
        for question_hits in new_scores:
            best_hit = next((i for i, x in enumerate(question_hits) if x), None)
            if best_hit is not None:
                top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
        top_k_hits = [v / len(origin_scores) for v in top_k_hits]
        logger.info('After rerank Validation results: top k documents hits %s', top_k_hits)
        logger.info({'top1': top_k_hits[0],'top5': top_k_hits[4], 'top10': top_k_hits[9],
                    'top20': top_k_hits[19], 'top50': top_k_hits[49], 'top100':top_k_hits[99]})
        
        top_k_hits = [0] * len(origin_scores[0])
        for question_hits in origin_scores:
            best_hit = next((i for i, x in enumerate(question_hits) if x), None)
            if best_hit is not None:
                top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
        top_k_hits = [v / len(origin_scores) for v in top_k_hits]
        logger.info('Before rerank Validation results: top k documents hits %s', top_k_hits)
        logger.info({'top1': top_k_hits[0],'top5': top_k_hits[4], 'top10': top_k_hits[9],
                    'top20': top_k_hits[19], 'top50': top_k_hits[49], 'top100':top_k_hits[99]})
    return qids_to_ranked_candidate_passages



def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list:",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--out_result_path",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--q_text_path",
        default=0,
        type=str,
        help="Number of epoch to train, if specified will use training data instead of ann",
    )
    parser.add_argument(
        "--random_pool",
        default=10,
        type=int,
        help="Number of epoch to train, if specified will use training data instead of ann",
    )
    # Other parameters
    parser.add_argument(
        "--eval_pickle_result_path", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=12,
        type=int,
        help="Number of floors of each tower in the two tower model",
    )

    parser.add_argument(
        "--corpus_path",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )


    parser.add_argument("--triplet", default=False, action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )

    parser.add_argument(
        "--optimizer",
        default="adamW",
        type=str,
        help="Optimizer - lamb or adamW",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_checkpointing", default=False, action="store_true",)

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    # ----------------- End of Doc Ranking HyperParam ------------------
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    args = parser.parse_args()

    return args


def set_env(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)


def load_model(args):
    # Prepare GLUE task
    args.output_mode = "classification"
    label_list = ["0", "1"]
    num_labels = len(label_list)

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # args.model_type = args.model_type.lower()

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True)
    model = get_bert_reranker_components(args)

    if args.model_name_or_path is not None:
        saved_state = load_states_from_checkpoint(args.model_name_or_path)
        # Todo
        model.load_state_dict(saved_state.model_dict, strict=False)
        # global_step = _load_saved_state(model, optimizer, scheduler, saved_state)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    model.to(args.device)
    logger.info("Inference parameters %s", args)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
    #     )
    return tokenizer, model


def main():
    args = get_arguments()
    set_env(args)
    

    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    if is_first_worker():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    dist.barrier()
    log_path = os.path.join(args.output_dir, 'log.txt')
    # sh = logging.StreamHandler()
    handler = logging.FileHandler(log_path, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.addHandler(sh)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    print(logger)

    tokenizer, model = load_model(args)
    evaluate_dev(args, model, tokenizer, passage=None)
    # global_step = train(args, model, tokenizer)
    # logger.info(" global_step = %s", global_step)

    if args.local_rank != -1:
        dist.barrier()


if __name__ == "__main__":
    main()
    logger.info(" train done")

