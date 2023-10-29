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
import pickle
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
from utils.marco_until import (
    Rocketqa_v2Dataset
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

import six
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def csv_reader(fd, delimiter='\t', trainer_id=0, trainer_num=1):
    def gen():
        for i, line in tqdm(enumerate(fd)):
            if i % trainer_num == trainer_id:
                slots = line.rstrip('\n').split(delimiter)
                if len(slots) == 1:
                    yield slots,
                else:
                    yield slots
    return gen()
def load_train_reference_from_stream(input_file, trainer_id=0, trainer_num=1):
    """Reads a tab separated value file."""
    print(input_file)
    with open(input_file, 'r', encoding='utf8') as f:
        reader = csv_reader(f, trainer_id=trainer_id, trainer_num=trainer_num)
        #headers = 'query_id\tpos_id\tneg_id'.split('\t')

        #Example = namedtuple('Example', headers)
        qrel = {}
        for [topicid, _, docid, rel] in reader:
            topicid = int(topicid)
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(int(docid))
            else:
                qrel[topicid] = [int(docid)]
    return qrel


def load_marcodoc_reference_from_stream(path_to_reference):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    qids_to_relevant_passageids = {}
    with open(path_to_reference, 'r') as f:
        for l in f:
            l = l.strip().split(' ')
            qid = int(l[0])
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            qids_to_relevant_passageids[qid].append(int(l[2][1:]))
    return qids_to_relevant_passageids
def read_real(real_path):
    qids_to_relevant_passageids = {}
    with open(real_path, 'r') as f:
        for l in f:
            try:
                l = l.strip().split('\t')
                qid = int(l[0])
                if qid in qids_to_relevant_passageids:
                    pass
                else:
                    qids_to_relevant_passageids[qid] = []
                qids_to_relevant_passageids[qid].append(int(l[1]))
            except:
                raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids

class MarcoDoc_reranker_infer_Dataset(Dataset):
    def __init__(self, result_file_path, query_path, corpus_path, tokenizer, max_seq_length=512,
                 trainer_id=0, trainer_num=1, p_text=None, p_title=None):
        self.result_file_path = result_file_path
        self.query_path = query_path
        self.result_data = self._read_example(trainer_id, trainer_num)
        self.query_dic = self.read_qstring()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_q_length = 32
        self.p_fulltext = self.load_id_text(os.path.join(corpus_path, 'msmarco-docs.tsv'))

    def _read_example(self,trainer_id=0, trainer_num=1):
        """Reads a tab separated value file."""
        examples_split = []
        with open(self.result_file_path, 'rb') as f:
            qids_to_ranked_candidate_passages,_ = pickle.load(f)
        for i, qid in enumerate(qids_to_ranked_candidate_passages):
            if i % trainer_num == trainer_id:
                examples = {}
                examples['query_id'] = int(qid)
                examples['candidate_pid'] = qids_to_ranked_candidate_passages[qid]
                examples['candidate_pid'] = [int(id) for id in examples['candidate_pid']]
                examples_split.append(examples)

        return examples_split

    def load_id_text(self, file_name):
        pids_to_doc = {}
        with open(file_name, 'r') as f:
            for l in f:
                line = l.strip().split('\t')
                if len(line) == 4:
                    pid = int(line[0][1:])
                    url = line[1].rstrip()
                    title = line[2].rstrip()
                    p_text = line[3].rstrip()
                    full_text = url + "<sep>" + title + "<sep>" + p_text
                    full_text = full_text[:10000]
                    pids_to_doc[pid] = full_text
                elif len(line) == 3:
                    pid = int(line[0][1:])
                    url = line[1].rstrip()
                    title_or_text = line[2].rstrip()
                    # NOTE: This linke is copied from ANCE,
                    # but I think it's better to use <s> as the separator,
                    full_text = url + "<sep>" + title_or_text
                    # keep only first 10000 characters, should be sufficient for any
                    # experiment that uses less than 500 - 1k tokens
                    full_text = full_text[:10000]
                    pids_to_doc[pid] = full_text
                elif len(line) == 2:
                    pid = int(line[0][1:])
                    url = line[1].rstrip()
                    # NOTE: This linke is copied from ANCE,
                    # but I think it's better to use <s> as the separator,
                    full_text = url + "<sep>"
                    # keep only first 10000 characters, should be sufficient for any
                    # experiment that uses less than 500 - 1k tokens
                    full_text = full_text[:10000]
                    pids_to_doc[pid] = full_text
                else:
                    pass
        return pids_to_doc

    def read_qstring(self):
        q_string = {}
        with open(self.query_path, 'r', encoding='utf-8') as file:
            for num, line in enumerate(file):
                line = line.strip('\n')  # 删除换行符
                line = line.split('\t')
                q_string[int(line[0])] = line[1]
        return q_string


    def __getitem__(self, index):
        json_sample = self.result_data[index]
        qid = json_sample['query_id']
        query = convert_to_unicode(self.query_dic[qid])
        query = normalize_question(query)

        candidate_list = json_sample['candidate_pid'][:100]
        p_candi_list = [convert_to_unicode(self.p_fulltext[candidate]) for candidate in candidate_list]

        ctx_token_ids = [self.tokenizer.encode(ctx, add_special_tokens=True,
                                               max_length=self.max_seq_length, truncation=True,
                                               pad_to_max_length=False) for ctx in p_candi_list]
        question_token_ids = self.tokenizer.encode(query, add_special_tokens=True,
                                                   max_length=self.max_q_length, truncation=True,
                                                   pad_to_max_length=False)
        def remove_special_token(token_id):
            if token_id[-1] == self.tokenizer.sep_token_id:
                return token_id[1:-1]
            else:
                return token_id[1:]

        c_e_token_ids = [question_token_ids + remove_special_token(ctx_token_id) for ctx_token_id in ctx_token_ids]
        c_e_token_ids = torch.LongTensor(
            [temp + [self.tokenizer.pad_token_id] * (self.max_seq_length + self.max_q_length - len(temp)) for temp in
             c_e_token_ids])
        candidate_list = [int(id) for id in candidate_list]
        return c_e_token_ids, candidate_list, int(qid)

    def __len__(self):
        return len(self.result_data)

    @classmethod
    def get_collate_fn(cls, args):
        def create_biencoder_input2(features):
            q_num = len(features)
            doc_per_question = features[0][0].size(0)
            ctx_tensor_out = torch.cat([feature[0] for feature in features])
            ctx_tensor_out = ctx_tensor_out.reshape(q_num, doc_per_question, -1)
            candidate_lists = [feature[1] for feature in features]
            qid = [feature[2] for feature in features]

            return {'teacher': [ctx_tensor_out,(ctx_tensor_out != 0).long(), candidate_lists, qid]}

        return create_biencoder_input2

from utils.dpr_utils import all_gather_list
def evaluate_dev(args, model, tokenizer, passage=None):
    if args.load_cache !=True:
        dev_dataset = MarcoDoc_reranker_infer_Dataset(args.result_data_path, args.query_path, args.corpus_path,
                                            tokenizer, max_seq_length=args.max_seq_length,
                                            trainer_id=args.local_rank,trainer_num=args.world_size)

        print("gpu all :", args.world_size)
        print("info  ------gpu num:", args.local_rank, "  dataset len:", len(dev_dataset))
        dev_sample = RandomSampler(dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, sampler=dev_sample,
                                    collate_fn=MarcoDoc_reranker_infer_Dataset.get_collate_fn(args),
                                    batch_size=args.per_gpu_train_batch_size, num_workers=10, shuffle=False)


        model.eval()

        total_qid_list = []
        total_candidate_list = []
        total_new_candidate_list = []
        with torch.no_grad():


            for i, batch in enumerate(tqdm(dev_dataloader)):
                batch_teacher = batch['teacher']
                # q_ids, d_ids_lists = batch_teacher[:2]
                inputs_teacher = {"input_ids": batch_teacher[0].long().to(args.device),
                                "attention_mask": batch_teacher[1].long().to(args.device)}
                '''
                '''
                print(inputs_teacher['input_ids'].shape)
                '''
                '''
                    
                output_teacher = model(**inputs_teacher)
                binary_logits, relevance_logits, _ = output_teacher
                # rank
                topk_sim, topk_index = relevance_logits.topk(100)
                new_candidate_index_list = topk_index.tolist()
                candidate_lists = batch_teacher[2]
                new_candidate_lists = np.array(batch_teacher[2])
                qid_list = batch_teacher[3]
                for i, candidate_list in enumerate(new_candidate_lists):
                    new_candidate_lists[i] = candidate_list[new_candidate_index_list[i]]
                new_candidate_lists = new_candidate_lists.tolist()

                total_new_candidate_list.extend(new_candidate_lists)
                total_candidate_list.extend(candidate_lists)
                total_qid_list.extend(qid_list)

            example_dic = {}
            for index, qid in enumerate(total_qid_list):
                example_dic[qid] = [total_new_candidate_list[index], total_candidate_list[index]]

            pickle_path = os.path.join(args.output_dir, str(dist.get_rank()) + 'temp.pick')
            with open(pickle_path, 'wb') as handle:
                pickle.dump(example_dic, handle, protocol=4)
            dist.barrier()            

    qid_to_candidate_dic = {}
    if is_first_worker():
        for i in tqdm(range(args.world_size)):
            pickle_path = os.path.join(args.output_dir, str(i) + 'temp.pick')
            with open(pickle_path, 'rb') as handle:
                item = pickle.load(handle)
                qid_to_candidate_dic = {**qid_to_candidate_dic,**item}

        ori_qid_to_candidate = {}
        new_qid_to_candidate = {}
        for qid in qid_to_candidate_dic:
            new_qid_to_candidate[qid] = qid_to_candidate_dic[qid][0]
            ori_qid_to_candidate[qid] = qid_to_candidate_dic[qid][1]

        # score
        qids_to_relevant_passageids = load_marcodoc_reference_from_stream(args.real_data_path)
        ori_score = compute_metrics(qids_to_relevant_passageids, ori_qid_to_candidate)
        new_score = compute_metrics(qids_to_relevant_passageids, new_qid_to_candidate)
        output_path = os.path.join(args.output_dir, "reranker_train_result_dict.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(new_qid_to_candidate, f)

        result = {}
        result['ori_score'] = ori_score
        result['new_score'] = new_score

        output_path = os.path.join(args.output_dir, "reranker_train_eval_result.json")
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Compute MRR metric
    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    MaxMRRRank = 10
    all_scores = {}
    MRR = 0
    qids_with_relevant_passages = 0
    ranking = []
    recall_q_top1 = set()
    recall_q_top5 = set()
    recall_q_top20 = set()
    recall_q_top50 = set()
    recall_q_top100 = set()
    recall_q_all = set()

    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            ranking.append(0)
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, MaxMRRRank):
                if candidate_pid[i] in target_pid:
                    MRR += 1.0 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
            for i, pid in enumerate(candidate_pid):
                if pid in target_pid:
                    recall_q_all.add(qid)
                    if i < 100:
                        recall_q_top100.add(qid)
                    if i < 50:
                        recall_q_top50.add(qid)
                    if i < 20:
                        recall_q_top20.add(qid)
                    if i < 5:
                        recall_q_top5.add(qid)
                    if i == 0:
                        recall_q_top1.add(qid)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

    MRR = MRR / len(qids_to_relevant_passageids)
    recall_top1 = len(recall_q_top1) * 1.0 / len(qids_to_relevant_passageids)
    recall_top5 = len(recall_q_top5) * 1.0 / len(qids_to_relevant_passageids)
    recall_top20 = len(recall_q_top20) * 1.0 / len(qids_to_relevant_passageids)
    recall_top50 = len(recall_q_top50) * 1.0 / len(qids_to_relevant_passageids)
    recall_top100 = len(recall_q_top100) * 1.0 / len(qids_to_relevant_passageids)
    recall_all = len(recall_q_all) * 1.0 / len(qids_to_relevant_passageids)
    all_scores['MRR @10'] = MRR
    all_scores["recall@1"] = recall_top1
    all_scores["recall@5"] = recall_top5
    all_scores["recall@20"] = recall_top20
    all_scores["recall@50"] = recall_top50
    all_scores["recall@100"] = recall_top100
    all_scores["recall@all"] = recall_all
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    return all_scores



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
        "--result_data_path",
        default=None,
        type=str,
        required=True,
        help="The output of the dual-encoder model",
    )
    parser.add_argument(
        "--real_data_path",
        default=None,
        type=str,
        required=True,
        help="real label data path",
    )
    parser.add_argument(
        "--query_path",
        default=None,
        type=str,
        required=True,
        help="qid to query string data path",
    )
    parser.add_argument(
        "--corpus_path",
        default="",
        type=str,
        help="corpus path",
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
        "--max_seq_length",
        default=256,
        type=int,
        help="max length of passage",
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
    parser.add_argument(
        "--load_cache",
        action="store_true",
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

