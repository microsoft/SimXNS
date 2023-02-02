import sys

sys.path += ['../']
import pandas as pd
from sklearn.metrics import roc_curve, auc
import gzip
import copy
import torch
from torch import nn
import torch.distributed as dist
from tqdm import tqdm, trange
import os
from os import listdir
from os.path import isfile, join
import json
import logging
import random
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
#  
from multiprocessing import Process
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import re
from typing import List, Set, Dict, Tuple, Callable, Iterable, Any
import collections
logger = logging.getLogger(__name__)
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])
class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]

class InputFeaturesPair(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(
            self,
            input_ids_a,
            attention_mask_a=None,
            token_type_ids_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            token_type_ids_b=None,
            label=None):
        self.input_ids_a = input_ids_a
        self.attention_mask_a = attention_mask_a
        self.token_type_ids_a = token_type_ids_a

        self.input_ids_b = input_ids_b
        self.attention_mask_b = attention_mask_b
        self.token_type_ids_b = token_type_ids_b

        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def getattr_recursive(obj, name):
    for layer in name.split("."):
        if hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj


def barrier_array_merge(
        args,
        data_array,
        merge_axis=0,
        prefix="",
        load_cache=False,
        only_load_in_master=False):
    # data array: [B, any dimension]
    # merge alone one axis

    if args.local_rank == -1:
        return data_array

    if not load_cache:
        rank = args.rank
        if is_first_worker():
            if not os.path.exists(args.cache_dir):
                os.makedirs(args.cache_dir)

        dist.barrier()  # directory created
        pickle_path = os.path.join(
            args.cache_dir,
            "{1}_data_obj_{0}.pb".format(
                str(rank),
                prefix))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(data_array, handle, protocol=4)

        # make sure all processes wrote their data before first process
        # collects it
        dist.barrier()

    data_array = None

    data_list = []

    # return empty data
    if only_load_in_master:
        if not is_first_worker():
            dist.barrier()
            return None

    for i in range(
            args.world_size):  # TODO: dynamically find the max instead of HardCode
        pickle_path = os.path.join(
            args.cache_dir,
            "{1}_data_obj_{0}.pb".format(
                str(i),
                prefix))
        try:
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                data_list.append(b)
        except BaseException:
            continue

    data_array_agg = np.concatenate(data_list, axis=merge_axis)
    dist.barrier()
    return data_array_agg


def pad_ids(input_ids, attention_mask, token_type_ids, max_length,
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            mask_padding_with_zero=True):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length
    padding_type = [pad_token_segment_id] * padding_length
    padding_attention = [0 if mask_padding_with_zero else 1] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_type_ids = token_type_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
            attention_mask = padding_attention + attention_mask
            token_type_ids = padding_type + token_type_ids
        else:
            input_ids = input_ids + padding_id
            attention_mask = attention_mask + padding_attention
            token_type_ids = token_type_ids + padding_type

    return input_ids, attention_mask, token_type_ids


# to reuse pytrec_eval, id must be string
def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def concat_key(all_list, key, axis=0):
    return np.concatenate([ele[key] for ele in all_list], axis=axis)


def get_checkpoint_no(checkpoint_path):
    nums = re.findall(r'\d+', checkpoint_path)
    return int(nums[-1]) if len(nums) > 0 else 0


def get_latest_ann_data(ann_data_path):
    ANN_PREFIX = "train_ann_data_"
    if not os.path.exists(ann_data_path):
        return -1, None
    files = list(next(os.walk(ann_data_path))[2])
    num_start_pos = len(ANN_PREFIX)
    data_no_list = [int(s[num_start_pos:])
                    for s in files if s[:num_start_pos] == ANN_PREFIX]
    if len(data_no_list) > 0:
        data_no = max(data_no_list)
        return data_no, os.path.join(ann_data_path, "train_ann_data_" + str(data_no))
    return -1, None

def numbered_byte_file_generator(base_path, file_no, record_size):
    for i in range(file_no):
        with open('{}_split{}'.format(base_path, i), 'rb') as f:
            while True:
                b = f.read(record_size)
                if not b:
                    # eof
                    break
                yield b


class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(
                meta['embedding_size']) * self.dtype.itemsize + 4
        if seed >= 0:
            self.ix_array = np.random.RandomState(
                seed).permutation(self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError(
                "Index {} is out of bound for cached embeddings of size {}".format(
                    key, self.total_number))
        self.f.seek(key * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number


class StreamingDataset(IterableDataset):
    def __init__(self, elements, fn, distributed=True):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas = -1
        self.distributed = distributed

    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            print("Not running in distributed mode")
        for i, element in enumerate(self.elements):
            if self.distributed and self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            records = self.fn(element, i)
            for rec in records:
                yield rec

def GetProcessingFn(args, query=False):
    def fn(vals, i):
        passage_len, passage = vals
        max_len = args.max_seq_length

        pad_len = max(0, max_len - passage_len)
        token_type_ids = [0] * passage_len + [0] * pad_len
        attention_mask = passage != 0

        passage_collection = [(i, passage, attention_mask, token_type_ids)]

        query2id_tensor = torch.tensor([f[0] for f in passage_collection], dtype=torch.long)
        all_input_ids_a = torch.tensor([f[1] for f in passage_collection], dtype=torch.int)
        all_attention_mask_a = torch.tensor([f[2] for f in passage_collection], dtype=torch.bool)
        all_token_type_ids_a = torch.tensor([f[3] for f in passage_collection], dtype=torch.uint8)

        dataset = TensorDataset(all_input_ids_a, all_attention_mask_a, all_token_type_ids_a, query2id_tensor)

        return [ts for ts in dataset]

    return fn

def GetTrainingDataProcessingFn(args, query_cache, passage_cache, shuffle=True):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        query_data = GetProcessingFn(args, query=True)(query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(args, query=False)(passage_cache[pos_pid], pos_pid)[0]

        if shuffle:
            random.shuffle(neg_pids)

        neg_data = GetProcessingFn(args, query=False)(passage_cache[neg_pids[0]], neg_pids[0])[0]
        yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2])
        yield (query_data[0], query_data[1], query_data[2], neg_data[0], neg_data[1], neg_data[2])

    return fn

def GetTrainingDataProcessingFn_zh(args, query_cache, passage_cache, shuffle=True):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(args, query=True)(query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(args, query=False)(passage_cache[pos_pid], pos_pid)[0]

        if shuffle:
            random.shuffle(neg_pids)

        neg_data = GetProcessingFn(args, query=False)(passage_cache[neg_pids[0]], neg_pids[0])[0]
        yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2])
        yield (query_data[0], query_data[1], query_data[2], neg_data[0], neg_data[1], neg_data[2])

    return fn

def GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache, shuffle=True):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(args, query=True)(query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(args, query=False)(passage_cache[pos_pid], pos_pid)[0]

        if shuffle:
            random.shuffle(neg_pids)

        neg_data = GetProcessingFn(args, query=False)(passage_cache[neg_pids[0]], neg_pids[0])[0]
        yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2],
               neg_data[0], neg_data[1], neg_data[2])

    return fn
    
class OriginDataset(Dataset):
    def __init__(self, elements, fn):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas = -1

    def __getitem__(self, index):
        return self.fn(self.elements[index], index)
    def __len__(self):
        return len(self.elements)
def GetDevDataProcessingFn_ZH(args, query_cache, passage_cache,number_neg = 20, shuffle=False):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])

        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        if len(line_arr)<3:
            hard_neg_pids = [random.randint(0, 20000000) for i in range(number_neg)]
            if pos_pid in hard_neg_pids:
                hard_neg_pids.remove(pos_pid)
        else:
            hard_neg_pids = line_arr[3].split(',')
            hard_neg_pids = [int(hard_neg_pid) for hard_neg_pid in hard_neg_pids]

        query_data = GetProcessingFn(args, query=True)(query_cache[qid], qid)[0]

        pos_data = GetProcessingFn(args, query=False)(passage_cache[pos_pid], pos_pid)[0]

        if shuffle:
            random.shuffle(neg_pids)

        
        neg_data = [GetProcessingFn(args, query=False)(passage_cache[neg_pid], neg_pid)[0] for neg_pid in neg_pids[:number_neg//2]]
        hard_neg_data = [GetProcessingFn(args, query=False)(passage_cache[hard_neg_pid], hard_neg_pid)[0] for hard_neg_pid in hard_neg_pids[:number_neg//2]]
        return (query_data,pos_data,neg_data,hard_neg_data)
    return fn
    
def dev_batcher(device, is_training=False):
    def batcher_f(features):
        question_tensors = []
        question_mask = []
        ctx_tensors = []
        ctx_mask = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []
        for feature in features:
            question = feature[0][0]
            question_att_mask = feature[0][1]
            pos_ctxs_tensor = [feature[1][0]]
            neg_ctxs_tensor = [neg_ctx[0] for neg_ctx in feature[2]]
            hard_ctxs_tensor = [hard_ctx[0] for hard_ctx in feature[3]]
            pos_att_mask = [feature[1][1]]
            neg_ctxs_att_mask = [neg_ctx[1] for neg_ctx in feature[2]]
            hard_ctxs_neg_att_mask = [hard_ctx[1] for hard_ctx in feature[3]]

            all_ctxs = pos_ctxs_tensor + hard_ctxs_tensor + neg_ctxs_tensor
            all_ctxs_msk = pos_att_mask + hard_ctxs_neg_att_mask + neg_ctxs_att_mask

            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_ctxs_tensor)

            current_ctxs_len = len(ctx_tensors)
            ctx_tensors.extend(all_ctxs)
            ctx_mask.extend(all_ctxs_msk)

            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                    current_ctxs_len + hard_negatives_start_idx,
                    current_ctxs_len + hard_negatives_end_idx,
                )
                ]
            )
            question_tensors.append(question)
            question_mask.append(question_att_mask)
        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctxs_mask_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_mask], dim=0)
        questions_mask_tensor = torch.cat([q.view(1, -1) for q in question_mask], dim=0)

        max_q_len = questions_mask_tensor.sum(-1).max()
        max_d_len = questions_mask_tensor.sum(-1).max()
        questions_tensor = questions_tensor[:, :max_q_len]
        questions_mask_tensor = questions_mask_tensor[:, :max_q_len]
        ctxs_tensor = ctxs_tensor[:, :max_d_len]
        ctxs_mask_tensor = ctxs_mask_tensor[:, :max_d_len]
        return questions_tensor, questions_mask_tensor, ctxs_tensor, ctxs_mask_tensor, positive_ctx_indices, hard_neg_ctx_indices

    return batcher_f


def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples
    
def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question
def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    return ctx_text

import unicodedata
def _normalize(text):
    return unicodedata.normalize('NFD', text)

class TraditionDataset(Dataset):
    def __init__(self, file_path, tokenizer,num_hard_negatives=1, num_easy_negatives=0, is_training=True,
            max_seq_length =256 ,max_q_length=32,shuffle_positives=False):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = self.load_data()
        self.is_training = is_training
        self.num_hard_negatives = num_hard_negatives
        self.num_easy_negatives = num_easy_negatives
        self.max_seq_length = max_seq_length
        self.max_q_length = max_q_length
        self.shuffle_positives = shuffle_positives
    def load_data(self):
        with open(self.file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
            print('Aggregated data size: {}'.format(len(data)))
        # filter those without positive ctx
        pre_data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("cleaned data size: {} after positive ctx".format(len(pre_data)))
        pre_data = [r for r in pre_data if len(r['hard_negative_ctxs']) > 0]
        logger.info("Total cleaned data size: {}".format(len(pre_data)))
        return pre_data

    def __getitem__(self, index):
        json_sample = self.data[index]
        query = normalize_question(json_sample["question"])
        positive_ctxs = json_sample["positive_ctxs"]
        negative_ctxs = (
            json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        )
        hard_negative_ctxs = (
            json_sample["hard_negative_ctxs"]
            if "hard_negative_ctxs" in json_sample
            else []
        )
        easy_negative_ctxs = (
            json_sample["easy_negative_ctxs"]
            if "easy_negative_ctxs" in json_sample
            else []
        )
        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs + easy_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        def create_passage(ctx: dict):
            return BiEncoderPassage( ctx["text"],
                ctx["title"],
            )
        positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]
        easy_negative_passages = [create_passage(ctx) for ctx in easy_negative_ctxs]
        if self.is_training:
            random.shuffle(hard_negative_passages)
            random.shuffle(easy_negative_passages)
        if len(hard_negative_passages) < self.num_hard_negatives:
            hard_negative_passages = hard_negative_passages*self.num_hard_negatives
        if len(easy_negative_passages) < self.num_easy_negatives:
            easy_negative_passages = easy_negative_passages*self.num_easy_negatives
        hard_neg_ctxs = hard_negative_passages[0:self.num_hard_negatives]
        easy_neg_ctxs = easy_negative_passages[0:self.num_easy_negatives]
        if self.shuffle_positives:
            positive_passagese_ctx = random.choice(positive_passages)
        else:
            positive_passagese_ctx = positive_passages[0]

        if self.num_easy_negatives != 0:
            ctxs = [positive_passagese_ctx] + hard_neg_ctxs + easy_neg_ctxs
        else:
            ctxs = [positive_passagese_ctx] + hard_neg_ctxs
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
        
        answers = [self.tokenizer.encode(_normalize(single_answer),add_special_tokens=False) for single_answer in json_sample['answers']]
        start_ctx = len(question_token_ids)
        start_end = [[start_ctx, len(c_e_token_id)] for c_e_token_id in c_e_token_ids]
        return question_token_ids,ctx_token_ids,c_e_token_ids,answers,start_end
    
    def __len__(self):
        return len(self.data)
    @classmethod
    def get_collate_fn(cls,args):
        def create_biencoder_input2(features):
            q_list = []
            d_list = []
            c_e_input_list = []
            positive_ctx_indices = []
            c_e_ctx_start_end = []
            for index, feature in enumerate(features):
                positive_ctx_indices.append(len(d_list))
                q_list.append(feature[0]) 
                d_list.extend(feature[1])
                c_e_input_list.extend(feature[2])
                c_e_ctx_start_end.append(feature[4])
            max_q_len = max([len(q) for q in q_list])
            max_d_len = max([len(d) for d in d_list])
            max_c_e_len = max([len(d) for d in c_e_input_list])
            q_list = [q+[0]*(max_q_len-len(q)) for q in q_list]
            d_list = [d+[0]*(max_d_len-len(d)) for d in d_list]
            c_e_list = [c_e+[0]*(max_c_e_len-len(c_e)) for c_e in c_e_input_list]

            q_tensor = torch.LongTensor(q_list)
            doc_tensor = torch.LongTensor(d_list)
            ctx_tensor_out = torch.LongTensor(c_e_list)
            q_num,d_num = len(q_list),len(d_list)
            tgt_tensor = torch.zeros((len(c_e_list)), dtype=torch.long)
            tgt_tensor[positive_ctx_indices] = 1
            ctx_tensor_out = ctx_tensor_out.reshape(q_num,d_num//q_num,-1)
            tgt_tensor = tgt_tensor.reshape(q_num,d_num//q_num)
            return {'reranker': [ctx_tensor_out, 
                                (ctx_tensor_out!= 0).long(), tgt_tensor],
                    'retriever': [q_tensor, (q_tensor!= 0).long(), doc_tensor, 
                                (doc_tensor!= 0).long(), positive_ctx_indices],
                    "answers": [feature[3] for feature in features],
                    "reranker_ctx_start_end":c_e_ctx_start_end}
        return create_biencoder_input2

def tokenize_to_file_v2(args, i, num_process, in_path, out_path, line_fn):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    list_sum = []
    with open(in_path, 'r', encoding='utf-8') as in_f:
        for idx, line in tqdm(enumerate(in_f)):
            if idx==0 or idx % num_process != i:
                continue
            p_id,token_ids = line_fn(args, line, tokenizer)
            list_sum.append((p_id,token_ids))
    with open('{}_split{}'.format(out_path, i), 'wb') as out_f:
        pickle.dump(list_sum, out_f)

def tokenize_to_file(args, i, num_process, in_path, out_path, line_fn):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt',
                                                                                     encoding='utf8') as in_f, \
            open('{}_split{}'.format(out_path, i), 'wb') as out_f:
        for idx, line in enumerate(in_f):
            if idx % num_process != i:
                continue
            out_f.write(line_fn(args, line, tokenizer))

def multi_file_process(args, num_process, in_path, out_path, line_fn):
    processes = []
    for i in range(num_process):
        p = Process(
            target=tokenize_to_file,
            args=(
                args,
                i,
                num_process,
                in_path,
                out_path,
                line_fn,
            ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [data]

    world_size = dist.get_world_size()
    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
