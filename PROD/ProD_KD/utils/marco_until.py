from collections import namedtuple
import sys
import os
from tqdm import tqdm, trange
import torch
import gzip
import copy
import json
import random
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)
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

def _read_tsv(input_file, trainer_id=0, trainer_num=1):
    """Reads a tab separated value file."""
    with open(input_file, 'r', encoding='utf8') as f:
        reader = csv_reader(f, trainer_id=trainer_id, trainer_num=trainer_num)
        headers = 'query\ttitle_pos\tpara_pos\ttitle_neg\tpara_neg\tlabel'.split('\t')
        text_indices = [
            index for index, h in enumerate(headers) if h != "label"
        ]
        Example = namedtuple('Example', headers)

        examples = []
        for cnt, line in enumerate(reader):
            example = Example(*line)
            examples.append(example)
        return examples

from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
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

def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question
def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    return ctx_text

class Rocketqa_v2Dataset(Dataset):
    def __init__(self, file_path, tokenizer, num_hard_negatives=1,
                     is_training=True, corpus_path='/colab_space/fanshuai/KDmarco/coCondenser-marco/marco',
                     max_seq_length=256, max_q_length=32, p_text=None, p_title=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = self.load_data()
        self.is_training = is_training
        self.num_hard_negatives = num_hard_negatives
        self.max_seq_length = max_seq_length
        self.max_q_length = max_q_length

        self.p_text = self.load_id_text(os.path.join(corpus_path,'para.txt')) if p_text is None else p_text
        self.p_title = self.load_id_text(os.path.join(corpus_path,'para.title.txt')) if p_title is None else p_text
        logger.info("Total data ready to train...")
        
    def load_data(self):
        with open(self.file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
            print('Aggregated data size: {}'.format(len(data)))
        # filter those without positive ctx
        pre_data = [r for r in data if len(r["pos_id"]) > 0]
        logger.info("cleaned data size: {} after positive ctx".format(len(pre_data)))
        pre_data = [r for r in pre_data if len(r['neg_id']) > 0]
        logger.info("Total cleaned data size: {}".format(len(pre_data)))
        return pre_data

    def load_id_text(self,file_name):
        """load tsv files"""
        id_text = {}
        with open(file_name) as inp:
            for line in tqdm(inp):
                line = line.strip()
                id, text = line.split('\t')
                id = int(id)
                id_text[id] = text
        return id_text

    def __getitem__(self, index):
        sample = self.data[index]

        query = convert_to_unicode(sample['query_string'])
        query = normalize_question(query)

        pos_ids_list = sample['pos_id']
        neg_ids_list = sample['neg_id']
        if self.is_training:
            # pos_id = int(random.choice(pos_ids_list))
            random.shuffle(neg_ids_list)

        if len(neg_ids_list) < self.num_hard_negatives:
            neg_ids_list = neg_ids_list * self.num_hard_negatives

        pos_id = int(pos_ids_list[0])
        neg_ids_list = neg_ids_list[0:self.num_hard_negatives]

        title_pos = convert_to_unicode(self.p_title.get(pos_id,'-'))
        para_pos = convert_to_unicode(self.p_text[pos_id])

        p_neg_list = [[convert_to_unicode(self.p_title.get(int(neg_id),'-')),
                       convert_to_unicode(self.p_text[int(neg_id)])] for neg_id in neg_ids_list]

        title_text_pairs = [[title_pos,para_pos]] + p_neg_list
        ctx_token_ids = [self.tokenizer.encode(ctx[0], text_pair=ctx[1], add_special_tokens=True,
                                        max_length=self.max_seq_length,truncation=True,
                                        pad_to_max_length=False) for ctx in title_text_pairs]
        question_token_ids = self.tokenizer.encode(query,add_special_tokens=True,
                                        max_length=self.max_q_length,truncation=True,
                                        pad_to_max_length=False)
        def remove_special_token(token_id):
            if token_id[-1] == self.tokenizer.sep_token_id:
                return token_id[1:-1]
            else:
                return token_id[1:]
        c_e_token_ids = [question_token_ids + remove_special_token(ctx_token_id) for ctx_token_id in ctx_token_ids]
        
        # padding
        question_token_ids = torch.LongTensor(question_token_ids + [self.tokenizer.pad_token_id]*(self.max_q_length-len(question_token_ids)))
        ctx_ids = torch.LongTensor([ctx_token_id + [self.tokenizer.pad_token_id]*(self.max_seq_length-len(ctx_token_id)) for ctx_token_id in ctx_token_ids])
        c_e_token_ids = torch.LongTensor([temp + [self.tokenizer.pad_token_id]*(self.max_seq_length + self.max_q_length-len(temp)) for temp in c_e_token_ids])

        return question_token_ids,ctx_ids,c_e_token_ids
    
    def __len__(self):
        return len(self.data)
    @classmethod
    def get_collate_fn(cls,args):
        def create_biencoder_input2(features):
            doc_per_question = features[0][1].size(0)
            q_tensor = torch.stack([feature[0] for feature in features],dim=0)
            doc_tensor = torch.cat([feature[1] for feature in features])
            ctx_tensor_out = torch.cat([feature[2] for feature in features])    

            positive_ctx_indices =[i*doc_per_question for i in range(len(features))]

            q_num,d_num = q_tensor.size(0),doc_tensor.size(0)
            tgt_tensor = torch.zeros((d_num), dtype=torch.long)
            tgt_tensor[positive_ctx_indices] = 1
            ctx_tensor_out = ctx_tensor_out.reshape(q_num,d_num//q_num,-1)
            tgt_tensor = tgt_tensor.reshape(q_num,d_num//q_num)
            return {'retriever': [q_tensor, (q_tensor!= 0).long(), doc_tensor, 
                                (doc_tensor!= 0).long(), positive_ctx_indices],
                    'reranker': [ctx_tensor_out, 
                                (ctx_tensor_out!= 0).long(), tgt_tensor],}
        return create_biencoder_input2


class MarcoDoc_Dataset(Dataset):
    def __init__(self, file_path, tokenizer, num_hard_negatives=1,
                 is_training=True, corpus_path='/colab_space/fanshuai/KDmarco/MarcoDoc',
                 max_seq_length=256, max_q_length=32):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = self.load_data()
        self.is_training = is_training
        self.num_hard_negatives = num_hard_negatives
        self.max_seq_length = max_seq_length
        self.max_q_length = max_q_length

        self.p_fulltext = self.load_id_text(os.path.join(corpus_path, 'msmarco-docs.tsv'))
        logger.info("Total data ready to train...")

    def load_data(self):
        with open(self.file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
            print('Aggregated data size: {}'.format(len(data)))
        # filter those without positive ctx
        pre_data = [r for r in data if len(r["pos_id"]) > 0]
        logger.info("cleaned data size: {} after positive ctx".format(len(pre_data)))
        pre_data = [r for r in pre_data if len(r['neg_id']) > 0]
        logger.info("Total cleaned data size: {}".format(len(pre_data)))
        return pre_data

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

    def __getitem__(self, index):
        sample = self.data[index]

        query = convert_to_unicode(sample['query_string'])
        query = normalize_question(query)

        pos_ids_list = sample['pos_id']
        neg_ids_list = sample['neg_id']
        if self.is_training:
            # pos_id = int(random.choice(pos_ids_list))
            random.shuffle(neg_ids_list)

        if len(neg_ids_list) < self.num_hard_negatives:
            neg_ids_list = neg_ids_list * self.num_hard_negatives

        pos_id = int(pos_ids_list[0])
        neg_ids_list = neg_ids_list[0:self.num_hard_negatives]

        pos_doc_text = convert_to_unicode(self.p_fulltext[pos_id])

        neg_doc_text = [self.p_fulltext[neg_id] for neg_id in neg_ids_list]

        all_doc_text = [pos_doc_text] + neg_doc_text
        ctx_token_ids = [self.tokenizer.encode(ctx, add_special_tokens=True,
                                               max_length=self.max_seq_length, truncation=True,
                                               pad_to_max_length=False) for ctx in all_doc_text]
        question_token_ids = self.tokenizer.encode(query, add_special_tokens=True,
                                                   max_length=self.max_q_length, truncation=True,
                                                   pad_to_max_length=False)

        def remove_special_token(token_id):
            if token_id[-1] == self.tokenizer.sep_token_id:
                return token_id[1:-1]
            else:
                return token_id[1:]

        c_e_token_ids = [question_token_ids + remove_special_token(ctx_token_id) for ctx_token_id in ctx_token_ids]

        # padding
        question_token_ids = torch.LongTensor(
            question_token_ids + [self.tokenizer.pad_token_id] * (self.max_q_length - len(question_token_ids)))
        ctx_ids = torch.LongTensor(
            [ctx_token_id + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(ctx_token_id)) for ctx_token_id
             in ctx_token_ids])
        c_e_token_ids = torch.LongTensor(
            [temp + [self.tokenizer.pad_token_id] * (self.max_seq_length + self.max_q_length - len(temp)) for temp in
             c_e_token_ids])

        return question_token_ids, ctx_ids, c_e_token_ids

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_collate_fn(cls, args):
        def create_biencoder_input2(features):
            doc_per_question = features[0][1].size(0)
            q_tensor = torch.stack([feature[0] for feature in features], dim=0)
            doc_tensor = torch.cat([feature[1] for feature in features])
            ctx_tensor_out = torch.cat([feature[2] for feature in features])

            positive_ctx_indices = [i * doc_per_question for i in range(len(features))]

            q_num, d_num = q_tensor.size(0), doc_tensor.size(0)
            tgt_tensor = torch.zeros((d_num), dtype=torch.long)
            tgt_tensor[positive_ctx_indices] = 1
            ctx_tensor_out = ctx_tensor_out.reshape(q_num, d_num // q_num, -1)
            tgt_tensor = tgt_tensor.reshape(q_num, d_num // q_num)
            return {'retriever': [q_tensor, (q_tensor != 0).long(), doc_tensor,
                                  (doc_tensor != 0).long(), positive_ctx_indices],
                    'reranker': [ctx_tensor_out,
                                 (ctx_tensor_out != 0).long(), tgt_tensor], }

        return create_biencoder_input2