from collections import namedtuple
import sys
import os
from tqdm import tqdm
import torch
import random
def csv_reader(fd, delimiter='\t', trainer_id=0, trainer_num=1):
    outputs = []
    for i, line in tqdm(enumerate(fd)):
        if i % trainer_num == trainer_id:
            slots = line.rstrip('\n').split(delimiter)
            if len(slots) == 1:
                outputs.append(slots,)
            else:
                outputs.append(slots)
    return outputs

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

class RocketqaDataset(Dataset):
    def __init__(self, file_path, tokenizer,num_hard_negatives=1, trainer_id=0, trainer_num=1, is_training=True):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = _read_tsv(file_path,trainer_id,trainer_num)
        self.is_training = is_training
        self.num_hard_negatives = num_hard_negatives

    def __getitem__(self, index):
        sample = self.data[index]
        query = convert_to_unicode(sample.query)
        title_pos = convert_to_unicode(sample.title_pos)
        para_pos = convert_to_unicode(sample.para_pos)
        title_neg = convert_to_unicode(sample.title_neg)
        para_neg = convert_to_unicode(sample.para_neg)

        title_text_pairs = [[title_pos,para_pos],[title_neg,para_neg]]
        ctx_token_ids = [self.tokenizer.encode(ctx[0], text_pair=ctx[1], add_special_tokens=True,
                                        max_length=128,truncation=True,
                                        pad_to_max_length=False) for ctx in title_text_pairs]
        question_token_ids = self.tokenizer.encode(query,add_special_tokens=True,
                                        max_length=32,truncation=True,
                                        pad_to_max_length=False)
        def remove_special_token(token_id):
            if token_id[-1] == self.tokenizer.sep_token_id:
                return token_id[1:-1]
            else:
                return token_id[1:]
        c_e_token_ids = [question_token_ids + remove_special_token(ctx_token_id) for ctx_token_id in ctx_token_ids]
        
        # padding
        question_token_ids = torch.LongTensor(question_token_ids + [0]*(32-len(question_token_ids)))
        ctx_ids = torch.LongTensor([ctx_token_id + [0]*(128-len(ctx_token_id)) for ctx_token_id in ctx_token_ids])

        c_e_token_ids = torch.LongTensor([temp + [0]*(160-len(temp)) for temp in c_e_token_ids])
        return question_token_ids,ctx_ids,c_e_token_ids
    
    def __len__(self):
        return len(self.data)
    @classmethod
    def get_collate_fn(cls,args):
        def create_biencoder_input2(features):
            q_tensor = torch.stack([feature[0] for feature in features],dim=0)
            doc_tensor = torch.cat([feature[1] for feature in features])
            ctx_tensor_out = torch.cat([feature[2] for feature in features])    
            positive_ctx_indices =[i*2 for i in range(len(features))]

            q_num,d_num = q_tensor.size(0),doc_tensor.size(0)
            tgt_tensor = torch.zeros((d_num), dtype=torch.long)
            tgt_tensor[positive_ctx_indices] = 1
            ctx_tensor_out = ctx_tensor_out.reshape(q_num,d_num//q_num,-1)
            tgt_tensor = tgt_tensor.reshape(q_num,d_num//q_num)
            return {'student': [q_tensor, (q_tensor!= 0).long(), doc_tensor, 
                                (doc_tensor!= 0).long(), positive_ctx_indices],
                    'teacher': [ctx_tensor_out, 
                                (ctx_tensor_out!= 0).long(), tgt_tensor],}
        return create_biencoder_input2

class Rocketqa_v2Dataset(Dataset):
    def __init__(self, file_path, tokenizer,num_hard_negatives=1, max_seq_length=128,
                     trainer_id=0, trainer_num=1, is_training=True,
                     corpus_path = '/quantus-nfs/zh/AN_dpr/data_train/',rand_pool = 50,
                     p_text=None,p_title=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = self._read_example(file_path,trainer_id,trainer_num)
        self.is_training = is_training
        self.num_hard_negatives = num_hard_negatives
        self.rand_pool = rand_pool
        self.max_seq_length=max_seq_length

        self.p_text = self.load_id_text(os.path.join(corpus_path,'para.txt')) if p_text is None else p_text
        self.p_title = self.load_id_text(os.path.join(corpus_path,'para.title.txt')) if p_title is None else p_text
        
    def _read_example(self, input_file, trainer_id=0, trainer_num=1):
        """Reads a tab separated value file."""
        with open(input_file, 'r', encoding='utf8') as f:
            reader = csv_reader(f, trainer_id=trainer_id, trainer_num=trainer_num)
            headers = 'query_id\tquery_string\tpos_id\tneg_id'.split('\t')

            Example = namedtuple('Example', headers)
            examples = []
            for cnt, line in enumerate(reader):
                example = Example(*line)
                examples.append(example)
        return examples

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

        query = convert_to_unicode(sample.query_string)

        pos_ids_list = sample.pos_id.split(',')
        neg_ids_list = sample.neg_id.split(',')
        neg_ids_list = neg_ids_list[:self.rand_pool]
        if self.is_training:
            pos_id = int(random.choice(pos_ids_list))
            random.shuffle(neg_ids_list)
        else:
            pos_id = int(pos_ids_list[0])
        neg_ids_list = neg_ids_list[0:self.num_hard_negatives]
        if len(neg_ids_list)!=self.num_hard_negatives:
            neg_ids_list = (neg_ids_list*self.num_hard_negatives)[:self.num_hard_negatives]

        title_pos =  convert_to_unicode(self.p_title.get(pos_id,'-'))
        para_pos =  convert_to_unicode(self.p_text[pos_id])

        p_neg_list = [[convert_to_unicode(self.p_title.get(int(neg_id),'-')),
                       convert_to_unicode(self.p_text[int(neg_id)])] for neg_id in neg_ids_list]

        title_text_pairs = [[title_pos,para_pos]] + p_neg_list
        ctx_token_ids = [self.tokenizer.encode(ctx[0], text_pair=ctx[1], add_special_tokens=True,
                                        max_length=self.max_seq_length,truncation=True,
                                        pad_to_max_length=False) for ctx in title_text_pairs]
        question_token_ids = self.tokenizer.encode(query,add_special_tokens=True,
                                        max_length=32,truncation=True,
                                        pad_to_max_length=False)
        def remove_special_token(token_id):
            if token_id[-1] == self.tokenizer.sep_token_id:
                return token_id[1:-1]
            else:
                return token_id[1:]
        c_e_token_ids = [question_token_ids + remove_special_token(ctx_token_id) for ctx_token_id in ctx_token_ids]
        
        # padding
        question_token_ids = torch.LongTensor(question_token_ids + [self.tokenizer.pad_token_id]*(32-len(question_token_ids)))
        ctx_ids = torch.LongTensor([ctx_token_id + [self.tokenizer.pad_token_id]*(self.max_seq_length-len(ctx_token_id)) for ctx_token_id in ctx_token_ids])
        c_e_token_ids = torch.LongTensor([temp + [self.tokenizer.pad_token_id]*(self.max_seq_length+32-len(temp)) for temp in c_e_token_ids])

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


class Doc_v2Dataset(Dataset):
    def __init__(self, file_path, tokenizer, num_hard_negatives=1,
                 trainer_id=0, trainer_num=1, is_training=True,
                 corpus_path='/quantus-nfs/zh/AN_dpr/data_train/', rand_pool=50,
                 p_text=None, p_title=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = self._read_example(file_path, trainer_id, trainer_num)
        self.is_training = is_training
        self.num_hard_negatives = num_hard_negatives
        self.rand_pool = rand_pool

        self.p_text = self.load_id_text(os.path.join(corpus_path, 'msmarco-docs.tsv')) if p_text is None else p_text

    def _read_example(self, input_file, trainer_id=0, trainer_num=1):
        """Reads a tab separated value file."""
        with open(input_file, 'r', encoding='utf8') as f:
            reader = csv_reader(f, trainer_id=trainer_id, trainer_num=trainer_num)
            headers = 'query_id\tquery_string\tpos_id\tneg_id'.split('\t')

            Example = namedtuple('Example', headers)
            examples = []
            for cnt, line in enumerate(reader):
                example = Example(*line)
                examples.append(example)
        return examples

    def load_id_text(self, file_name):
        """load tsv files"""
        id_text = {}
        with open(file_name) as inp:
            for line in tqdm(inp):
                line_arr = line.split('\t')
                p_id = int(line_arr[0][1:])  # remove "D"

                url = line_arr[1].rstrip()
                title = line_arr[2].rstrip()
                p_text = line_arr[3].rstrip()
                # NOTE: This linke is copied from ANCE,
                # but I think it's better to use <s> as the separator,
                full_text = url + "<sep>" + title + "<sep>" + p_text
                # keep only first 10000 characters, should be sufficient for any
                # experiment that uses less than 500 - 1k tokens
                full_text = full_text[:10000]

                id_text[p_id] = full_text
        return id_text

    def __getitem__(self, index):
        sample = self.data[index]

        query = convert_to_unicode(sample.query_string)

        pos_ids_list = sample.pos_id.split(',')
        neg_ids_list = sample.neg_id.split(',')
        neg_ids_list = neg_ids_list[:self.rand_pool]
        if self.is_training:
            pos_id = int(random.choice(pos_ids_list))
            random.shuffle(neg_ids_list)
        else:
            pos_id = int(pos_ids_list[0])
        neg_ids_list = neg_ids_list[0:self.num_hard_negatives]

        para_pos = convert_to_unicode(self.p_text[pos_id])

        p_neg_list = [convert_to_unicode(self.p_text[int(neg_id)]) for neg_id in neg_ids_list]

        title_text_pairs = [para_pos] + p_neg_list
        ctx_token_ids = [self.tokenizer.encode(ctx, add_special_tokens=True,
                                               max_length=512, truncation=True,
                                               pad_to_max_length=False) for ctx in title_text_pairs]
        question_token_ids = self.tokenizer.encode(query, add_special_tokens=True,
                                                   max_length=128, truncation=True,
                                                   pad_to_max_length=False)

        def remove_special_token(token_id):
            if token_id[-1] == self.tokenizer.sep_token_id:
                return token_id[1:-1]
            else:
                return token_id[1:]

        c_e_token_ids = [question_token_ids + remove_special_token(ctx_token_id) for ctx_token_id in ctx_token_ids]
        c_e_token_ids = [ele[:512] for ele in c_e_token_ids]

        # padding
        question_token_ids = torch.LongTensor(
            question_token_ids + [self.tokenizer.pad_token_id] * (128 - len(question_token_ids)))
        ctx_ids = torch.LongTensor(
            [ctx_token_id + [self.tokenizer.pad_token_id] * (512 - len(ctx_token_id)) for ctx_token_id in
             ctx_token_ids])
        c_e_token_ids = torch.LongTensor([temp + [self.tokenizer.pad_token_id] * (512 - len(temp)) for temp in c_e_token_ids])

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
            return {'student': [q_tensor, (q_tensor != 1).long(), doc_tensor,
                                (doc_tensor != 1).long(), positive_ctx_indices],
                    'teacher': [ctx_tensor_out,
                                (ctx_tensor_out != 1).long(), tgt_tensor], }

        return create_biencoder_input2

class Rocketqa_v3Dataset(Dataset):
    def __init__(self, file_path, tokenizer, num_hard_negatives=1,
                 trainer_id=0, trainer_num=1, is_training=True,
                 corpus_path='/quantus-nfs/zh/AN_dpr/data_train/', rand_pool=50,
                 p_text=None, p_title=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = self._read_example(file_path, trainer_id, trainer_num)
        self.is_training = is_training
        self.num_hard_negatives = num_hard_negatives
        self.rand_pool = rand_pool
        self.ordinal = True

        self.p_text = self.load_id_text(os.path.join(corpus_path, 'para.txt')) if p_text is None else p_text
        self.p_title = self.load_id_text(os.path.join(corpus_path, 'para.title.txt')) if p_title is None else p_text

    def _read_example(self, input_file, trainer_id=0, trainer_num=1):
        """Reads a tab separated value file."""
        with open(input_file, 'r', encoding='utf8') as f:
            reader = csv_reader(f, trainer_id=trainer_id, trainer_num=trainer_num)
            headers = 'query_id\tquery_string\tpos_id\tneg_id'.split('\t')

            Example = namedtuple('Example', headers)
            examples = []
            for cnt, line in enumerate(reader):
                example = Example(*line)
                examples.append(example)
        return examples

    def load_id_text(self, file_name):
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

        query = convert_to_unicode(sample.query_string)

        pos_ids_list = sample.pos_id.split(',')
        neg_ids_list = sample.neg_id.split(',')
        if self.ordinal:
            neg_ids_list = neg_ids_list[:self.rand_pool]
        else:
            neg_ids_list = neg_ids_list[self.rand_pool:]
        if self.is_training:
            pos_id = int(random.choice(pos_ids_list))
            random.shuffle(neg_ids_list)
        else:
            pos_id = int(pos_ids_list[0])

        neg_ids_list = neg_ids_list[0:self.num_hard_negatives]

        title_pos = convert_to_unicode(self.p_title.get(pos_id, '-'))
        para_pos = convert_to_unicode(self.p_text[pos_id])

        p_neg_list = [[convert_to_unicode(self.p_title.get(int(neg_id), '-')),
                       convert_to_unicode(self.p_text[int(neg_id)])] for neg_id in neg_ids_list]

        title_text_pairs = [[title_pos, para_pos]] + p_neg_list
        ctx_token_ids = [self.tokenizer.encode(ctx[0], text_pair=ctx[1], add_special_tokens=True,
                                               max_length=128, truncation=True,
                                               pad_to_max_length=False) for ctx in title_text_pairs]
        question_token_ids = self.tokenizer.encode(query, add_special_tokens=True,
                                                   max_length=32, truncation=True,
                                                   pad_to_max_length=False)

        def remove_special_token(token_id):
            if token_id[-1] == self.tokenizer.sep_token_id:
                return token_id[1:-1]
            else:
                return token_id[1:]

        c_e_token_ids = [question_token_ids + remove_special_token(ctx_token_id) for ctx_token_id in ctx_token_ids]

        # padding
        question_token_ids = torch.LongTensor(
            question_token_ids + [self.tokenizer.pad_token_id] * (32 - len(question_token_ids)))
        ctx_ids = torch.LongTensor(
            [ctx_token_id + [self.tokenizer.pad_token_id] * (128 - len(ctx_token_id)) for ctx_token_id in
             ctx_token_ids])
        c_e_token_ids = torch.LongTensor(
            [temp + [self.tokenizer.pad_token_id] * (160 - len(temp)) for temp in c_e_token_ids])

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
            return {'student': [q_tensor, (q_tensor != 0).long(), doc_tensor,
                                (doc_tensor != 0).long(), positive_ctx_indices],
                    'teacher': [ctx_tensor_out,
                                (ctx_tensor_out != 0).long(), tgt_tensor], }

        return create_biencoder_input2
