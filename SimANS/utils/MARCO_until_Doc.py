from collections import namedtuple
import os
from tqdm import tqdm
import torch
import random
import math

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


from torch.utils.data import Dataset
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

class Doc_v2Dataset(Dataset):
    def __init__(self, file_path, tokenizer, num_hard_negatives=1, a=0.5, b=0,
                 trainer_id=0, trainer_num=1, is_training=True,
                 corpus_path='', rand_pool=50,
                 p_text=None, p_title=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = self._read_example(file_path, trainer_id, trainer_num)
        self.is_training = is_training
        self.num_hard_negatives = num_hard_negatives
        self.rand_pool = rand_pool
        self.a = a
        self.b = b

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

        pos_pairs_list = sample.pos_id.split(',')
        neg_pairs_list = sample.neg_id.split(',')
        neg_pairs_list = [(int(pair.split()[0]), float(pair.split()[1])) for pair in neg_pairs_list]
        if self.is_training:
            pos_id, pos_score = random.choice(pos_pairs_list).split()
        else:
            pos_id, pos_score = pos_pairs_list[0].split()
        pos_id, pos_score = int(pos_id), float(pos_score)

        if pos_score == 0:
            neg_ids_list = [pair[0] for pair in neg_pairs_list[-self.num_hard_negatives:]]
        else:
            neg_candidates = []
            neg_scores = []
            for pair in neg_pairs_list:
                neg_id, neg_score = pair
                neg_score = math.exp(-(neg_score - pos_score + self.b) ** 2 * self.a)
                neg_candidates.append(neg_id)
                neg_scores.append(neg_score)

            neg_ids_list = set()
            while len(neg_ids_list) < self.num_hard_negatives:
                neg_ids_list = neg_ids_list.union(
                    random.choices(neg_candidates, weights=neg_scores, k=self.num_hard_negatives))
                new_neg_candidates = []
                new_neg_scores = []
                for neg_id, neg_score in zip(neg_candidates, neg_scores):
                    if neg_id not in neg_ids_list:
                        new_neg_candidates.append(neg_id)
                        new_neg_scores.append(neg_score)
                neg_candidates = new_neg_candidates
                neg_scores = new_neg_scores

            neg_ids_list = list(neg_ids_list)[0:self.num_hard_negatives]

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
        c_e_token_ids = torch.LongTensor(
            [temp + [self.tokenizer.pad_token_id] * (512 - len(temp)) for temp in c_e_token_ids])

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
