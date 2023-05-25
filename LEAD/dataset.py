from torch.utils.data import Dataset, DataLoader, RandomSampler
import os
import random
import torch
import json
import logging
import collections
import csv
import sys
from tqdm import tqdm
import unicodedata

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])

def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question

def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    return ctx_text

def _normalize(text):
    return unicodedata.normalize('NFD', text)

def load_data_academic(args):
    passage_path = args.passage_path
    if not os.path.exists(passage_path):
        logger.info(f'{passage_path} does not exist')
        return
    logger.info(f'Loading passages from: {passage_path}')
    passages = []
    with open(passage_path) as fin:
        reader = csv.reader(fin, delimiter='\t')
        for k, row in enumerate(reader):
            if not row[0] == 'id':
                try:
                    passages.append((int(row[0]) - 1, row[1], row[2]))
                except:
                    logger.warning(f'The following input line has not been correctly loaded: {row}')

    test_questions = []
    test_answers = []
    logger.info("Loading test answers")
    with open(args.test_file, "r", encoding="utf-8") as ifile:
        # file format: question, answers
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            test_questions.append(row[0])
            test_answers.append(eval(row[1]))

    return (passages, test_questions, test_answers)

class TraditionDataset(Dataset):
    def __init__(self, args, file_path, tokenizer, num_hard_negatives=1, is_training=True,
                 max_seq_length=256, max_q_length=64, shuffle_positives=False):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.data = self.load_data()
        self.passage = self.load_corpus_passage(args.passage_path)
        self.is_training = is_training
        self.num_hard_negatives = num_hard_negatives
        self.max_doc_length = max_seq_length
        self.max_query_length = max_q_length
        self.shuffle_positives = shuffle_positives

        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tokenizer.convert_tokens_to_ids('[unused1]')
        self.D_marker_token, self.D_marker_token_id = '[D]', self.tokenizer.convert_tokens_to_ids('[unused2]')
        self.mask_token, self.mask_token_id = self.tokenizer.mask_token, self.tokenizer.mask_token_id
        self.pad_token, self.pad_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id

    def load_data(self):
        with open(self.file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
            print('Aggregated data size: {}'.format(len(data)))
        # filter those without positive ctx
        pre_data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("cleaned data size: {} after positive ctx".format(len(pre_data)))
        pre_data = [r for r in pre_data if len(r['hard_negative_ctxs']) > 0]
        logger.info("Total cleaned data size: {}".format(len(pre_data)))
        del data
        return pre_data

    def load_corpus_passage(self, passage_path):
        if not os.path.exists(passage_path):
            logger.info(f'{passage_path} does not exist')
            return
        logger.info(f'Loading passages from: {passage_path}')
        passages = {}
        with open(passage_path) as fin:
            reader = csv.reader(fin, delimiter='\t')
            epoch_iterator = tqdm(reader, desc="Iteration")
            for k, row in enumerate(epoch_iterator):
                if not row[0] == 'id':
                    try:
                        passages[int(row[0])-1] = {}
                        passages[int(row[0])-1]['text'] = row[1]
                        passages[int(row[0])-1]['title'] = row[2]
                    except:
                        logger.warning(f'The following input line has not been correctly loaded: {row}')
        return passages

    def __getitem__(self, index):
        json_sample = self.data[index]
        query = normalize_question(json_sample["question"])
        positive_ctxs = [self.passage[int(elem)] for elem in json_sample["positive_ctxs"]]
        hard_negative_ctxs = (
            [self.passage[int(elem)] for elem in json_sample["hard_negative_ctxs"]]
            if "hard_negative_ctxs" in json_sample
            else []
        )
        positive_passages = [ctx for ctx in positive_ctxs]
        hard_negative_passages = [ctx for ctx in hard_negative_ctxs]
        if self.is_training:
            random.shuffle(hard_negative_passages)
        if len(hard_negative_passages) < self.num_hard_negatives:
            hard_negative_passages = hard_negative_passages * self.num_hard_negatives
        hard_neg_ctxs = hard_negative_passages[0:self.num_hard_negatives]
        if self.shuffle_positives:
            positive_passagese_ctx = random.choice(positive_passages)
        else:
            positive_passagese_ctx = positive_passages[0]
        ctxs = [positive_passagese_ctx] + hard_neg_ctxs
        ctx_token_ids = [self.tokenizer.encode(ctx['title'], text_pair=ctx['text'].strip(), add_special_tokens=True,
                                               max_length=self.max_doc_length, truncation=True,
                                               pad_to_max_length=False) for ctx in ctxs]
        ctx_token_ids_col = [[self.D_marker_token_id] + elem[1:] for elem in ctx_token_ids]

        question_token_ids = self.tokenizer.encode(query)[:self.max_query_length]
        question_token_ids_col = [self.Q_marker_token_id] + question_token_ids[1:]

        def remove_special_token(token_id):
            if token_id[-1] == self.tokenizer.sep_token_id:
                return token_id[1:-1]
            else:
                return token_id[1:]

        c_e_token_ids = [question_token_ids + remove_special_token(ctx_token_id) for ctx_token_id in ctx_token_ids]

        answers = None
        # answers = [self.tokenizer.encode(_normalize(single_answer), add_special_tokens=False) for single_answer in
        #            json_sample['answers']]
        start_ctx = len(question_token_ids)
        start_end_ce = [[start_ctx, len(c_e_token_id)] for c_e_token_id in c_e_token_ids]
        start_end_de = [[start_ctx, len(elem)] for elem in ctx_token_ids]
        return question_token_ids, question_token_ids_col, ctx_token_ids, ctx_token_ids_col, c_e_token_ids, answers, start_end_ce, start_end_de

    def __len__(self):
        return len(self.data)
        # return 100

    def get_collate_fn(self, features):
        q_list = []
        q_list_col = []
        d_list = []
        d_list_col = []
        c_e_input_list = []
        positive_ctx_indices = []
        c_e_ctx_start_end = []
        d_e_ctx_start_end = []
        for index, feature in enumerate(features):
            positive_ctx_indices.append(len(d_list))
            q_list.append(feature[0].copy())
            q_list_col.append(feature[1].copy())
            d_list.extend(feature[2].copy())
            d_list_col.extend(feature[3].copy())
            c_e_input_list.extend(feature[4].copy())
            c_e_ctx_start_end.append(feature[6].copy())
            d_e_ctx_start_end.append(feature[7].copy())

        max_q_len = self.max_query_length
        max_d_len = self.max_doc_length
        max_c_e_len = max([len(d) for d in c_e_input_list])

        q_list = [q + [self.pad_token_id] * (max_q_len - len(q)) for q in q_list]
        q_list_col = [q + [self.mask_token_id] * (max_q_len - len(q)) for q in q_list_col] # padding

        d_list = [d + [self.pad_token_id] * (max_d_len - len(d)) for d in d_list]
        d_list_col = [d + [self.pad_token_id] * (max_d_len - len(d)) for d in d_list_col] # padding

        c_e_list = [c_e + [self.pad_token_id] * (max_c_e_len - len(c_e)) for c_e in c_e_input_list]

        q_tensor = torch.LongTensor(q_list)
        q_tensor_col = torch.LongTensor(q_list_col)

        doc_tensor = torch.LongTensor(d_list)
        doc_tensor_col = torch.LongTensor(d_list_col)

        ctx_tensor_out = torch.LongTensor(c_e_list)
        q_num, d_num = len(q_list), len(d_list)
        tgt_tensor = torch.zeros((len(c_e_list)), dtype=torch.long)
        tgt_tensor[positive_ctx_indices] = 1
        ctx_tensor_out = ctx_tensor_out.reshape(q_num, d_num // q_num, -1)
        tgt_tensor = tgt_tensor.reshape(q_num, d_num // q_num)

        c_e_ctx_start_end = torch.LongTensor(c_e_ctx_start_end)
        # c_e_ctx_start_end = c_e_ctx_start_end.view(-1, c_e_ctx_start_end.shape[2])
        d_e_ctx_start_end = torch.LongTensor(d_e_ctx_start_end)
        # d_e_ctx_start_end = d_e_ctx_start_end.view(-1, d_e_ctx_start_end.shape[2])

        return {'reranker': [ctx_tensor_out,
                             (ctx_tensor_out != 0).long(), tgt_tensor],
                'retriever': [q_tensor, (q_tensor != 0).long(), doc_tensor, (doc_tensor != 0).long(),
                              q_tensor_col, (q_tensor_col != 0).long(), doc_tensor_col, (doc_tensor_col != 0).long(), positive_ctx_indices],
                "answers": [feature[5] for feature in features],
                "ce_ctx_start_end": c_e_ctx_start_end,
                "de_ctx_start_end": d_e_ctx_start_end}

class Coldual_Inference_Academic(Dataset):
    def __init__(self, args, model_type, query_doc_pair, passages, test_questions, tokenizer, max_seq_length=256, max_q_length=32):
        self.max_doc_length = args.max_doc_length
        self.max_query_length = args.max_query_length

        self.questions = test_questions
        self.passages = passages

        ## ColBERT
        self.model_type = model_type
        self.query_doc_pair = query_doc_pair

        self.tokenizer = tokenizer
        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tokenizer.convert_tokens_to_ids('[unused1]')
        self.D_marker_token, self.D_marker_token_id = '[D]', self.tokenizer.convert_tokens_to_ids('[unused2]')
        self.mask_token, self.mask_token_id = self.tokenizer.mask_token, self.tokenizer.mask_token_id
        self.cls_token, self.cls_token_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.pad_token, self.pad_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id

    def __getitem__(self, index):
        ## query preparation
        query_id = self.query_doc_pair[index][0]
        query = self.questions[query_id].replace("’", "'")
        query_tokens_ids = self.tokenizer.encode(query)[:self.max_query_length]

        if 'colbert' in self.model_type:
            query_tokens_ids = [self.Q_marker_token_id] + query_tokens_ids[1:]

        ## document preparation
        doc_id = self.query_doc_pair[index][1]
        ctx_token_ids = self.tokenizer.encode(self.passages[doc_id][2][:2000], text_pair=self.passages[doc_id][1][:2000].strip(), add_special_tokens=True,
                                               max_length=self.max_doc_length, truncation=True,
                                               pad_to_max_length=False)

        if 'colbert' in self.model_type:
            ctx_token_ids = [self.D_marker_token_id] + ctx_token_ids[1:]

        def remove_special_token(token_id):
            if token_id[-1] == self.tokenizer.sep_token_id:
                return token_id[1:-1]
            else:
                return token_id[1:]

        ## remove [CLS] token in doc
        ## c_e_token_ids: [[query,pos], [query,neg1],..., [query,negN]]
        c_e_token_ids = query_tokens_ids + remove_special_token(ctx_token_ids)

        return query_id, query_tokens_ids, doc_id, ctx_token_ids, c_e_token_ids, self.query_doc_pair[index]

    def __len__(self):
        return len(self.query_doc_pair)
        # return 1000

    def get_collate_fn(self, features):
        q_num = 0
        query_list = []
        query_id_list = []
        doc_list = []
        doc_id_list = []
        c_e_token_ids_list = []
        query_doc_pair_all = []

        for index, feature in enumerate(features):
            # reranker
            q_num += 1
            query_id_list.append(feature[0])
            query_list.append(feature[1])
            doc_id_list.append(feature[2])
            doc_list.append(feature[3])
            c_e_token_ids_list.append(feature[4])
            query_doc_pair_all.append(feature[5])

        ##############
        # reranker
        ##############
        max_c_e_len = max([len(c_e_token_ids) for c_e_token_ids in c_e_token_ids_list])

        # [Batch_size, Max_len]
        c_e_list = [ce + [0] * (max_c_e_len - len(ce)) for ce in c_e_token_ids_list]  # padding

        # [Batch_size, Max_len]
        ctx_tensor_out = torch.LongTensor(c_e_list)

        ##############
        # dual encoder
        ##############

        # [Batch_size, max_q_len]
        if 'colbert' in self.model_type:
            q_list = [q + [self.mask_token_id] * (self.max_query_length - len(q)) for q in query_list]  # padding
        else:
            q_list = [q + [self.pad_token_id] * (self.max_query_length - len(q)) for q in query_list]  # padding

        # [Batch_size, topK, max_d_len]
        d_list = [d + [self.pad_token_id] * (self.max_doc_length - len(d)) for d in doc_list]  # padding

        # [Batch_size, max_q_len]
        q_tensor = torch.LongTensor(q_list)
        # [Batch_size, topK, max_d_len]
        doc_tensor = torch.LongTensor(d_list)

        return {'q_tensor': q_tensor,
                'doc_tensor': doc_tensor,
                'ce_tensor': ctx_tensor_out,
                'q_tensor_mask': (q_tensor != 0).long(),
                'doc_tensor_mask': (doc_tensor != 0).long(),
                'ce_tensor_mask': (ctx_tensor_out != 0).long(),
                'query_id_list': query_id_list,
                'doc_id_list': doc_id_list,
                'query_doc_pair': query_doc_pair_all
                }
