# -*- coding: utf-8 -*-
import logging
import numpy as np
import torch
from torch import nn
import random
import collections
from torch.optim import AdamW, Adam
import torch.distributed as dist
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from utils.lamb import Lamb
import unicodedata
from utils.dpr_utils import all_gather_list
from utils.metric_utils import compute_metric
from utils.data_utils import load_passage
def _normalize(text):
    return unicodedata.normalize('NFD', text)

logger = logging.getLogger("__main__")
# logger = logging.getLogger(__name__)
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])
BiEncoderPassageQueries = collections.namedtuple("BiEncoderPassageQueries", ["text", "title", "queries"])
def sum_main(x, opt):
    if opt.world_size > 1:
        # torch.distributed.reduce(x, 0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def normalize_question(question: str) -> str:
    question = question.replace("’", "'")
    return question

def normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    return ctx_text

def get_optimizer(optimizer, model: nn.Module, weight_decay: float = 0.0, lr: float = 0, adam_epsilon=1e-8) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if optimizer == "adamW":
        return AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    elif optimizer == "lamb":
        return Lamb(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    else:
        raise Exception("optimizer {0} not recognized! Can only be lamb or adamW".format(optimizer))


def is_first_worker(): # cpu or single card or the first card of multiple cards
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


class TraditionDataset(Dataset):
    def __init__(self, file_path, retriever_tokenizer, num_hard_negatives=1, is_training=True, passages_path=None,
                 max_seq_length =128, max_query_length=32, shuffle_positives=False, 
                 expand_doc_w_query=False, expand_corpus=False, top_k_query=1, append=False, 
                 gold_query_prob = 0, select_generated_query='random', metric='rouge-l', query_path=None, 
                 prepare_generator_inputs = False, generator_tokenizer=None, generator_max_seq_length =0, generator_max_query_length=0,
                 delimiter = ' ', n_sample=0, total_part=0, filter_threshold=1, psg_id_2_query_dict={}):
        self.file_path = file_path
        # reranker and retriever share the same tokenizer
        self.retriever_tokenizer = retriever_tokenizer

        self.data = self.load_data()
        if len(self.data)>0 and type(self.data[0]['hard_negative_ctxs'][0]) != dict:
            self.passages = load_passage(passages_path)

        self.is_training = is_training
        self.num_hard_negatives = num_hard_negatives
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.shuffle_positives = shuffle_positives

        self.expand_doc_w_query= expand_doc_w_query
        self.expand_corpus= expand_corpus
        self.top_k_query= top_k_query
        self.append= append 
        self.gold_query_prob= gold_query_prob
        self.select_generated_query= select_generated_query
        self.metric= metric
        self.query_path= query_path

        self.prepare_generator_inputs = prepare_generator_inputs
        self.generator_tokenizer= generator_tokenizer
        self.generator_max_seq_length = generator_max_seq_length if generator_max_seq_length>0 else max_seq_length
        self.generator_max_query_length = generator_max_query_length if generator_max_query_length>0 else max_query_length 
        if self.prepare_generator_inputs:
            assert self.generator_tokenizer is not None, 'Please specify the generator tokenizer.'
        
        self.delimiter= delimiter
        self.n_sample = n_sample

        self.total_part = total_part
        self.filter_threshold = filter_threshold
        self.psg_id_2_query_dict = psg_id_2_query_dict
        if self.expand_doc_w_query or self.expand_corpus:
            self.max_query =  80 
            # Format for each line: psg_id\tquery\tquery\tquery...\query\n
            self.psg_id_2_query_dict = load_dataset('text', data_files=self.query_path)['train']
            print(f'Finish loading queries.')
            if self.expand_doc_w_query:
                logger.info(f'Expand the doc with the top {self.top_k_query} queries from {self.query_path}.')

            if self.expand_corpus and self.gold_query_prob<1 and self.select_generated_query!='random':
                self.metric_obj = compute_metric()
            

    def reset_select_generated_query(self, global_step, max_steps, total_part, select_generated_query):
        N = total_part
        if select_generated_query == 'gradual-gold':
            iter_steps = max_steps//(N+1)
            i = max(N - global_step//iter_steps, 0)
            self.select_generated_query = f'{i}-part'
            if i ==0:
                # use gold query
                self.gold_query_prob = 1
            else:
                self.gold_query_prob = 0 # use generated query
            return i
        if select_generated_query == 'gradual':
            iter_steps = max_steps//N
            i = max(N - global_step//iter_steps, 1)
            self.select_generated_query = f'{i}-part'
            self.gold_query_prob = 0 # use generated query
            return i

        return -1


    def load_data(self):
        with open(self.file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f'Aggregated data size: {len(data)}.')
        # filter those without positive ctx
        pre_data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info(f"Cleaned data size: {len(pre_data)} after removing instances without positive ctx.")
        pre_data = [r for r in pre_data if len(r['hard_negative_ctxs']) > 0]
        logger.info(f"Total cleaned data size: {len(pre_data)} after removing instances without hard negative ctx.")
        return pre_data

    def create_passage(self, ctx: dict):
        return BiEncoderPassage(ctx["text"], ctx["title"])

    def create_passage_queries(self, ctx: dict, query, gold_query_prob, select_generated_query, index, total_part, filter_threshold=1):
        psg_id = ctx["psg_id"] if "psg_id" in ctx else ctx["passage_id"]

        contents = self.psg_id_2_query_dict[int(psg_id)]['text'].strip().split('\t')
        passage_id = int(contents[0]) 
        assert passage_id == int(psg_id)
        query_list = contents[1:][:self.max_query]
        if self.expand_corpus:
            if self.is_training:
                # remove the repetitive query and the gold query
                query_list = list(set(query_list) - set([query])) 
                assert len(query_list)>=1
                N = len(query_list)
                if select_generated_query == 'batch-uniform':
                    i = index%N + 1
                    select_generated_query = f'{i}-th'
                    gold_query_prob = 0
                elif select_generated_query == 'batch-uniform-gold':
                    i = index%(N+1) + 1
                    select_generated_query = f'{i}-th'
                    if i==N+1:
                        gold_query_prob = 1
                    else:
                        gold_query_prob = 0
                elif '-part' in select_generated_query:
                    k = int(select_generated_query.split('-part')[0])
                    end = int((N/total_part)*k)
                    start =  int((N/total_part)*(k-1))
                    if start == end:
                        i = end+1
                    else:
                        i = random.randint(start+1, end)
                    a=select_generated_query
                    select_generated_query = f'{i}-th'
                else:
                    pass
                if select_generated_query =='gold':
                    assert gold_query_prob == 1
                if random.random()<gold_query_prob:
                    queries = query # use gold query
                else:
                    # if len(query_list) <2:
                    #     return None
                    # different strategies to choose the query
                    if select_generated_query == 'first':
                        queries = query_list[0]
                    elif select_generated_query == 'random':
                        queries = random.choice(query_list)
                    else:
                        # compute the metric score between the generated query and gold query
                        score_list = self.metric_obj.compute_metric_score(query, query_list, self.metric)
                        
                        new_score_list = []
                        new_query_list = []
                        min_s = 10000
                        min_q = None
                        for s, q in zip(score_list, query_list):
                            if s<=filter_threshold:
                                new_score_list.append(s)
                                new_query_list.append(q)
                            if s < min_s:
                                min_s = s
                                min_q = q
                        if len(new_score_list) == 0:
                            score_list = [min_s]
                            query_list = [min_q]
                        else:
                            score_list = new_score_list
                            query_list = new_query_list
                        
                        score_list = torch.tensor(np.array(score_list))
                        if 'top-' in select_generated_query:
                            k = int(select_generated_query.split('top-')[1])
                            k = min(k, len(query_list))
                            values, indices = torch.topk(score_list, k)
                        elif 'bottom-' in select_generated_query:
                            k = int(select_generated_query.split('bottom-')[1])
                            k = min(k, len(query_list))
                            values, indices = torch.topk(-score_list, k)
                        elif '-th' in select_generated_query:
                            k = int(select_generated_query.split('-th')[0])
                            k = min(k, len(query_list))
                            values, indices = torch.topk(score_list, k)
                            indices = indices[k-1:k]
                            values = values[k-1:k]
                        else:
                            raise ValueError()
                        
                        sampled_index = random.choice(indices.tolist())
                        queries = query_list[sampled_index]
            else:
                queries = query_list[0]

        elif self.expand_doc_w_query:
            if self.is_training:
                # remove the repetitive query and the gold query
                query_list = list(set(query_list) - set([query])) 
            else:
                query_list = list(set(query_list)) 

            if len(query_list)<self.top_k_query:
                query_list = (query_list*self.top_k_query)[:self.top_k_query]

            queries = " ".join(query_list[:self.top_k_query])
        else:
            raise ValueError('')

        return BiEncoderPassageQueries(ctx["text"], ctx["title"], queries)

    def __getitem__(self, index):
        json_sample = self.data[index]
        query = normalize_question(json_sample["question"])
        positive_ctxs = json_sample["positive_ctxs"]
        # negative_ctxs = (
        #     json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        # )
        hard_negative_ctxs = (
            json_sample["hard_negative_ctxs"] if "hard_negative_ctxs" in json_sample else []
        )
        if self.n_sample>0:
            hard_negative_ctxs = hard_negative_ctxs[:self.n_sample]
        if self.is_training:
            random.shuffle(hard_negative_ctxs)
        hard_negative_ctxs = hard_negative_ctxs[:self.num_hard_negatives]
        if type(hard_negative_ctxs[0])!=dict:
            new_hard_negative_ctxs = []
            for psg_id in hard_negative_ctxs:
                ctx = self.passages[int(psg_id)] # psg_id, text, title
                assert ctx[0] == int(psg_id)
                new_hard_negative_ctxs.append({"title": ctx[2], "text":ctx[1], "passage_id": int(psg_id)})
            hard_negative_ctxs = new_hard_negative_ctxs

        for ctx in positive_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        if len(hard_negative_ctxs) < self.num_hard_negatives:
            hard_negative_ctxs = hard_negative_ctxs*self.num_hard_negatives
        hard_negative_ctxs = hard_negative_ctxs[0:self.num_hard_negatives]
        
        if self.shuffle_positives:
            positive_ctx = random.choice(positive_ctxs)
        else:
            positive_ctx = positive_ctxs[0]
        # print(index, self.gold_query_prob, self.select_generated_query)
        if self.expand_doc_w_query or self.expand_corpus:
            positive_passagese_ctx = self.create_passage_queries(positive_ctx, query, self.gold_query_prob, self.select_generated_query, index, self.total_part, self.filter_threshold) 
            hard_neg_ctxs = [self.create_passage_queries(ctx, query, self.gold_query_prob, self.select_generated_query, index, self.total_part, self.filter_threshold) for ctx in hard_negative_ctxs]
        else:
            positive_passagese_ctx = self.create_passage(positive_ctx)
            hard_neg_ctxs = [self.create_passage(ctx) for ctx in hard_negative_ctxs]


        # create inputs for generator
        if self.prepare_generator_inputs:
            generator_encoder_token_ids = self.generator_tokenizer.encode(positive_passagese_ctx.title.strip() + ' ' + positive_passagese_ctx.text.strip(), 
                                                                            add_special_tokens=True,
                                                                            max_length=self.generator_max_seq_length, 
                                                                            truncation=True, padding=False) 
            generator_decoder_token_ids = self.generator_tokenizer.encode(query, add_special_tokens=True, 
                                                                           max_length=self.generator_max_query_length, 
                                                                           truncation=True, padding=False)
        else:
            generator_encoder_token_ids = None
            generator_decoder_token_ids = None

        # inputs for the dual-encoder model
        ctxs = [positive_passagese_ctx] + hard_neg_ctxs

        def get_content(ctx, delimiter):
            if self.expand_doc_w_query or self.expand_corpus:
                if self.append:
                    content = ctx.title.strip() + delimiter + ctx.text.strip() + delimiter + ctx.queries.strip()
                else:
                    content = ctx.queries.strip() + delimiter+ ctx.title.strip() + delimiter + ctx.text.strip() 
            else:
                content = ctx.title.strip() + delimiter + ctx.text.strip() 
            return content    

        ctx_token_ids = [self.retriever_tokenizer.encode(get_content(ctx, delimiter= self.delimiter), 
                                                add_special_tokens=True, max_length=self.max_seq_length, 
                                                truncation=True, padding=False) for ctx in ctxs]

        question_token_ids = self.retriever_tokenizer.encode(query, max_length=self.max_query_length, truncation=True, padding=False)
        

        # inputs for the cross-encoder model
        c_e_token_ids = [question_token_ids + ctx_token_id[1:] for ctx_token_id in ctx_token_ids]

        answers = [self.retriever_tokenizer.encode(_normalize(single_answer), add_special_tokens=False) if type(single_answer)==str else single_answer 
                  for single_answer in json_sample['answers']]
        start_ctx = len(question_token_ids)
        start_end = [[start_ctx, len(c_e_token_id)] for c_e_token_id in c_e_token_ids]

        return question_token_ids, ctx_token_ids, c_e_token_ids, answers, start_end, \
               generator_encoder_token_ids, generator_decoder_token_ids, \
               query, positive_passagese_ctx.title.strip() + self.delimiter + positive_passagese_ctx.text.strip()
    
    def __len__(self):
        return len(self.data)
    @classmethod
    def get_collate_fn(cls, args):
        def create_inputs(features):
            """
            create inputs for the cross-encoder ranker and the dual-encoder retriever.
            """
            q_list = []
            d_list = []
            c_e_input_list = []
            positive_ctx_indices = []
            c_e_ctx_start_end = []
            # generator inputs
            prepare_generator_inputs = True
            for feature in features:
                positive_ctx_indices.append(len(d_list))
                q_list.append(feature[0]) 
                d_list.extend(feature[1])
                c_e_input_list.extend(feature[2])
                c_e_ctx_start_end.append(feature[4])

                if feature[5] is None:
                    prepare_generator_inputs = False

            if prepare_generator_inputs:
                encoder_input_list = [torch.tensor(np.array(feature[5]), dtype=torch.long) for feature in features]
                # create encoder_inputs, attention_mask, decoder_inputs, decoder_labels
                _mask = pad_sequence(encoder_input_list, batch_first=True, padding_value=-100)

                attention_mask = torch.zeros(_mask.shape, dtype=torch.float32)
                attention_mask = attention_mask.masked_fill(_mask != -100, 1)
                encoder_inputs = pad_sequence(encoder_input_list, batch_first=True, padding_value=0)

                decoder_input_list = [torch.tensor(np.array(feature[6][:-1]), dtype=torch.long) for feature in features]
                decoder_label_list = [torch.tensor(np.array(feature[6][1:]), dtype=torch.long) for feature in features]
                decoder_inputs = pad_sequence(decoder_input_list, batch_first=True, padding_value=0)
                decoder_labels = pad_sequence(decoder_label_list, batch_first=True, padding_value=-100)

                query_list = [feature[7] for feature in features]
                doc_list = [feature[8] for feature in features]
                generator_inputs =  {'input_ids': encoder_inputs, 
                                    'attention_mask': attention_mask, 
                                    'decoder_input_ids': decoder_inputs, 
                                    'labels': decoder_labels}
            else:
                generator_inputs = None
                query_list = None
                doc_list = None


            max_q_len = max([len(q) for q in q_list])
            max_d_len = max([len(d) for d in d_list])
            max_c_e_len = max([len(d) for d in c_e_input_list])
            q_list = [q+[0]*(max_q_len-len(q)) for q in q_list]
            d_list = [d+[0]*(max_d_len-len(d)) for d in d_list]
            c_e_list = [c_e+[0]*(max_c_e_len-len(c_e)) for c_e in c_e_input_list]

            q_tensor = torch.LongTensor(np.array(q_list))
            doc_tensor = torch.LongTensor(np.array(d_list))
            ctx_tensor_out = torch.LongTensor(np.array(c_e_list))
            q_num, d_num = len(q_list),len(d_list)
            tgt_tensor = torch.zeros((len(c_e_list)), dtype=torch.long)
            tgt_tensor[positive_ctx_indices] = 1
            ctx_tensor_out = ctx_tensor_out.reshape(q_num,d_num//q_num,-1)
            tgt_tensor = tgt_tensor.reshape(q_num,d_num//q_num)
            return {
                    'reranker': [ctx_tensor_out, 
                                (ctx_tensor_out!= 0).long(), tgt_tensor],
                    'retriever': [q_tensor, (q_tensor!= 0).long(), doc_tensor, 
                                 (doc_tensor!= 0).long(), positive_ctx_indices],
                    "answers": [feature[3] for feature in features],
                    "reranker_ctx_start_end": c_e_ctx_start_end,
                    'generator': generator_inputs, 
                    'query_list': query_list, 
                    'doc_list': doc_list
                    }

        return create_inputs




class RerankerQueryDataset(Dataset):
    """
    This dataset provides data with the following formats:
    positive data instance: gold_query + gold passage
    negative data instance: generated query + gold passage
    """
    def __init__(self, file_path, retriever_tokenizer, num_hard_negatives=1, is_training=True,
                 max_seq_length =128, max_query_length=32, shuffle_positives=False, 
                 metric='rouge-l', select_negative_query='random',  select_positive_query='gold', 
                 query_path=None):
        """
        metric: metric used to compute the scores between the gold query and the generated queries.
        select_negative_query: how to select the negative queries from the generated queries. Suport 'random', 'top', and 'bottom' methods.
        select_positive_query: how the select the positive query. Suport 'gold' and 'max' methods.
        """
        self.file_path = file_path
        # reranker and retriever share the same tokenizer
        self.retriever_tokenizer = retriever_tokenizer

        self.data = self.load_data()
        self.is_training = is_training
        self.num_hard_negatives = num_hard_negatives
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.shuffle_positives = shuffle_positives

        self.metric= metric
        self.select_negative_query = select_negative_query
        self.select_positive_query = select_positive_query
        self.query_path= query_path
        
        max_query =  80 if self.is_training else 5 
        self.psg_id_2_query_dict = {}
        with open(self.query_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                contents = line.strip().split('\t')
                passage_id = int(contents[0]) 
                query_list = contents[1:][:max_query]
                self.psg_id_2_query_dict[passage_id] = query_list
        print(f'Finish loading queries.')
        self.metric_obj = compute_metric()


    def load_data(self):
        with open(self.file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f'Aggregated data size: {len(data)}.')
        # filter those without positive ctx
        pre_data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info(f"Cleaned data size: {len(pre_data)} after positive ctx.")

        return pre_data

    def __getitem__(self, index):
        json_sample = self.data[index]
        query = normalize_question(json_sample["question"])
        positive_ctxs = json_sample["positive_ctxs"]

        for ctx in positive_ctxs:
            if "title" not in ctx:
                ctx["title"] = None
        
        if self.shuffle_positives:
            ctx = random.choice(positive_ctxs)
        else:
            ctx = positive_ctxs[0]
    
        # compute the metric score between the generated query and gold query
        psg_id = ctx["psg_id"] if "psg_id" in ctx else ctx["passage_id"]
        query_list = self.psg_id_2_query_dict[int(psg_id)]

        if self.is_training:
            # remove the repetive query and the gold query
            p = query_list
            query_list = list(set(query_list) - set([query]))
            if len(query_list) <2:
                return None

            score_list = self.metric_obj.compute_metric_score(query, query_list, self.metric)
            score_list = [(q, s)  for s, q in zip(score_list, query_list)]
            scores = sorted(score_list, key = lambda x: x[1], reverse=True)
            # first select positive query
            if self.select_positive_query == 'gold':
                pos_query = query
            elif self.select_positive_query == 'max':
                pos_query = scores[0][0]
                pos_score = scores[0][1]
                scores = scores[1:]
            else:
                raise ValueError('')

            if len(scores) < self.num_hard_negatives:
                scores = scores*self.num_hard_negatives
                # scores = scores[0:self.num_hard_negatives]
            max_num = self.num_hard_negatives
            # select negative queries
            if self.select_negative_query == 'random':
                s_for_pick = random.sample(scores, max_num)
                neg_query_list = [s[0] for s in s_for_pick]
                neg_score_list = [s[1] for s in s_for_pick]
            elif self.select_negative_query == 'top':
                neg_query_list =  [s[0] for s in scores[:max_num]]
                neg_score_list = [s[1] for s in scores[:max_num]]
            elif self.select_negative_query == 'bottom':
                neg_query_list =  [s[0] for s in scores[-max_num:]]
                neg_score_list = [s[1] for s in scores[-max_num:]]
            elif self.select_negative_query == 'top-bottom':
                n_top = max_num // 2
                n_bottom = max_num - n_top
                neg_query_list = [s[0] for s in scores[:n_top]]
                neg_score_list = [s[1] for s in scores[:n_top]]
                neg_query_list +=  [s[0] for s in scores[-n_bottom:]]
                neg_score_list += [s[1] for s in scores[-n_bottom:]]
            else:
                raise ValueError('')
        else:
            pos_query = query
            neg_query_list = query_list
        # print(scores, len(scores))
        # print(pos_query, neg_query_list, neg_score_list)
        # exit()
        pos_content = ctx['title'].strip() + ' ' + ctx['text'].strip()
        query_list = [pos_query] + neg_query_list
        ctx_token_ids = self.retriever_tokenizer.encode(pos_content, add_special_tokens=True, max_length=self.max_seq_length, 
                                                        truncation=True, padding=False) 

        query_token_ids_list = [self.retriever_tokenizer.encode(query, add_special_tokens=True, 
                                                               max_length=self.max_query_length, truncation=True, padding=False)
                                                               for query in query_list]
        
        c_e_token_ids = [query_token_ids + ctx_token_ids[1:] for query_token_ids in query_token_ids_list]
        
        return c_e_token_ids,
    
    def __len__(self):
        return len(self.data)
    
    @classmethod
    def get_collate_fn(cls, args):
        def create_inputs(features):
            """
            create inputs for the cross-encoder ranker and the dual-encoder retriever.
            """
            c_e_input_list = []
            positive_ctx_indices = []
            q_num = 0
            for feature in features:
                if feature is None:
                    continue
                q_num +=1
                positive_ctx_indices.append(len(c_e_input_list))
                c_e_input_list.extend(feature[0])
            max_c_e_len = max([len(d) for d in c_e_input_list])

            c_e_list = [c_e+[0]*(max_c_e_len-len(c_e)) for c_e in c_e_input_list]

            ctx_tensor_out = torch.LongTensor(c_e_list)
            d_num = len(c_e_input_list)
            tgt_tensor = torch.zeros((len(c_e_list)), dtype=torch.long)
            tgt_tensor[positive_ctx_indices] = 1
            ctx_tensor_out = ctx_tensor_out.reshape(q_num,d_num//q_num,-1)
            # print(ctx_tensor_out.shape)
            tgt_tensor = tgt_tensor.reshape(q_num,d_num//q_num)
            return {'reranker': [ctx_tensor_out, 
                                (ctx_tensor_out!= 0).long(), tgt_tensor]
                    }

        return create_inputs



class RerankDataset(Dataset):
    """
    this is used to compute the sentence tokenization
    """

    def __init__(self, tokenizer, passages, psg_id_2_query_dict,  max_seq_length =128, max_query_length=32):
        self.tokenizer = tokenizer
        self.passages = passages
        self.psg_id_2_query_dict = psg_id_2_query_dict
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length

        assert len(self.passages) == len(self.psg_id_2_query_dict)
        self.psg_id_list = sorted(list(passages.keys()))

    def __getitem__(self, idx):
        psg_id = self.psg_id_list[idx]
        passage = self.passages[psg_id]
        query_list = self.psg_id_2_query_dict[psg_id]
        # title + body
        pos_content = passage[2].strip() + ' ' + passage[1].strip()
        ctx_token_ids = self.tokenizer.encode(pos_content, add_special_tokens=True, max_length=self.max_seq_length, 
                                                        truncation=True, padding=False) 

        query_token_ids_list = [self.tokenizer.encode(query, add_special_tokens=True, 
                                                               max_length=self.max_query_length, truncation=True, padding=False)
                                                               for query in query_list]
        
        c_e_token_ids = [query_token_ids + ctx_token_ids[1:] for query_token_ids in query_token_ids_list]
        
        return c_e_token_ids, psg_id

    def __len__(self):
        return len(self.psg_id_list)

    def collate_fn(self, features):
        """
        create inputs for the cross-encoder ranker and the dual-encoder retriever.
        """
        c_e_input_list = []
        psg_id_list = []
        for feature in features:
            c_e_input_list.extend(feature[0])
            psg_id_list.append(feature[1])
        max_c_e_len = max([len(d) for d in c_e_input_list])

        c_e_list = [c_e+[0]*(max_c_e_len-len(c_e)) for c_e in c_e_input_list]

        ctx_tensor_out = torch.LongTensor(c_e_list)
        d_num = len(c_e_input_list)
        q_num = len(features)

        ctx_tensor_out = ctx_tensor_out.reshape(q_num,d_num//q_num,-1)
        return ctx_tensor_out, (ctx_tensor_out!= 0).long(), psg_id_list
               