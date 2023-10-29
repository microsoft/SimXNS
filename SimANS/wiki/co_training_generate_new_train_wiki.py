import faiss
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger("__main__")
import torch
from utils.util import (
    is_first_worker,
)
import os
import numpy as np
import torch.distributed as dist
import csv
import json
from utils.dpr_utils import SimpleTokenizer, has_answer, Eval_Tool


class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        return example[0], example[1], example[2]


class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        retrived_doc_title = [x[2] for x in batch]
        retrived_doc_text = [x[1] for x in batch]
        tokenized_docs = self.tokenizer(
            retrived_doc_title,
            retrived_doc_text,
            max_length=self.maxlength,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        text_ids = tokenized_docs['input_ids']
        text_mask = tokenized_docs['attention_mask']
        return index, text_ids, text_mask


class Question_dataset(torch.utils.data.Dataset):
    def __init__(self, questions, tokenizer):
        self.questions = questions
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        query = self.questions[index].replace("’", "'")
        question_token_ids = self.tokenizer.encode(query)
        return index, question_token_ids

    def __len__(self):
        return len(self.questions)

    @classmethod
    def get_collate_fn(cls, args):
        def fn(features):
            max_q_len = max([len(feature[1]) for feature in features])
            q_list = [feature[1] + [0] * (max_q_len - len(feature[1])) for feature in features]
            q_tensor = torch.LongTensor(q_list)
            id_list = [feature[0] for feature in features]
            return np.array(id_list), q_tensor, (q_tensor != 0).long()

        return fn


def embed_passages(args, passages, model, tokenizer):
    batch_size = 512
    collator = TextCollator(tokenizer, args.max_seq_length)
    dataset = TextDataset(passages)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=15, collate_fn=collator)
    total = 0
    allids, allembeddings = [], []
    with torch.no_grad():
        for k, (ids, text_ids, text_mask) in enumerate(tqdm(dataloader)):
            inputs = {"input_ids": text_ids.long().to(args.device), "attention_mask": text_mask.long().to(args.device)}
            if hasattr(model, 'module'):
                embs = model.module.body_emb(**inputs)
            else:
                embs = model.body_emb(**inputs)
            embeddings = embs.detach().cpu()
            total += len(ids)

            allids.append(ids)
            allembeddings.append(embeddings)
            if k % 100 == 0:
                logger.info('Encoded passages %d', total)

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    allids = np.array([x for idlist in allids for x in idlist])
    return allids, allembeddings


class V_dataset(torch.utils.data.Dataset):
    def __init__(self, questions, passages, answers, closest_docs,
                 similar_scores, query_embedding2id, passage_embedding2id):
        self.questions = questions
        self.passages = passages
        self.answers = answers
        self.closest_docs = closest_docs
        self.similar_scores = similar_scores
        self.query_embedding2id = query_embedding2id
        self.passage_embedding2id = passage_embedding2id
        tok_opts = {}
        self.tokenizer = SimpleTokenizer(**tok_opts)

    def __getitem__(self, query_idx):
        query_id = self.query_embedding2id[query_idx]
        doc_ids = [self.passage_embedding2id[pidx] for pidx in self.closest_docs[query_idx]]
        hits = []
        temp_result_dict = {}
        temp_result_dict['id'] = str(query_id)
        temp_result_dict['question'] = self.questions[query_id]
        temp_result_dict['answers'] = self.answers[query_id]
        temp_result_dict['ctxs'] = []
        for i, doc_id in enumerate(doc_ids):
            text, title = self.passages[doc_id]
            hits.append(has_answer(self.answers[query_id], text, self.tokenizer))
            temp_result_dict['ctxs'].append({'d_id': str(doc_id),
                                             'text': text,
                                             'title': title,
                                             'score': str(self.similar_scores[query_idx, i]),
                                             'hit': str(hits[-1])})
        return hits, [query_id, doc_ids], temp_result_dict

    def __len__(self):
        return self.closest_docs.shape[0]

    @classmethod
    def collect_fn(cls, ):
        def create_biencoder_input2(features):
            scores = [feature[0] for feature in features]
            result_list = [feature[1] for feature in features]
            result_dict_list = [feature[2] for feature in features]
            return scores, result_list, result_dict_list

        return create_biencoder_input2


def validate(questions, passages, answers, closest_docs, similar_scores, query_embedding2id, passage_embedding2id):
    passage_text = {}
    for passage in passages:
        passage_text[passage[0]] = (passage[1], passage[2])
    v_dataset = V_dataset(questions, passage_text, answers, closest_docs, similar_scores, query_embedding2id,
                          passage_embedding2id)
    v_dataloader = DataLoader(v_dataset, 128, shuffle=False, num_workers=15, collate_fn=V_dataset.collect_fn())
    final_scores = []
    final_result_list = []
    final_result_dict_list = []
    for k, (scores, result_list, result_dict_list) in enumerate(tqdm(v_dataloader)):
        final_scores.extend(scores)  
        final_result_list.extend(result_list)
        final_result_dict_list.extend(result_dict_list)
    logger.info('Per question validation results len=%d', len(final_scores))
    n_docs = len(closest_docs[0])
    top_k_hits = [0] * n_docs
    for question_hits in final_scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(closest_docs) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)

    return top_k_hits, final_scores, Eval_Tool.get_matrics(final_scores), final_result_dict_list


def reform_out(result_dict_list, q_pos_dict):
    transfer_list = []
    for infer_result in tqdm(result_dict_list):
        if 'passage_id' not in infer_result.keys():
            q_id = infer_result["id"]
        else:
            q_id = infer_result["passage_id"]
        q_str = infer_result["question"]
        q_answer = infer_result["answers"]
        positive_ctxs = []
        negative_ctxs = []
        real_true_id = 0
        if q_str in q_pos_dict.keys():
            real_true_dic = q_pos_dict[q_str]
            # real_true_doc_id = real_true_dic['passage_id'] if 'passage_id' in real_true_dic.keys() else real_true_dic['id']
            if 'passage_id' not in real_true_dic.keys() and 'id' in real_true_dic.keys():
                real_true_dic['passage_id'] = real_true_dic['id']
            elif 'psg_id' in real_true_dic.keys():
                real_true_dic['passage_id'] = real_true_dic['psg_id']
            real_true_dic['score'] = str(0)
            real_true_id = int(real_true_dic['passage_id'])
            positive_ctxs.append(real_true_dic)

        for doc in infer_result['ctxs']:
            doc_text = doc['text']
            doc_title = doc['title']
            if doc['hit'] == "True":
                if int(doc['d_id'])==real_true_id-1:
                    positive_ctxs[0]['score'] = str(doc['score'])
                else:
                    positive_ctxs.append({'title': doc_title, 'text': doc_text, 'passage_id': doc['d_id'], 'score': str(doc['score'])})
            else:
                negative_ctxs.append(
                    {'title': doc_title, 'text': doc_text, 'passage_id': doc['d_id'], 'score': str(doc['score'])})

        transfer_list.append(
            {
                "q_id": str(q_id), "question": q_str, "answers": q_answer, "positive_ctxs": positive_ctxs,
                "hard_negative_ctxs": negative_ctxs, "negative_ctxs": []
            }
        )
    return transfer_list


class RenewTools:
    def __init__(self, passages_path, tokenizer, output_dir, temp_dir):
        self.passages = self.load_passage(passages_path)
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        if is_first_worker():
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
        dist.barrier()

    def get_passage_embedding(self, args, model):
        if args.load_cache:
            pass
        else:
            shard_size = len(self.passages) // args.world_size
            start_idx = args.rank * shard_size
            end_idx = start_idx + shard_size
            if args.rank == args.world_size - 1:
                end_idx = len(self.passages)
            passages_piece = self.passages[start_idx:end_idx]
            logger.info(f'Embedding generation for {len(passages_piece)} passages from idx {start_idx} to {end_idx}')
            allids, allembeddings = embed_passages(args, passages_piece, model, self.tokenizer)
            pickle_path = os.path.join(self.temp_dir,
                                       "{1}_data_obj_{0}.pb".format(str(args.rank), 'psg_embed'))
            with open(pickle_path, 'wb') as handle:
                pickle.dump(allembeddings, handle, protocol=4)
            pickle_path = os.path.join(self.temp_dir,
                                       "{1}_data_obj_{0}.pb".format(str(args.rank), 'psg_embed_id'))
            with open(pickle_path, 'wb') as handle:
                pickle.dump(allids, handle, protocol=4)
            logger.info(f'Total passages processed {len(allids)}. Written to {pickle_path}.')
        # dist.barrier()
        passage_embedding, passage_embedding_id = None, None
        if is_first_worker():
            logger.info('load_passage_begin')
            passage_embedding_list = []
            passage_embedding_id_list = []
            for i in tqdm(range(args.world_size)):  # TODO: dynamically find the max instead of HardCode
                pickle_path = os.path.join(self.temp_dir, "{1}_data_obj_{0}.pb".format(str(i), 'psg_embed'))
                with open(pickle_path, 'rb') as handle:
                    b = pickle.load(handle)
                    passage_embedding_list.append(b)
            logger.info('load_passage_id_begin')
            for i in tqdm(range(args.world_size)):  # TODO: dynamically find the max instead of HardCode
                pickle_path = os.path.join(self.temp_dir, "{1}_data_obj_{0}.pb".format(str(i), 'psg_embed_id'))
                with open(pickle_path, 'rb') as handle:
                    b = pickle.load(handle)
                    passage_embedding_id_list.append(b)
            passage_embedding = np.concatenate(passage_embedding_list, axis=0)
            passage_embedding_id = np.concatenate(passage_embedding_id_list, axis=0)
            logger.info('load_passage_done')
        return passage_embedding, passage_embedding_id

    def get_question_embeddings(self, args, questions, model):
        dataset = Question_dataset(questions, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=512, drop_last=False,
                                num_workers=15, collate_fn=Question_dataset.get_collate_fn(args))
        total = 0
        allids, allembeddings = [], []
        with torch.no_grad():
            for k, (ids, text_ids, text_mask) in enumerate(tqdm(dataloader)):
                inputs = {"input_ids": text_ids.long().to(args.device),
                          "attention_mask": text_mask.long().to(args.device)}
                if hasattr(model, 'module'):
                    embs = model.module.query_emb(**inputs)
                else:
                    embs = model.query_emb(**inputs)
                embeddings = embs.detach().cpu()
                total += len(ids)
                allids.append(ids)
                allembeddings.append(embeddings)
                if k % 1000 == 0:
                    logger.info('Encoded question %d', total)

        allembeddings = torch.cat(allembeddings, dim=0).numpy()
        allids = np.array([x for idlist in allids for x in idlist])
        return allembeddings, allids

    def get_new_faiss_index(self, args, passage_embedding, passage_embedding2id):

        logger.info("***** Begin passage_embedding reorder *****")
        new_passage_embedding = passage_embedding.copy()
        for i in range(passage_embedding.shape[0]):
            new_passage_embedding[passage_embedding2id[i]] = passage_embedding[i]
        del (passage_embedding)
        passage_embedding = new_passage_embedding
        passage_embedding2id = np.arange(passage_embedding.shape[0])
        logger.info("***** end passage_embedding reorder  *****")

        dim = passage_embedding.shape[1]
        faiss.omp_set_num_threads(90)
        cpu_index = faiss.IndexFlatIP(dim)

        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        gpu_index_flat = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index,
            co=co
        )
        logger.info("***** begin add passages  *****")
        gpu_index_flat.add(passage_embedding.astype(np.float32))
        logger.info("***** end build index  *****")
        return gpu_index_flat, passage_embedding2id

    def load_passage(self, passage_path):
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
                        passages.append((int(row[0]) - 1,row[1], row[2]))
                    except:
                        logger.warning(f'The following input line has not been correctly loaded: {row}')
        return passages
    def get_question_embedding(self,args,model,qa_path,mode='train'):
        train_questions = []
        train_answers = []
        logger.info("Loading"+mode+"answers")
        with open(qa_path, "r", encoding="utf-8") as ifile:
            # file format: question, answers
            reader = csv.reader(ifile, delimiter='\t')
            for row in reader:
                train_questions.append(row[0])
                train_answers.append(eval(row[1]))

        train_question_embedding, train_question_embedding2id = self.get_question_embeddings(args, train_questions,
                                                                                             model)
        return train_questions,train_answers,train_question_embedding, train_question_embedding2id
    # given q_a path ,generate
    def get_question_topk(self, train_questions,train_answers,
                                train_question_embedding, 
                                train_question_embedding2id,
                                golden_path, gpu_index_flat, passage_embedding2id,
                                mode='train', step_num=0):
        faiss.omp_set_num_threads(90)
        similar_scores, train_I = gpu_index_flat.search(train_question_embedding.astype(np.float32),
                                                        100)  # I: [number of queries, topk]

        top_k_hits, scores, result_dict, result_dict_list = \
            validate(train_questions, self.passages, train_answers, train_I,
                     similar_scores, train_question_embedding2id, passage_embedding2id)

        logger.info("writing new " + mode + " top-k")
        output_path = os.path.join(self.output_dir, mode + "_result_dict_list_" + str(step_num) + ".json")
        with open(output_path, 'w') as f:
            json.dump(result_dict_list, f, indent=2)


        logger.info("writing new " + mode + " data")
        ndcg_output_path = os.path.join(self.output_dir, mode + "_eval_result" + str(step_num) + ".json")
        with open(ndcg_output_path, 'w') as f:
            json.dump({'top1': top_k_hits[0], 'top5': top_k_hits[4],
                       'top20': top_k_hits[19], 'top100': top_k_hits[99], 'result_dict': result_dict}, f, indent=2)

        if mode == 'test':
            pass
        else:
            with open(golden_path, "r", encoding="utf-8") as ifile:
                # file format: question, answers
                golden_list = json.load(ifile)
            q_pos_dict = {}
            for example in golden_list:
                if len(example['positive_ctxs'])>0:
                    q_pos_dict[example['question']] = example['positive_ctxs'][0]
            transfer_list = reform_out(result_dict_list, q_pos_dict)


            output_path = os.path.join(self.output_dir, mode + '_ce_' + str(step_num) + '.json')
            with open(output_path, 'w') as f:
                json.dump(transfer_list, f, indent=2)

        return result_dict_list

    def get_question_topk_golden(self, args, model, qa_path, golden_list, gpu_index_flat, passage_embedding2id,
                                 mode='train', step_num=0):
        train_questions = []
        train_answers = []
        logger.info("Loading"+mode+"answers")
        with open(qa_path, "r", encoding="utf-8") as ifile:
            # file format: question, answers
            reader = csv.reader(ifile, delimiter='\t')
            for row in reader:
                train_questions.append(row[0])
                train_answers.append(eval(row[1]))

        train_question_embedding, train_question_embedding2id = self.get_question_embeddings(args, train_questions,
                                                                                             model)

        similar_scores, train_I = gpu_index_flat.search(train_question_embedding.astype(np.float32),
                                                        100)  # I: [number of queries, topk]

        top_k_hits, scores, result_dict, result_dict_list = \
            validate(train_questions, self.passages, train_answers, train_I,
                     similar_scores, train_question_embedding2id, passage_embedding2id)

        logger.info("writing new" + mode + "data")
        ndcg_output_path = os.path.join(self.output_dir, mode + "_eval_result" + str(step_num) + ".json")
        with open(ndcg_output_path, 'w') as f:
            json.dump({'top1': top_k_hits[0], 'top5': top_k_hits[4],
                       'top20': top_k_hits[19], 'top100': top_k_hits[99], 'result_dict': result_dict}, f, indent=2)

        if mode == 'test':
            pass
        else:
            q_pos_dict = {}
            for example in golden_list:
                q_pos_dict[example['question']] = example['positive_ctxs'][0]
            transfer_list = reform_out(result_dict_list, q_pos_dict)

            output_path = os.path.join(self.output_dir, mode + '_result_dict_list_' + str(step_num) + '.json')
            with open(output_path, 'w') as f:
                json.dump(result_dict_list, f, indent=2)

            output_path = os.path.join(self.output_dir, mode + '_ce_' + str(step_num) + '.json')
            with open(output_path, 'w') as f:
                json.dump(transfer_list, f, indent=2)

        return result_dict_list

    def read_train_pos(self, ground_truth_path):
        origin_train_path = ground_truth_path
        with open(origin_train_path, "r", encoding="utf-8") as ifile:
            # file format: question, answers
            train_list = json.load(ifile)
            train_q_pos_dict = {}
            for example in train_list:
                if len(example['positive_ctxs']) == 0 or "positive_ctxs" not in example.keys():
                    continue
                train_q_pos_dict[example['question']] = example['positive_ctxs'][0]
        return train_q_pos_dict
