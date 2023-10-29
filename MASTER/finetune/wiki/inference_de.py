import argparse
import sys

from torch.utils.data.dataset import Dataset

sys.path += ['../']
import json
import logging
import os
import csv
import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from tqdm import tqdm
import torch.distributed as dist
from utils.util import (
    is_first_worker,
)
from model.models import BiBertEncoder
from utils.dpr_utils import load_states_from_checkpoint, get_model_obj, SimpleTokenizer, has_answer
import pickle
from transformers import (
    BertTokenizer
)
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)
import faiss


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
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        dist.barrier()  # directory created
        pickle_path = os.path.join(
            args.output_dir,
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
            args.output_dir,
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


def load_data(args):
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
    return (passages)


def load_model(args):
    # Prepare GLUE task
    args.output_mode = "classification"
    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True)
    model = BiBertEncoder(args)

    saved_state = load_states_from_checkpoint(args.eval_model_dir)
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

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    return tokenizer, model


class Question_dataset(Dataset):
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


def get_question_embeddings(args, questions, tokenizer, model):
    dataset = Question_dataset(questions, tokenizer)
    sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.per_gpu_eval_batch_size,
                            collate_fn=Question_dataset.get_collate_fn(args), num_workers=0, shuffle=False)
    epoch_iterator = tqdm(dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    embedding2id = []
    embedding = []
    if args.local_rank != -1:
        dist.barrier()
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(epoch_iterator):
            id_list = batch[0]
            inputs = {"input_ids": batch[1].long().to(args.device), "attention_mask": batch[2].long().to(args.device)}
            embs = model.module.query_emb(**inputs)
            embs = embs.detach().cpu().numpy()
            embedding2id.append(id_list)
            embedding.append(embs)
    embedding = np.concatenate(embedding, axis=0)
    embedding2id = np.concatenate(embedding2id, axis=0)

    full_embedding = barrier_array_merge(args, embedding, prefix="question_" + str(0) + "_" + "_emb_p_",
                                         load_cache=False, only_load_in_master=True)
    full_embedding2id = barrier_array_merge(args, embedding2id, prefix="question_" + str(0) + "_" + "_embid_p_",
                                            load_cache=False, only_load_in_master=True)
    if args.local_rank != -1:
        dist.barrier()
    return full_embedding, full_embedding2id


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


def embed_passages(opt, passages, model, tokenizer):
    batch_size = opt.per_gpu_eval_batch_size
    collator = TextCollator(tokenizer, opt.max_seq_length)
    dataset = TextDataset(passages)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=10, collate_fn=collator)
    total = 0
    allids, allembeddings = [], []
    with torch.no_grad():
        for k, (ids, text_ids, text_mask) in enumerate(tqdm(dataloader)):
            inputs = {"input_ids": text_ids.long().to(opt.device), "attention_mask": text_mask.long().to(opt.device)}
            embs = model.module.body_emb(**inputs)
            embeddings = embs.detach().cpu()
            total += len(ids)

            allids.append(ids)
            allembeddings.append(embeddings)
            if k % 100 == 0:
                logger.info('Encoded passages %d', total)

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    allids = np.array([x for idlist in allids for x in idlist])
    return allids, allembeddings


def get_passage_embedding(args, passages, model, tokenizer):
    if not args.load_cache:
        shard_size = len(passages) // args.world_size
        start_idx = args.local_rank * shard_size
        end_idx = start_idx + shard_size
        if args.local_rank == args.world_size - 1:
            end_idx = len(passages)
        passages_piece = passages[start_idx:end_idx]
        logger.info(f'Embedding generation for {len(passages_piece)} passages from idx {start_idx} to {end_idx}')
        allids, allembeddings = embed_passages(args, passages_piece, model, tokenizer)
        if is_first_worker():
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
        dist.barrier()
        pickle_path = os.path.join(args.output_dir,
                                   "{1}_data_obj_{0}.pb".format(str(args.local_rank), 'passage_embedding'))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(allembeddings, handle, protocol=4)
        pickle_path = os.path.join(args.output_dir,
                                   "{1}_data_obj_{0}.pb".format(str(args.local_rank), 'passage_embedding_id'))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(allids, handle, protocol=4)
        logger.info(f'Total passages processed {len(allids)}. Written to {pickle_path}.')
        dist.barrier()

    passage_embedding, passage_embedding_id = None, None
    if is_first_worker():
        passage_embedding_list = []
        passage_embedding_id_list = []
        for i in range(args.world_size):  # TODO: dynamically find the max instead of HardCode
            pickle_path = os.path.join(args.output_dir, "{1}_data_obj_{0}.pb".format(str(i), 'passage_embedding'))
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                passage_embedding_list.append(b)
        for i in range(args.world_size):  # TODO: dynamically find the max instead of HardCode
            pickle_path = os.path.join(args.output_dir, "{1}_data_obj_{0}.pb".format(str(i), 'passage_embedding_id'))
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                passage_embedding_id_list.append(b)
        passage_embedding = np.concatenate(passage_embedding_list, axis=0)
        passage_embedding_id = np.concatenate(passage_embedding_id_list, axis=0)
    return passage_embedding, passage_embedding_id


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
        if q_str in q_pos_dict.keys():
            real_true_dic = q_pos_dict[q_str]
            # real_true_doc_id = real_true_dic['passage_id'] if 'passage_id' in real_true_dic.keys() else real_true_dic['id']
            if 'passage_id' not in real_true_dic.keys() and 'id' in real_true_dic.keys():
                real_true_dic['passage_id'] = real_true_dic['id']
            elif 'psg_id' in real_true_dic.keys():
                real_true_dic['passage_id'] = real_true_dic['psg_id']
            positive_ctxs.append(real_true_dic)

        for doc in infer_result['ctxs']:
            doc_text = doc['text']
            doc_title = doc['title']
            if doc['hit'] == "True":
                positive_ctxs.append(
                    {'title': doc_title, 'text': doc_text, 'passage_id': doc['d_id'], 'score': str(doc['score'])})
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


def generate_new_embeddings(args, tokenizer, model):
    # passage_text, test_questions, test_answers = preloaded_data
    passages = load_data(args)
    logger.info("***** inference of passages *****")

    passage_embedding, passage_embedding2id = get_passage_embedding(args, passages, model, tokenizer)
    logger.info("***** Done passage inference *****")

    logger.info("***** inference of test query *****")
    test_questions = []
    test_answers = []
    logger.info("Loading test answers")
    with open(args.test_qa_path, "r", encoding="utf-8") as ifile:
        # file format: question, answers
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            test_questions.append(row[0])
            test_answers.append(eval(row[1]))
    test_question_embedding, test_question_embedding2id = get_question_embeddings(args, test_questions, tokenizer,
                                                                                  model)

    '''train dataset_generation'''
    if args.train_qa_path is not None:
        logger.info("***** inference of train query *****")
        train_questions = []
        train_answers = []
        logger.info("Loading train answers")
        with open(args.train_qa_path, "r", encoding="utf-8") as ifile:
            # file format: question, answers
            reader = csv.reader(ifile, delimiter='\t')
            for row in reader:
                train_questions.append(row[0])
                train_answers.append(eval(row[1]))
        train_question_embedding, train_question_embedding2id = get_question_embeddings(args, train_questions,
                                                                                        tokenizer, model)

    '''eval dataset_generation'''
    if args.dev_qa_path is not None:
        logger.info("***** inference of dev query *****")
        dev_answers = []
        dev_questions = []
        logger.info("Loading dev answers")
        with open(args.dev_qa_path, "r", encoding="utf-8") as ifile:
            # file format: question, answers
            reader = csv.reader(ifile, delimiter='\t')
            for row in reader:
                dev_questions.append(row[0])
                dev_answers.append(eval(row[1]))
        dev_question_embedding, dev_question_embedding2id = get_question_embeddings(args, dev_questions, tokenizer,
                                                                                    model)

    ''' test eval'''
    if is_first_worker():
        passage_text = {}
        for passage in passages:
            passage_text[passage[0]] = (passage[1], passage[2])

        dim = passage_embedding.shape[1]
        print('passage embedding shape: ' + str(passage_embedding.shape))
        logger.info("***** Begin passage_embedding reorder *****")
        new_passage_embedding = passage_embedding.copy()
        for i in range(passage_embedding.shape[0]):
            new_passage_embedding[passage_embedding2id[i]] = passage_embedding[i]
        del (passage_embedding)
        passage_embedding = new_passage_embedding
        passage_embedding2id = np.arange(passage_embedding.shape[0])
        logger.info("***** Begin passage_embedding reorder  *****")

        logger.info("***** Begin ANN Index build *****")
        top_k = args.top_k
        faiss.omp_set_num_threads(args.thread_num)
        cpu_index = faiss.IndexFlatIP(dim)

        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        gpu_index_flat = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index,
            co=co
        )
        gpu_index_flat.add(passage_embedding.astype(np.float32))
        cpu_index = gpu_index_flat
        # cpu_index.add(passage_embedding.astype(np.float32))
        # output_path = os.path.join(args.output_dir, 'faiss.index')
        # faiss.write_index(cpu_index, output_path)
        logger.info("***** Done ANN Index *****")

        logger.info("***** Begin test ANN Index *****")
        faiss.omp_set_num_threads(args.thread_num)
        similar_scores, dev_I = cpu_index.search(test_question_embedding.astype(np.float32),
                                                 100)  # I: [number of queries, topk]
        logger.info("***** Done test ANN search *****")

        logger.info("***** Begin test validate *****")
        top_k_hits, scores, result_dict, result_dict_list = \
            validate(test_questions, passage_text, test_answers, dev_I,
                     similar_scores, test_question_embedding2id, passage_embedding2id)
        logger.info("***** Done test validate *****")

        logger.info("***** Done test validate *****")
        ndcg_output_path = os.path.join(args.output_dir, "test_eval_result.json")
        with open(ndcg_output_path, 'w') as f:
            json.dump({'top1': top_k_hits[0], 'top5': top_k_hits[4],
                       'top20': top_k_hits[19], 'top100': top_k_hits[99], 'result_dict': result_dict}, f, indent=2)
        output_path = os.path.join(args.output_dir, 'test_result_dict_list.json')
        with open(output_path, 'w') as f:
            json.dump(result_dict_list, f, indent=2)

        output_path = os.path.join(args.output_dir, "test_hit_result.pkl")
        torch.save(scores, output_path)

        if args.train_qa_path is not None:
            logger.info("***** Begin train ANN Index *****")
            faiss.omp_set_num_threads(args.thread_num)
            similar_scores, train_I = cpu_index.search(train_question_embedding.astype(np.float32),
                                                       100)  # I: [number of queries, topk]
            logger.info("***** Done train ANN search *****")

            logger.info("***** Begin train validate *****")
            top_k_hits, scores, result_dict, result_dict_list = \
                validate(train_questions, passage_text, train_answers, train_I,
                         similar_scores, train_question_embedding2id, passage_embedding2id)
            logger.info("***** Done train validate *****")

            logger.info("***** Done train validate *****")
            ndcg_output_path = os.path.join(args.output_dir, "train_eval_result.json")
            with open(ndcg_output_path, 'w') as f:
                json.dump({'top1': top_k_hits[0], 'top5': top_k_hits[4],
                           'top20': top_k_hits[19], 'top100': top_k_hits[99], 'result_dict': result_dict}, f, indent=2)
            output_path = os.path.join(args.output_dir, 'train_result_dict_list.json')
            with open(output_path, 'w') as f:
                json.dump(result_dict_list, f, indent=2)
            output_path = os.path.join(args.output_dir, "train_hit_result.pkl")
            torch.save(scores, output_path)

            if args.write_hardneg==True and args.golden_train_qa_path:
                with open(args.golden_train_qa_path, "r", encoding="utf-8") as ifile:
                    # file format: question, answers
                    golden_list = json.load(ifile)
                q_pos_dict = {}
                for example in golden_list:
                    if len(example['positive_ctxs']) > 0:
                        q_pos_dict[example['question']] = example['positive_ctxs'][0]
                transfer_list = reform_out(result_dict_list, q_pos_dict)

                output_path = os.path.join(args.output_dir, 'train_ce_hardneg.json')
                with open(output_path, 'w') as f:
                    json.dump(transfer_list, f, indent=2)

        if args.dev_qa_path is not None:
            logger.info("***** Begin dev ANN Index *****")
            faiss.omp_set_num_threads(args.thread_num)
            similar_scores, dev_I = cpu_index.search(dev_question_embedding.astype(np.float32),
                                                     100)  # I: [number of queries, topk]
            logger.info("***** Done dev ANN search *****")

            logger.info("***** Begin dev validate *****")
            top_k_hits, scores, result_dict, result_dict_list = \
                validate(dev_questions, passage_text, dev_answers, dev_I,
                         similar_scores, dev_question_embedding2id, passage_embedding2id)
            logger.info("***** Done dev validate *****")

            logger.info("***** Done dev validate *****")
            ndcg_output_path = os.path.join(args.output_dir, "dev_eval_result.json")
            with open(ndcg_output_path, 'w') as f:
                json.dump({'top1': top_k_hits[0], 'top5': top_k_hits[4],
                           'top20': top_k_hits[19], 'top100': top_k_hits[99], 'result_dict': result_dict}, f, indent=2)
            output_path = os.path.join(args.output_dir, 'dev_result_dict_list.json')
            with open(output_path, 'w') as f:
                json.dump(result_dict_list, f, indent=2)
            output_path = os.path.join(args.output_dir, "dev_hit_result.pkl")
            torch.save(scores, output_path)

            if args.write_hardneg==True and args.golden_dev_qa_path:
                with open(args.golden_dev_qa_path, "r", encoding="utf-8") as ifile:
                    # file format: question, answers
                    golden_list = json.load(ifile)
                q_pos_dict = {}
                for example in golden_list:
                    if len(example['positive_ctxs']) > 0:
                        q_pos_dict[example['question']] = example['positive_ctxs'][0]
                transfer_list = reform_out(result_dict_list, q_pos_dict)

                output_path = os.path.join(args.output_dir, 'dev_ce_hardneg.json')
                with open(output_path, 'w') as f:
                    json.dump(transfer_list, f, indent=2)


def GeneratePassaageID(args, passages, answers, query_embedding2id, passage_embedding2id, closest_docs,
                       training_query_positive_id):
    query_negative_passage = {}
    query_positive_passage = {}
    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    for query_idx in tqdm(range(closest_docs.shape[0])):
        query_id = query_embedding2id[query_idx]

        pos_pid = training_query_positive_id[query_id]
        doc_ids = [passage_embedding2id[pidx] for pidx in closest_docs[query_idx]]

        query_negative_passage[query_id] = []
        query_positive_passage[query_id] = []
        neg_cnt = 0
        pos_cnt = 0
        for doc_id in doc_ids:
            if doc_id == pos_pid:
                continue
            if doc_id in query_negative_passage[query_id]:
                continue
            if neg_cnt >= args.negative_sample:
                break

            text = passages[doc_id][0]

            if not has_answer(answers[query_id], text, tokenizer):
                query_negative_passage[query_id].append(doc_id)
                neg_cnt += 1
            else:
                if pos_cnt < 10:
                    query_positive_passage[query_id].append(doc_id)
                    pos_cnt += 1
    return query_negative_passage, query_positive_passage


class V_dataset(Dataset):
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
    v_dataset = V_dataset(questions, passages, answers, closest_docs, similar_scores, query_embedding2id,
                          passage_embedding2id)
    v_dataloader = DataLoader(v_dataset, 128, shuffle=False, num_workers=20, collate_fn=V_dataset.collect_fn())
    final_scores = []
    final_result_list = []
    final_result_dict_list = []
    for k, (scores, result_list, result_dict_list) in enumerate(tqdm(v_dataloader)):
        final_scores.extend(scores)  # 等着func的计算结果
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


import math


class Eval_Tool:
    @classmethod
    def MRR_n(cls, results_list, n):
        mrr_100_list = []
        for hits in results_list:
            score = 0
            for rank, item in enumerate(hits[:n]):
                if item:
                    score = 1.0 / (rank + 1.0)
                    break
            mrr_100_list.append(score)
        return sum(mrr_100_list) / len(mrr_100_list)

    @classmethod
    def MAP_n(cls, results_list, n):
        MAP_n_list = []
        for predict in results_list:
            ap = 0
            hit_num = 1
            for rank, item in enumerate(predict[:n]):
                if item:
                    ap += hit_num / (rank + 1.0)
                    hit_num += 1
            ap /= n
            MAP_n_list.append(ap)
        return sum(MAP_n_list) / len(MAP_n_list)

    @classmethod
    def DCG_n(cls, results_list, n):
        DCG_n_list = []
        for predict in results_list:
            DCG = 0
            for rank, item in enumerate(predict[:n]):
                if item:
                    DCG += 1 / math.log2(rank + 2)
            DCG_n_list.append(DCG)
        return sum(DCG_n_list) / len(DCG_n_list)

    @classmethod
    def nDCG_n(cls, results_list, n):
        nDCG_n_list = []
        for predict in results_list:
            nDCG = 0
            for rank, item in enumerate(predict[:n]):
                if item:
                    nDCG += 1 / math.log2(rank + 2)
            nDCG /= sum([math.log2(i + 2) for i in range(n)])
            nDCG_n_list.append(nDCG)
        return sum(nDCG_n_list) / len(nDCG_n_list)

    @classmethod
    def P_n(cls, results_list, n):
        p_n_list = []
        for predict in results_list:
            true_num = 0
            for rank, item in enumerate(predict[:n]):
                if item:
                    true_num += 1
            p = true_num / n
            p_n_list.append(p)
        return sum(p_n_list) / len(p_n_list)

    @classmethod
    def get_matrics(cls, results_list):
        p_list = [1, 5, 10, 20, 50, 100]
        metrics = {'MRR_n': cls.MRR_n,
                   'MAP_n': cls.MAP_n,
                   'DCG_n': cls.DCG_n, 'nDCG_n': cls.nDCG_n, 'P_n': cls.P_n}
        result_dict = {}
        for metric_name, fuction in metrics.items():
            for p in p_list:
                temp_result = fuction(results_list, p)
                result_dict[metric_name + '@_' + str(p)] = temp_result
        return result_dict


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_qa_path",
        default=None,
        type=str,
        required=True,
        help="test_qa_path",
    )
    parser.add_argument(
        "--train_qa_path",
        default=None,
        type=str,
        required=False,
        help="test_qa_path",
    )
    parser.add_argument(
        "--dev_qa_path",
        default=None,
        type=str,
        required=False,
        help="test_qa_path",
    )
    parser.add_argument(
        "--golden_train_qa_path",
        default=None,
        type=str,
        required=False,
        help="test_qa_path",
    )
    parser.add_argument(
        "--golden_dev_qa_path",
        default=None,
        type=str,
        required=False,
        help="test_qa_path",
    )
    # Required parameters
    parser.add_argument(
        "--eval_model_dir",
        default=None,
        type=str,
        required=True,
        help="Initial model dir, will use this if no checkpoint is found in model_dir",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training data will be written",
    )
    parser.add_argument(
        "--passage_path",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training data will be written",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=128,
        type=int,
        help="The starting output file number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
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
        "--mode",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--load_cache",
        action="store_true",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--thread_num",
        type=int,
        default=90,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
    )
    parser.add_argument("--write_hardneg", type=bool, default=False)
    args = parser.parse_args()

    return args


def set_env(args):
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

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

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


def evaluate(args, tokenizer, model):
    if is_first_worker():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    if args.local_rank != -1:
        dist.barrier()
    logger.info("start eval")
    logger.info("eval checkpoint at " + args.eval_model_dir)
    generate_new_embeddings(args, tokenizer, model)
    logger.info("finished eval")


def main():
    args = get_arguments()
    set_env(args)
    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    if args.local_rank != 0:
        dist.barrier()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.local_rank == 0:
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
    evaluate(args, tokenizer, model)


if __name__ == "__main__":
    main()
