import argparse
import sys

from torch.utils.data.dataset import Dataset

sys.path += ['../']
import json
import logging
import os
from os.path import isfile, join
import csv
import numpy as np
import torch
import pytrec_eval
import json
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from tqdm import tqdm
import torch.distributed as dist
import pickle
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import regex

logger = logging.getLogger(__name__)
import faiss
from util import set_env, get_arguments, load_model, Eval_Tool, is_first_worker, SimpleTokenizer
import transformers
transformers.logging.set_verbosity_error()
csv.field_size_limit(sys.maxsize)

class Question_dataset(Dataset):
    def __init__(self, args, questions, tokenizer):
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_query_len = args.max_query_length

    def __getitem__(self, index):
        query = self.questions[index].replace("’", "'")
        question_token_ids = self.tokenizer.encode(query)[:self.max_query_len]
        return index, question_token_ids

    def __len__(self):
        return len(self.questions)
        # return 10

    @classmethod
    def get_collate_fn(cls, args):
        def fn(features):
            max_q_len = max([len(feature[1]) for feature in features])
            q_list = [feature[1] + [0] * (max_q_len - len(feature[1])) for feature in features]
            q_tensor = torch.LongTensor(q_list)
            id_list = [feature[0] for feature in features]
            return np.array(id_list), q_tensor, (q_tensor != 0).long()

        return fn

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data):
        self.data = data

    def __len__(self):
        return len(self.data)
        # return 10

    def __getitem__(self, index):
        example = self.data[index]
        return example[0], example[1], example[2]

class V_dataset(Dataset):
    def __init__(self, test_file_name, questions, passages, answers, closest_docs,
                 similar_scores, query_embedding2id, passage_embedding2id):
        self.questions = questions
        self.passages = passages
        self.test_file_name = test_file_name
        self.answers = answers
        self.closest_docs = closest_docs
        self.similar_scores = similar_scores
        self.query_embedding2id = query_embedding2id
        self.passage_embedding2id = passage_embedding2id
        tok_opts = {}

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
            hits.append(str(doc_id) in temp_result_dict['answers'])
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

class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        retrived_doc_title = [x[2] for x in batch]
        retrived_doc_text = [x[1] for x in batch]
        tokenized_docs = self.tokenizer(
            retrived_doc_title[:2000],
            retrived_doc_text[:2000],
            max_length=self.maxlength,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        text_ids = tokenized_docs['input_ids']
        text_mask = tokenized_docs['attention_mask']
        return index, text_ids, text_mask

def beir_evaluate(qrels, results, k_values):
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

    for eval in [ndcg, _map, recall, precision]:
        logging.info("\n")
        for k in eval.keys():
            logging.info("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision

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


def get_question_embeddings(args, questions, tokenizer, model):
    dataset = Question_dataset(args, questions, tokenizer)
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
            embs, _, _ = model.query_emb(args.model_type, **inputs)
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


def embed_passages(opt, passages, model, tokenizer):
    batch_size = opt.per_gpu_eval_batch_size
    collator = TextCollator(tokenizer, opt.max_doc_length)
    dataset = TextDataset(passages)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=10, collate_fn=collator)
    total = 0
    allids, allembeddings = [], []

    with torch.no_grad():
        for k, (ids, text_ids, text_mask) in enumerate(tqdm(dataloader)):
            inputs = {"input_ids": text_ids.long().to(opt.device), "attention_mask": text_mask.long().to(opt.device)}
            embs, _, _ = model.body_emb(opt.model_type, **inputs)
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
    if os.path.exists(os.path.join(args.output_dir, "passage_embedding_data_obj_0.pb")):
        args.load_cache = True

    if not args.load_cache:
        shard_size = len(passages) // args.world_size
        start_idx = args.local_rank * shard_size if args.local_rank >= 0 else 0
        end_idx = start_idx + shard_size
        if args.local_rank == args.world_size - 1 or args.local_rank == -1:
            end_idx = len(passages)
        passages_piece = passages[start_idx:end_idx]
        logger.info(f'Embedding generation for {len(passages_piece)} passages from idx {start_idx} to {end_idx}')
        allids, allembeddings = embed_passages(args, passages_piece, model, tokenizer)
        if is_first_worker():
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        if (args.local_rank != -1):
            dist.barrier()
        pickle_path = os.path.join(args.output_dir,
                                   "{1}_data_obj_{0}.pb".format(str(args.local_rank if args.local_rank >= 0 else 0), 'passage_embedding'))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(allembeddings, handle, protocol=4)
        pickle_path = os.path.join(args.output_dir,
                                   "{1}_data_obj_{0}.pb".format(str(args.local_rank if args.local_rank >= 0 else 0), 'passage_embedding_id'))
        with open(pickle_path, 'wb') as handle:
            pickle.dump(allids, handle, protocol=4)
        logger.info(f'Total passages processed {len(allids)}. Written to {pickle_path}.')
        if (args.local_rank != -1):
            dist.barrier()

    passage_embedding, passage_embedding_id = None, None
    if is_first_worker():
        passage_embedding_list = []
        passage_embedding_id_list = []
        for i in range(args.world_size if args.local_rank != -1 else 8):  # TODO: dynamically find the max instead of HardCode
            pickle_path = os.path.join(args.output_dir, "{1}_data_obj_{0}.pb".format(str(i), 'passage_embedding'))
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                passage_embedding_list.append(b)
        for i in range(args.world_size if args.local_rank != -1 else 8):  # TODO: dynamically find the max instead of HardCode
            pickle_path = os.path.join(args.output_dir, "{1}_data_obj_{0}.pb".format(str(i), 'passage_embedding_id'))
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                passage_embedding_id_list.append(b)
        passage_embedding = np.concatenate(passage_embedding_list, axis=0)
        passage_embedding_id = np.concatenate(passage_embedding_id_list, axis=0)
    return passage_embedding, passage_embedding_id


def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Compute MRR metric"""
    MaxMRRRank = 10
    all_scores = {}
    MRR = 0
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
    all_scores['record'] = str(round(recall_top1*100, 2)) + " / " + str(round(recall_top5*100, 2)) + " / " + str(round(recall_top20*100, 2)) + \
        " / " + str(round(recall_top50*100, 2)) + " / " + str(round(recall_top100*100, 2)) + " / " + str(round(recall_all*100, 2))
    return all_scores


def load_reference_from_stream(path_to_reference):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    dev_query_positive_id = {}
    with open(path_to_reference, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter=" ")
        count = 0
        for [topicid, _,docid, rel] in tsvreader:
            topicid = int(topicid)
            docid = int(docid)
            if str(count) not in dev_query_positive_id:
                dev_query_positive_id[str(count)] = {}
            dev_query_positive_id[str(count)][str(docid)] = int(rel)
            count += 1
    return dev_query_positive_id


def generate_new_embeddings(args, tokenizer, model):
    # passage_text, test_questions, test_answers = preloaded_data
    passages = load_data(args)
    logger.info("***** inference of passages *****")

    ## passage_embedding2id: [reverted index] : original id in test file
    passage_embedding, passage_embedding2id = get_passage_embedding(args, passages, model, tokenizer)
    logger.info("***** Done passage inference *****")

    logger.info("***** inference of test query *****")
    test_questions = []
    test_answers = []
    logger.info("Loading test answers")
    with open(args.test_file, "r", encoding="utf-8") as ifile:
        # file format: question, answers
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            test_questions.append(row[0])
            test_answers.append(eval(row[1]))

    ## test_question_embedding2id: [reverted index] - original id in test file
    test_question_embedding, test_question_embedding2id = get_question_embeddings(args, test_questions, tokenizer, model)

    ''' test eval'''
    if is_first_worker():
    # if False:
        passage_text = {}
        for passage in passages:
            passage_text[passage[0]] = (passage[1], passage[2])

        dim = passage_embedding.shape[1]
        print('passage embedding shape: ' + str(passage_embedding.shape))
        ## the aim of reorder is to let dev_I variable directly map to the docid
        logger.info("***** Begin passage_embedding reorder *****")
        new_passage_embedding = passage_embedding.copy()
        for i in range(passage_embedding.shape[0]):
            new_passage_embedding[passage_embedding2id[i]] = passage_embedding[i]
        del (passage_embedding)
        passage_embedding = new_passage_embedding
        ## passage_embedding2id: [original id] : original id in test file
        passage_embedding2id = np.arange(passage_embedding.shape[0])
        logger.info("***** Begin passage_embedding reorder  *****")

        logger.info("***** Begin ANN Index build *****")
        top_k = args.top_k
        faiss.omp_set_num_threads(args.thread_num)
        cpu_index = faiss.IndexFlatIP(dim)
        logger.info("***** Begin cpu_index *****")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        gpu_index_flat = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index,
            co=co
        )
        logger.info("***** Begin faiss *****")
        gpu_index_flat.add(passage_embedding.astype(np.float32))
        cpu_index = gpu_index_flat
        logger.info("***** Done ANN Index *****")

        logger.info("***** Begin test ANN Index *****")
        faiss.omp_set_num_threads(args.thread_num)
        # dev_I: [number of queries, top_k], top_k default 1000,
        # index is the corsponding to test_question_embedding2id and test_question_embedding
        # [reverted index] - original passage id
        similar_scores, dev_I = cpu_index.search(test_question_embedding.astype(np.float32), top_k)
        logger.info("***** Done test ANN search *****")

        if 'ms' in args.test_file or 'md' in args.test_file:
            search_result_topk = 1000
        else:
            search_result_topk = 1000

        logger.info(f"***** Saving Top-{search_result_topk} Search Result *****")
        all_doc_list = []
        for i in range(len(test_question_embedding2id)):
            query = test_question_embedding2id[i]
            doc_list = [passage_embedding2id[index] for index in dev_I[i][:search_result_topk]]
            id1 = [query] + doc_list
            all_doc_list.append(id1)

        pickle_path = os.path.join(args.output_dir, f"search_result_{search_result_topk}.pb")
        with open(pickle_path, 'wb') as handle:
            pickle.dump(all_doc_list, handle, protocol=4)
        logger.info("***** Done Search Result *****")

######################################################################################################################################################
        # qids_to_relevant_passageids = {}
        # for i in range(len(test_answers)):
        #     qids_to_relevant_passageids[str(i)] = {str(elem):1 for elem in test_answers[i]}
        #
        # qids_to_ranked_candidate_passages = {}
        # for index, ranked_candidate_passages in enumerate(dev_I):
        #     qids_to_ranked_candidate_passages[str(test_question_embedding2id[index])] = {str(ranked_candidate_passages[i]): -i for i in range(len(ranked_candidate_passages))}

        # merge 输出 dev_I 和test_question_embedding2id，输出：字典
        # qids_to_ranked_candidate_passages = {}
        # for index, ranked_candidate_passages in enumerate(dev_I):
        #     qids_to_ranked_candidate_passages[str(test_question_embedding2id[index])] = list(ranked_candidate_passages)

        ## Some other metrics
        # ms_mrr = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)

    ######################################################################################################################################################

        logger.info("***** Begin test validate *****")
        top_k_hits, scores, result_dict, result_dict_list = \
            validate(args.test_file, test_questions, passage_text, test_answers, dev_I,
                     similar_scores, test_question_embedding2id, passage_embedding2id)
        logger.info("***** Done test validate *****")

        result_dict['top1'] = top_k_hits[0]
        result_dict['top5'] = top_k_hits[4]
        result_dict['top20'] = top_k_hits[19]
        result_dict['top50'] = top_k_hits[49]
        result_dict['top100'] = top_k_hits[99]
        result_dict['top1000'] = top_k_hits[999]
        result_dict['MAP@1000'] = result_dict['MAP_n@_1000']
        ndcg_output_path = os.path.join(args.output_dir, f"test_eval_result_{top_k}.json")
        with open(ndcg_output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)


def validate(test_file_name, questions, passages, answers, closest_docs, similar_scores, query_embedding2id, passage_embedding2id):
    v_dataset = V_dataset(test_file_name, questions, passages, answers, closest_docs, similar_scores, query_embedding2id,
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

    # logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(closest_docs) for v in top_k_hits]
    # logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    with open('final_result_dict_list.pkl', 'wb') as f:
        pickle.dump(final_result_dict_list, f)
    f.close()
    return top_k_hits, final_scores, Eval_Tool.get_matrics(final_scores), final_result_dict_list


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
    if args.local_rank != 0 and args.local_rank != -1:
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
