import argparse
import sys

from torch.utils.data.dataset import Dataset

sys.path += ['../']
import json
import logging
import os
from os.path import isfile, join
import random
import time
import csv
import numpy as np
import torch
from pathlib import Path
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
#  
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch import nn
from utils.util import (
    is_first_worker,
)
from model.models import BiEncoderNllLoss, BiBertEncoder, HFBertEncoder
from utils.dpr_utils import load_states_from_checkpoint, get_model_obj
import random
import pickle
from transformers import (
    BertTokenizer
)
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
logger = logging.getLogger(__name__)
import faiss

def load_id_text(file_name):
    """load tsv files"""
    id_text = {}
    with open(file_name) as inp:
        for line in tqdm(inp):
            line = line.strip()
            id, text = line.split('\t')
            id_text[id] = text
    return id_text
def load_data(args):
    passage_title_path = os.path.join(args.passage_path,"para.title.txt")
    passage_ctx_path = os.path.join(args.passage_path,"para.txt")
    passage_title = load_id_text(passage_title_path)
    passages = []
    with open(passage_ctx_path) as inp:
        for line in tqdm(inp):
            line = line.strip()
            id, text = line.split('\t')
            # return id , passage , passage title
            passages.append((int(id), text,passage_title.get(id, '-')))
    return passages

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

    args.model_type = args.model_type

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
    model.eval()
    return tokenizer, model



def get_question_embeddings(args, questions, tokenizer, model):
    batch_size = args.per_gpu_eval_batch_size

    dataset = Question_dataset(questions,tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                             num_workers=20, collate_fn=Question_dataset.get_collate_fn(args))
    total = 0
    allids, allembeddings = [], []
    with torch.no_grad():
        for k, (ids, text_ids, text_mask) in enumerate(tqdm(dataloader)):
            inputs = {"input_ids": text_ids.long().to(args.device), "attention_mask": text_mask.long().to(args.device)}
            embs = model.module.query_emb(**inputs)
            embeddings = embs.detach().cpu()
            total += len(ids)

            allids.append(ids)
            allembeddings.append(embeddings)
            if k % 100 == 0:
                logger.info('Encoded question %d', total)

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    allids = np.array([x for idlist in allids for x in idlist])
    return allembeddings,allids 


def get_passage_embedding(args, passages, model, tokenizer):
    if args.load_cache:
        pass
    else:
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
    passage_embedding,passage_embedding_id = None,None
    if is_first_worker():
        passage_embedding_list = []
        passage_embedding_id_list = []
        for i in range(args.world_size):  # TODO: dynamically find the max instead of HardCode
            pickle_path = os.path.join(args.output_dir,"{1}_data_obj_{0}.pb".format(str(i),'passage_embedding'))
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                passage_embedding_list.append(b)
        for i in range(args.world_size):  # TODO: dynamically find the max instead of HardCode
            pickle_path = os.path.join(args.output_dir,"{1}_data_obj_{0}.pb".format(str(i),'passage_embedding_id'))
            with open(pickle_path, 'rb') as handle:
                b = pickle.load(handle)
                passage_embedding_id_list.append(b)
        passage_embedding = np.concatenate(passage_embedding_list, axis=0)
        passage_embedding_id = np.concatenate(passage_embedding_id_list, axis=0)
    dist.barrier()
    return passage_embedding, passage_embedding_id

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

class Question_dataset(Dataset):
    def __init__(self, questions, tokenizer,maxlength=32):
        self.questions = questions
        self.tokenizer = tokenizer
        self.maxlength = maxlength
    def __getitem__(self, index):
        example = self.questions[index]
        input_ids = self.tokenizer.encode(example[1], add_special_tokens=True,
                                        max_length=self.maxlength, truncation=True,
                                       padding='max_length',return_tensors='pt') 
        return example[0], input_ids

    def __len__(self):
        return len(self.questions)

    @classmethod
    def get_collate_fn(cls, args):
        def fn(features):
            id_list = [feature[0] for feature in features]
            q_tensor = torch.cat([feature[1] for feature in features])
            return np.array(id_list), q_tensor, (q_tensor != 0).long()
        return fn


class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,tokenizer,maxlength =128):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlength = maxlength
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        title = convert_to_unicode(example[2])
        text = convert_to_unicode(example[1])
        input_ids = self.tokenizer.encode(title, text_pair=text, add_special_tokens=True,
                                        max_length=self.maxlength, truncation=True,
                                       padding='max_length',return_tensors='pt') 
        return example[0],input_ids
    @classmethod
    def get_collate_fn(cls, args):
        def fn(features):
            id_list = [feature[0] for feature in features]
            input_ids = torch.cat([feature[1] for feature in features])
            return np.array(id_list), input_ids, (input_ids!= 0).long()
        return fn
    

def embed_passages(opt, passages, model, tokenizer):
    batch_size = opt.per_gpu_eval_batch_size
    dataset = TextDataset(passages, tokenizer, opt.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=20, collate_fn=TextDataset.get_collate_fn(opt))
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


def generate_new_embeddings(args, tokenizer, model):
    # passage_text, test_questions, test_answers = preloaded_data

    passages = load_data(args)
    logger.info("***** inference of passages *****")

    if args.load_cache and is_first_worker():
        passage_embedding, passage_embedding2id = get_passage_embedding(args, passages, model,tokenizer)
    else:
        passage_embedding, passage_embedding2id = get_passage_embedding(args, passages, model,tokenizer)
    logger.info("***** Done passage inference *****")
    if args.test_qa_path is not None:
        logger.info("***** inference of test query *****")

        logger.info("Loading test question")
        test_questions = []
        with open(args.test_qa_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                id, text = line.split('\t')
                test_questions.append([int(id),text])
        test_question_embedding, test_question_embedding2id = get_question_embeddings(args, test_questions, 
                                                                                        tokenizer, model)

    '''train dataset_generation'''
    if args.train_qa_path is not None:
        logger.info("***** inference of train query *****")
        train_questions = []
        logger.info("Loading train question")
        with open(args.train_qa_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                id, text = line.split('\t')
                train_questions.append([int(id),text])
        train_question_embedding, train_question_embedding2id = get_question_embeddings(args, train_questions,
                                                                                        tokenizer, model)

    '''eval dataset_generation'''
    if args.dev_qa_path is not None:
        logger.info("***** inference of dev query *****")
        dev_questions = []
        logger.info("Loading dev answers")
        with open(args.dev_qa_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                id, text = line.split('\t')
                dev_questions.append([int(id),text])
        dev_question_embedding, dev_question_embedding2id = get_question_embeddings(args, dev_questions, tokenizer,
                                                                                    model)

    ''' test eval'''
    if is_first_worker():
        passage_text = {}
        for passage in passages:
            passage_text[passage[0]] = (passage[1],passage[2])

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
        # faiss.omp_set_num_threads(args.thread_num)
        cpu_index = faiss.IndexFlatIP(dim)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        # co.useFloat16 = True
        gpu_index_flat = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index,
            co=co
        )
        # gpu_index_flat = faiss.index_cpu_to_all_gpus(cpu_index)
        gpu_index_flat.add(passage_embedding.astype(np.float32))
        # output_path = os.path.join(args.output_dir, 'faiss.index')
        # faiss.write_index(cpu_index, output_path)
        logger.info("***** Done ANN Index *****")

        if args.test_qa_path is not None:
            logger.info("***** Begin test ANN Index *****")
            # faiss.omp_set_num_threads(args.thread_num)
            similar_scores, dev_I = gpu_index_flat.search(test_question_embedding.astype(np.float32),
                                                    args.top_k)  # I: [number of queries, topk]
            logger.info("***** Done test ANN search *****")

            logger.info("***** Begin test validate *****")
            #merge 输出 dev_I 和test_question_embedding2id，输出：字典
            qids_to_ranked_candidate_passages  = {}
            qids_to_ranked_candidate_scores = {}
            for index,ranked_candidate_passages in enumerate(dev_I):
                qids_to_ranked_candidate_passages[test_question_embedding2id[index]] = ranked_candidate_passages
                qids_to_ranked_candidate_scores[test_question_embedding2id[index]] = similar_scores[index]

            output_path = os.path.join(args.output_dir, 'test_result_dict_list.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump([qids_to_ranked_candidate_passages,qids_to_ranked_candidate_scores], f)

            # ground truth path
            # data_dir = os.path.abspath(os.path.dirname(args.test_qa_path))
            ground_truth_path = args.ground_truth_path
            # ground_truth_path = os.path.join(data_dir,'qrels.dev.tsv')
            qids_to_relevant_passageids = load_reference_from_stream(ground_truth_path)
            all_scores = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
            logger.info("***** Done test validate *****")

            logger.info("***** Done test validate *****")
            ndcg_output_path = os.path.join(args.output_dir, "test_eval_result.json")
            with open(ndcg_output_path, 'w') as f:
                json.dump(all_scores, f, indent=2)

        if args.train_qa_path is not None:
            logger.info("***** Begin train ANN Index *****")
            # faiss.omp_set_num_threads(args.thread_num)
            similar_scores, train_I = gpu_index_flat.search(train_question_embedding.astype(np.float32),
                                                       200)  # I: [number of queries, topk]
            logger.info("***** Done train ANN search *****")

            logger.info("***** Begin train validate *****")
            qids_to_ranked_candidate_passages  = {}
            qids_to_ranked_candidate_scores = {}
            for index,ranked_candidate_passages in enumerate(train_I):
                qids_to_ranked_candidate_passages[train_question_embedding2id[index]] = ranked_candidate_passages
                qids_to_ranked_candidate_scores[train_question_embedding2id[index]] = similar_scores[index]

            output_path = os.path.join(args.output_dir, 'train_result_dict_list.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump([qids_to_ranked_candidate_passages,qids_to_ranked_candidate_scores], f)

            train_ground_truth_path = args.train_ground_truth_path
            # ground_truth_path = os.path.join(data_dir,'qrels.dev.tsv')
            qids_to_relevant_passageids = load_train_reference_from_stream(train_ground_truth_path)
            all_scores = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
            ndcg_output_path = os.path.join(args.output_dir, "train_eval_result.json")
            with open(ndcg_output_path, 'w') as f:
                json.dump(all_scores, f, indent=2)

            logger.info("***** Done train validate *****")

        # if args.dev_qa_path is not None:
        #     logger.info("***** Begin dev ANN Index *****")
        #     faiss.omp_set_num_threads(args.thread_num)
        #     similar_scores, dev_I = gpu_index_flat.search(dev_question_embedding.astype(np.float32),
        #                                              1000)  # I: [number of queries, topk]
        #     logger.info("***** Done dev ANN search *****")
        #
        #     logger.info("***** Begin dev validate *****")
        #     qids_to_ranked_candidate_passages  = {}
        #     qids_to_ranked_candidate_scores = {}
        #     for index,ranked_candidate_passages in enumerate(dev_I):
        #         qids_to_ranked_candidate_passages[dev_question_embedding2id[index]] = ranked_candidate_passages
        #         qids_to_ranked_candidate_scores[dev_question_embedding2id[index]] = similar_scores[index]
        #
        #     output_path = os.path.join(args.output_dir, 'dev_result_dict_list.pkl')
        #
        #     with open(output_path, 'wb') as f:
        #         pickle.dump([qids_to_ranked_candidate_passages,qids_to_ranked_candidate_scores], f)
        #
        #     # ground truth path
        #     data_dir = os.path.abspath(os.path.dirname(args.dev_qa_path))
        #     ground_truth_path = os.path.join(data_dir,'qrels.dev.tsv')
        #     qids_to_relevant_passageids = load_reference_from_stream(ground_truth_path)
        #     all_scores = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        #     logger.info("***** Done dev validate *****")
        #     logger.info(all_scores)
        #     logger.info("***** Done dev validate *****")
        #     ndcg_output_path = os.path.join(args.output_dir, "dev_eval_result.json")
        #     with open(ndcg_output_path, 'w') as f:
        #         json.dump(all_scores, f, indent=2)

def load_reference_from_stream(path_to_reference):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    qids_to_relevant_passageids = {}
    with open(path_to_reference, 'r') as f:
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
    all_scores['record'] = str(round(recall_top1*100, 2)) + " / " + str(round(recall_top5*100, 2)) +" / " + str(round(recall_top20*100, 2)) +" / " + str(round(recall_top50*100, 2)) +" / " + str(round(recall_top100*100, 2)) +" / " + str(round(recall_all*100, 2))
    return all_scores

import math
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_qa_path",
        default=None,
        type=str,
        required=False,
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
        help="model type",
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
        "--ground_truth_path",
        default=None,
        type=str,
        required=True,
        help="The oground truth path",
    )
    parser.add_argument(
        "--train_ground_truth_path",
        default=None,
        type=str,
        required=True,
        help="Train data ground truth path",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=12,
        type=int,
        help="num layer of model",
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
        default=1000,
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
    dist.barrier()


if __name__ == "__main__":
    main()
