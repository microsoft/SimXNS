import sys

from torch.utils.data.dataset import Dataset

sys.path += ['../']
import json
import logging
import os
import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
#
from tqdm import tqdm
import torch.distributed as dist
from utils.util import (
    is_first_worker,
)
import pickle
from torch.utils.data import DataLoader

logger = logging.getLogger("__main__")
import faiss
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
                 data,tokenizer,maxlength = 128):
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


def embed_passages(args, passages, model, tokenizer):
    batch_size = 256
    dataset = TextDataset(passages, tokenizer, 128)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=20,
                            collate_fn=TextDataset.get_collate_fn(args))
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
            if k % 1000 == 0:
                logger.info('Encoded passages %d', total)

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    allids = np.array([x for idlist in allids for x in idlist])
    return allids, allembeddings


def load_pos_examples(q_type='train'):
    """positive examples(only for MSMARCO)"""
    pos_qp = {}
    file = os.path.join('data/MS-Pas/qrels.' + q_type + '.tsv')
    with open(file) as inp:
        for line in inp:
            line = line.strip()
            qid, pid = line.split('\t')
            qid, pid = int(qid), int(pid)
            if qid not in pos_qp:
                pos_qp[qid] = []
            pos_qp[qid].append(pid)
    print('positive qids: %s' % len(pos_qp))
    # additional positive examples(collect by literal match)
    pos_qp_add = {}
    if q_type == 'train':
        file_add = os.path.join('data/MS-Pas/qrels.' + q_type + '.addition.tsv')
        with open(file_add) as inp:
            for line in inp:
                qid, pid = line.strip().split('\t')
                qid, pid = int(qid), int(pid)
                if qid not in pos_qp_add:
                    pos_qp_add[qid] = []
                pos_qp_add[qid].append(pid)
    else:
        pass
    return pos_qp, pos_qp_add


def write_to_file(qids_to_ranked_candidate_passages, qids_to_ranked_candidate_scores,
                  q_text, pos_qp, pos_qp_add, q_type, save_path='/quantus-nfs/zh/AN_dpr/data_train/',
                  global_step = 0):
    q_text_dict={}
    for item in q_text:
        q_text_dict[item[0]]=item[1]
    result_dict_list = []
    for i, (q_id, p_id_list) in enumerate(tqdm(qids_to_ranked_candidate_passages.items())):
        temp_result_dict = {}
        temp_result_dict['q_id'] = str(q_id)
        temp_result_dict['question'] = q_text_dict[q_id]
        temp_result_dict['positive_ctxs_id'] = []
        temp_result_dict['hard_negative_ctxs_id'] = []
        temp_result_dict['hard_negative_ctxs_score'] = []
        temp_pos_list = pos_qp[q_id] + pos_qp_add.get(q_id, [])
        temp_pos = {ele:0 for ele in temp_pos_list}

        score_list = qids_to_ranked_candidate_scores[q_id]
        for j, (doc_id, doc_score) in enumerate(zip(p_id_list[:200], score_list[:200])):
            if doc_id in temp_pos:
                temp_pos[doc_id] = doc_score
            else:
                # text, title = p_text[doc_id],p_title.get(doc_id, '-')
                temp_result_dict['hard_negative_ctxs_id'].append((str(doc_id), str(doc_score)))
                # temp_result_dict['hard_negative_ctxs_score'].append(str(qids_to_ranked_candidate_scores[q_id][j]))

        for doc_id in temp_pos:
            # text, title = p_text[doc_id], p_title.get(doc_id, '-')
            temp_result_dict['positive_ctxs_id'].append((str(doc_id), str(temp_pos[doc_id])))

        #else:
        #    add_neg_num = 15 - len(temp_result_dict['hard_negative_ctxs_id'])
        #    temp_result_dict['hard_negative_ctxs_id'].extend(hard_negatives_set[-add_neg_num:])
        result_dict_list.append(temp_result_dict)
    out_path = os.path.join(save_path, q_type+'_ce_'+str(global_step) + '.tsv')
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, temp_result_dict in enumerate(tqdm(result_dict_list)):
            f.write('%s\t%s\t%s\t%s\n' % (temp_result_dict['q_id'],
                                          temp_result_dict['question'],
                                          ",".join([pair[0]+' '+pair[1] for pair in temp_result_dict['positive_ctxs_id']]),
                                          ",".join([pair[0]+' '+pair[1] for pair in temp_result_dict['hard_negative_ctxs_id']])))


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
    recall_q_top50 = set()
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
                    if i < 50:
                        recall_q_top50.add(qid)
                    if i == 0:
                        recall_q_top1.add(qid)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

    MRR = MRR / len(qids_to_relevant_passageids)
    recall_top1 = len(recall_q_top1) * 1.0 / len(qids_to_relevant_passageids)
    recall_top50 = len(recall_q_top50) * 1.0 / len(qids_to_relevant_passageids)
    recall_all = len(recall_q_all) * 1.0 / len(qids_to_relevant_passageids)
    all_scores['MRR @10'] = MRR
    all_scores["recall@1"] = recall_top1
    all_scores["recall@50"] = recall_top50
    all_scores["recall@all"] = recall_all
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    return all_scores

def load_id_text(file_name):
    """load tsv files"""
    id_text = {}
    with open(file_name) as inp:
        for line in tqdm(inp):
            line = line.strip()
            id, text = line.split('\t')
            id_text[id] = text
    return id_text

class RenewTools:
    def __init__(self, passages_title_path,passages_ctx_path, tokenizer, output_dir, temp_dir):
        self.passages = self.load_passage(passages_title_path,passages_ctx_path)
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
                                       "{1}_data_obj_{0}.pb".format(str(args.rank), 'passage_embedding'))
            with open(pickle_path, 'wb') as handle:
                pickle.dump(allembeddings, handle, protocol=4)
            pickle_path = os.path.join(self.temp_dir,
                                       "{1}_data_obj_{0}.pb".format(str(args.rank), 'passage_embedding_id'))
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
                pickle_path = os.path.join(self.temp_dir, "{1}_data_obj_{0}.pb".format(str(i), 'passage_embedding'))
                with open(pickle_path, 'rb') as handle:
                    b = pickle.load(handle)
                    passage_embedding_list.append(b)
            logger.info('load_passage_id_begin')
            for i in tqdm(range(args.world_size)):  # TODO: dynamically find the max instead of HardCode
                pickle_path = os.path.join(self.temp_dir, "{1}_data_obj_{0}.pb".format(str(i), 'passage_embedding_id'))
                with open(pickle_path, 'rb') as handle:
                    b = pickle.load(handle)
                    passage_embedding_id_list.append(b)
            passage_embedding = np.concatenate(passage_embedding_list, axis=0)
            passage_embedding_id = np.concatenate(passage_embedding_id_list, axis=0)
            logger.info('load_passage_done')
        return passage_embedding, passage_embedding_id

    def get_question_embeddings_sub(self, args, questions, model):
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
        # co.useFloat16 = True
        gpu_index_flat = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index,
            co=co
        )
        logger.info("***** begin add passages  *****")
        gpu_index_flat.add(passage_embedding.astype(np.float32))
        logger.info("***** end build index  *****")
        return gpu_index_flat, passage_embedding2id

    def load_passage(self, passages_path,passages_ctx_path):
        passage_title = load_id_text(passages_path)
        passages = []
        with open(passages_ctx_path) as inp:
            for line in tqdm(inp):
                line = line.strip()
                id, text = line.split('\t')
                passages.append((int(id), text,passage_title.get(id, '-')))
        return passages

    def get_question_embedding(self,args,model,qa_path,mode='train'):
        train_questions = []
        logger.info("Loading "+mode+" question")
        with open(qa_path, "r", encoding="utf-8") as ifile:
            for line in tqdm(ifile):
                line = line.strip()
                id, text = line.split('\t')
                train_questions.append([int(id),text])

        train_question_embedding, train_question_embedding2id = self.get_question_embeddings_sub(args, train_questions,
                                                                                             model)
        return train_questions,train_question_embedding, train_question_embedding2id

    # given q_a path ,generate
    def get_question_topk(self, train_questions,
                                train_question_embedding,
                                train_question_embedding2id,
                                golden_path, gpu_index_flat, passage_embedding2id,
                                mode='train', step_num=0):
        faiss.omp_set_num_threads(90)
        if mode == 'train':
            similar_scores, train_I = gpu_index_flat.search(train_question_embedding.astype(np.float32),
                                                        200)
        else:
            similar_scores, train_I = gpu_index_flat.search(train_question_embedding.astype(np.float32),
                                                        1000)  # I: [number of queries, topk]

        qids_to_ranked_candidate_passages  = {}
        qids_to_ranked_candidate_scores = {}
        for index,ranked_candidate_passages in enumerate(train_I):
            qids_to_ranked_candidate_passages[train_question_embedding2id[index]] = ranked_candidate_passages
            qids_to_ranked_candidate_scores[train_question_embedding2id[index]] = similar_scores[index]
        print(golden_path)
        qids_to_relevant_passageids = load_reference_from_stream(golden_path)
        all_scores = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)

        logger.info("***** Done "+mode+" validate *****")
        logger.info(all_scores)
        logger.info("***** Done "+mode+" validate *****")
        ndcg_output_path = os.path.join(self.output_dir, mode + "_eval_result"+str(step_num)+".json")
        with open(ndcg_output_path, 'w') as f:
            json.dump(all_scores, f, indent=2)

        train_pos_qp, train_pos_qp_add = load_pos_examples(mode)
        write_to_file(qids_to_ranked_candidate_passages, qids_to_ranked_candidate_scores, train_questions,
                          train_pos_qp, train_pos_qp_add, q_type=mode,
                          save_path=self.output_dir,global_step=step_num)
