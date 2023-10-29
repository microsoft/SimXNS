import faiss
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset, DataLoader
import logging
import torch
import os
import numpy as np
import torch.distributed as dist
import csv
import json
import sys
from datasets import load_dataset


pyfile_path = os.path.abspath(__file__) 
pyfile_dir = os.path.dirname(os.path.dirname(pyfile_path)) # equals to the path '../'
sys.path.append(pyfile_dir)
from utils.dpr_utils import SimpleTokenizer, has_answer, Eval_Tool
from utils.util import is_first_worker, normalize_question
from utils.data_utils import load_passage
from utils.evaluate_trec import load_qrel_data, EvalDevQuery


logger = logging.getLogger("__main__")

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        # psg_id, text, title
        return [example[0], example[1], example[2]]


class TextCollator(object):
    def __init__(self, tokenizer, max_seq_length=144, max_query_length=32, delimiter=' '):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.delimiter = delimiter

    def __call__(self, batch):
        index = [x[0] for x in batch]
        # title + body
        retrived_doc = [x[2].strip()+ self.delimiter + x[1].strip() for x in batch]
        tokenized_docs = self.tokenizer(
            retrived_doc,
            max_length=self.max_seq_length,
            truncation=True,
            padding=True,
            return_tensors='pt',
        )
        text_ids = tokenized_docs['input_ids']
        text_mask = tokenized_docs['attention_mask']

        return index, text_ids, text_mask


class QuestionDataset(Dataset):
    def __init__(self, questions, tokenizer, max_query_length):
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length

    def __getitem__(self, index):
        query = normalize_question(self.questions[index])
        question_token_ids = self.tokenizer.encode(query, max_length=self.max_query_length, truncation=True, padding=False)
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
    collator = TextCollator(tokenizer, args.max_seq_length, args.max_query_length, args.delimiter)
    dataset = TextDataset(passages)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=15, collate_fn=collator)
    total = 0
    allids, allembeddings = [], []
    with torch.no_grad():
        for k, (ids, text_ids, text_mask) in enumerate(tqdm(dataloader, desc="Encoded passages", disable=args.local_rank not in [-1, 0])):
            inputs = {"input_ids": text_ids.long().to(args.device), "attention_mask": text_mask.long().to(args.device)}
            if hasattr(model, 'module'):
                embs = model.module.body_emb(**inputs)
            else:
                embs = model.body_emb(**inputs)
            embeddings = embs.detach().cpu()
            total += len(ids)

            allids.extend(ids)
            allembeddings.extend(embeddings)
    
    s = sorted(zip(allids, allembeddings), key=lambda x: x[0])
    allids = [e[0] for e in s]
    allembeddings = np.array([e[1].numpy() for e in s])
    return allids, allembeddings


class VDataset(Dataset):
    def __init__(self, questions, passages, answers, closest_docs,
                 similar_scores, query_ids, real_question_ids, is_msmarco_data, passage_ids):
        # query_ids: List[int]
        # passage_ids: List[str]

        self.questions = questions
        self.passages = passages
        self.answers = answers
        self.closest_docs = closest_docs
        self.similar_scores = similar_scores
        self.query_ids = query_ids
        self.real_question_ids = real_question_ids
        self.is_msmarco_data =is_msmarco_data
        self.passage_ids = passage_ids
        
        tok_opts = {}
        self.tokenizer = SimpleTokenizer(**tok_opts)

        if self.is_msmarco_data:
            assert len(self.real_question_ids) == len(self.answers)

    def __getitem__(self, query_idx):
        # map the query_idx to the real order of self.questions and self.answers
        query_id = self.query_ids[query_idx] 
        # passage_ids maps index to  original psg_id
        doc_ids = [self.passage_ids[pidx] for pidx in self.closest_docs[query_idx]] 

        hits = []
        temp_result_dict = {}
        temp_result_dict['q_id'] = self.real_question_ids[query_id] if self.is_msmarco_data else str(query_id)
        temp_result_dict['question'] = self.questions[query_id]
        temp_result_dict['answers'] = self.answers[query_id]
        temp_result_dict['ctxs'] = []
        for i, doc_id in enumerate(doc_ids):
            passage = self.passages[str(doc_id)] #To be compatible with old calculation results
            text = passage[1]
            title = passage[2]
            score = str(self.similar_scores[query_idx, i]) 
            if self.is_msmarco_data: # for msmarco
                assert type(self.answers[query_id][0])==int
                if int(doc_id) in self.answers[query_id]: 
                    hits.append(True)
                else:
                    hits.append(False)
            else:
                assert type(self.answers[query_id][0])==str# for nq and tq
                hits.append(has_answer(self.answers[query_id], text, self.tokenizer))

            if i<200:
                temp_result_dict['ctxs'].append({'d_id': str(doc_id),
                                                'text': text,
                                                'title': title,
                                                'score': score,
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


def validate(questions, passages, answers, closest_docs, similar_scores, query_ids, real_question_ids, is_msmarco_data, passage_ids):
    v_dataset = VDataset(questions, passages, answers, closest_docs, similar_scores, query_ids, real_question_ids, is_msmarco_data, 
                          passage_ids)
    v_dataloader = DataLoader(v_dataset, 128, shuffle=False, drop_last=False, num_workers=15, collate_fn=VDataset.collect_fn())
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


def reform_out(result_dict_list, q_pos_dict, is_msmarco_data=False):
    transfer_list = []
    for infer_result in tqdm(result_dict_list):
        
        q_id = infer_result["q_id"]
        q_str = infer_result["question"]
        q_answer = infer_result["answers"]
        positive_ctxs = []
        negative_ctxs = []
        retrieved_positive_ctxs = []
        if q_str in q_pos_dict.keys():
            if is_msmarco_data:
                for real_true_dic in q_pos_dict[q_str]:
                    if 'passage_id' not in real_true_dic.keys() and 'id' in real_true_dic.keys():
                        real_true_dic['passage_id'] = real_true_dic['id']
                    elif 'psg_id' in real_true_dic.keys():
                        real_true_dic['passage_id'] = real_true_dic['psg_id']
                    positive_ctxs.append(real_true_dic)
            else:
                real_true_dic = q_pos_dict[q_str]
                if 'passage_id' not in real_true_dic.keys() and 'id' in real_true_dic.keys():
                    real_true_dic['passage_id'] = real_true_dic['id']
                elif 'psg_id' in real_true_dic.keys():
                    real_true_dic['passage_id'] = real_true_dic['psg_id']
                positive_ctxs.append(real_true_dic)

        for doc in infer_result['ctxs']:
            doc_text = doc['text']
            doc_title = doc['title']
            if doc['hit'] == "True":
                pass
                # if is_msmarco_data:
                #     retrieved_positive_ctxs.append({'title': doc_title, 'text': doc_text, 'passage_id': doc['d_id'], 'score': str(doc['score'])})
                # else:
                #     positive_ctxs.append(
                #         {'title': doc_title, 'text': doc_text, 'passage_id': doc['d_id'], 'score': str(doc['score'])})
            else:
                # negative_ctxs.append(
                #     {
                #      'title': doc_title,  
                #      'text': doc_text, 
                #      'passage_id': doc['d_id'], 
                #      'score': str(doc['score'])})
                # we only save the passage_id to save memory
                negative_ctxs.append(int(doc['d_id']))
        if is_msmarco_data:
            transfer_list.append(
                {
                    "q_id": q_id, "question": q_str, "answers": q_answer, "positive_ctxs": positive_ctxs, 
                    # "retrieved_positive_ctxs": retrieved_positive_ctxs,
                    "hard_negative_ctxs": negative_ctxs
                }
            )
        else:
            transfer_list.append(
                {
                    "q_id": q_id, "question": q_str, "answers": q_answer, "positive_ctxs": positive_ctxs,
                    "hard_negative_ctxs": negative_ctxs
                }
            )
        
    return transfer_list


class RenewTools:
    def __init__(self, args, passages_path, tokenizer, output_dir, temp_dir, 
                 expand_doc_w_query=False, expand_corpus=False, top_k_query=1, 
                 append=False,  query_path=None, delimiter=' ', 
                 evaluate_beir=False,  beir_data_path=None):

        self.passages_path = passages_path
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        
        if is_first_worker():
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
        dist.barrier()
        self.expand_doc_w_query = expand_doc_w_query
        self.expand_corpus = expand_corpus
        self.top_k_query = top_k_query
        self.append = append
        self.query_path = query_path

        self.delimiter = delimiter

        self.evaluate_beir = evaluate_beir
        self.beir_data_path = beir_data_path
        self.load_corpus()

        # to save memory, only load part data for each gpu
        shard_size = len(self.passages) // args.world_size
        start_idx = args.local_rank * shard_size
        end_idx = start_idx + shard_size
        if args.local_rank == args.world_size - 1:
            end_idx = len(self.passages)
        self.start_idx = start_idx
        self.end_idx = end_idx

        if self.evaluate_beir:
            # only for beir benchmark evaluation
            from beir.datasets.data_loader import GenericDataLoader
            from beir.retrieval.evaluation import EvaluateRetrieval
            # note the beir psg_id cannot converted into int
            self.sorted_key_list = sorted([psg_id for psg_id in self.passages])[start_idx:end_idx]
        else:# for nq, tq, ms-marco
            self.sorted_key_list = sorted([int(psg_id) for psg_id in self.passages])[start_idx:end_idx]
            self.sorted_key_list = [str(psg_id) for psg_id in self.sorted_key_list]

        self.passages_piece = {}
        for psg_id in self.sorted_key_list:
            self.passages_piece[psg_id] = self.passages[psg_id]
        del self.passages

        self.list_len = 1
        if self.expand_doc_w_query or self.expand_corpus:
            self.psg_id_2_query_dict = {}
            print(f'Load query from {self.query_path}.')
            with open(self.query_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    contents = line.strip().split('\t')
                    passage_id = contents[0] 
                    if passage_id not in self.passages_piece:
                        continue
                    query_list = contents[1:]
                    query_list = list(set(query_list))
                    if len(query_list)<self.top_k_query:
                        query_list = (query_list * self.top_k_query)
                    self.psg_id_2_query_dict[passage_id] = query_list[:self.top_k_query]
            print(f'Finish loading queries.')

            # get the expanded corpus
            if self.expand_corpus:
                def _f():
                    expanded_passages = {}
                    for i in range(self.top_k_query):
                        print(f'Load the {i+1}-th query.')
                        for psg_id, passage in self.passages_piece.items():
                            assert psg_id == passage[0]
                            query = self.psg_id_2_query_dict[psg_id][i]
                            if self.append: # append queries to the doc
                                # if self.evaluate_beir:
                                #     new_passage = [passage[0], passage[1].strip() + '\t\t\t' + query.strip(), passage[2]]
                                # else:
                                new_passage = [passage[0], passage[1].strip() + self.delimiter + query.strip(), passage[2]]
                            else: # prepend queries to the doc
                                new_passage = [passage[0], passage[1], query.strip() + self.delimiter + passage[2].strip()]
                            expanded_passages[psg_id] = new_passage
                        yield expanded_passages
                self.expanded_passages_list = _f()
                self.list_len = self.top_k_query

                logger.info(f'Expand the corpus to {self.top_k_query} times.')
            elif self.expand_doc_w_query:
                self.expanded_passages = {}
                logger.info(f'Expand the doc with the top {self.top_k_query} queries from {self.query_path}.')
                for psg_id, passage in self.passages_piece.items():
                    assert psg_id == passage[0]
                    queries = " ".join(self.psg_id_2_query_dict[psg_id][:self.top_k_query])
                    if self.append: # append queries to the doc
                        new_passage = [passage[0], passage[1].strip() + self.delimiter + queries.strip(), passage[2]]
                    else: # prepend queries to the doc
                        new_passage = [passage[0], passage[1], queries.strip() + self.delimiter + passage[2].strip()]
                    self.expanded_passages[psg_id] = new_passage
            else:
                raise ValueError('')
    
    def load_corpus(self):
        if self.evaluate_beir:
            # Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]
            self.corpus, self.queries, self.qrels = GenericDataLoader(data_folder=self.beir_data_path).load(split="test")
            self.question_ids = list(self.queries.keys())
            self.questions = [self.queries[e] for e in self.question_ids]
            # psg_id, text, title
            self.passages = {}
            for psg_id in self.corpus:
                text = self.corpus[psg_id]['text']
                title = self.corpus[psg_id]['title']
                self.passages[psg_id] = [psg_id, text.strip() if text else '', title.strip() if title else '']
            del self.corpus
            self.query_path = os.path.join(self.beir_data_path, 'generated_query/query.tsv')
        else:
            # {psg_id: [psg_id, text, title]}
            self.passages = load_passage(self.passages_path)

    def compute_passage_embedding(self, args, model):
        if self.expand_doc_w_query:
            passages_list = [self.expanded_passages]
        elif self.expand_corpus:
            passages_list = self.expanded_passages_list
        else:
            passages_list = [self.passages_piece]
        if args.load_cache:
            pass
        else:
            for i, passages in enumerate(passages_list):
                assert len(passages) == len(self.sorted_key_list)
                passages_piece = [passages[str(e)] for e in self.sorted_key_list]
                del passages
                logger.info(f'Embedding generation for {len(passages_piece)} passages from idx {self.start_idx} to {self.end_idx}')
                if self.list_len==1:
                    pickle_path = os.path.join(self.temp_dir, f"psg_embed_gpu_{args.local_rank}.pb")
                else: 
                    pickle_path = os.path.join(self.temp_dir, f"psg_embed_gpu_{args.local_rank}_{i+1}_query.pb")
                print(pickle_path)
                pickle_path2 = os.path.join(self.temp_dir, f"psg_embed_id_gpu_{args.local_rank}.pb")
                allids, allembeddings = embed_passages(args, passages_piece, model, self.tokenizer)
                with open(pickle_path, 'wb') as handle:
                    pickle.dump(allembeddings, handle, protocol=4)
                logger.info(f'Total passages processed {len(allids)}. Written to {pickle_path}.')

                if i==0:
                    with open(pickle_path2, 'wb') as handle:
                        pickle.dump(allids, handle, protocol=4)
                del allids
                del allembeddings

        return self.list_len
                
    def get_passage_embedding(self, args, K):
        if is_first_worker():
            logger.info('load passage begin')

            for i in range(K):
                _passage_embedding_list = []
                _passage_id_list = []
                for j in tqdm(range(args.world_size)):  # TODO: dynamically find the max instead of HardCode

                    if K==1:
                        pickle_path = os.path.join(self.temp_dir, f"psg_embed_gpu_{j}.pb")
                    else: 
                        pickle_path = os.path.join(self.temp_dir, f"psg_embed_gpu_{j}_{i+1}_query.pb")
                    print(pickle_path)
                    with open(pickle_path, 'rb') as handle:
                        b = pickle.load(handle)
                        _passage_embedding_list.append(b)
                    if i==0:
                        pickle_path = os.path.join(self.temp_dir, f"psg_embed_id_gpu_{j}.pb")
                        with open(pickle_path, 'rb') as handle:
                            b = pickle.load(handle)
                            _passage_id_list += list(b) # To be compatible with old calculation results

                passage_embeddings = np.concatenate(_passage_embedding_list, axis=0)
                if i== 0:
                    # Note the passage_ids is a list, which records the original psg_id, List[str]
                    passage_ids = _passage_id_list
                yield passage_embeddings, passage_ids, K


    def get_question_embeddings(self, args, questions, model):
        dataset = QuestionDataset(questions, self.tokenizer, args.max_query_length)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False, num_workers=15, collate_fn=QuestionDataset.get_collate_fn(args))
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
                if (k+1) % 1000 == 0:
                    logger.info('Encoded question %d', total)

        question_embeddings = torch.cat(allembeddings, dim=0).numpy()
        # Note the question_ids is a numpy array, which records the computing order of the questions. 
        # Since the question dataloader gets question in order, question_ids = np.array([0, 1,2,3,...])
        question_ids = [x for idlist in allids for x in idlist]
        return question_embeddings, question_ids

    def get_question_embedding(self, args, model, qa_path, mode='train'):
        questions = []
        answers = []
        real_question_ids = []
        logger.info("Loading "+mode+" answers")
        is_msmarco_data = False
        with open(qa_path, "r", encoding="utf-8") as fr:
            for line in fr:
                break
            if len(line.strip().split('\t'))==3:
                is_msmarco_data = True
        if is_msmarco_data:
            logger.warning("You are loading answers for MSMARCO or TREC data")
        with open(qa_path, "r", encoding="utf-8") as ifile:
            # file format: question, answers
            reader = csv.reader(ifile, delimiter='\t')
            for row in reader:
                if is_msmarco_data: # for marco
                    assert len(row) ==3
                    questions.append(row[0])
                    answers.append([int(psg_id) for psg_id in eval(row[1])]) # the positive id list
                    real_question_ids.append(row[2]) # real q_id
                else: # for nq and tq
                    assert len(row) ==2
                    questions.append(row[0])
                    answers.append(eval(row[1]))
        logger.warning(f"{qa_path} has {len(answers)} queries")
        question_embeddings, question_ids = self.get_question_embeddings(args, questions, model)
        return questions,answers,question_embeddings, question_ids, real_question_ids, is_msmarco_data
    
    # given q_a path ,generate
    def get_question_topk(self, questions, answers,
                                question_embeddings, 
                                question_ids,
                                real_question_ids, is_msmarco_data, 
                                golden_path, embedding_and_id_generator, 
                                mode='train', step_num=0, 
                                evaluate_trec=False, 
                                query_positive_id_path=None, prefix=None):
        
        def _infer(k, K, passage_embedding, real_question_ids):
            dim = passage_embedding.shape[1]
            faiss.omp_set_num_threads(90)
            cpu_index = faiss.IndexFlatIP(dim)
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = False
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
            # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
            logger.info("***** begin add passages  *****")
            gpu_index.add(passage_embedding.astype(np.float32))
            logger.info(f"***** end build the {k+1}-th index  *****")

            # k ==0 denotes the first; k==-1 denotes average
            similar_scores, train_I = gpu_index.search(question_embeddings.astype(np.float32), 1000)  # I: [number of queries, topk]
            similar_scores_list.append(similar_scores)
            train_I_list.append(train_I)
            if k>=1:
                similar_scores = np.concatenate(similar_scores_list, axis=1)
                train_I = np.concatenate(train_I_list, axis=1)
                ind = np.argsort(-similar_scores, axis=1) 
                similar_scores = np.take_along_axis(similar_scores, ind, axis=1)
                train_I = np.take_along_axis(train_I, ind, axis=1)

                # choose the top 1000
                new_similar_scores_list = []
                new_train_I_list = []
                for i in range(train_I.shape[0]):
                    unique_ids = set()
                    id_list = []
                    score_list = []
                    for j in range(train_I.shape[1]):
                        if train_I[i, j] not in unique_ids:
                            unique_ids.add(train_I[i, j])
                            id_list.append(train_I[i, j])
                            score_list.append(similar_scores[i, j])
                        if len(unique_ids)==1000:
                            break
                    new_similar_scores_list.append(score_list)
                    new_train_I_list.append(id_list)
                similar_scores = np.array(new_similar_scores_list)
                train_I = np.array(new_train_I_list)
            
            if self.evaluate_beir:
                results = {}
                for idx in range(len(self.question_ids)):
                    scores = [float(score) for score in similar_scores[idx]]
                    doc_ids = [passage_ids[doc_id] for doc_id in train_I[idx]]
                    results[self.question_ids[idx]] = dict(zip(doc_ids, scores))

                retriever = EvaluateRetrieval(None, score_function="dot") # or "cos_sim" for cosine similarity
                #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
                ndcg, _map, recall, precision = retriever.evaluate(self.qrels, results, retriever.k_values)

                if K==1:
                    ndcg_output_path = os.path.join(self.output_dir, mode + "_eval_result" + str(step_num) + ".json")
                else:
                    ndcg_output_path = os.path.join(self.output_dir, mode + "_eval_result" + str(step_num) + f'_{k+1}_{K}_query' + ".json")

                with open(ndcg_output_path, 'w') as f:
                    json.dump(ndcg, f, indent=2)

            elif evaluate_trec: # evaluate the TREC-19, 20 test set.
                dev_query_positive_id = load_qrel_data(query_positive_id_path)
                topN = 1000 
                # convert 
                _real_question_ids = [int(real_question_ids[e]) for e in question_ids]
                result = EvalDevQuery(_real_question_ids, passage_ids, dev_query_positive_id, train_I, topN)
                final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate,  Ahole_rate, metrics, prediction = result

                print("NDCG@10:" + str(final_ndcg))
                print("map@10:" + str(final_Map))
                print("pytrec_mrr:" + str(final_mrr))
                print("recall@"+str(topN)+":" + str(final_recall))
                print("hole rate@10:" + str(hole_rate))
                print("hole rate:" + str(Ahole_rate))
                assert prefix is not None
                if K==1:
                    ndcg_output_path = os.path.join(self.output_dir, prefix + "_eval_result" + str(step_num) + ".json")
                else:
                    ndcg_output_path = os.path.join(self.output_dir, prefix + "_eval_result" + str(step_num) + f'_{k+1}_query' + ".json")

                with open(ndcg_output_path, 'w') as f:
                    json.dump({'NDCG@10': final_ndcg, 'map@10': final_Map,
                              'pytrec_mrr': final_mrr, "recall@"+str(topN): final_recall, 
                              'hole rate@10': hole_rate, 'hole rate': Ahole_rate}, f, indent=2)

            else: # ms-marco, nq, tq
                self.load_corpus()
                top_k_hits, scores, result_dict, result_dict_list = \
                    validate(questions, self.passages, answers, train_I,
                            similar_scores, question_ids, real_question_ids, is_msmarco_data, passage_ids)

                logger.info("writing new " + mode + " data")
                if K==1:
                    ndcg_output_path = os.path.join(self.output_dir, mode + "_eval_result" + str(step_num) + ".json")
                else:
                    ndcg_output_path = os.path.join(self.output_dir, mode + "_eval_result" + str(step_num) + f'_{k+1}_query' + ".json")
                    # if k==-1:
                    #     ndcg_output_path = os.path.join(self.output_dir, mode + "_eval_result" + str(step_num) + f'_ave_query' + ".json")
                    # if k==-2:
                    #     ndcg_output_path = os.path.join(self.output_dir, mode + "_eval_result" + str(step_num) + f'_max_query' + ".json")
                    # if k==-3:
                    #     ndcg_output_path = os.path.join(self.output_dir, mode + "_eval_result" + str(step_num) + f'_median_query' + ".json")


                with open(ndcg_output_path, 'w') as f:
                    json.dump({'top1': top_k_hits[0], 'top5': top_k_hits[4],
                            'top20': top_k_hits[19], 'top50': top_k_hits[49], 
                            'top100': top_k_hits[99], 'top1000': top_k_hits[999], 'result_dict': result_dict}, f, indent=2)
                
                if mode == 'test':
                    pass
                else:
                    if k+1>1 or golden_path is None:
                        pass
                    else:
                        with open(golden_path, "r", encoding="utf-8") as ifile:
                            # file format: question, answers
                            golden_list = json.load(ifile)
                        q_pos_dict = {}
                        for example in golden_list:
                            if len(example['positive_ctxs'])>0:
                                if is_msmarco_data:
                                    q_pos_dict[example['question']] = example['positive_ctxs']
                                else:
                                    q_pos_dict[example['question']] = example['positive_ctxs'][0]
                        transfer_list = reform_out(result_dict_list, q_pos_dict, is_msmarco_data)

                        if K==1:
                            output_path = os.path.join(self.output_dir, mode + '_' + str(step_num) + '.json')
                        else:
                            output_path = os.path.join(self.output_dir, mode + '_' + str(step_num) + f'_{k+1}_query' + '.json')

                        with open(output_path, 'w') as f:
                            json.dump(transfer_list, f, indent=2)

        similar_scores_list = []
        train_I_list = []
        # typical representation
        ave_passage_embedding = 0
        # tmp_list = []

        for k, (passage_embedding, passage_ids, K) in enumerate(embedding_and_id_generator):
            ave_passage_embedding += passage_embedding
            _infer(k, K, passage_embedding, real_question_ids)

        if K>1:
            _infer(-1, K, ave_passage_embedding/K, real_question_ids)
            # _infer(-3, K, np.median(np.stack(tmp_list, axis=0), axis=0), real_question_ids)
            # _infer(-1, K, np.stack(tmp_list, axis=0).mean(axis=0), real_question_ids)
            # _infer(-2, K, np.stack(tmp_list, axis=0).max(axis=0), real_question_ids)
        return None


