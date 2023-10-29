"""
This file is used to evaluate the trec-19, 20 test set.
"""
import pytrec_eval

def load_qrel_data(query_positive_id_path):
    dev_query_positive_id = {}
    with open(query_positive_id_path, 'r', encoding='utf8') as f:
        for line in f:
            query_id, _, doc_id, rel = line.split()
            query_id = int(query_id)
            doc_id = int(doc_id)
            if query_id not in dev_query_positive_id:
                dev_query_positive_id[query_id] = {}
            dev_query_positive_id[query_id][doc_id] = int(rel)
    return dev_query_positive_id

def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict
def EvalDevQuery(query_embedding2id, passage_embedding2id, dev_query_positive_id, I_nearest_neighbor,topN):
    prediction = {} #[qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)

    total = 0
    labeled = 0
    Atotal = 0
    Alabeled = 0
    qids_to_ranked_candidate_passages = {} 
    for query_idx in range(len(I_nearest_neighbor)): 
        seen_pid = set()
        query_id = query_embedding2id[query_idx]
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        rank = 0
        
        if query_id in qids_to_ranked_candidate_passages:
            pass    
        else:
            # By default, all PIDs in the list of 1000 are 0. Only override those that are given
            tmp = [0] * 1000
            qids_to_ranked_candidate_passages[query_id] = tmp
                
        for idx in selected_ann_idx:
            pred_pid = passage_embedding2id[idx]
            
            if not pred_pid in seen_pid:
                # this check handles multiple vector per document
                qids_to_ranked_candidate_passages[query_id][rank]=pred_pid
                Atotal += 1
                if pred_pid not in dev_query_positive_id[query_id]:
                    Alabeled += 1
                if rank < 10:
                    total += 1
                    if pred_pid not in dev_query_positive_id[query_id]:
                        labeled += 1
                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(dev_query_positive_id), {'map_cut', 'ndcg_cut', 'recip_rank','recall'})

    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction))
    
    qids_to_relevant_passageids = {}
    for qid in dev_query_positive_id:
        qid = int(qid)
        if qid in qids_to_relevant_passageids:
            pass
        else:
            qids_to_relevant_passageids[qid] = []
            for pid in dev_query_positive_id[qid]:
                if pid>0:
                    qids_to_relevant_passageids[qid].append(pid)
            
    # ms_mrr = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)

    ndcg = 0
    Map = 0
    mrr = 0
    recall = 0
    recall_1000 = 0

    for k in result.keys():
        eval_query_cnt += 1
        ndcg += result[k]["ndcg_cut_10"]
        Map += result[k]["map_cut_10"]
        mrr += result[k]["recip_rank"]
        recall += result[k]["recall_"+str(topN)]

    final_ndcg = ndcg / eval_query_cnt
    final_Map = Map / eval_query_cnt
    final_mrr = mrr / eval_query_cnt
    final_recall = recall / eval_query_cnt
    hole_rate = labeled/total
    Ahole_rate = Alabeled/Atotal

    return final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, Ahole_rate, result, prediction