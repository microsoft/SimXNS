import pickle
import json
from tqdm import tqdm
import os
import sys

# data file
# qid \t qstring \t pos_id \t neg_id
# type : str

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

def read_result(result_file_path):
    with open(result_file_path, 'rb') as f:
        if "list" in result_file_path:
            qids_to_ranked_candidate_passages, _ = pickle.load(f)
        elif "dict" in result_file_path:
            qids_to_ranked_candidate_passages = pickle.load(f)
    return qids_to_ranked_candidate_passages

def load_data(data_file_path):
    with open(data_file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        print('Aggregated data size: {}'.format(len(data)))
    # filter those without positive ctx
    pre_data = [r for r in data if len(r["pos_id"]) > 0]
    print("cleaned data size: {} after positive ctx".format(len(pre_data)))
    pre_data = [r for r in pre_data if len(r['neg_id']) > 0]
    print("Total cleaned data size: {}".format(len(pre_data)))
    return pre_data

def divide_data(qid_to_candidate_dic, qids_to_relevant_pids):
    ranking = []
    recall_q_top1 = set()
    recall_q_2ti = set()
    recall_q_2t5 = set()
    recall_q_2t10 = set()
    recall_q_2t15 = set()
    recall_q_6t20 = set()
    recall_q_21t50 = set()
    recall_q_51t100 = set()
    recall_q_101tall = set()

    for qid in qid_to_candidate_dic:
        if qid in qids_to_relevant_pids:
            ranking.append(0)
            target_pid = qids_to_relevant_pids[qid]
            candidate_pid = qid_to_candidate_dic[qid]
            for i, pid in enumerate(candidate_pid):
                if pid in target_pid:
                    if i < 1000 and i >= 100:
                        recall_q_101tall.add(qid)
                    elif i < 100 and i >= 50:
                        recall_q_51t100.add(qid)
                    elif i < 50 and i >= 20:
                        recall_q_21t50.add(qid)
                    elif i < 20 and i >= 5:
                        recall_q_6t20.add(qid)
                    elif i < 5 and i >= 1:
                        recall_q_2t5.add(qid)
                    elif i == 0:
                        recall_q_top1.add(qid)
                    break

    for qid in qid_to_candidate_dic:
        if qid in qids_to_relevant_pids:
            ranking.append(0)
            target_pid = qids_to_relevant_pids[qid]
            candidate_pid = qid_to_candidate_dic[qid]
            for i, pid in enumerate(candidate_pid):
                if pid in target_pid:
                    if i < 15 and i >= 1:
                        recall_q_2t15.add(qid)
                    elif i == 0:
                        recall_q_top1.add(qid)
                    break

    for qid in qid_to_candidate_dic:
        if qid in qids_to_relevant_pids:
            ranking.append(0)
            target_pid = qids_to_relevant_pids[qid]
            candidate_pid = qid_to_candidate_dic[qid]
            for i, pid in enumerate(candidate_pid):
                if pid in target_pid:
                    if i < 10 and i >= 1:
                        recall_q_2t10.add(qid)
                    break

    for qid in qid_to_candidate_dic:
        if qid in qids_to_relevant_pids:
            ranking.append(0)
            target_pid = qids_to_relevant_pids[qid]
            candidate_pid = qid_to_candidate_dic[qid]
            for i, pid in enumerate(candidate_pid):
                if pid in target_pid:
                    if i < 2 and i >= 1:
                        recall_q_2ti.add(qid)
                    break
    print("top1 data num :", len(recall_q_top1))
    print("top2 to topi data num :", len(recall_q_2ti))
    print("top2 to top5 data num :", len(recall_q_2t5))
    print("top2 to top10 data num :", len(recall_q_2t10))
    print("top2 to top15 data num :", len(recall_q_2t15))
    print("top6 to top20 data num :", len(recall_q_6t20))
    print("top21 to top50 data num :", len(recall_q_21t50))
    print("top51 to top100 data num :", len(recall_q_51t100))
    print("top101 to top1000 data num :", len(recall_q_101tall))
    print()


    qid_divide_dic = {}
    qid_divide_dic['top1'] = recall_q_top1
    qid_divide_dic['2ti'] = recall_q_2ti
    qid_divide_dic['2t5'] = recall_q_2t5
    qid_divide_dic['2t10'] = recall_q_2t10
    qid_divide_dic['2t15'] = recall_q_2t15
    qid_divide_dic['6t20'] = recall_q_6t20
    qid_divide_dic['21t50'] = recall_q_21t50
    qid_divide_dic['51t100'] = recall_q_51t100
    qid_divide_dic['101tall'] = recall_q_101tall

    return qid_divide_dic


def main(result_file_path1, result_file_path2, data_file_path, train_ground_truth_path, output_dir):
    t1_qidset = set()
    t2_qidset = set()

    t2_better_set = set()

    qid_to_candidate_dic1 = read_result(result_file_path1)
    qid_to_candidate_dic2 = read_result(result_file_path2)

    train_data = load_data(data_file_path)
    qids_to_relevant_pids = load_train_reference_from_stream(train_ground_truth_path)

    # top1 / top2-top5 / top6-top20 / top30-top50 / top50-top100 / top100-topall
    qid_divide_dic1 = divide_data(qid_to_candidate_dic1, qids_to_relevant_pids)
    qid_divide_dic2 = divide_data(qid_to_candidate_dic2, qids_to_relevant_pids)

    print("top1 commen qid num:", len(qid_divide_dic1['top1'] & qid_divide_dic2['top1']))
    print("top2 to top5 commen qid num:", len(qid_divide_dic1['2t5'] & qid_divide_dic2['2t5']))
    print("top6 to top20 commen qid num:", len(qid_divide_dic1['6t20'] & qid_divide_dic2['6t20']))
    print("top21 to top50 commen qid num:", len(qid_divide_dic1['21t50'] & qid_divide_dic2['21t50']))
    print("top51 to top100 commen qid num:", len(qid_divide_dic1['51t100'] & qid_divide_dic2['51t100']))
    print("top101 to top1000 commen qid num:", len(qid_divide_dic1['101tall'] & qid_divide_dic2['101tall']))
    print()

    t2_better_set = qid_divide_dic1['2t15'] & qid_divide_dic2['top1']
    print("t2_better_set len:", len(t2_better_set))

    with open(data_file_path, 'r', encoding="utf-8") as f:
        dataset = json.load(f)
        print('Aggregated data size: {}'.format(len(dataset)))

    t2_better_data = []

    for data in dataset:
        if int(data['query_id']) in t2_better_set:
            t2_better_data.append(data)



    print("check if data right...")
    if len(t2_better_data) != len(t2_better_set):
        print("data error")
        exit(0)
    print("data set success!")

    t2_better_output_dir = output_dir
    with open(t2_better_output_dir, 'w', encoding='utf-8') as f:
        json.dump(t2_better_data, f, indent=2)




if __name__ == "__main__":
    result_file_path_s = sys.argv[1]
    result_file_path_t = sys.argv[2]
    data_file_path = sys.argv[3]
    train_ground_truth_path = sys.argv[4]
    output_dir = sys.argv[5]
    main(result_file_path_s, result_file_path_t, data_file_path, train_ground_truth_path, output_dir)
    print("data division done")