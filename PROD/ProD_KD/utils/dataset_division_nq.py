import pickle
import json
from tqdm import tqdm
import os
import sys


# data file
# qid \t qstring \t pos_id \t neg_id
# type : str
# result_file_path1="/colab_space/fanshuai/KDnq/result/24CEt6DE_LwF_5e-5/20000/train_result_dict_list.json"
# result_file_path2="/colab_space/fanshuai/KDnq/result/Ranker_24layer/10000trainrank/train_rerank_dict_list.json"
# data_file_path="/colab_space/fanshuai/KDnq/result/24CEt6DE_LwF_5e-5/20000/train_nq_flash4.json"
# output_dir = "/colab_space/fanshuai/KDnq/result/Ranker_24layer/10000div/"

def read_result(result_file_path):
    result_data = []
    with open(result_file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        for i, example in tqdm(enumerate(data)):
            result_data.append(example)

    return result_data

def load_data(data_file_path):
    with open(data_file_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        print('Aggregated data size: {}'.format(len(data)))
    # filter those without positive ctx
    return data

# def divide_data(result_data_student):
#     ranking = []
#     recall_q_top1 = set()
#     recall_q_2ti = set()
#     recall_q_2t5 = set()
#     recall_q_2t10 = set()
#     recall_q_2t15 = set()
#     recall_q_6t20 = set()
#     recall_q_21t50 = set()
#     recall_q_51t100 = set()
#     recall_q_101tall = set()
#
#     for data in result_data_student:
#         qid = data['id']
#         for i, ctx in enumerate(data['ctxs']):
#             if ctx['hit'] == 'True':
#                 if i < 100 and i >= 50:
#                     recall_q_51t100.add(qid)
#                 elif i < 50 and i >= 20:
#                     recall_q_21t50.add(qid)
#                 elif i < 20 and i >= 5:
#                     recall_q_6t20.add(qid)
#                 elif i < 5 and i >= 1:
#                     recall_q_2t5.add(qid)
#                 elif i == 0:
#                     recall_q_top1.add(qid)
#                 break
#
#     print("top1 data num :", len(recall_q_top1))
#     print("top2 to topi data num :", len(recall_q_2ti))
#     print("top2 to top5 data num :", len(recall_q_2t5))
#     print("top2 to top10 data num :", len(recall_q_2t10))
#     print("top2 to top15 data num :", len(recall_q_2t15))
#     print("top6 to top20 data num :", len(recall_q_6t20))
#     print("top21 to top50 data num :", len(recall_q_21t50))
#     print("top51 to top100 data num :", len(recall_q_51t100))
#     print("top101 to top1000 data num :", len(recall_q_101tall))
#     print()
#
#
#     qid_divide_dic = {}
#     qid_divide_dic['top1'] = recall_q_top1
#     qid_divide_dic['2ti'] = recall_q_2ti
#     qid_divide_dic['2t5'] = recall_q_2t5
#     qid_divide_dic['2t10'] = recall_q_2t10
#     qid_divide_dic['2t15'] = recall_q_2t15
#     qid_divide_dic['6t20'] = recall_q_6t20
#     qid_divide_dic['21t50'] = recall_q_21t50
#     qid_divide_dic['51t100'] = recall_q_51t100
#     qid_divide_dic['101tall'] = recall_q_101tall
#
#     return qid_divide_dic

def divide_data(result_data_student, result_data_teacher):
    ranking = []
    t2_all_better = set()
    t2_15_better = set()

    qid_data_dic={}
    for data in result_data_teacher:
        qid_data_dic[data['id']] = data

    for data in result_data_student:
        qid = data['id']

        s_rank = 99999
        t_rank = 99999
        for i,ctx in enumerate(data['ctxs']):
            if ctx['hit'] == 'True':
                s_rank = i
                break

        t_data = qid_data_dic[qid]
        for i,ctx in enumerate(t_data['ctxs']):
            if ctx['hit'] == 'True':
                t_rank = i
                break

        if t_rank < s_rank and t_rank < 15:
            t2_15_better.add(qid)

        if t_rank < s_rank and t_rank < 100:
            t2_all_better.add(qid)

    print("t2_15_better len ", len(t2_15_better))
    print("t2_31_better len ", len(t2_all_better))
    print()

    qid_divide_dic = {}
    qid_divide_dic['t2_15_better'] = t2_15_better
    qid_divide_dic['t2_all_better'] = t2_all_better

    return qid_divide_dic

def main(result_file_path1, result_file_path2, data_file_path, output_dir):
    t1_qidset = set()
    t2_qidset = set()

    t2_better_set = set()

    result_data_student = read_result(result_file_path1)
    result_data_teacher = read_result(result_file_path2)

    # train_data = load_data(data_file_path)

    # top1 / top2-top5 / top6-top20 / top30-top50 / top50-top100 / top100-topall
    #qid_divide_dic1 = divide_data(result_data_student)
    #qid_divide_dic2 = divide_data(result_data_teacher)
    qid_divide_dic = divide_data(result_data_student, result_data_teacher)

    t2_better_set = qid_divide_dic['t2_15_better']

    print("t2_better_set len:", len(t2_better_set))

    with open(data_file_path, 'r', encoding="utf-8") as f:
        dataset = json.load(f)
        print('Aggregated data size: {}'.format(len(dataset)))

    t2_better_data = []

    for data in dataset:
        if data['q_id'] in t2_better_set:
            t2_better_data.append(data)

    print(len(t2_better_data))

    print("check if data right...")
    if len(t2_better_data) != len(t2_better_set):
        print("data error")
        exit(0)
    print("data set success!")

    t2_better_output_dir = os.path.join(output_dir, "flash4_top15_better.json")
    with open(t2_better_output_dir, 'w', encoding='utf-8') as f:
        json.dump(t2_better_data, f, indent=2)





if __name__ == "__main__":
    result_file_path_s = sys.argv[1]
    result_file_path_t = sys.argv[2]
    data_file_path = sys.argv[3]
    output_dir = sys.argv[4]
    main(result_file_path_s, result_file_path_t, data_file_path, output_dir)
    print("data division done")