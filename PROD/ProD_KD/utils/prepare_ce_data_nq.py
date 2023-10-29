import json
import sys
from tqdm import tqdm
def load_data(data_path=None):
    assert data_path
    with open(data_path, 'r',encoding='utf-8') as fin:
        data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    return examples

def read_train_pos(ground_truth_path):
    origin_train_path = ground_truth_path
    with open(origin_train_path, "r", encoding="utf-8") as ifile:
        # file format: question, answers
        train_list = json.load(ifile)
        train_q_pos_dict = {}
        for example in train_list:
            if len(example['positive_ctxs'])==0 or "positive_ctxs" not in example.keys():
                continue
            train_q_pos_dict[example['question']]=example['positive_ctxs'][0]
    return train_q_pos_dict

def reform_out(examples,outfile, ground_truth_path):
    train_q_pos_dict = read_train_pos(ground_truth_path)
    transfer_list = []
    easy_ctxs = []
    for infer_result in tqdm(examples):
        if 'passage_id' not in infer_result.keys():
            q_id = infer_result["id"]
        else:
            q_id = infer_result["passage_id"]
        q_str = infer_result["question"]
        q_answer = infer_result["answers"]
        positive_ctxs = []
        negative_ctxs = []
        if q_str in train_q_pos_dict.keys():
            # ground true
            real_true_dic = train_q_pos_dict[q_str]
            # real_true_doc_id = real_true_dic['passage_id'] if 'passage_id' in real_true_dic.keys() else real_true_dic['id']
            if 'passage_id' not in real_true_dic.keys() and 'id' in real_true_dic.keys():
                real_true_dic['passage_id'] = real_true_dic['id']
            elif 'psg_id' in  real_true_dic.keys():
                real_true_dic['passage_id'] = real_true_dic['psg_id']
            positive_ctxs.append(real_true_dic)

        for doc in infer_result['ctxs']:
            doc_text = doc['text']
            doc_title = doc['title']
            if doc['hit']=="True":
                positive_ctxs.append({'title':doc_title,'text':doc_text,'passage_id':doc['d_id'],'score':str(doc['score'])})
            else:
                negative_ctxs.append({'title':doc_title,'text':doc_text,'passage_id':doc['d_id'],'score':str(doc['score'])})
                # easy_ctxs.append({'title':doc_title,'text':doc_text,'passage_id':doc['d_id'],'score':str(doc['score'])})
    
        transfer_list.append(
            {
                "q_id":str(q_id), "question":q_str, "answers" :q_answer ,"positive_ctxs":positive_ctxs,"hard_negative_ctxs":negative_ctxs,"negative_ctxs":[]
            }
        )

    print("total data to ce train: ", len(transfer_list))
    # print("easy data to ce train: ", len(easy_list))
    print("hardneg num:", len(transfer_list[0]["hard_negative_ctxs"]))
    # print("easyneg num:", len(transfer_list[0]["easy_negative_ctxs"]))

    with open(outfile, 'w',encoding='utf-8') as f:
        json.dump(transfer_list, f,indent=2)
    return 
if __name__ == "__main__":
    inference_results = sys.argv[1]
    outfile = sys.argv[2]
    ground_truth_path =  sys.argv[3]
    examples = load_data(inference_results)
    reform_out(examples,outfile,ground_truth_path)


