import json
from tqdm import tqdm
import random
import pickle
import sys

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

def read_qrel(relevance_file):
    qrel = {}
    with open(relevance_file, encoding='utf8') as f:
        tsvreader = csv_reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]
    return qrel

def read_qstring(query_file):
    q_string = {}
    with open(query_file, 'r', encoding='utf-8') as file:
        for num, line in enumerate(file):
            line = line.strip('\n')
            line = line.split('\t')
            q_string[line[0]] = line[1]
    return q_string

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

def main(relevance_file, query_file, result_file, outfile, neg_num):
    with open(result_file, 'rb') as f:
        qids_to_ranked_candidate_passages, qids_to_ranked_candidate_scores = pickle.load(f)
    qids_to_relevant_passageids = load_train_reference_from_stream(relevance_file)

    datalist = []
    q_string = read_qstring(query_file)
    for qid in qids_to_ranked_candidate_passages:
        examples = {}
        if qid in qids_to_relevant_passageids:
            target_pid = qids_to_relevant_passageids[qid]
            examples['query_id'] = str(qid)
            examples['pos_id'] = [str(id) for id in target_pid]
            examples['query_string'] = q_string[examples['query_id']]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            examples['neg_id'] = []

            for i, pid in enumerate(candidate_pid):
                if pid in target_pid:
                    # print(target_pid, pid)
                    continue
                else:
                    if len(examples['neg_id']) < neg_num:
                        examples['neg_id'].append(str(pid))
                    else:
                        break
            if len(examples['query_id'])!=0 and len(examples['pos_id'])!=0 and len(examples['neg_id'])!=0:
                datalist.append(examples)


    # print(num,len(qids_to_ranked_candidate_passages),num/len(qids_to_ranked_candidate_passages))
    print("data len:", len(datalist))
    print("data keys:", datalist[0].keys())
    print("data info:", datalist[0])
    print("data info:", datalist[100])
    print("data info:", datalist[1000])

    for data in datalist:
        for id in data['neg_id']:
            if int(id) in data['pos_id']:
                print("data error")
                print(data)
                exit(0)
            if str(id) in data['pos_id']:
                print("data error")
                print(data)
                exit(0)
    print("Correctness check completed")


    with open(outfile, 'w',encoding='utf-8') as f:
        json.dump(datalist, f,indent=2)

if __name__ == "__main__":
    relevance_file = sys.argv[1]
    query_file = sys.argv[2]
    result_file = sys.argv[3]
    outfile = sys.argv[4]
    neg_num = int(sys.argv[5])
    main(relevance_file, query_file, result_file, outfile, neg_num)