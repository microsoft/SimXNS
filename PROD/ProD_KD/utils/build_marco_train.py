import json
from tqdm import tqdm
import random

relevance_file = "/colab_space/fanshuai/KDmarco/coCondenser-marco/marco/qrels.train.tsv"
query_file = "/colab_space/fanshuai/KDmarco/coCondenser-marco/marco/train.query.txt"
negative_file = "/colab_space/fanshuai/KDmarco/coCondenser-marco/marco/train.negatives.tsv"
outfile = "/colab_space/fanshuai/KDmarco/coCondenser-marco/marco/marco_train.json"
n_sample = 30

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
            line = line.strip('\n')  # 删除换行符
            line = line.split('\t')
            q_string[line[0]] = line[1]
    return q_string

datalist = []
qrel = read_qrel(relevance_file)
q_string = read_qstring(query_file)
with open(negative_file, 'r', encoding='utf8') as nf:
    reader = csv_reader(nf)
    for cnt, line in enumerate(reader):
        examples = {}
        q = line[0]
        nn = line[1]
        nn = nn.split(',')
        random.shuffle(nn)
        nn = nn[:n_sample]
        examples['query_id'] = q
        examples['query_string'] = q_string[q]
        examples['pos_id'] = qrel[q]
        examples['neg_id'] = nn
        if len(examples['query_id'])!=0 and len(examples['pos_id'])!=0 and len(examples['neg_id'])!=0:
            datalist.append(examples)

print("data len:", len(datalist))
print("data keys:", datalist[0].keys())
print("data info:", datalist[0])

with open(outfile, 'w',encoding='utf-8') as f:
    json.dump(datalist, f,indent=2)