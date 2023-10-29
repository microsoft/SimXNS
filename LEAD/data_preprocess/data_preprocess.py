import json
from tqdm import tqdm, trange
from random import sample

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

def read_qrel_train(relevance_file):
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

def read_qrel_train_2(relevance_file):
    qrel = {}
    with open(relevance_file, encoding='utf8') as f:
        tsvreader = csv_reader(f, delimiter=" ")
        for [topicid, _, docid, rel] in tsvreader:
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]
    return qrel


def read_qrel_dev(relevance_file):
    qrel = {}
    with open(relevance_file, encoding='utf8') as f:
        tsvreader = csv_reader(f, delimiter="\t")
        for [topicid, docid] in tsvreader:
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

def read_dstring(corpus_file):
    d_string = {}
    with open(corpus_file, 'r', encoding='utf-8') as file:
        for num, line in enumerate(file):
            line = line.strip('\n')  # 删除换行符
            line = line.split('\t')
            d_string[line[0]] = [line[1], line[2]]
    return d_string

def read_dstring_2(corpus_file):
    d_string = {}
    with open(corpus_file, 'r', encoding='utf-8') as file:
        for num, line in enumerate(file):
            line = line.strip('\n')  # 删除换行符
            line = line.split('\t')
            d_string[line[0]] = [line[2], line[3]]
    return d_string


def construct_mspas():
    train_relevance_file = "mspas/qrels.train.tsv"
    train_query_file = "mspas/train.query.txt"
    train_negative_file = "mspas/train.negatives.tsv"
    corpus_file = "mspas/corpus.tsv"
    dev_relevance_file = "mspas/qrels.dev.tsv"
    dev_query_file = "mspas/dev.query.txt"
    n_sample = 100

    ## qid: docid
    train_qrel = read_qrel_train(train_relevance_file)

    ## qid: query
    train_q_string = read_qstring(train_query_file)

    ## qid: docid
    dev_qrel = read_qrel_dev(dev_relevance_file)

    ## qid: query
    dev_q_string = read_qstring(dev_query_file)

    # docid: doc
    d_string = read_dstring(corpus_file)

    ## qid: [docid1, ...]
    negative = {}
    with open(train_negative_file, 'r', encoding='utf8') as nf:
        reader = csv_reader(nf)
        for cnt, line in enumerate(reader):
            q = line[0]
            nn = line[1]
            nn = nn.split(',')
            negative[q] = nn

    train_file = open("mspas/psgs_w100.tsv", 'w')
    for did in d_string:
        train_file.write('\t'.join([str(int(did) + 1), d_string[did][1], d_string[did][0]]) + '\n')
        train_file.flush()

    train_data_list = []
    for qid in tqdm(negative):
        example = {}
        example['question'] = train_q_string[qid]
        example['answers'] = train_qrel[qid]
        example['positive_ctxs'] = [did for did in train_qrel[qid]]
        example['hard_negative_ctxs'] = [did for did in negative[qid][:n_sample]]
        example['negative_ctxs'] = []
        train_data_list.append(example)

    with open('mspas/biencoder-mspas-train.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json.dumps(train_data_list, indent=4))


    train_data_list = []
    for qid in tqdm(train_q_string):
        example = {}
        example['question'] = train_q_string[qid]
        example['answers'] = train_qrel[qid]
        example['positive_ctxs'] = [did for did in train_qrel[qid]]
        example['negative_ctxs'] = []
        train_data_list.append(example)

    with open('mspas/biencoder-mspas-train-full.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json.dumps(train_data_list, indent=4))

    train_file = open("mspas/mspas-test.qa.csv", 'w')
    for qid in dev_qrel:
        train_file.write('\t'.join([dev_q_string[qid], str(dev_qrel[qid])]) + '\n')
        train_file.flush()


def construct_msdoc():
    train_relevance_file = "msdoc/msmarco-doctrain-qrels.tsv"
    train_query_file = "msdoc/msmarco-doctrain-queries.tsv"
    corpus_file = "msdoc/msmarco-docs.tsv"
    dev_relevance_file = "msdoc/msmarco-docdev-qrels.tsv"
    dev_query_file = "msdoc/msmarco-docdev-queries.tsv"
    n_sample = 100

    ## qid: docid
    train_qrel = read_qrel_train_2(train_relevance_file)

    ## qid: query
    train_q_string = read_qstring(train_query_file)

    ## qid: docid
    dev_qrel = read_qrel_train_2(dev_relevance_file)

    ## qid: query
    dev_q_string = read_qstring(dev_query_file)

    # docid: doc
    d_string = read_dstring_2(corpus_file)

    docid_int = {}
    int_docid = {}
    idx = 0
    for docid in d_string:
        docid_int[docid] = idx
        int_docid[idx] = docid
        idx += 1

    ## qid: [docid1, ...]
    negative = {}
    all_doc_list = [elem for elem in docid_int]

    for docid in train_qrel:
        negative[docid] = sample(all_doc_list, n_sample)

    train_data_list = []
    for qid in tqdm(train_qrel):
        example = {}
        example['question'] = train_q_string[qid]
        example['answers'] = [docid_int[did] for did in train_qrel[qid]]
        example['positive_ctxs'] = [docid_int[did] for did in train_qrel[qid]]
        example['hard_negative_ctxs'] = [docid_int[did] for did in negative[qid][:n_sample]]
        example['negative_ctxs'] = []
        train_data_list.append(example)

    with open('msdoc/biencoder-msdoc-train.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json.dumps(train_data_list, indent=4))

    train_file = open("msdoc/psgs_w100.tsv", 'w')
    for did in int_docid:
        train_file.write('\t'.join([str(did + 1), d_string[int_docid[did]][1], d_string[int_docid[did]][0]]) + '\n')
        train_file.flush()

    train_file = open("msdoc/msdoc-test.qa.csv", 'w')
    for qid in dev_qrel:
        train_file.write('\t'.join([dev_q_string[qid], str([str(docid_int[elem]) for elem in dev_qrel[qid]])]) + '\n')
        train_file.flush()


print('Start Processing')
construct_mspas()
construct_msdoc()
print('Finish Processing')