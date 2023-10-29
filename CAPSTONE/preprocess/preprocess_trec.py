from tqdm import tqdm
import json
import csv
import datasets
from dataclasses import dataclass
from argparse import ArgumentParser
import os
import random
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
import logging

def read_queries(queries):
    print(f'Load the query from {queries}.')
    qmap = OrderedDict()
    with open(queries, 'r', encoding='utf8') as f:
        for l in f:
            qid, qry = l.strip().split('\t')
            qmap[qid] = qry
    print(f'Train set has {len(qmap)} queries.')
    return qmap


def read_positive(relevance_file):
    print(f'Load the positive docs for queries from {relevance_file}.')
    query_2_pos_doc_dict = {}
    with open(relevance_file, 'r', encoding='utf8') as fr:
        for line in fr:
            query_id, _, doc_id, _ = line.split()
            if query_id in query_2_pos_doc_dict:
                query_2_pos_doc_dict[query_id].append(doc_id)
            else:
                query_2_pos_doc_dict[query_id] = [doc_id]
    print(f'{len(query_2_pos_doc_dict)} queries have positive docs.')
    return query_2_pos_doc_dict

def generate_qa_files(output_path, queries, query_2_pos_doc_dict):
    with open(output_path, 'w', encoding='utf8') as fw:
        for q_id, query in queries.items():
            if q_id not in query_2_pos_doc_dict:
                continue
            positive_doc_id_list = query_2_pos_doc_dict[q_id]
            fw.write(f"{query}\t{json.dumps(positive_doc_id_list)}\t{q_id}\n")

if __name__ == '__main__':
    random.seed(0)
    parser = ArgumentParser()
    args = parser.parse_args()

    # trec_19
    data_dir = './trec_19'
    queries = read_queries(os.path.join(data_dir, 'msmarco-test2019-queries.tsv'))
    query_2_pos_doc_dict = read_positive(os.path.join(data_dir, '2019qrels-pass.txt'))
    generate_qa_files(os.path.join(data_dir, 'test2019.qa.csv'), queries, query_2_pos_doc_dict)

    # trec_20
    data_dir = './trec_20'
    queries = read_queries(os.path.join(data_dir, 'msmarco-test2020-queries.tsv'))
    query_2_pos_doc_dict = read_positive(os.path.join(data_dir, '2020qrels-pass.txt'))
    generate_qa_files(os.path.join(data_dir, 'test2020.qa.csv'), queries, query_2_pos_doc_dict)