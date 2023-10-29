"""
This script aims to convert the data format into the 
"""
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
# logger = logging.getLogger("__main__")
logger = logging.getLogger(__name__)
@dataclass
class MSMARCOPreProcessor:
    data_dir: str
    split: str
    
    columns = ['text_id', 'title', 'text']
    title_field = 'title'
    text_field = 'text'

    def __post_init__(self):
        assert self.split in ['train', 'dev']
        self.query_file =  os.path.join(self.data_dir, f'{self.split}.query.txt')
        self.queries = self.read_queries(self.query_file)

        
        self.collection_file = os.path.join(self.data_dir, f'corpus.tsv')
        print(f'Loading passages from: {self.collection_file}')
        self.collection = datasets.load_dataset(
            'csv',
            data_files=self.collection_file,
            column_names=self.columns,
            delimiter='\t',
        )['train']
        print(f'the copus has {len(self.collection)} passages.')

        self.relevance_file = os.path.join(self.data_dir, f'qrels.{self.split}.tsv')
        self.query_2_pos_doc_dict = self.read_positive(self.relevance_file)
        if self.split =='train':
            self.negative_file = os.path.join(self.data_dir, f'train.negatives.tsv')
            self.query_2_neg_doc_dict = self.read_negative(self.negative_file)
        else:
            self.negative_file = None
            self.query_2_neg_doc_dict = None

    @staticmethod
    def read_negative(negative_file):
        print(f'Load the negative docs for queries from {negative_file}.')
        query_2_neg_doc_dict = {}
        with open(negative_file, 'r', encoding='utf8') as fr:
            for line in fr:
                query_id, negative_doc_id_list = line.strip().split('\t')
                negative_doc_id_list = negative_doc_id_list.split(',')
                random.shuffle(negative_doc_id_list)
                query_2_neg_doc_dict[query_id] = negative_doc_id_list
        print(f'{len(query_2_neg_doc_dict)} queries have negative docs.')
        return query_2_neg_doc_dict

    @staticmethod
    def read_queries(queries):
        print(f'Load the query from {queries}.')
        qmap = OrderedDict()
        with open(queries, 'r', encoding='utf8') as f:
            for l in f:
                qid, qry = l.strip().split('\t')
                qmap[qid] = qry
        print(f'Train set has {len(qmap)} queries.')
        return qmap


    @staticmethod
    def read_positive(relevance_file):
        print(f'Load the positive docs for queries from {relevance_file}.')
        query_2_pos_doc_dict = {}
        with open(relevance_file, 'r', encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            if 'train' in relevance_file:
                print('train')
                for [query_id, _, doc_id, rel] in tsvreader:
                    if query_id in query_2_pos_doc_dict:
                        assert rel == "1"
                        query_2_pos_doc_dict[query_id].append(doc_id)
                    else:
                        query_2_pos_doc_dict[query_id] = [doc_id]
            else: # dev
                print('dev')
                for [query_id, doc_id] in tsvreader:
                    if query_id in query_2_pos_doc_dict:
                        query_2_pos_doc_dict[query_id].append(doc_id)
                    else:
                        query_2_pos_doc_dict[query_id] = [doc_id]
        print(f'{len(query_2_pos_doc_dict)} queries have positive docs.')
        return query_2_pos_doc_dict
    
    
    def reform_data(self, n_sample, output_path):
        output_list = []
        for query_id in self.queries:
            positive_doc_id_list = self.query_2_pos_doc_dict[query_id]
            if self.query_2_neg_doc_dict is not None and query_id in self.query_2_neg_doc_dict:
                negative_doc_id_list = self.query_2_neg_doc_dict[query_id][:n_sample]
            else:
                negative_doc_id_list = []

            q_str = self.queries[query_id]
            q_answer = 'nil'
            positive_ctxs = []
            negative_ctxs = []
            for passage_id in negative_doc_id_list:
                entry = self.collection[int(passage_id)]
                title = entry[self.title_field]
                title = "" if title is None else title
                body = entry[self.text_field]

                negative_ctxs.append(
                    {'title': title, 'text': body, 'passage_id': passage_id, 'score': 'nil'}
                    )

            for passage_id in positive_doc_id_list:
                entry = self.collection[int(passage_id)]
                title = entry[self.title_field]
                title = "" if title is None else title
                body = entry[self.text_field]

                positive_ctxs.append(
                    {'title': title, 'text': body, 'passage_id': passage_id, 'score': 'nil'}
                    )

            output_list.append(
                {
                    "q_id": query_id, "question": q_str, "answers": q_answer, "positive_ctxs": positive_ctxs,
                    "hard_negative_ctxs": negative_ctxs, "negative_ctxs": []
                }
                )
        
        
        with open(output_path, 'w', encoding='utf8') as f:
            json.dump(output_list, f, indent=2)


    def generate_qa_files(self, output_path):
        with open(output_path, 'w', encoding='utf8') as fw:
            for q_id, query in self.queries.items():
                positive_doc_id_list = self.query_2_pos_doc_dict[q_id]
                fw.write(f"{query}\t{json.dumps(positive_doc_id_list)}\t{q_id}\n")
    
    def reform_corpus(self, output_path):
        # original format: psg_id, title, text
        # reformed format: psg_id, text, title (to consistent with nq and tq)
        with open(self.collection_file, 'r', encoding='utf8') as fr, open(output_path, 'w', encoding='utf8') as fw:
            for line in fr:
                l = line.strip().split('\t')
                assert len(l) == 3
                fw.write(f'{l[0]}\t{l[2]}\t{l[1]}\n')

    
if __name__ == '__main__':
    random.seed(0)
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./marco')
    parser.add_argument('--output_path', type=str, default='./reformed_marco')
    parser.add_argument('--n_sample', type=int, default=30, help='number of selected negative examples.')

    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    
    processor = MSMARCOPreProcessor(
        data_dir=args.data_dir,
        split='train'
    )
    processor.reform_data(args.n_sample, os.path.join(args.output_path, 'biencoder-marco-train.json'))
    processor.generate_qa_files(os.path.join(args.output_path, 'marco-train.qa.csv'))

    processor = MSMARCOPreProcessor(
        data_dir=args.data_dir,
        split='dev'
    )
    processor.reform_corpus( os.path.join(args.output_path, 'corpus.tsv'))
    processor.reform_data(args.n_sample, os.path.join(args.output_path, 'biencoder-marco-dev.json'))
    processor.generate_qa_files(os.path.join(args.output_path, 'marco-dev.qa.csv'))