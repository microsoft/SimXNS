# -*- coding: utf-8 -*-

"""
This file aims to generate queries for a given document, which users may ask based on the given document.
Input: a document
Output: top-k queries, 
"""
from dataclasses import dataclass, field
import csv
import pickle
from turtle import title
from transformers import (
    HfArgumentParser,
)

import os
import sys
import json
import random
from tabulate import tabulate
from tqdm import tqdm
import logging

pyfile_path = os.path.abspath(__file__) 
pyfile_dir = os.path.dirname(os.path.dirname(pyfile_path)) # equals to the path '../'
sys.path.append(pyfile_dir)
from utils.util import normalize_question, set_seed, get_optimizer, sum_main
from utils.data_utils import load_passage

# logger = logging.getLogger(__name__)
logger = logging.getLogger("__main__")

@dataclass
class Arguments:
    dir_name: str = field(
        default="",
        metadata={
            "help": (
                "The path containing the generated queries."
            )
        },
    )

    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})

    # for cross train
    num_split: int = field(default=1, 
                           metadata={"help": "Split the train set into num_split parts. "
                                     "Note There is no overlap between different parts in terms of the positive doc."
                                     "Suppose num_split=3, we need to train three different generators. "
                                     "The first generator is tained on the data_2 and data_3, "
                                     "the second is trained on the data_1 and data_3, the third is trained on data_1 and data_2."})
    



def main():
    parser = HfArgumentParser((Arguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, = parser.parse_args_into_dataclasses()
    print(args)
    removed_passage_ids_list = []
    # removed_passage_id_path = os.path.join('checkpoints', args.dir_name, f'0_th_generator/removed_passge_id_for_0_th_generator.json')
    # removed_passage_id_set = set(json.load(open(removed_passage_id_path, 'r', encoding='utf-8')))

    # removed_passage_id_path = os.path.join('checkpoints', args.dir_name, f'1_th_generator/removed_passge_id_for_1_th_generator.json')
    # removed_passage_id_set2 = set(json.load(open(removed_passage_id_path, 'r', encoding='utf-8')))
    # print(len(removed_passage_id_set), len(removed_passage_id_set2), len(removed_passage_id_set|removed_passage_id_set2), len(removed_passage_id_set&removed_passage_id_set2))
    # exit()
    union_set = set()
    for i in range(args.num_split):
        # query_path = os.path.join(args.dir_name, f'{i}_th_generator/corpus/top_k_10_rs_10.json')
        removed_passage_id_path = os.path.join('checkpoints', args.dir_name, f'{i}_th_generator/removed_passge_id_for_{i}_th_generator.json')
        print(os.path.exists(removed_passage_id_path))
        removed_passage_ids = json.load(open(removed_passage_id_path, 'r', encoding='utf-8'))
        removed_passage_ids_list.append(removed_passage_ids)
        union_set = union_set | set(removed_passage_ids)
    

    final_queries_dict = {}
    for i in range(args.num_split):
        query_path = os.path.join('checkpoints', args.dir_name, f'{i}_th_generator/corpus/top_k_10_rs_10.json')
        removed_passage_ids = set(removed_passage_ids_list[i])
        queries_list =  json.load(open(query_path, 'r',  encoding='utf-8'))
        for e in queries_list:
            passage_id = e['passage_id']
            generated_queries = e['generated_queries']
            if passage_id in removed_passage_ids or passage_id not in union_set:
                final_queries_dict[passage_id] = final_queries_dict.get(passage_id, []) + generated_queries
    merge_output_data = []
    for k, v in final_queries_dict.items():
        merge_output_data.append({'passage_id': k, 'generated_queries':v})
    sorted_merge_output_data = sorted(merge_output_data, key=lambda x: int(x['passage_id']))
    with open(os.path.join('checkpoints', args.dir_name, f'top_k_10_rs_10.json'), 'w',  encoding='utf-8') as fw:
        json.dump(sorted_merge_output_data, fw, indent=2)


if __name__ == "__main__":
    main()