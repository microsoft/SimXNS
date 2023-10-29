import logging
import os
import sys
import csv
import json
import datasets
from dataclasses import dataclass
from collections import OrderedDict
csv.field_size_limit(sys.maxsize)
logger = logging.getLogger("__main__")
# logger = logging.getLogger(__name__)

def load_passage(passage_path:str):
    """
    for nq, tq, and msmarco
    """
    if not os.path.exists(passage_path):
        logger.info(f'{passage_path} does not exist')
        return
    logger.info(f'Loading passages from: {passage_path}')
    passages = OrderedDict()
    with open(passage_path, 'r', encoding='utf8') as fin:
        reader = csv.reader(fin, delimiter='\t')
        for row in reader:
            if row[0] != 'id':
                try:
                    if len(row) == 3:
                        # psg_id, text, title
                        passages[row[0]] = (row[0], row[1].strip(), row[2].strip())
                    else:
                         # psg_id, text, title, original psg_id in the psgs_w100.tsv file
                        passages[row[3]] = (row[3], row[1].strip(), row[2].strip())
                except Exception:
                    logger.warning(f'The following input line has not been correctly loaded: {row}')
    logger.info(f'{passage_path} has {len(passages)} passages.')
    return passages
