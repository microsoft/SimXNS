import openai
import json
import time
from tqdm import trange
import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter
from rouge import Rouge
import json
import openai
import backoff
import os
from multiprocessing.pool import ThreadPool
import threading
import time
import datetime
from msal import PublicClientApplication, SerializableTokenCache
import json
import os
import atexit
import requests
from utils import *
from retrieval_utils import *
import _thread
from contextlib import contextmanager
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


import signal

cache = {}
default_engine = None

from transformers import AutoTokenizer, AutoModel

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def readfiles(infile):

    if infile.endswith('json'):
        lines = json.load(open(infile, 'r', encoding='utf8'))
    elif infile.endswith('jsonl'):
        lines = open(infile, 'r', encoding='utf8').readlines()
        lines = [json.loads(l) for l in lines]
    else:
        raise NotImplementedError

    if len(lines[0]) == 1 and lines[0].get('prompt'):
        lines = lines[1:] ## skip prompt line

    return lines

def write(log_file, text):
    log_file.write(text + '\n')
    log_file.flush()

def create_directory(path):
    try:
        os.makedirs(path)
        print(f"Created directory: {path}")
    except FileExistsError:
        print(f"Directory already exists: {path}")

def sparse_retrieval(searcher_sparse, query, top_K):
    hits_sparse = searcher_sparse.search(query)
    result = []
    for i in range(top_K):
        result.append([field.stringValue() for field in hits_sparse[i].lucene_document.getFields()][1])
    return result

def retrieval(questions, top_k, tokenizer, model, cpu_index, passage_embedding2id, passages, args):
    question_embedding = get_question_embeddings(questions, tokenizer, model, args)
    _, dev_I = cpu_index.search(question_embedding.astype(np.float32), top_k)  # I: [number of queries, topk]
    topk_document = [passage_embedding2id[dev_I[0][index]] for index in range(len(dev_I[0]))]
    
    return [passages[doc][1] for doc in topk_document]

def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits

def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

def _normalize(text):
    return unicodedata.normalize('NFD', text)

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1(prediction, ground_truths):
    return max([f1_score(prediction, gt) for gt in ground_truths])

def rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def rl(prediction, ground_truths):
    return max([rougel_score(prediction, gt) for gt in ground_truths])

def run_inference_openai(inputs_with_prompts):
    for _ in range(200):
        try:
            completions = openai.ChatCompletion.create(
                          model="gpt-3.5-turbo",
                          messages=[
                              {"role": "system", "content": "You are a helpful AI assistant."},
                              {"role": "user", "content": inputs_with_prompts},
                            ],
                          temperature=0,
                        )
        except:
            continue

    outputs = [c["message"]['content'] for c in completions["choices"]]
    return outputs, completions['usage']['total_tokens']

def print_args(args):
    print("=================================================================")
    print("======================General Setting=========================")
    print(f"Dataset:".rjust(30) + "  " + str(args.dataset))
    print(f"Data Path:".rjust(30) + "  " + str(args.data_path))
    print(f"Task:".rjust(30) + "  " + str(args.task))
    print(f"Retrieval TopK:".rjust(30) + "  " + str(args.topK))
    print(f"Beam Size:".rjust(30) + "  " + str(args.beam_size))
    print(f"Beam Depth:".rjust(30) + "  " + str(args.beam_Depth))
    print(f"Ask Question Num:".rjust(30) + "  " + str(args.ask_question_num))
    print(f"Threshold:".rjust(30) + "  " + str(args.threshold))
    print(f"Device:".rjust(30) + "  " + str(args.device))
    print(f"Summary:".rjust(30) + "  " + str(args.summary))
    print(f"Retrieval Type:".rjust(30) + "  " + str(args.retrieval_type))
    print(f"Identifier:".rjust(30) + "  " + str(args.unique_identifier))
    print(f"Save File:".rjust(30) + "  " + str(args.save_file))
    print("=================================================================")
    print("=================================================================")

def set_openai(args):
    openai.api_key = args.apikey
