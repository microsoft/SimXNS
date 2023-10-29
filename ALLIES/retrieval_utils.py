import sys

from torch.utils.data.dataset import Dataset

sys.path += ['../']
import json
import logging
import os
from os.path import isfile, join
import csv
import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from tqdm import tqdm, trange
import torch.distributed as dist
from torch.serialization import default_restore_location

import pickle

logger = logging.getLogger(__name__)
import faiss
import transformers
from transformers import (
    BertConfig,
    BertTokenizer,
    __version__
)
from modeling_bert import BertModel

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T
from transformers.activations import ACT2FN
import collections


CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset', 'epoch',
                                          'encoder_params'])

transformers.logging.set_verbosity_error()
csv.field_size_limit(sys.maxsize)

class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        self.init_weights()
        self.version = int(__version__.split('.')[0])

    @classmethod
    def init_encoder(cls, dropout: float = 0.1, pretrained_model_name=None):
        pretrained_model_name = 'Luyu/co-condenser-marco'

        model_path = pretrained_model_name
        cfg = BertConfig.from_pretrained(pretrained_model_name)

        cfg.output_hidden_states = True
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        return cls.from_pretrained(model_path, config=cfg)

    def forward(self, **kwargs):
        hidden_states = None
        ## all_layer_hidden： [Layer_num, Batch_size, Seq_len, Embed_size]
        ## all_layer_hidden_all_head： [Layer_num, Batch_size, Head_num, Seq_len, Embed_size_per_head]
        result, _ = super().forward(**kwargs)
        sequence_output = result.last_hidden_state
        all_layer_attention_map = result.attentions
        pooled_output = sequence_output[:, 0, :]
        # all_layer_hidden_adapter = [self.linear_adapter[i](all_layer_hidden[i]) for i in range(len(all_layer_hidden))]
        ## After Permutation: all_layer_hidden： [Batch_size, Layer_num, Seq_len, Embed_size]
        ## After Permutation: all_layer_hidden_all_head： [Batch_size, Layer_num, Head_num, Seq_len, Embed_size_per_head]
        return sequence_output, pooled_output, hidden_states, all_layer_attention_map


class BiBertEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """
    def __init__(self, device):
        super(BiBertEncoder, self).__init__()
        self.pretrained_model_name = 'Luyu/co-condenser-marco'
        self.device = device
        self.question_model = HFBertEncoder.init_encoder()

        self.ctx_model = self.question_model

    def query_emb(self, mode, input_ids, attention_mask):
        if 'colbert' in mode:
            col_output, all_layer_hidden, _ = self.question_model(mode, self.device, input_ids=input_ids, attention_mask=attention_mask)
            return col_output, all_layer_hidden, None
        else:
            _, pooled_output, _, _ = self.question_model(input_ids=input_ids, attention_mask=attention_mask)
            return pooled_output, None, None

    def body_emb(self, mode, input_ids, attention_mask):
        if 'colbert' in mode:
            col_output, all_layer_hidden, mask_doc = self.ctx_model(mode, self.device, input_ids=input_ids, attention_mask=attention_mask)
            return col_output, all_layer_hidden, mask_doc
        else:
            _, pooled_output, _, _ = self.ctx_model(input_ids=input_ids, attention_mask=attention_mask)
            return pooled_output, None, None

    def forward(self, query_ids, attention_mask_q, doc_ids=None, attention_mask_d=None):
        q_embs, _, _ = self.query_emb(self.model_type+'_query', query_ids, attention_mask_q)
        d_embs, _, mask_doc = self.body_emb(self.model_type+'_doc', doc_ids, attention_mask_d)
        return q_embs, d_embs, None, None, mask_doc


def load_model(args):

    # Model definition
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BiBertEncoder(args.device)
    saved_state = load_states_from_checkpoint(args.dr_path)

    model.load_state_dict(saved_state.model_dict, strict=False)

    model.to(args.device)

    return tokenizer, model


def load_states_from_checkpoint(model_file: str):
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)


def load_data():
    passage_path = f'{args.data_path}/psgs_w100.tsv'
    passages = []
    with open(passage_path) as fin:
        reader = csv.reader(fin, delimiter='\t')
        for k, row in enumerate(reader):
            if not row[0] == 'id':
                try:
                    passages.append((int(row[0]) - 1, row[1], row[2]))
                except:
                    pass
    return passages


def get_question_embeddings(questions, tokenizer, model, args):
    model.eval()
    with torch.no_grad():
        token_id = [tokenizer.encode(questions)]

        inputs = {"input_ids": torch.LongTensor(token_id).to(args.device), "attention_mask": (torch.LongTensor(token_id) != 0).long().to(args.device)}
        embs, _, _ = model.query_emb('dual_encoder', **inputs)
        test_question_embedding = embs.detach().cpu().numpy()

    return test_question_embedding


def get_passage_embedding(args):
    with open(f'{args.passage_embdding_path}/passage_embedding.pb', 'rb') as handle:
        passage_embedding = pickle.load(handle)
    with open(f'{args.passage_embdding_path}/passage_embedding2id.pb', 'rb') as handle:
        passage_embedding_id = pickle.load(handle)

    return passage_embedding, passage_embedding_id


def prepare_dr(args):
    tokenizer, model, cpu_index, passage_embedding2id, passages = None,  None,  None,  None,  None 
    if args.retrieval_type =='retrieve':
        # Prepare the dense retriever
        print('Preparing the dense retriever')
        tokenizer, model = load_model(args)
        passages = load_data()
        passage_embedding, passage_embedding2id = get_passage_embedding(args)
        dim = passage_embedding.shape[1]

        new_passage_embedding = passage_embedding.copy()
        for i in trange(passage_embedding.shape[0]):
            new_passage_embedding[passage_embedding2id[i]] = passage_embedding[i]
        del (passage_embedding)
        passage_embedding = new_passage_embedding
        passage_embedding2id = np.arange(passage_embedding.shape[0])
        thread_num = 90
        faiss.omp_set_num_threads(thread_num)
        cpu_index = faiss.IndexFlatIP(dim)
        if args.device == 'cuda':
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            print('Building Index')
            gpu_index_flat = faiss.index_cpu_to_all_gpus(  # build the index
                cpu_index,
                co=co
            )
            gpu_index_flat.add(passage_embedding.astype(np.float32))
            cpu_index = gpu_index_flat
        else:
            cpu_index.add(passage_embedding.astype(np.float32))
        print('Preparing Done!')
    return tokenizer, model, cpu_index, passage_embedding2id, passages