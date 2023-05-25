import sys
sys.path += ['../']
sys.path += ['../../']
import argparse
import logging
import os
import torch
from torch.nn import functional as F
from torch.nn import MSELoss
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd()))))
import torch.distributed as dist
from models import BiBertEncoder, HFBertEncoder, Reranker
from transformers import BertTokenizer
import pickle
import random
import numpy as np
logger = logging.getLogger(__name__)
from torch.serialization import default_restore_location
from torch import nn
import re
from dataset import TraditionDataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import math
from random import sample
import collections


CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset', 'epoch',
                                          'encoder_params'])

def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model


def sum_main(x, opt):
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x

def text_clean(text):
    text = text.replace("#n#", " ").replace("<sep>", " ").replace("#tab#", " ").replace("#r#", " ").replace("\t", " ")
    return " ".join(text.split())

def url_clean(raw_url):
    """
    https events kgw com Battle of the Bands Sheridan Education Foundation 336639274 html
    -> events kgw battle of the bands sheridan education foundation
    """
    if not raw_url:
        return raw_url

    url = raw_url.lower().replace("https", "").replace("http", "")
    url_list = url.replace("www", "").replace("com", "").replace("html", "").replace("htm", "").split()
    try:
        # deal with `list index out of range` error
        last_item = url_list[-1]
        if len(last_item) >= 8 and bool(re.search(r"\d", last_item)):
            url_list = url_list[:-1]
    except Exception:
        return ''

    return " ".join(url_list)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

class Eval_Tool:
    @classmethod
    def MRR_n(cls, results_list, n):
        mrr_100_list = []
        for hits in results_list:
            score = 0
            for rank, item in enumerate(hits[:n]):
                if item:
                    score = 1.0 / (rank + 1.0)
                    break
            mrr_100_list.append(score)
        return sum(mrr_100_list) / len(mrr_100_list)

    @classmethod
    def MAP_n(cls, results_list, n):
        MAP_n_list = []
        for predict in results_list:
            ap = 0
            hit_num = 1
            for rank, item in enumerate(predict[:n]):
                if item:
                    ap += hit_num / (rank + 1.0)
                    hit_num += 1
            if ap != 0:
                ap /= (hit_num-1)
            else:
                ap = 0
            MAP_n_list.append(ap)
        return sum(MAP_n_list) / len(MAP_n_list)

    @classmethod
    def DCG_n(cls, results_list, n):
        DCG_n_list = []
        for predict in results_list:
            DCG = 0
            for rank, item in enumerate(predict[:n]):
                if item:
                    DCG += 1 / math.log2(rank + 2)
            DCG_n_list.append(DCG)
        return sum(DCG_n_list) / len(DCG_n_list)

    @classmethod
    def nDCG_n(cls, results_list, n):
        nDCG_n_list = []
        for predict in results_list:
            nDCG = 0
            for rank, item in enumerate(predict[:n]):
                if item:
                    nDCG += 1 / math.log2(rank + 2)
            nDCG /= sum([math.log2(i + 2) for i in range(n)])
            nDCG_n_list.append(nDCG)
        return sum(nDCG_n_list) / len(nDCG_n_list)

    @classmethod
    def P_n(cls, results_list, n):
        p_n_list = []
        for predict in results_list:
            true_num = 0
            for rank, item in enumerate(predict[:n]):
                if item:
                    true_num += 1
            p = true_num / n
            p_n_list.append(p)
        return sum(p_n_list) / len(p_n_list)

    @classmethod
    def get_matrics(cls, results_list):
        p_list = [1, 5, 10, 20, 50, 100, 1000]
        metrics = {'MRR_n': cls.MRR_n,
                   'MAP_n': cls.MAP_n,
                   'DCG_n': cls.DCG_n, 'nDCG_n': cls.nDCG_n, 'P_n': cls.P_n}
        result_dict = {}
        for metric_name, fuction in metrics.items():
            for p in p_list:
                temp_result = fuction(results_list, p)
                result_dict[metric_name + '@_' + str(p)] = temp_result
        print(result_dict)
        return result_dict

class SimpleTokenizer:
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)

def gather_tensor(args, t):
    gathered = [torch.empty_like(t) for _ in range(torch.distributed.get_world_size())]
    dist.all_gather(gathered, t)
    gathered[args.rank] = t
    return torch.cat(gathered, dim=0)

def select_layer(args):
    ## student layer selection
    selected_index_list_db, selected_index_list_teacher = None, None
    if args.layer_selection_random:
        if 'large' not in args.pretrained_model_name:
            if args.add_linear:
                all_layer_index = [i for i in range(7)]
            else:
                all_layer_index = [i for i in range(6)]
            selected_index_list_db = sorted(sample(all_layer_index, args.disitll_layer_num))

            if args.add_linear:
                all_layer_index = [i for i in range(13)]
            else:
                all_layer_index = [i for i in range(12)]
            selected_index_list_teacher = sorted(sample(all_layer_index, args.disitll_layer_num))
        else:
            if args.add_linear:
                all_layer_index = [i for i in range(13)]
            else:
                all_layer_index = [i for i in range(12)]
            selected_index_list_db = sorted(sample(all_layer_index, args.disitll_layer_num))

            if args.add_linear:
                all_layer_index = [i for i in range(25)]
            else:
                all_layer_index = [i for i in range(24)]
            selected_index_list_teacher = sorted(sample(all_layer_index, args.disitll_layer_num))

    elif args.layer_selection_last:
        ## pas distill 5 layers, doc distill 6 layers
        if 'ms' in args.train_file:
            selected_index_list_db = [1, 2, 3, 4, 5]
            selected_index_list_teacher = [7, 8, 9, 10, 11]
        else:
            selected_index_list_db = [0, 1, 2, 3, 4, 5]
            selected_index_list_teacher = [6, 7, 8, 9, 10, 11]

    elif args.layer_selection_skip:
        if 'ms' in args.train_file:
            selected_index_list_db = [1, 2, 3, 4, 5]
            selected_index_list_teacher = [0, 2, 4, 6, 8]
        else:
            selected_index_list_db = [0, 1, 2, 3, 4, 5]
            selected_index_list_teacher = [0, 2, 4, 6, 8, 10]

    print(selected_index_list_db)
    print(selected_index_list_teacher)
    return selected_index_list_db, selected_index_list_teacher

def evaluate_dev(args, model, tokenizer):
    # Valid dataset
    if args.use_academic:
        val_dataset = TraditionDataset(args.valid_file, tokenizer, num_hard_negatives=args.num_hard_negatives,
                                         max_seq_length=args.max_doc_length, shuffle_positives=args.shuffle_positives)
    # else:
    #     val_dataset = BingMetaSubSetWithHardTrain(args.train_q2d_file, args.train_docpool_file, tokenizer,
    #                                                 args.model_type,
    #                                                 max_doc_length=args.max_doc_length,
    #                                                 num_hard_negatives=args.num_hard_negatives,
    #                                                 negative_q2d_file=args.negative_q2d_file,
    #                                                 negative_docpool_file=args.negative_docpool_file)

    val_sample = RandomSampler(val_dataset) if args.local_rank == -1 else DistributedSampler(val_dataset)

    ## when using DDP, each process should hold train_batch_size samples,so the real batch size should equal to train_batch_size * gpu_num
    val_dataloader = DataLoader(val_dataset, sampler=val_sample,
                                collate_fn=val_dataset.get_collate_fn,
                                batch_size=args.train_batch_size, num_workers=16)

    epoch_iterator = tqdm(val_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

    model.eval()
    correct_predictions_count_all = 0
    example_num = 0
    total_loss = 0

    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            if 'cross_encoder' in args.model_type:
                batch_reranker = batch['reranker']
                inputs_reranker = {"input_ids": batch_reranker[0].long().to(args.device),
                                   "attention_mask": batch_reranker[1].long().to(args.device)}
                output_reranker, _= model(**inputs_reranker)
                relevance_logits = output_reranker
                relevance_target = torch.zeros(relevance_logits.size(0), dtype=torch.long).to(args.device)
                loss_fct = torch.nn.CrossEntropyLoss()
                relative_loss = loss_fct(relevance_logits, relevance_target)
                total_loss += relative_loss
                max_score, max_idxs = torch.max(relevance_logits, 1)
                correct_predictions_count = (max_idxs == 0).sum()
                correct_predictions_count_all += correct_predictions_count
                example_num += batch['reranker'][1].size(0)

            elif 'dual_encoder' in args.model_type:
                batch_retriever = batch['retriever']
                inputs_retriever = {"query_ids": batch_retriever[0].long().to(args.device),
                                    "attention_mask_q": batch_retriever[1].long().to(args.device),
                                    "doc_ids": batch_retriever[2].long().to(args.device),
                                    "attention_mask_d": batch_retriever[3].long().to(args.device)}
                local_q_vector, local_ctx_vectors, _, _, _, _, _ = model(**inputs_retriever)

                relevance_logits = torch.matmul(local_q_vector, local_ctx_vectors.transpose(0, 1))

                q_num = local_q_vector.size(0)
                scores = relevance_logits.view(q_num, -1)
                sample_num = local_ctx_vectors.size(0) // local_q_vector.size(0)
                softmax_scores = F.log_softmax(scores, dim=1)
                target = torch.arange(0, local_q_vector.size(0) * sample_num, sample_num, device=relevance_logits.device,
                                      dtype=torch.long)
                loss = F.nll_loss(
                    softmax_scores,
                    target,
                    reduction='mean',
                )
                total_loss += loss
                max_score, max_idxs = torch.max(relevance_logits, 1)
                correct_predictions_count = (max_idxs == target).sum()
                correct_predictions_count_all += correct_predictions_count
                example_num += batch['reranker'][1].size(0)

            elif 'distilbert' in args.model_type:
                batch_retriever = batch['retriever']
                inputs_retriever = {"query_ids": batch_retriever[0].long().to(args.device),
                                    "attention_mask_q": batch_retriever[1].long().to(args.device),
                                    "doc_ids": batch_retriever[2].long().to(args.device),
                                    "attention_mask_d": batch_retriever[3].long().to(args.device)}
                local_q_vector, local_ctx_vectors, _, _, _, _, _ = model(**inputs_retriever)

                relevance_logits = torch.matmul(local_q_vector, local_ctx_vectors.transpose(0, 1))

                q_num = local_q_vector.size(0)
                scores = relevance_logits.view(q_num, -1)
                sample_num = local_ctx_vectors.size(0) // local_q_vector.size(0)
                softmax_scores = F.log_softmax(scores, dim=1)
                target = torch.arange(0, local_q_vector.size(0) * sample_num, sample_num, device=relevance_logits.device,
                                      dtype=torch.long)
                loss = F.nll_loss(
                    softmax_scores,
                    target,
                    reduction='mean',
                )
                total_loss += loss
                max_score, max_idxs = torch.max(relevance_logits, 1)
                correct_predictions_count = (max_idxs == target).sum()
                correct_predictions_count_all += correct_predictions_count
                example_num += batch['reranker'][1].size(0)

            elif 'colbert' in args.model_type:
                batch_retriever = batch['retriever']
                inputs_retriever = {"query_ids": batch_retriever[4].long().to(args.device),
                                    "attention_mask_q": batch_retriever[5].long().to(args.device),
                                    "doc_ids": batch_retriever[6].long().to(args.device),
                                    "attention_mask_d": batch_retriever[7].long().to(args.device)}
                local_q_vector, local_ctx_vectors, _, _, _, _, _ = model(**inputs_retriever)
                attention_mask = inputs_retriever['attention_mask_d']
                logits_1 = torch.matmul(local_q_vector.view(local_q_vector.shape[0] * local_q_vector.shape[1], -1),
                                        local_ctx_vectors.permute(2, 0, 1).view(local_ctx_vectors.shape[2], -1))

                ## logits_1: [batch_size * max_query_len, batch_size * (neg_doc_num + 1) * max_doc_len]
                logits_1_mask = logits_1.index_fill_(1, torch.nonzero(
                    attention_mask.view(attention_mask.shape[0] * attention_mask.shape[1]) == 0).squeeze(1), -1e4)

                ## logits_2: [batch_size, max_query_len, batch_size * (neg_doc_num + 1), max_doc_len]
                logits_2 = logits_1_mask.view(local_q_vector.shape[0], local_q_vector.shape[1], local_ctx_vectors.shape[0],
                                              local_ctx_vectors.shape[1])

                ## logits: [batch_size, batch_size * (neg_doc_num + 1)]
                relevance_logits = logits_2.max(3).values.sum(1)

                q_num = local_q_vector.size(0)
                scores = relevance_logits.view(q_num, -1)
                sample_num = local_ctx_vectors.size(0) // local_q_vector.size(0)
                softmax_scores = F.log_softmax(scores, dim=1)
                target = torch.arange(0, local_q_vector.size(0) * sample_num, sample_num, device=relevance_logits.device,
                                      dtype=torch.long)
                loss = F.nll_loss(
                    softmax_scores,
                    target,
                    reduction='mean',
                )
                total_loss += loss
                max_score, max_idxs = torch.max(relevance_logits, 1)
                correct_predictions_count = (max_idxs == target).sum()
                correct_predictions_count_all += correct_predictions_count
                example_num += batch['reranker'][1].size(0)

    example_num = torch.tensor(1).to(relevance_logits) * example_num
    total_loss = torch.tensor(1).to(relevance_logits) * total_loss
    correct_predictions_count_all = torch.tensor(1).to(relevance_logits) * correct_predictions_count_all
    correct_predictions_count_all = sum_main(correct_predictions_count_all, args)
    example_num = sum_main(example_num, args)
    total_loss = sum_main(total_loss, args)
    total_loss = total_loss / step
    correct_ratio = float(correct_predictions_count_all / example_num)
    logger.info('NLL Validation: loss = %f. correct prediction ratio  %d/%d ~  %f', total_loss,
                correct_predictions_count_all.item(),
                example_num.item(),
                correct_ratio
                )
    model.train()

    return total_loss, correct_ratio

def get_loss_cross(args, relevance_logits):
    relevance_target = torch.zeros(relevance_logits.size(0), dtype=torch.long).to(args.device)
    loss_fct = torch.nn.CrossEntropyLoss()
    relative_loss = loss_fct(relevance_logits, relevance_target)
    return relative_loss

def get_loss_dual(args, local_q_vector, local_ctx_vectors, attention_mask, reduction='mean'):
    if args.local_rank != -1:
        all_q_vector = gather_tensor(args, local_q_vector.contiguous())
        all_ctx_vectors = gather_tensor(args, local_ctx_vectors.contiguous())
    else:
        all_q_vector = local_q_vector
        all_ctx_vectors = local_ctx_vectors

    if 'colbert' in args.model_type:
        ## all_q_vector:[batch_size, max_query_len, embedding_dim]
        ## all_ctx_vectors: [batch_size * (neg_doc_num + 1), max_doc_len, embedding_dim]
        ## logits_1: [batch_size * max_query_len, batch_size * (neg_doc_num + 1) * max_doc_len]
        logits_1 = torch.matmul(all_q_vector.view(all_q_vector.shape[0] * all_q_vector.shape[1], -1), all_ctx_vectors.permute(2, 0, 1).view(all_ctx_vectors.shape[2], -1))

        ## logits_1: [batch_size * max_query_len, batch_size * (neg_doc_num + 1) * max_doc_len]
        logits_1_mask = logits_1.index_fill_(1, torch.nonzero(attention_mask.view(attention_mask.shape[0] * attention_mask.shape[1]) == 0).squeeze(1), -1e4)

        ## logits_2: [batch_size, max_query_len, batch_size * (neg_doc_num + 1), max_doc_len]
        logits_2 = logits_1_mask.view(all_q_vector.shape[0], all_q_vector.shape[1], all_ctx_vectors.shape[0], all_ctx_vectors.shape[1])

        ## logits: [batch_size, batch_size * (neg_doc_num + 1)]
        logits = logits_2.max(3).values.sum(1)
    else:
        logits = torch.matmul(all_q_vector, all_ctx_vectors.transpose(0, 1))

    q_num = all_q_vector.size(0)
    scores = logits.view(q_num, -1)
    sample_num = all_ctx_vectors.size(0) // all_q_vector.size(0)
    softmax_scores = F.log_softmax(scores, dim=1)
    target = torch.arange(0, all_q_vector.size(0) * sample_num, sample_num, device=logits.device, dtype=torch.long)
    loss = F.nll_loss(
        softmax_scores,
        target,
        reduction=reduction,
    )
    return loss

def compute_dual_regularization_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
    loss = (p_loss + q_loss) / 2
    return loss

def colbert_score(q_embs_col, d_embs_col, attention_mask, q_num):
    logits_1 = torch.matmul(q_embs_col.reshape(q_embs_col.shape[0] * q_embs_col.shape[1], -1), d_embs_col.permute(2, 0, 1).reshape(d_embs_col.shape[2], -1))
    logits_1_mask = logits_1.index_fill_(1, torch.nonzero(attention_mask.view(attention_mask.shape[0] * attention_mask.shape[1]) == 0).squeeze(1), -9e9)
    logits_2 = logits_1_mask.view(q_embs_col.shape[0], q_embs_col.shape[1], d_embs_col.shape[0], d_embs_col.shape[1])
    logits_col = logits_2.max(3).values.sum(1)
    scores_col = logits_col.view(q_num, -1)
    return scores_col

def attention_map_loss(batch, last_context_layer_col_d, last_context_layer_col_q, last_attention_map, doc_mask_col):
    ## Add attention map distillation
    ce_span = [elem_ for elem in batch['ce_ctx_start_end'] for elem_ in elem]
    de_span = [elem_ for elem in batch['de_ctx_start_end'] for elem_ in elem]
    ## [Batch_size, Head_number, (1+num_neg), Max_passage_len, Embed_size]
    doc_token_embedding = last_context_layer_col_d.view(last_context_layer_col_q.shape[0], -1,
                                                        last_context_layer_col_d.shape[1],
                                                        last_context_layer_col_d.shape[2],
                                                        last_context_layer_col_d.shape[3]).permute(0, 2, 1, 3, 4)
    ## [Batch_size, (1+num_neg), Head_number, Max_passage_len, Max_query_len]
    query_doc_attention = torch.einsum('ijkl,ijmnl->ijmnk', [last_context_layer_col_q, doc_token_embedding]).permute(
        [0, 2, 1, 3, 4])
    ## [Batch_size*(1+num_neg), Head_number, Max_query_len, Max_passage_len]
    query_doc_attention = query_doc_attention.contiguous().view(-1, query_doc_attention.shape[2],
                                                                query_doc_attention.shape[3],
                                                                query_doc_attention.shape[4]).permute(0, 1, 3, 2)
    ## [Batch_size*(1+num_neg), Head_number, Max_query_len, Max_passage_len]
    doc_mask = doc_mask_col.squeeze(2).unsqueeze(1).unsqueeze(1).repeat(1, query_doc_attention.shape[1],
                                                                        query_doc_attention.shape[2], 1)
    query_doc_attention = query_doc_attention.masked_fill_(mask=(doc_mask == 0).bool(), value=-1e9)

    ## attention loss of each query doc pair
    ce_col_attention_loss_list = []
    for i in range(len(de_span)):
        ##  [Head_num, Query_len, Doc_len]
        query_doc_attention_instance = query_doc_attention[i][:, 1:de_span[i][0] - 1, 1:de_span[i][1] - 1]
        last_attention_map_instance = last_attention_map[i][:, 1:ce_span[i][0] - 1, ce_span[i][0]:ce_span[i][1]].clone()
        ##  Need mask index
        mask_index = (query_doc_attention_instance[0, 0, :] != -1e9).nonzero()
        ##  [Head_num, Query_len, Non_masked_doc_len]
        query_doc_attention_instance = torch.index_select(query_doc_attention_instance, 2, mask_index.squeeze())
        last_attention_map_instance = torch.index_select(last_attention_map_instance, 2, mask_index.squeeze())
        ce_col_attention_loss_list.append(F.kl_div(F.log_softmax(query_doc_attention_instance, dim=-1),
                                                   F.softmax(last_attention_map_instance, dim=-1),
                                                   reduction='batchmean'))
    return ce_col_attention_loss_list

def query_doc_attention_map(args, q_all_layer_hidden, d_all_layer_hidden, query_len, doc_len, selected_index_list):
    # q_all_layer_hidden_dual: [Query_num, Layer_num, Max_q_len, Dim]
    # d_all_layer_hidden_dual: [Doc_num, Layer_num, Max_d_len, Dim]

    # q_all_layer_hidden_col: [Query_num, Layer_num, Max_q_len, Dim]
    # d_all_layer_hidden_col: [Doc_num, Layer_num, Max_d_len, Dim]
    q_all_layer_hidden = q_all_layer_hidden.permute([1, 0, 2, 3])
    d_all_layer_hidden = d_all_layer_hidden.permute([1, 0, 2, 3])
    query_doc_attention = [torch.einsum('ijk,mnk->ijnm', [q_all_layer_hidden[i], d_all_layer_hidden[i]]).permute([0, 3, 1, 2])
                           for i in selected_index_list]

    # query_doc_attention_mask: [Query_num, Doc_num, Max_q_len, Max_d_len]
    query_doc_attention_mask = torch.ones(query_doc_attention[0].shape).to(args.device)
    for i in range(query_doc_attention[0].shape[0]):
        for j in range(query_doc_attention[0].shape[1]):
            query_doc_attention_mask[i, j, :query_len[i], :doc_len[j]] = 0

    query_doc_attention_score = []
    query_doc_attention_target = []
    for i in range(len(query_doc_attention)):
        query_doc_attention[i] = query_doc_attention[i].masked_fill(mask=query_doc_attention_mask.bool(), value=torch.tensor(-1e9))
        query_doc_attention_score.append(F.log_softmax(query_doc_attention[i], dim=-1))
        query_doc_attention_target.append(F.softmax(query_doc_attention[i] / args.temperature, dim=-1))
        query_doc_attention[i] = F.softmax(query_doc_attention[i], dim=-1)

    return query_doc_attention, query_doc_attention_score, query_doc_attention_target

def virt_loss(query_doc_attention_t, query_doc_attention_s, distill_para):
    mse_loss_fn = MSELoss(reduction='sum')
    loss_t_s_attention = 0
    for i in range(len(query_doc_attention_t)):
        loss_t_s_attention += mse_loss_fn(query_doc_attention_s[i], query_doc_attention_t[i])
    loss_t_s_attention = loss_t_s_attention / len(query_doc_attention_t) / query_doc_attention_t[0].shape[0] / query_doc_attention_t[0].shape[1] * distill_para
    return loss_t_s_attention

def layer_score_dis_loss(args, target, softmax_targets_teacher_all_layer, softmax_scores_teacher_all_layer,
                         softmax_targets_student_all_layer, softmax_scores_student_all_layer, distill_para, loss_dict):
    loss_t_s_layer_dis_list = []
    loss_t_s_layer_lambda_list = []
    for index in range(len(softmax_targets_teacher_all_layer)):
        loss_tmp_1 = F.kl_div(softmax_scores_student_all_layer[index], softmax_targets_teacher_all_layer[index], reduction='batchmean')
        loss_lambda_1 = F.nll_loss(softmax_scores_teacher_all_layer[index], target, reduction='mean')
        loss_dict[f'loss_score_layer_dis_{index}'] = loss_tmp_1
        loss_t_s_layer_dis_list.append(loss_tmp_1)
        loss_t_s_layer_lambda_list.append(-loss_lambda_1 / args.layer_temperature)

    loss_t_s_layer_lambda_list = F.softmax(torch.tensor(loss_t_s_layer_lambda_list), dim=0)

    loss_t_s, loss_s_t = 0, 0
    for i in range(len(loss_t_s_layer_dis_list)):
        if args.layer_score_reweight:
            loss_t_s += loss_t_s_layer_lambda_list[i] * loss_t_s_layer_dis_list[i]
        else:
            loss_t_s += distill_para * loss_t_s_layer_dis_list[i]

    return loss_t_s, loss_s_t

def distill_loss(args, q_embs_col, q_embs_dual, q_embs_db, d_embs_col, d_embs_dual, d_embs_db, doc_mask_col,
                 selected_index_list_db, selected_index_list_teacher, attention_map_ce,
                 q_all_layer_hidden_dual, d_all_layer_hidden_dual, q_all_layer_hidden_db, d_all_layer_hidden_db, q_all_layer_hidden_col, d_all_layer_hidden_col,
                 all_layer_score_ce, attention_mask, output_reranker, batch, reduction='mean'):
    len_info = batch['de_ctx_start_end'].to(args.device)

    if args.local_rank != -1 and not args.distill_ce:
        len_info = gather_tensor(args, len_info.contiguous())
        if args.distill_db:
            q_embs_db = gather_tensor(args, q_embs_db.contiguous())
            d_embs_db = gather_tensor(args, d_embs_db.contiguous())
            q_all_layer_hidden_db = gather_tensor(args, q_all_layer_hidden_db.contiguous())
            d_all_layer_hidden_db = gather_tensor(args, d_all_layer_hidden_db.contiguous())

        if args.distill_de:
            q_embs_dual = gather_tensor(args, q_embs_dual.contiguous())
            d_embs_dual = gather_tensor(args, d_embs_dual.contiguous())
            q_all_layer_hidden_dual = gather_tensor(args, q_all_layer_hidden_dual.contiguous())
            d_all_layer_hidden_dual = gather_tensor(args, d_all_layer_hidden_dual.contiguous())

        if args.distill_col:
            q_embs_col = gather_tensor(args, q_embs_col.contiguous())
            d_embs_col = gather_tensor(args, d_embs_col.contiguous())
            q_all_layer_hidden_col = gather_tensor(args, q_all_layer_hidden_col.contiguous())
            d_all_layer_hidden_col = gather_tensor(args, d_all_layer_hidden_col.contiguous())

    query_len = [elem[0][0] for elem in len_info.cpu().numpy().tolist()]
    doc_len = [elem[1] for tuple_ in len_info.cpu().numpy().tolist() for elem in tuple_]

    ########################################################################################################################################################################
    ## Prepare loss
    ########################################################################################################################################################################

    ## distilbert
    if args.distill_db:
        q_num = q_embs_db.size(0)
        sample_num = d_embs_db.size(0) // q_embs_db.size(0)
        target = torch.arange(0, q_num * sample_num, sample_num, device=args.device, dtype=torch.long)

        logits_db = torch.matmul(q_embs_db, d_embs_db.transpose(0, 1))
        scores_db = logits_db.view(q_num, -1)
        softmax_scores_db = F.log_softmax(scores_db, dim=-1)
        softmax_target_db = F.softmax(scores_db / args.temperature, dim=-1)

        softmax_scores_db_selected = []
        softmax_targets_db_selected = []
        if args.distill_de_db_layer_score or args.distill_col_db_layer_score or args.distill_ce_db_layer_score:
            for layer_index in selected_index_list_db:
                logits_db_ = torch.matmul(q_all_layer_hidden_db[:, layer_index, 0, :], d_all_layer_hidden_db[:, layer_index, 0, :].transpose(0, 1))
                scores_db_ = logits_db_.view(q_num, -1)
                softmax_scores_db_ = F.log_softmax(scores_db_, dim=-1)
                softmax_target_db_ = F.softmax(scores_db_ / args.temperature, dim=-1)
                softmax_scores_db_selected.append(softmax_scores_db_)
                softmax_targets_db_selected.append(softmax_target_db_)

        # VIRT loss
        if args.distill_col_db_attention or args.distill_de_db_attention or args.distill_ce_db_attention:
            query_doc_attention_db, _, _ = query_doc_attention_map(args, q_all_layer_hidden_db, d_all_layer_hidden_db, query_len, doc_len, selected_index_list_db)

    ## dual_encoder
    if args.distill_de:
        q_num = q_embs_dual.size(0)
        sample_num = d_embs_dual.size(0) // q_embs_dual.size(0)
        target = torch.arange(0, q_num * sample_num, sample_num, device=args.device, dtype=torch.long)

        logits_dual = torch.matmul(q_embs_dual, d_embs_dual.transpose(0, 1))
        scores_dual = logits_dual.view(q_num, -1)
        softmax_scores_dual = F.log_softmax(scores_dual, dim=-1)
        softmax_target_dual = F.softmax(scores_dual / args.temperature, dim=-1)

        softmax_scores_de_selected = []
        softmax_targets_de_selected = []
        if args.distill_de_db_layer_score:
            for layer_index in selected_index_list_teacher:
                logits_dual_ = torch.matmul(q_all_layer_hidden_dual[:, layer_index, 0, :], d_all_layer_hidden_dual[:, layer_index, 0, :].transpose(0, 1))
                scores_dual_ = logits_dual_.view(q_num, -1)
                softmax_scores_dual_ = F.log_softmax(scores_dual_, dim=-1)
                softmax_target_dual_ = F.softmax(scores_dual_ / args.temperature, dim=-1)
                softmax_scores_de_selected.append(softmax_scores_dual_)
                softmax_targets_de_selected.append(softmax_target_dual_)

        # VIRT attention
        if args.distill_de_db_attention:
            query_doc_attention_de, _, _ = query_doc_attention_map(args, q_all_layer_hidden_dual, d_all_layer_hidden_dual, query_len, doc_len, selected_index_list_teacher)

    ## col_bert
    if args.distill_col:
        q_num = q_embs_col.size(0)
        sample_num = d_embs_col.size(0) // q_embs_col.size(0)
        target = torch.arange(0, q_num * sample_num, sample_num, device=args.device, dtype=torch.long)
        scores_col = colbert_score(q_embs_col, d_embs_col, attention_mask, q_num)
        softmax_scores_col = F.log_softmax(scores_col, dim=-1)
        softmax_target_col = F.softmax(scores_col / args.temperature, dim=-1)

        softmax_scores_col_selected = []
        softmax_targets_col_selected = []
        if args.distill_col_db_layer_score:
            for layer_index in selected_index_list_teacher:
                scores_col_ = colbert_score(q_all_layer_hidden_col[:, layer_index, :, :], d_all_layer_hidden_col[:, layer_index, :, :], attention_mask, q_num)
                softmax_scores_col_ = F.log_softmax(scores_col_, dim=-1)
                softmax_target_col_ = F.softmax(scores_col_ / args.temperature, dim=-1)
                softmax_scores_col_selected.append(softmax_scores_col_)
                softmax_targets_col_selected.append(softmax_target_col_)

        # VIRT loss
        if args.distill_col_db_attention:
            query_doc_attention_col, _, _ = query_doc_attention_map(args, q_all_layer_hidden_col, d_all_layer_hidden_col, query_len, doc_len, selected_index_list_teacher)

    ## cross encoder
    if args.distill_ce:
        softmax_scores_ce = F.log_softmax(output_reranker, dim=-1)
        softmax_target_ce = F.softmax(output_reranker / args.temperature, dim=-1)
        target = torch.zeros(softmax_scores_ce.size(0), dtype=torch.long).to(args.device)

        softmax_scores_ce_selected = []
        softmax_targets_ce_selected = []
        if args.distill_ce_db_layer_score:
            for layer_index in selected_index_list_teacher:
                softmax_scores_ce_ = F.log_softmax(all_layer_score_ce[layer_index], dim=-1)
                softmax_target_ce_ = F.softmax(all_layer_score_ce[layer_index] / args.temperature, dim=-1)
                softmax_scores_ce_selected.append(softmax_scores_ce_)
                softmax_targets_ce_selected.append(softmax_target_ce_)

        # VIRT loss
        if args.distill_ce_db_attention:
            query_doc_attention_ce = []
            for i in selected_index_list_teacher:
                layer_attention_map_ce = torch.mean(attention_map_ce[i], dim=1)
                all_instance_attention_map = []
                for j in range(layer_attention_map_ce.shape[0]):
                    query_len = batch['ce_ctx_start_end'][int(j/batch['ce_ctx_start_end'].shape[1])][int(j%batch['ce_ctx_start_end'].shape[1])][0].item()
                    doc_idx = batch['ce_ctx_start_end'][int(j/batch['ce_ctx_start_end'].shape[1])][int(j%batch['ce_ctx_start_end'].shape[1])][1].item()
                    instance_attention_map = layer_attention_map_ce[j, 1:query_len-1, query_len:doc_idx]
                    instance_attention_map = torch.cat([instance_attention_map,
                                                        torch.ones(instance_attention_map.shape[0], args.max_doc_length - instance_attention_map.shape[1]).to(args.device)*-1e9], dim=1)
                    instance_attention_map = torch.cat([instance_attention_map,
                                                        torch.ones(args.max_query_length - instance_attention_map.shape[0], instance_attention_map.shape[1]).to(args.device)*-1e9], dim=0)
                    all_instance_attention_map.append(F.softmax(instance_attention_map, dim=-1).unsqueeze(0))

                query_doc_attention_ce.append(torch.cat(all_instance_attention_map, 0))

        ## We need to reshape the softmax of de for distillation purpose, mainly to remove cross-batch negatives
        if args.distill_db:
            scores_db = torch.stack([scores_db[i, i * sample_num: (i + 1) * sample_num] for i in range(scores_db.shape[0])], dim=0)
            softmax_scores_db = F.log_softmax(scores_db, dim=-1)
            softmax_target_db = F.softmax(scores_db / args.temperature, dim=-1)

            softmax_scores_db_selected = []
            softmax_targets_db_selected = []
            for layer_index in selected_index_list_db:
                logits_db_ = torch.matmul(q_all_layer_hidden_db[:, layer_index, 0, :], d_all_layer_hidden_db[:, layer_index, 0, :].transpose(0, 1))
                scores_db_ = logits_db_.view(q_num, -1)
                scores_db_ = torch.stack([scores_db_[i, i * sample_num: (i + 1) * sample_num] for i in range(scores_db_.shape[0])], dim=0)
                softmax_scores_db_ = F.log_softmax(scores_db_, dim=-1)
                softmax_target_db_ = F.softmax(scores_db_ / args.temperature, dim=-1)
                softmax_scores_db_selected.append(softmax_scores_db_)
                softmax_targets_db_selected.append(softmax_target_db_)


            if args.distill_ce_db_attention:
                all_query_doc_attention_all = []
                for layer_index in range(len(query_doc_attention_db)):
                    query_doc_attention_db_layer = query_doc_attention_db[layer_index]
                    query_doc_attention_layer = []
                    for i in range(query_doc_attention_db_layer.shape[0]):
                        for j in range(sample_num):
                            query_doc_attention_layer.append(query_doc_attention_db_layer[i, sample_num*i + j].unsqueeze(0))
                    all_query_doc_attention_all.append(torch.cat(query_doc_attention_layer, 0))
                query_doc_attention_db = all_query_doc_attention_all

    ########################################################################################################################################################################
    ## Sum loss
    ########################################################################################################################################################################
    loss_dict = {}
    loss = 0

    if args.distill_db:
        ## loss_db
        loss_db = F.nll_loss(softmax_scores_db, target, reduction=reduction)
        loss_dict['loss_db'] = loss_db
        loss += args.distill_para_db * loss_db

    if args.distill_de:
        if args.train_de:
            ## loss_de
            loss_dual = F.nll_loss(softmax_scores_dual, target, reduction=reduction)
            loss_dict['loss_dual'] = loss_dual
            loss += args.distill_para_de * loss_dual

        if args.distill_db:
            loss_de_db_dis = F.kl_div(softmax_scores_db, softmax_target_dual, reduction='batchmean')
            loss_db_de_dis = F.kl_div(softmax_scores_dual, softmax_target_db, reduction='batchmean')
            if args.train_de:
                loss += args.distill_para_de_db_dis * (loss_de_db_dis + loss_db_de_dis)
            else:
                loss += args.distill_para_de_db_dis * (loss_de_db_dis)
            loss_dict['loss_de_db_dis'], loss_dict['loss_db_de_dis'] = loss_de_db_dis, loss_db_de_dis

            ## loss_db_de_layer_dis & loss_de_db_layer_dis
            if args.distill_de_db_layer_score:
                loss_de_db_layer_dis, loss_db_de_layer_dis = layer_score_dis_loss(args, target, softmax_targets_de_selected, softmax_scores_de_selected,
                                                                                  softmax_targets_db_selected, softmax_scores_db_selected, args.distill_para_de_db_layer_score, loss_dict)
                loss += (loss_de_db_layer_dis + loss_db_de_layer_dis)
                loss_dict[f'loss_de_db_layer_dis'], loss_dict[f'loss_db_de_layer_dis'] = loss_de_db_layer_dis, loss_db_de_layer_dis
            
            ## loss_db_de_virt_dis & loss_de_db_virt_dis
            if args.distill_de_db_attention:
                loss_de_db_attention = virt_loss(query_doc_attention_de, query_doc_attention_db, args.distill_para_de_db_attention)
                loss_dict['loss_de_db_attention'] = loss_de_db_attention
                loss += (loss_de_db_attention)

    if args.distill_col:
        if args.train_col:
            ## loss_col
            loss_col = F.nll_loss(softmax_scores_col, target, reduction=reduction)
            loss += args.distill_para_col * loss_col
            loss_dict['loss_col'] = loss_col

        ## loss_col_db_layer_dis & loss_db_col_layer_dis
        if args.distill_db:
            loss_col_db_dis = F.kl_div(softmax_scores_db, softmax_target_col, reduction='batchmean')
            loss_db_col_dis = F.kl_div(softmax_scores_col, softmax_target_db, reduction='batchmean')
            if args.train_col:
                loss += args.distill_para_col_db_dis * (loss_col_db_dis + loss_db_col_dis)
            else:
                loss += args.distill_para_col_db_dis * (loss_col_db_dis)
            loss_dict['loss_col_db_dis'], loss_dict['loss_db_col_dis'] = loss_col_db_dis, loss_db_col_dis

            if args.distill_col_db_layer_score:
                loss_col_db_layer_dis, loss_db_col_layer_dis = layer_score_dis_loss(args, target, softmax_targets_col_selected, softmax_scores_col_selected,
                                                                                    softmax_targets_db_selected, softmax_scores_db_selected,
                                                                                    args.distill_para_col_db_layer_score, loss_dict)
                loss_dict[f'loss_col_db_layer_dis'], loss_dict[f'loss_db_col_layer_dis'] = loss_col_db_layer_dis, loss_db_col_layer_dis
                loss += (loss_col_db_layer_dis + loss_db_col_layer_dis)

            if args.distill_col_db_attention:
                loss_col_db_attention = virt_loss(query_doc_attention_col, query_doc_attention_db, args.distill_para_col_db_attention)
                loss_dict['loss_col_db_attention']= loss_col_db_attention
                loss += (loss_col_db_attention)

    if args.distill_ce:
        if args.train_ce:
            ## loss_ce
            loss_ce = F.nll_loss(softmax_scores_ce, target, reduction=reduction)
            loss_dict['loss_ce'] = loss_ce
            loss += args.distill_para_ce * loss_ce

        ##  loss_ce_de_dis & loss_de_ce_dis
        if args.distill_db:
            loss_ce_db_dis = F.kl_div(softmax_scores_db, softmax_target_ce, reduction='batchmean')
            loss_db_ce_dis = F.kl_div(softmax_scores_ce, softmax_target_db, reduction='batchmean')
            if args.train_ce:
                loss += args.distill_para_ce_db_dis * (loss_ce_db_dis + loss_db_ce_dis)
            else:
                loss += args.distill_para_ce_db_dis * (loss_ce_db_dis)
            loss_dict['loss_ce_db_dis'], loss_dict['loss_db_ce_dis'] = loss_ce_db_dis, loss_db_ce_dis

            if args.distill_ce_db_layer_score:
                loss_ce_db_layer_dis, loss_db_ce_layer_dis = layer_score_dis_loss(args, target, softmax_targets_ce_selected, softmax_scores_ce_selected,
                                                                                  softmax_targets_db_selected, softmax_scores_db_selected,
                                                                                  args.distill_para_ce_db_layer_score, loss_dict)

                loss_dict[f'loss_ce_db_layer_dis'], loss_dict[f'loss_db_ce_layer_dis'] = loss_ce_db_layer_dis, loss_db_ce_layer_dis
                loss += (loss_ce_db_layer_dis + loss_db_ce_layer_dis)

            if args.distill_ce_db_attention:
                loss_ce_db_attention = virt_loss(query_doc_attention_ce, query_doc_attention_db, args.distill_para_ce_db_attention)
                loss_dict['loss_ce_db_attention'] = loss_ce_db_attention
                loss += (loss_ce_db_attention)

    return loss, loss_dict

def fwd_pass(args, model, batch):
    batch_reranker = tuple(t.to(args.device) for t in batch['reranker'])
    inputs_reranker = {"input_ids": batch_reranker[0].long(), "attention_mask": batch_reranker[1].long()}
    output_reranker = model(**inputs_reranker)
    relevance_logits = output_reranker
    relevance_target = torch.zeros(relevance_logits.size(0), dtype=torch.long).to(args.device)
    loss_fct = torch.nn.CrossEntropyLoss()
    relative_loss = loss_fct(relevance_logits, relevance_target)
    return relative_loss.item()

def _save_checkpoint(args, model, optimizer, scheduler, step: int, unique_identifier, model_type) -> str:
    offset = step
    epoch = 0
    model_to_save = get_model_obj(model)
    if not os.path.exists(args.output_dir + '/' + unique_identifier):
        os.makedirs(args.output_dir + '/' + unique_identifier)

    cp = os.path.join(args.output_dir + '/' + unique_identifier, f'{model_type}-checkpoint-' + str(offset))

    meta_params = {}

    state = CheckpointState(model_to_save.state_dict(),
                            optimizer.state_dict(),
                            scheduler.state_dict(),
                            offset,
                            epoch, meta_params
                            )
    torch.save(state._asdict(), cp)
    logger.info('Saved checkpoint at %s', cp)
    return cp

def _load_saved_state(model, optimizer, scheduler, saved_state: CheckpointState):
    epoch = saved_state.epoch
    step = saved_state.offset
    logger.info('Loading checkpoint @ step=%s', step)

    model_to_load = get_model_obj(model)
    logger.info('Loading saved model state ...')
    model_to_load.load_state_dict(saved_state.model_dict)  # set strict=False if you use extra projection
    scheduler.load_state_dict(saved_state.scheduler_dict)
    return step

def load_states_from_checkpoint_ict(model_file: str) -> CheckpointState:
    from torch.serialization import default_restore_location
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    new_stae_dict = {}
    for key, value in state_dict['model']['query_model']['language_model'].items():
        new_stae_dict['question_model.' + key] = value
    for key, value in state_dict['model']['context_model']['language_model'].items():
        new_stae_dict['ctx_model.' + key] = value
    return new_stae_dict

def load_states_from_checkpoint(model_file: str):
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)

def get_optimizer(args, model: nn.Module, weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.optimizer == "adamW":
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        raise Exception("optimizer {0} not recognized! Can only be lamb or adamW".format(args.optimizer))

def get_bert_reranker_components(args, **kwargs):
    encoder = HFBertEncoder.init_encoder_from_my_model(
        args,
    )
    hidden_size = encoder.config.hidden_size
    reranker = Reranker(encoder, hidden_size)
    return reranker

def load_model(args):
    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # Model definition
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=True if args.tokenizer_name == 'bert-base-uncased' else False)

    if (args.model_type == 'cross_encoder'):
        model = get_bert_reranker_components(args)
    elif (args.model_type == 'dual_encoder' or args.model_type == 'colbert' or args.model_type == 'distilbert'):
        model = BiBertEncoder(args)
    # print(model)
    if args.model_name_or_path_ict is not None:
        saved_state = load_states_from_checkpoint_ict(args.model_name_or_path_ict)
        model.load_state_dict(saved_state, strict=False)
    elif args.load_continue_train_path is not None:
        saved_state = load_states_from_checkpoint(args.load_continue_train_path)
        model.load_state_dict(saved_state.model_dict, strict=False)
    elif args.eval_model_dir is not None:
        saved_state = load_states_from_checkpoint(args.eval_model_dir)
        model.load_state_dict(saved_state.model_dict, strict=False)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    return tokenizer, model

def loads_data(buf):
    return pickle.loads(buf)

def set_env(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()
    else:
        args.world_size = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

def get_arguments():
    parser = argparse.ArgumentParser()
    ## Important hyper parameters
    parser.add_argument("--unique_identifier", default='', type=str, required=False, help="Model identifier")
    parser.add_argument("--dataset", default='', type=str, required=False, help="Model identifier")
    parser.add_argument("--max_doc_length", default=256, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_query_length", default=32, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--disitll_layer_num", default=5, type=int, help="distill layer number")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_hard_negatives", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--warm_up_ratio", default=0.1, type=float, help="The warm_up_ratio for Adam.")
    parser.add_argument("--refresh_index", default=1, type=int, help="the refresh index of the refresh")

    ## Distillation parameters
    parser.add_argument("--layer_selection_random", action="store_true")
    parser.add_argument("--layer_selection_last", action="store_true")
    parser.add_argument("--layer_selection_skip", action="store_true")
    parser.add_argument("--layer_score_reweight", action="store_true")
    parser.add_argument("--add_linear", action="store_true")
    parser.add_argument("--temperature", default=1, type=float, help="The temperature of distillation")
    parser.add_argument("--layer_temperature", default=1, type=float, help="The temperature of layer distillation")
    parser.add_argument("--distill_de_path", default=None, type=str)
    parser.add_argument("--distill_db_path", default=None, type=str)
    parser.add_argument("--distill_col_path", default=None, type=str)
    parser.add_argument("--distill_ce_path", default=None, type=str)

    ## Distill models
    parser.add_argument("--distill_db", action="store_true")
    parser.add_argument("--train_db", action="store_true")
    parser.add_argument("--distill_ce", action="store_true")
    parser.add_argument("--train_ce", action="store_true")
    parser.add_argument("--distill_col", action="store_true")
    parser.add_argument("--train_col", action="store_true")
    parser.add_argument("--distill_de", action="store_true")
    parser.add_argument("--train_de", action="store_true")
    parser.add_argument("--distill_attention", action="store_true")

    ## GT loss
    parser.add_argument("--distill_para_de", default=0, type=float, help="The distillation parameter of loss_de")
    parser.add_argument("--distill_para_db", default=0, type=float, help="The distillation parameter of loss_db")
    parser.add_argument("--distill_para_col", default=0, type=float, help="The distillation parameter of loss_col")
    parser.add_argument("--distill_para_ce", default=0, type=float, help="The distillation parameter of loss_ce")
    ## Response_based loss
    parser.add_argument("--distill_para_col_db_dis", default=0, type=float, help="The distillation parameter of loss_col_db_dis")
    parser.add_argument("--distill_para_de_db_dis", default=0, type=float, help="The distillation parameter of loss_de_db_dis")
    parser.add_argument("--distill_para_ce_db_dis", default=0, type=float, help="The distillation parameter of loss_ce_db_dis")
    ## Layer score loss
    parser.add_argument("--distill_de_db_layer_score", action="store_true")
    parser.add_argument("--distill_col_db_layer_score", action="store_true")
    parser.add_argument("--distill_ce_db_layer_score", action="store_true")
    parser.add_argument("--distill_para_ce_db_layer_score", default=0, type=float, help="The distillation parameter of loss_col_de_layer_dis")
    parser.add_argument("--distill_para_col_db_layer_score", default=0, type=float, help="The distillation parameter of loss_col_db_layer_dis")
    parser.add_argument("--distill_para_de_db_layer_score", default=0, type=float, help="The distillation parameter of loss_de_db_layer_dis")
    ## VIRT loss
    parser.add_argument("--distill_ce_db_attention", action="store_true")
    parser.add_argument("--distill_de_db_attention", action="store_true")
    parser.add_argument("--distill_col_db_attention", action="store_true")
    parser.add_argument("--distill_para_col_db_attention", default=0, type=float, help="The distillation parameter of loss_col_db_dis")
    parser.add_argument("--distill_para_ce_db_attention", default=0, type=float, help="The distillation parameter of loss_ce_db_dis")
    parser.add_argument("--distill_para_de_db_attention", default=0, type=float, help="The distillation parameter of loss_de_db_dis")

    ##  TREC year
    parser.add_argument("--year", default=2019, type=int, help="trec year information")

    # Other parameters
    # Required parameters
    parser.add_argument("--load_cache", action="store_true")
    parser.add_argument("--model_type", default=None, type=str, choices=['cross_encoder', 'colbert', 'dual_encoder', 'distilbert'], required=True, help="choose model type")
    parser.add_argument("--pretrained_model_name", default=None, type=str, required=True, help="Model type selected in the list:")
    parser.add_argument("--passage_path", default=None, type=str)
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--train_q2d_file", default=None, type=str)
    parser.add_argument("--train_docpool_file", default=None, type=str)
    parser.add_argument("--negative_q2d_file", default=None, type=str)
    parser.add_argument("--negative_docpool_file", default=None, type=str)
    parser.add_argument("--test_q2d_file", default=None, type=str, required=False, help="test_q2d_file")
    parser.add_argument("--test_docpool_file", default=None, type=str, required=False, help="Passage doc pool")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_model_dir", default=None, type=str, required=False, help="Initial model dir, will use this if no checkpoint is found in model_dir")
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int, help="The starting output file number")
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--search_result_path", default=None, type=str, required=False)
    parser.add_argument("--triplet", default=False, action="store_true", help="Whether to run training.")
    parser.add_argument("--log_dir", default=None, type=str, help="Tensorboard log dir")
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--optimizer", default="adamW", type=str, help="Optimizer - lamb or adamW")
    parser.add_argument("--save_hard_negatives_path", default=None, type=str)

    #  continue train
    parser.add_argument("--load_continue_train_path", default=None, type=str)
    parser.add_argument("--model_name_or_path_ict", default=None, type=str)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps",  default=300000, type=int, help="If > 0: set total number of training steps to perform")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--load_continue_train", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--gradient_checkpointing", default=False, action="store_true")
    parser.add_argument("--reset_global_step", default=False, action="store_true")
    parser.add_argument("--thread_num", type=int, default=90)
    parser.add_argument("--share_weight", action="store_true", help="Whether to share weight between query encoder and context encoder")
    parser.add_argument("--shuffle_positives", default=False, action="store_true", help="use single or re-warmup")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    args = parser.parse_args()

    return args