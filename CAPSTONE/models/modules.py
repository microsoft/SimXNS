import transformers
from transformers import (
    BertModel,
    BertConfig,
)
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import CrossEntropyLoss
import sys
import os
pyfile_path = os.path.abspath(__file__) 
pyfile_dir = os.path.dirname(os.path.dirname(pyfile_path)) # equals to the path '../'
sys.path.append(pyfile_dir)

from utils.dpr_utils import all_gather_list


class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.init_weights()
        self.version = int(transformers.__version__.split('.')[0])

    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1, model_type=None):
        if model_type is None:
            model_type = args.model_type
        cfg = BertConfig.from_pretrained(model_type)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        if int(transformers.__version__.split('.')[0]) >= 3:
            cfg.gradient_checkpointing = args.gradient_checkpointing
        return cls.from_pretrained(model_type, config=cfg)

    def forward(self, **kwargs):
        hidden_states = None
        result = super().forward(**kwargs)
        sequence_output = result.last_hidden_state + 0 * result.pooler_output.sum()
        pooled_output = sequence_output[:, 0, :] # cls represetation 
        return sequence_output, pooled_output, hidden_states


class BiBertEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """

    def __init__(self, args, model_type=None):
        super(BiBertEncoder, self).__init__()
        self.question_model = HFBertEncoder.init_encoder(args, model_type=model_type)
        self.question_model.gradient_checkpointing_enable()
        if hasattr(args, 'share_weight') and args.share_weight:
            self.ctx_model = self.question_model
        else:
            self.ctx_model = HFBertEncoder.init_encoder(args, model_type=model_type)
        self.ctx_model.gradient_checkpointing_enable()
    def query_emb(self, input_ids, attention_mask):
        _, pooled_output, _ = self.question_model(input_ids=input_ids, attention_mask=attention_mask)
        return pooled_output

    def body_emb(self, input_ids, attention_mask):
        _, pooled_output, _ = self.ctx_model(input_ids=input_ids, attention_mask=attention_mask)
        return pooled_output

    def forward(self, query_ids, attention_mask_q, input_ids_a=None, attention_mask_a=None, input_ids_b=None,
                attention_mask_b=None):
        if input_ids_b is None:
            q_embs = self.query_emb(query_ids, attention_mask_q)
            a_embs = self.body_emb(input_ids_a, attention_mask_a)
            return (q_embs, a_embs)
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1), (q_embs * b_embs).sum(-1).unsqueeze(1)],
                                 dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)


def init_weights(modules):
    for module in modules:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BiEncoderNllLoss(object):
    def calc(
            self,
            q_vectors,
            ctx_vectors,
            positive_idx_per_question: list,
            hard_negative_idx_per_question: list = None,
            loss_scale: float = None,
    ):
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = dot_product_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
                max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector, ctx_vectors):
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores

def dot_product_scores(q_vectors, ctx_vectors):
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r

class Reranker(nn.Module):

    def __init__(self, args, model_type=None):
        super(Reranker, self).__init__()

        encoder = HFBertEncoder.init_encoder(
            args, model_type=model_type
        )
        hidden_size = encoder.config.hidden_size

        self.encoder = encoder

        self.binary = nn.Linear(hidden_size, 2)
        self.qa_classifier = nn.Linear(hidden_size, 1)
        init_weights([self.binary, self.qa_classifier])

    def forward(self, input_ids: T, attention_mask: T):
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        N, M, L = input_ids.size()
        binary_logits, relevance_logits = self._forward(input_ids.view(N * M, L),
                                                        attention_mask.view(N * M, L))
        return binary_logits.view(N, M, 2), relevance_logits.view(N, M)

    def _forward(self, input_ids, attention_mask):
        # TODO: provide segment values
        sequence_output, _, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        binary_logits = self.binary(sequence_output[:, 0, :])
        rank_logits = self.qa_classifier(sequence_output[:, 0, :])
        return binary_logits, rank_logits


def calculate_dual_encoder_cont_loss(local_rank, local_q_vector, local_ctx_vectors, local_positive_idxs):
    """
    calculate the contrastive loss for the dual-encoder retriever
    """
    if torch.distributed.get_world_size() > 1:
        q_vector_to_send = (
            torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
        )
        ctx_vector_to_send = (
            torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()
        )

        global_question_ctx_vectors = all_gather_list(
            [
                q_vector_to_send,
                ctx_vector_to_send,
                local_positive_idxs,
            ],
            max_size=640000000,
        )

        global_q_vector = []
        global_ctxs_vector = []

        # ctxs_per_question = local_ctx_vectors.size(0)
        positive_idx_per_question = []
        # hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, positive_idx = item

            if i != local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                positive_idx_per_question.extend(
                    [v + total_ctxs for v in local_positive_idxs]
                )
            total_ctxs += ctx_vectors.size(0)
        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)
    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        positive_idx_per_question = local_positive_idxs

    loss_function = BiEncoderNllLoss()
    loss, is_correct = loss_function.calc(
        global_q_vector,
        global_ctxs_vector,
        positive_idx_per_question,
    )
    return loss, is_correct
