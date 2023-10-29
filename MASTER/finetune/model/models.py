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
        pooled_output = sequence_output[:, 0, :]
        return sequence_output, pooled_output, hidden_states


class BiBertEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """

    def __init__(self, args):
        super(BiBertEncoder, self).__init__()
        self.question_model = HFBertEncoder.init_encoder(args)
        if hasattr(args, 'share_weight') and args.share_weight:
            self.ctx_model = self.question_model
        else:
            self.ctx_model = HFBertEncoder.init_encoder(args)

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

    def forward_adv_triplet(self, query_ids, attention_mask_q, input_ids_a=None,
                            attention_mask_a=None, input_ids_embed_b=None, attention_mask_b=None):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        _, b_embs, _ = self.ctx_model(inputs_embeds=input_ids_embed_b, attention_mask=attention_mask_b)
        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1), (q_embs * b_embs).sum(-1).unsqueeze(1)],
                                 dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)

    def forward_adv_pairloss(self, query_ids, attention_mask_q, input_ids_a=None,
                             attention_mask_a=None, input_ids_embed_b=None, attention_mask_b=None):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        _, b_embs, _ = self.ctx_model(inputs_embeds=input_ids_embed_b, attention_mask=attention_mask_b)

        question_num = q_embs.size(0)
        neg_local_ctx_vectors = b_embs.reshape(question_num, b_embs.size(0) // question_num, -1)

        neg_simila = torch.einsum("bh,bdh->bd", [q_embs, neg_local_ctx_vectors])
        pos_simil = (q_embs * a_embs).sum(-1).unsqueeze(1)
        logit_matrix = torch.cat([pos_simil, neg_simila], dim=1)  # [B, 17]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 1]
        return (loss.mean(), lsm)

    def forward_adv_pairloss_mse(self, query_ids, attention_mask_q, input_ids_a=None,
                                 attention_mask_a=None, input_ids_embed_b=None, attention_mask_b=None):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        _, b_embs, _ = self.ctx_model(inputs_embeds=input_ids_embed_b, attention_mask=attention_mask_b)

        question_num = q_embs.size(0)
        neg_local_ctx_vectors = b_embs.reshape(question_num, b_embs.size(0) // question_num, -1)

        neg_simila = torch.einsum("bh,bdh->bd", [q_embs, neg_local_ctx_vectors])
        pos_simil = (q_embs * a_embs).sum(-1).unsqueeze(1)
        mse = nn.MSELoss()
        loss = mse(neg_simila, pos_simil)
        return (loss.mean(), 0)

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

    def __init__(self, encoder: nn.Module, hidden_size):
        super(Reranker, self).__init__()
        self.encoder = encoder
        self.binary = nn.Linear(hidden_size, 2)
        self.qa_classifier = nn.Linear(hidden_size, 1)
        init_weights([self.binary, self.qa_classifier])

    def forward(self, input_ids: T, attention_mask: T):
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        N, M, L = input_ids.size()
        binary_logits, relevance_logits, _, = self._forward(input_ids.view(N * M, L),
                                                            attention_mask.view(N * M, L))

        return binary_logits.view(N, M, 2), relevance_logits.view(N, M), None

    def _forward(self, input_ids, attention_mask):
        # TODO: provide segment values
        sequence_output, _, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        binary_logits = self.binary(sequence_output[:, 0, :])
        rank_logits = self.qa_classifier(sequence_output[:, 0, :])
        return binary_logits, rank_logits, None


def compute_loss(start_positions, end_positions, answer_mask, start_logits, end_logits, relevance_logits, N, M):
    start_positions = start_positions.view(N * M, -1)
    end_positions = end_positions.view(N * M, -1)
    answer_mask = answer_mask.view(N * M, -1)

    start_logits = start_logits.view(N * M, -1)
    end_logits = end_logits.view(N * M, -1)
    relevance_logits = relevance_logits.view(N * M)

    answer_mask = answer_mask.type(torch.FloatTensor).cuda()

    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)
    loss_fct = CrossEntropyLoss(reduce=False, ignore_index=ignored_index)

    # compute switch loss
    relevance_logits = relevance_logits.view(N, M)
    switch_labels = torch.zeros(N, dtype=torch.long).cuda()
    switch_loss = torch.sum(loss_fct(relevance_logits, switch_labels))

    # compute span loss
    start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask)
                    for (_start_positions, _span_mask)
                    in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_mask, dim=1))]

    end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask)
                  for (_end_positions, _span_mask)
                  in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_mask, dim=1))]
    loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + \
                  torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)

    loss_tensor = loss_tensor.view(N, M, -1).max(dim=1)[0]
    span_loss = _calc_mml(loss_tensor)
    return span_loss + switch_loss


def _calc_mml(loss_tensor):
    marginal_likelihood = torch.sum(torch.exp(
        - loss_tensor - 1e10 * (loss_tensor == 0).float()), 1)
    return -torch.sum(torch.log(marginal_likelihood +
                                torch.ones(loss_tensor.size(0)).cuda() * (marginal_likelihood == 0).float()))
