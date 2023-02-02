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
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig

class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.init_weights()
        self.version = int(transformers.__version__.split('.')[0])

    @classmethod
    def init_encoder(cls, args, role=None, dropout: float = 0.1, model_type=None, number_layers=0):
        if model_type is None:
            if role == 'student':
                model_type = args.model_type
            elif role == 'teacher':
                model_type = args.teacher_model_type
            elif role == 'double_teacher':
                model_type = args.double_teacher_pretrain
            else:
                print("no such type role")
                exit(0)

        # change to your own path or load from huggingface
        if model_type == "bert-base-uncased":
            config_path = "/colab_space/fanshuai/KDexp/pretrain_model/" + "bert-base-uncased" + "/config.json"
        elif model_type == "nghuyong/ernie-2.0-base-en":
            config_path = "/colab_space/fanshuai/KDexp/pretrain_model/" + "ernie_base" + "/config.json"
        elif model_type == "nghuyong/ernie-2.0-large-en":
            config_path = "/colab_space/fanshuai/KDexp/pretrain_model/" + "ernie_large" + "/config.json"
        else:
            print("no such type model")
            exit(0)
        cfg = BertConfig.from_json_file(config_path)
        # cfg = BertConfig.from_pretrained(model_type)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        if int(transformers.__version__.split('.')[0]) >= 3:
            cfg.gradient_checkpointing = args.gradient_checkpointing

        # 指定模型的层数
        if role is None:
            cfg.num_hidden_layers = number_layers
        else:
            if role == 'student':
                cfg.num_hidden_layers = args.student_num_hidden_layers
            elif role == 'teacher':
                cfg.num_hidden_layers = args.teacher_num_hidden_layers
            elif role == 'double_teacher':
                cfg.num_hidden_layers = args.double_teacher_num_hidden_layers
            else:
                print("no such type role")
                exit(0)

        # change to your own path or load from huggingface
        if model_type == "bert-base-uncased":
            model_path = "/colab_space/fanshuai/KDexp/pretrain_model/" + "bert-base-uncased"
        elif model_type == "nghuyong/ernie-2.0-base-en":
            model_path = "/colab_space/fanshuai/KDexp/pretrain_model/" + "ernie_base"
        elif model_type == "nghuyong/ernie-2.0-large-en":
            model_path = "/colab_space/fanshuai/KDexp/pretrain_model/" + "ernie_large"
        else:
            print("no such type model")
            exit(0)
        return cls.from_pretrained(model_path, config=cfg)

    def forward(self, **kwargs):
        hidden_states = None
        result = super().forward(**kwargs)
        sequence_output = result.last_hidden_state + 0 * result.pooler_output.sum()
        pooled_output = sequence_output[:, 0, :]
        return sequence_output, pooled_output, hidden_states

class HFDistilBertEncoder(DistilBertModel):
    def __init__(self, config):
        DistilBertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.init_weights()
        self.version = int(transformers.__version__.split('.')[0])

    @classmethod
    def init_encoder(cls, args, role, dropout: float = 0.1, model_type=None):
        if model_type is None:
            if role == 'student':
                model_type = args.model_type
            elif role == 'teacher':
                model_type = args.teacher_model_type
            elif role == 'double_teacher':
                model_type = args.double_teacher_pretrain
            else:
                print("no such type role")
                exit(0)

        if model_type == "distilbert-base-uncased":
            config_path = "/colab_space/fanshuai/KDexp/pretrain_model/" + "distilbert-base-uncased" + "/config.json"
        else:
            print("no such type model")
            exit(0)
        cfg = DistilBertConfig.from_json_file(config_path)
        # cfg = BertConfig.from_pretrained(model_type)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        if int(transformers.__version__.split('.')[0]) >= 3:
            cfg.gradient_checkpointing = args.gradient_checkpointing

        # if role == 'student':
        #     cfg.num_hidden_layers = args.student_num_hidden_layers
        # elif role == 'teacher':
        #     cfg.num_hidden_layers = args.teacher_num_hidden_layers
        # elif role == 'double_teacher':
        #     cfg.num_hidden_layers = args.double_teacher_num_hidden_layers
        # else:
        #     print("no such type role")
        #     exit(0)

        if model_type == "distilbert-base-uncased":
            model_path = "/colab_space/fanshuai/KDexp/pretrain_model/" + "distilbert-base-uncased"
        else:
            print("no such type model")
            exit(0)
        return cls.from_pretrained(model_path, config=cfg)

    def forward(self, **kwargs):
        hidden_states = None
        result = super().forward(**kwargs)
        sequence_output = result.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        return sequence_output, pooled_output, hidden_states

class ColBERT(nn.Module):
    def __init__(self, args, role, similarity_metric='cosine'):

        super(ColBERT, self).__init__()
        self.role = role
        if args.similarity_metric != None:
            self.similarity_metric = args.similarity_metric
        else:
            self.similarity_metric = similarity_metric

        self.question_model = HFBertEncoder.init_encoder(args, self.role)
        if hasattr(args, 'share_weight') and args.share_weight:
            self.ctx_model = self.question_model
        else:
            self.ctx_model = HFBertEncoder.init_encoder(args, self.role)

        self.dim = 128
        self.max_len = args.max_seq_length
        cfg = BertConfig.from_pretrained(args.model_type)
        self.q_linear = nn.Linear(cfg.hidden_size, self.dim, bias=False)
        self.ctx_linear = nn.Linear(cfg.hidden_size, self.dim, bias=False)

    def forward(self, query_ids, attention_mask_q, input_ids_a=None, attention_mask_a=None, input_ids_b=None,
                attention_mask_b=None):
        q_embs, q_hidden = self.query_emb(query_ids, attention_mask_q)
        a_embs, a_hidden = self.body_emb(input_ids_a, attention_mask_a)
        return (q_embs, a_embs, q_hidden, a_hidden)

    def query_emb(self, input_ids, attention_mask):
        Q_output, pooled_output, _ = self.question_model(input_ids=input_ids, attention_mask=attention_mask)
        Q_output = self.q_linear(Q_output)

        return pooled_output, torch.nn.functional.normalize(Q_output, p=2, dim=2)

    def body_emb(self, input_ids, attention_mask):

        D_output, pooled_output, _ = self.ctx_model(input_ids=input_ids, attention_mask=attention_mask)
        D_output = self.ctx_linear(D_output)

        D_output = D_output * attention_mask.unsqueeze(-1)

        return pooled_output, torch.nn.functional.normalize(D_output, p=2, dim=2)

    def score(self, query_ids, attention_mask_q, input_ids_a, attention_mask_a):
        _, Q_hidden = self.query_emb(query_ids, attention_mask_q)
        _, D_hidden = self.body_emb(input_ids_a, attention_mask_a)
        Doc_num = D_hidden.shape[0]
        Q_num = Q_hidden.shape[0]
        D_hidden = D_hidden.view(-1,self.dim)

        if self.similarity_metric == 'cosine':
            return ((Q_hidden @ D_hidden.permute(1,0)).view(Q_num,self.max_len,Doc_num,self.max_len).permute(0, 2, 1, 3)).max(3).values.sum(2)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q_hidden.unsqueeze(2) - D_hidden.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def init_weights(modules):
        for module in modules:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()


class BiBertEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """

    def __init__(self, args, role=None):
        super(BiBertEncoder, self).__init__()
        if role!=None:
            self.role = role
            if role=='student' and args.model_type=='distilbert-base-uncased':
                self.question_model = HFDistilBertEncoder.init_encoder(args, self.role)
                if hasattr(args, 'share_weight') and args.share_weight:
                    self.ctx_model = self.question_model
                else:
                    self.ctx_model = HFDistilBertEncoder.init_encoder(args, self.role)
            else:
                self.question_model = HFBertEncoder.init_encoder(args, self.role)
                if hasattr(args, 'share_weight') and args.share_weight:
                    self.ctx_model = self.question_model
                else:
                    self.ctx_model = HFBertEncoder.init_encoder(args, self.role)
        else:
            self.question_model = HFBertEncoder.init_encoder(args, model_type='nghuyong/ernie-2.0-base-en', number_layers=12)
            if hasattr(args, 'share_weight') and args.share_weight:
                self.ctx_model = self.question_model
            else:
                self.ctx_model = HFBertEncoder.init_encoder(args, model_type='nghuyong/ernie-2.0-base-en', number_layers=12)

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


class ColBERTKDLoss(object):
    def calc(
            self,
            args,
            q_vectors,
            ctx_vectors,
            teacher_q_hidden,
            teacher_ctx_hidden,
            teacher_ctx_mask,
            positive_idx_per_question: list,
            hard_negative_idx_per_question: list = None,
            loss_scale: float = None,
    ):

        scores = dot_product_scores(q_vectors, ctx_vectors)

        # Doc_num = teacher_ctx_hidden.shape[0]
        # q_max_len = teacher_q_hidden.shape[1]
        # ctx_max_len = teacher_ctx_hidden.shape[1]
        # dim = teacher_ctx_hidden.shape[2]
        # Q_num = teacher_q_hidden.shape[0]
        # teacher_ctx_hidden = teacher_ctx_hidden.view(-1, dim)
        #
        # if args.similarity_metric == 'cosine':
        #     teacher_scores = \
        #         ((teacher_q_hidden @ teacher_ctx_hidden.permute(1, 0)).view(
        #             Q_num, q_max_len, Doc_num, ctx_max_len).permute(0, 2, 1, 3)).max(3).values.sum(2)
        # else:
        #     print("error similarity_metric")
        #     exit(0)

        Q_num = teacher_q_hidden.shape[0]
        q_max_len = teacher_q_hidden.shape[1]
        teacher_scores = torch.einsum('qin,pjn->qipj', teacher_q_hidden, teacher_ctx_hidden)
        mask_index = (~teacher_ctx_mask.bool()).unsqueeze(0).unsqueeze(0).repeat(Q_num, q_max_len, 1, 1)
        teacher_scores[mask_index] = -9999
        teacher_scores, _ = teacher_scores.max(-1)
        teacher_scores = teacher_scores.sum(1)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        hard_loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )
        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
                max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()

        soft_loss = 0

        if teacher_q_hidden != None and teacher_ctx_hidden != None:
            if args.KD_type == "KD_softmax":
                soft_loss = self.kd_loss(scores, teacher_scores, args.TEMPERATURE, args.KD_type)
            elif args.KD_type == "KD_logit":
                soft_loss = self.kd_loss(scores, teacher_scores, args.TEMPERATURE, args.KD_type)
            elif args.KD_type == "DKD":
                if args.DKD_alpha != None and args.DKD_beta != None:
                    soft_loss = self.dkd_loss(
                        scores,
                        teacher_scores,
                        torch.tensor(positive_idx_per_question).to(max_idxs.device),
                        args.DKD_alpha,
                        args.DKD_beta,
                        args.TEMPERATURE,
                    )
                else:
                    print("DKD loss need refer DKD_alpha and DKD_beta in args")
                    exit(0)
            elif args.KD_type == "prob_loss":
                eps = 1e-7
                teacher_scores_p = F.softmax(teacher_scores/args.TEMPERATURE, dim=1)
                scores_p = F.softmax(scores, dim=1)
                soft_loss = -teacher_scores_p * torch.log(scores_p + eps)
                soft_loss = soft_loss.sum() / scores_p.size(0)
            else:
                print("no such type of KD loss,please check KD_type")
                exit(0)

        if teacher_q_hidden != None and teacher_ctx_hidden != None:
            loss = args.CE_WEIGHT * hard_loss + args.KD_WEIGHT * soft_loss
        else:
            loss = hard_loss

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

    @staticmethod
    def kd_loss(logits_student, logits_teacher, temperature, KD_type):
        if KD_type == 'KD_softmax':
            log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
            pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
            loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
            loss_kd *= temperature ** 2
        elif KD_type == 'KD_logit':
            KD_loss_fn = torch.nn.MSELoss(reduction='mean')
            loss_kd = 0.5 * KD_loss_fn(logits_student, logits_teacher)
        return loss_kd

    @staticmethod
    def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
        gt_mask = BiEncoderKDLoss._get_gt_mask(logits_student, target)
        other_mask = BiEncoderKDLoss._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = BiEncoderKDLoss.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = BiEncoderKDLoss.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )
        return alpha * tckd_loss + beta * nckd_loss

    @staticmethod
    def _get_gt_mask(logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    @staticmethod
    def _get_other_mask(logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    @staticmethod
    def cat_mask(t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

class UniKDLoss(object):
    def calc(
            self,
            args,
            q_vectors,
            ctx_vectors,
            teacher_q_vector,
            teacher_ctxs_vector,
            ce12_relevance_logits,
            ce24_relevance_logits,
            hard_negative_idx_per_question: list = None,
            loss_scale: float = None,
    ):

        #scores = dot_product_scores(q_vectors, ctx_vectors)
        retriever_ctx_vectors = ctx_vectors.reshape(q_vectors.size(0),
                                                                ctx_vectors.size(0) // q_vectors.size(0), -1)
        scores = torch.einsum("bh,bdh->bd", q_vectors, retriever_ctx_vectors)

        teacher_retriever_ctx_vectors = teacher_ctxs_vector.reshape(teacher_q_vector.size(0),
                                                    teacher_ctxs_vector.size(0) // teacher_q_vector.size(0), -1)
        de_scores = torch.einsum("bh,bdh->bd", teacher_q_vector, teacher_retriever_ctx_vectors)

        ce12_scores = ce12_relevance_logits
        ce24_scores = ce24_relevance_logits

        positive_idx_per_question = torch.zeros(q_vectors.size(0), dtype=torch.long).to(scores.device)
        #print("scores",scores.shape)
        #print("teacher_scores",teacher_scores.shape)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        # print("dual-encoder score shape:", softmax_scores.shape)
        # print("dual-encoder score", F.softmax(scores/4,dim=1))
        # print("cross encoder score shape:", teacher_scores.shape)
        # print("cross encoder score", F.softmax(teacher_scores/4, dim=1))

        hard_loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )
        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
                max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()

        de_loss = F.nll_loss(
            F.softmax(de_scores, dim=1),
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )
        ce12_loss = F.nll_loss(
            F.softmax(ce12_scores, dim=1),
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )
        ce24_loss = F.nll_loss(
            F.softmax(ce24_scores, dim=1),
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        de_loss_pct = de_loss / (de_loss + ce12_loss + ce24_loss)
        ce12_loss_pct = ce12_loss / (de_loss + ce12_loss + ce24_loss)
        ce24_loss_pct = ce24_loss / (de_loss + ce12_loss + ce24_loss)
        teacher_scores = de_loss_pct * de_scores + ce12_loss_pct * ce12_scores + ce24_loss_pct * ce24_scores

        soft_loss = 0
        if ce24_loss != None:
            if args.KD_type == "KD_softmax":
                soft_loss = self.kd_loss(scores, teacher_scores, args.TEMPERATURE, args.KD_type)

        if ce24_loss != None:
            loss = hard_loss + (1/(0.1+(de_loss + ce12_loss + ce24_loss)/3)) * soft_loss
        else:
            loss = hard_loss

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

    @staticmethod
    def kd_loss(logits_student, logits_teacher, temperature, KD_type):
        if KD_type == 'KD_softmax':
            log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
            pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
            loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
            loss_kd *= temperature ** 2
        return loss_kd


class MultiTeacherLoss(object):
    def calc(
            self,
            args,
            q_vectors,
            ctx_vectors,
            teacher_q_vector,
            teacher_ctxs_vector,
            ce12_relevance_logits,
            ce24_relevance_logits,
            hard_negative_idx_per_question: list = None,
            loss_scale: float = None,
    ):

        #scores = dot_product_scores(q_vectors, ctx_vectors)
        retriever_ctx_vectors = ctx_vectors.reshape(q_vectors.size(0),
                                                                ctx_vectors.size(0) // q_vectors.size(0), -1)
        scores = torch.einsum("bh,bdh->bd", q_vectors, retriever_ctx_vectors)

        teacher_retriever_ctx_vectors = teacher_ctxs_vector.reshape(teacher_q_vector.size(0),
                                                    teacher_ctxs_vector.size(0) // teacher_q_vector.size(0), -1)
        de_scores = torch.einsum("bh,bdh->bd", teacher_q_vector, teacher_retriever_ctx_vectors)

        ce12_scores = ce12_relevance_logits
        ce24_scores = ce24_relevance_logits

        positive_idx_per_question = torch.zeros(q_vectors.size(0), dtype=torch.long).to(scores.device)
        #print("scores",scores.shape)
        #print("teacher_scores",teacher_scores.shape)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        # print("dual-encoder score shape:", softmax_scores.shape)
        # print("dual-encoder score", F.softmax(scores/4,dim=1))
        # print("cross encoder score shape:", teacher_scores.shape)
        # print("cross encoder score", F.softmax(teacher_scores/4, dim=1))

        hard_loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )
        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
                max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()

        soft_loss = 0

        if ce24_scores != None:
            if args.KD_type == "KD_softmax":
                soft_loss = self.kd_loss(scores, de_scores, ce12_scores, ce24_scores, args.TEMPERATURE, args.KD_type)

        if ce24_scores != None:
            loss = args.CE_WEIGHT * hard_loss + args.KD_WEIGHT * soft_loss
        else:
            loss = hard_loss

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

    @staticmethod
    def kd_loss(logits_student, logits_de, logits_ce12, logits_ce24, temperature, KD_type):
        if KD_type == 'KD_softmax':
            log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
            pred_de = F.softmax(logits_de / temperature, dim=1)
            pred_ce12 = F.softmax(logits_ce12 / temperature, dim=1)
            pred_ce24 = F.softmax(logits_ce24 / temperature, dim=1)
            pred_teacher = (pred_de + pred_ce12 + pred_ce24) / 3
            loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
            loss_kd *= temperature ** 2
        return loss_kd


class CrossBERTKDLoss(object):
    def calc(
            self,
            args,
            q_vectors,
            ctx_vectors,
            relevance_logits,
            hard_negative_idx_per_question: list = None,
            loss_scale: float = None,
            LwF=False,
            ori_q_vector=None,
            ori_ctx_vectors=None,
    ):

        #scores = dot_product_scores(q_vectors, ctx_vectors)
        retriever_ctx_vectors = ctx_vectors.reshape(q_vectors.size(0),
                                                                ctx_vectors.size(0) // q_vectors.size(0), -1)
        scores = torch.einsum("bh,bdh->bd", q_vectors, retriever_ctx_vectors)

        if LwF:
            ori_ctx_vectors = ori_ctx_vectors.reshape(ori_q_vector.size(0),
                                                        ori_ctx_vectors.size(0) // ori_q_vector.size(0), -1)
            ori_scores = torch.einsum("bh,bdh->bd", ori_q_vector, ori_ctx_vectors)

        teacher_scores = relevance_logits

        positive_idx_per_question = torch.zeros(q_vectors.size(0), dtype=torch.long).to(scores.device)
        #print("scores",scores.shape)
        #print("teacher_scores",teacher_scores.shape)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        # print("dual-encoder score shape:", softmax_scores.shape)
        # print("dual-encoder score", F.softmax(scores/4,dim=1))
        # print("cross encoder score shape:", teacher_scores.shape)
        # print("cross encoder score", F.softmax(teacher_scores/4, dim=1))

        hard_loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )
        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
                max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()

        soft_loss = 0
        if relevance_logits != None:
            if args.KD_type == "KD_softmax":
                soft_loss = self.kd_loss(scores, teacher_scores, args.TEMPERATURE, args.KD_type)
            elif args.KD_type == "KD_logit":
                soft_loss = self.kd_loss(scores, teacher_scores, args.TEMPERATURE, args.KD_type)
            elif args.KD_type == "DKD":
                if args.DKD_alpha != None and args.DKD_beta != None:
                    soft_loss = self.dkd_loss(
                        scores,
                        teacher_scores,
                        torch.tensor(positive_idx_per_question).to(max_idxs.device),
                        args.DKD_alpha,
                        args.DKD_beta,
                        args.TEMPERATURE,
                    )
                else:
                    print("DKD loss need refer DKD_alpha and DKD_beta in args")
                    exit(0)
            elif args.KD_type == "prob_loss":
                eps = 1e-7
                teacher_scores_p = F.softmax(teacher_scores/args.TEMPERATURE, dim=1)
                scores_p = F.softmax(scores, dim=1)
                soft_loss = -teacher_scores_p * torch.log(scores_p + eps)
                soft_loss = soft_loss.sum() / scores_p.size(0)
            else:
                print("no such type of KD loss,please check KD_type")
                exit(0)

        if LwF:
            LwF_loss = self.kd_loss(scores, ori_scores, args.TEMPERATURE, args.KD_type)
            loss = args.CE_WEIGHT * hard_loss + args.KD_WEIGHT * soft_loss + args.LwF_WEIGHT * LwF_loss
        else:
            if relevance_logits != None:
                loss = args.CE_WEIGHT * hard_loss + args.KD_WEIGHT * soft_loss
            else:
                loss = hard_loss

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

    @staticmethod
    def kd_loss(logits_student, logits_teacher, temperature, KD_type):
        if KD_type == 'KD_softmax':
            log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
            pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
            loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
            loss_kd *= temperature ** 2
        elif KD_type == 'KD_logit':
            KD_loss_fn = torch.nn.MSELoss(reduction='mean')
            loss_kd = 0.5 * KD_loss_fn(logits_student, logits_teacher)
        return loss_kd

    @staticmethod
    def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
        gt_mask = BiEncoderKDLoss._get_gt_mask(logits_student, target)
        other_mask = BiEncoderKDLoss._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = BiEncoderKDLoss.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = BiEncoderKDLoss.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )
        return alpha * tckd_loss + beta * nckd_loss

    @staticmethod
    def _get_gt_mask(logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    @staticmethod
    def _get_other_mask(logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    @staticmethod
    def cat_mask(t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt


class Cross2CrossKDLoss(object):
    def calc(
        self,
        args,
        teacher_relevance_logits,
        teacher_binary_logits,
        relevance_logits,
        binary_logits,
        hard_negative_idx_per_question: list = None,
        loss_scale: float = None,
    ):

        scores = relevance_logits
        teacher_scores = teacher_relevance_logits

        positive_idx_per_question = torch.zeros(relevance_logits.size(0), dtype=torch.long).to(scores.device)
        # print("pos index:", positive_idx_per_question)
        # print("pos index shape:", positive_idx_per_question.shape)
        loss_fct = torch.nn.CrossEntropyLoss()
        hard_loss = loss_fct(relevance_logits, positive_idx_per_question)
        # print("hard_loss:", hard_loss)
        # print("hard_loss shape:", hard_loss)

        binary_logits = binary_logits.view(-1, 2)
        classfi_target = torch.ones(binary_logits.size(0), dtype=torch.long).to(args.device)
        classfi_target[::2] = 0
        classfi_loss = loss_fct(binary_logits, classfi_target)

        teacher_binary_logits = teacher_binary_logits.view(-1, 2)
        teacher_classfi_loss = loss_fct(teacher_binary_logits, classfi_target)

        softmax_scores = F.log_softmax(scores, dim=1)
        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
                max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()

        soft_loss = 0
        if teacher_relevance_logits != None:
            if args.KD_type == "KD_softmax":
                soft_loss = self.kd_loss(scores, teacher_scores, args.TEMPERATURE, args.KD_type)
            elif args.KD_type == "KD_logit":
                soft_loss = self.kd_loss(scores, teacher_scores, args.TEMPERATURE, args.KD_type)
            elif args.KD_type == "DKD":
                if args.DKD_alpha != None and args.DKD_beta != None:
                    soft_loss = self.dkd_loss(
                        scores,
                        teacher_scores,
                        torch.tensor(positive_idx_per_question).to(max_idxs.device),
                        args.DKD_alpha,
                        args.DKD_beta,
                        args.TEMPERATURE,
                    )
                else:
                    print("DKD loss need refer DKD_alpha and DKD_beta in args")
                    exit(0)
            else:
                print("no such type of KD loss,please check KD_type")
                exit(0)

        if teacher_relevance_logits != None:
            print("")
            loss = args.CE_WEIGHT * hard_loss + args.KD_WEIGHT * soft_loss + 0 * classfi_loss + 0*teacher_classfi_loss
        else:
            loss = hard_loss + 0 * classfi_loss+0*teacher_classfi_loss

        # loss = hard_loss + 0 * classfi_loss

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

    @staticmethod
    def kd_loss(logits_student, logits_teacher, temperature, KD_type):
        if KD_type == 'KD_softmax':
            log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
            pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
            loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
            loss_kd *= temperature ** 2
        elif KD_type == 'KD_logit':
            KD_loss_fn = torch.nn.MSELoss(reduction='mean')
            loss_kd = 0.5 * KD_loss_fn(logits_student, logits_teacher)
        return loss_kd

    @staticmethod
    def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
        gt_mask = BiEncoderKDLoss._get_gt_mask(logits_student, target)
        other_mask = BiEncoderKDLoss._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = BiEncoderKDLoss.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = BiEncoderKDLoss.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )
        return alpha * tckd_loss + beta * nckd_loss

    @staticmethod
    def _get_gt_mask(logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    @staticmethod
    def _get_other_mask(logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    @staticmethod
    def cat_mask(t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

class BiEncoderKDLoss(object):
    def calc(
            self,
            args,
            q_vectors,
            ctx_vectors,
            teacher_q_vector,
            teacher_ctxs_vector,
            positive_idx_per_question: list,
            hard_negative_idx_per_question: list = None,
            loss_scale: float = None,
    ):

        scores = dot_product_scores(q_vectors, ctx_vectors)
        teacher_scores = dot_product_scores(teacher_q_vector, teacher_ctxs_vector)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        hard_loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )
        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (
                max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)
        ).sum()

        soft_loss = 0
        '''
        KD loss
        '''
        if teacher_q_vector != None and teacher_ctxs_vector != None:
            if args.KD_type == "KD_softmax":
                soft_loss = self.kd_loss(scores, teacher_scores, args.TEMPERATURE, args.KD_type)
            elif args.KD_type == "KD_logit":
                soft_loss = self.kd_loss(scores, teacher_scores, args.TEMPERATURE, args.KD_type)
            elif args.KD_type == "DKD":
                if args.DKD_alpha != None and args.DKD_beta != None:
                    soft_loss = self.dkd_loss(
                        scores,
                        teacher_scores,
                        torch.tensor(positive_idx_per_question).to(max_idxs.device),
                        args.DKD_alpha,
                        args.DKD_beta,
                        args.TEMPERATURE,
                    )
                else:
                    print("DKD loss need refer DKD_alpha and DKD_beta in args")
                    exit(0)
            elif args.KD_type == "Bi_logit":
                soft_loss = self.bi_logit(q_vectors, ctx_vectors, teacher_q_vector, teacher_ctxs_vector)
            else:
                print("no such type of KD loss,please check KD_type")
                exit(0)

        if teacher_q_vector != None and teacher_ctxs_vector != None:
            loss = args.CE_WEIGHT * hard_loss + args.KD_WEIGHT * soft_loss
        else:
            loss = hard_loss

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

    @staticmethod
    def kd_loss(logits_student, logits_teacher, temperature, KD_type):
        if KD_type == 'KD_softmax':
            log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
            pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
            loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
            loss_kd *= temperature ** 2
        elif KD_type == 'KD_logit':
            KD_loss_fn = torch.nn.MSELoss(reduction='mean')
            loss_kd = 0.5 * KD_loss_fn(logits_student, logits_teacher)
        return loss_kd

    @staticmethod
    def bi_logit(student_q, student_ctxs, teacher_q, teacher_ctxs):
        KD_loss_fn = torch.nn.MSELoss(reduction='mean')
        q_KD_loss = 0.5 * KD_loss_fn(student_q, teacher_q)
        ctx_KD_loss = 0.5 * KD_loss_fn(student_ctxs, teacher_ctxs)
        loss_bi_logit = q_KD_loss + ctx_KD_loss
        return loss_bi_logit

    @staticmethod
    def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
        gt_mask = BiEncoderKDLoss._get_gt_mask(logits_student, target)
        other_mask = BiEncoderKDLoss._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = BiEncoderKDLoss.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = BiEncoderKDLoss.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
                * (temperature ** 2)
                / target.shape[0]
        )
        return alpha * tckd_loss + beta * nckd_loss

    @staticmethod
    def _get_gt_mask(logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    @staticmethod
    def _get_other_mask(logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    @staticmethod
    def cat_mask(t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

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

class ColBERTNllLoss(object):
    def calc(
            self,
            args,
            q_hidden,
            ctx_hidden,
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

        Doc_num = ctx_hidden.shape[0]
        q_max_len = q_hidden.shape[1]
        ctx_max_len = ctx_hidden.shape[1]
        dim = ctx_hidden.shape[2]
        Q_num = q_hidden.shape[0]
        ctx_hidden = ctx_hidden.view(-1, dim)
        if args.similarity_metric == 'cosine':
            scores = \
                ((q_hidden @ ctx_hidden.permute(1, 0)).view(Q_num, q_max_len, Doc_num, ctx_max_len).permute(0, 2, 1, 3)).max(3).values.sum(2)
        else:
            print("error similarity_metric")
            exit(0)

        if len(q_hidden.size()) > 1:
            q_num = q_hidden.size(0)
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
