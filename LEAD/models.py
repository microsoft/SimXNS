from transformers import (
    BertConfig,
    DistilBertConfig,
    BertTokenizerFast,
    __version__
)
from modeling_bert import BertModel
from modeling_distilbert import DistilBertModel
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T
import string
from transformers.activations import ACT2FN


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class HFDistilBertEncoder(DistilBertModel):
    def __init__(self, config):
        DistilBertModel.__init__(self, config)
        self.add_linear = config.add_linear
        if self.add_linear:
            self.linear = nn.Linear(config.hidden_size, 768)
        self.init_weights()
        self.version = int(__version__.split('.')[0])

    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1, pretrained_model_name=None):
        if pretrained_model_name is None:
            pretrained_model_name = args.pretrained_model_name
        cfg = BertConfig.from_pretrained(pretrained_model_name)
        cfg.output_hidden_states = True
        cfg.add_linear = args.add_linear
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        if int(__version__.split('.')[0]) >= 3:
            cfg.gradient_checkpointing = args.gradient_checkpointing
        return cls.from_pretrained(pretrained_model_name, config=cfg)

    @classmethod
    def init_encoder_from_my_model(cls, args, dropout: float = 0.1):
        cfg = DistilBertConfig.from_pretrained(args.model_path)
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        if int(__version__.split('.')[0]) >= 3:
            cfg.gradient_checkpointing = args.gradient_checkpointing
        return cls.from_pretrained(args.model_path, config=cfg)

    def forward(self, **kwargs):
        hidden_states = None
        result, all_layer_hidden, _ = super().forward(**kwargs)
        if self.add_linear:
            sequence_output = self.linear(result.last_hidden_state)
            all_layer_hidden = list(all_layer_hidden)
            all_layer_hidden.append(sequence_output)
        else:
            sequence_output = result.last_hidden_state
        all_layer_attention_map = result.attentions
        pooled_output = sequence_output[:, 0, :]

        return sequence_output, pooled_output, hidden_states, all_layer_attention_map, \
               torch.cat([elem.unsqueeze(0) for elem in all_layer_hidden], dim=0).permute([1,0,2,3])

class HFColBertEncoder(BertModel):
    def __init__(self, config, mask_punctuation=True, dim=128):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.version = int(__version__.split('.')[0])
        self.dim = dim
        self.add_linear = config.add_linear
        if self.add_linear:
            self.linear = nn.Linear(config.hidden_size, 768)
        if mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        self.init_weights()

    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1, pretrained_model_name=None):
        pretrained_model_name = args.pretrained_model_name
        model_path = pretrained_model_name
        cfg = BertConfig.from_pretrained(pretrained_model_name)

        cfg.add_linear = args.add_linear
        cfg.output_hidden_states = True
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        if int(__version__.split('.')[0]) >= 3:
            cfg.gradient_checkpointing = args.gradient_checkpointing
        return cls.from_pretrained(model_path, config=cfg)

    @classmethod
    def init_encoder_from_my_model(cls, args, dropout: float = 0.1):
        pretrained_model_name = args.pretrained_model_name
        model_path = pretrained_model_name
        cfg = BertConfig.from_pretrained(pretrained_model_name)

        cfg.add_linear = args.add_linear
        cfg.output_hidden_states = True
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        if int(__version__.split('.')[0]) >= 3:
            cfg.gradient_checkpointing = args.gradient_checkpointing
        return cls.from_pretrained(model_path, config=cfg)

    def forward(self, mode, device, **kwargs):
        ## all_layer_hidden： [Layer_num, Batch_size, Seq_len, Embed_size]
        ## all_layer_hidden_all_head： [Layer_num, Batch_size, Head_num, Seq_len, Embed_size_per_head]
        result, all_layer_hidden = super().forward(**kwargs)
        col_out = result[0]
        if self.add_linear:
            col_out = self.linear(result[0])
            all_layer_hidden = list(all_layer_hidden)
            all_layer_hidden.append(col_out)
        mask = None
        ## After Permutation: all_layer_hidden： [Batch_size, Layer_num, Seq_len, Embed_size]
        ## After Permutation: all_layer_hidden_all_head： [Batch_size, Layer_num, Head_num, Seq_len, Embed_size_per_head]
        return torch.nn.functional.normalize(col_out, p=2, dim=2), \
               torch.cat([elem.unsqueeze(0) for elem in all_layer_hidden], dim=0).permute([1, 0, 2, 3]), mask

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask

class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        self.init_weights()
        self.version = int(__version__.split('.')[0])

    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1, pretrained_model_name=None):
        pretrained_model_name = args.pretrained_model_name
        ## If use MASTER, please download the ckpt from https://uswvhd.blob.core.windows.net/anonymous/MASTER/MASTER-MARCO.tar.gz
        if pretrained_model_name == "master":
            config_path = "pretrain_models/MASTER-MARCO/config.json"
            model_path = "pretrain_models/MASTER-MARCO"
            cfg = BertConfig.from_json_file(config_path)
        else:
            model_path = pretrained_model_name
            cfg = BertConfig.from_pretrained(pretrained_model_name)

        cfg.output_hidden_states = True
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        if int(__version__.split('.')[0]) >= 3:
            cfg.gradient_checkpointing = args.gradient_checkpointing
        return cls.from_pretrained(model_path, config=cfg)

    @classmethod
    def init_encoder_from_my_model(cls, args, dropout: float = 0.1):
        pretrained_model_name = args.pretrained_model_name
        model_path = pretrained_model_name
        cfg = BertConfig.from_pretrained(pretrained_model_name)

        cfg.output_hidden_states = True
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        if int(__version__.split('.')[0]) >= 3:
            cfg.gradient_checkpointing = args.gradient_checkpointing
        return cls.from_pretrained(model_path, config=cfg)

    def forward(self, **kwargs):
        hidden_states = None
        ## all_layer_hidden： [Layer_num, Batch_size, Seq_len, Embed_size]
        ## all_layer_hidden_all_head： [Layer_num, Batch_size, Head_num, Seq_len, Embed_size_per_head]
        result, all_layer_hidden = super().forward(**kwargs)
        sequence_output = result.last_hidden_state
        all_layer_attention_map = result.attentions
        pooled_output = sequence_output[:, 0, :]
        # all_layer_hidden_adapter = [self.linear_adapter[i](all_layer_hidden[i]) for i in range(len(all_layer_hidden))]
        ## After Permutation: all_layer_hidden： [Batch_size, Layer_num, Seq_len, Embed_size]
        ## After Permutation: all_layer_hidden_all_head： [Batch_size, Layer_num, Head_num, Seq_len, Embed_size_per_head]
        return sequence_output, pooled_output, hidden_states, all_layer_attention_map, \
               torch.cat([elem.unsqueeze(0) for elem in all_layer_hidden], dim=0).permute([1, 0, 2, 3])

class BiBertEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """
    def __init__(self, args):
        super(BiBertEncoder, self).__init__()
        self.pretrained_model_name = args.pretrained_model_name
        self.model_type = args.model_type
        self.device = args.device
        if 'colbert' in args.model_type:
            self.question_model = HFColBertEncoder.init_encoder(args)
        elif 'distilbert' in args.model_type:
            self.question_model = HFDistilBertEncoder.init_encoder(args)
        else:
            self.question_model = HFBertEncoder.init_encoder(args)

        if hasattr(args, 'share_weight') and args.share_weight:
            self.ctx_model = self.question_model
        else:
            if 'colbert' in args.model_type:
                self.ctx_model = HFColBertEncoder.init_encoder(args)
            elif 'distilbert' in args.model_type:
                self.ctx_model = HFDistilBertEncoder.init_encoder(args)
            else:
                self.ctx_model = HFBertEncoder.init_encoder(args)

    def query_emb(self, mode, input_ids, attention_mask):
        if 'colbert' in mode:
            col_output, all_layer_hidden, _ = self.question_model(mode, self.device, input_ids=input_ids, attention_mask=attention_mask)
            return col_output, all_layer_hidden, None
        else:
            _, pooled_output, _, _, all_layer_hidden = self.question_model(input_ids=input_ids, attention_mask=attention_mask)
            return pooled_output, all_layer_hidden, None

    def body_emb(self, mode, input_ids, attention_mask):
        if 'colbert' in mode:
            col_output, all_layer_hidden, mask_doc = self.ctx_model(mode, self.device, input_ids=input_ids, attention_mask=attention_mask)
            return col_output, all_layer_hidden, mask_doc
        else:
            _, pooled_output, _, _, all_layer_hidden = self.ctx_model(input_ids=input_ids, attention_mask=attention_mask)
            return pooled_output, all_layer_hidden, None

    def forward(self, query_ids, attention_mask_q, doc_ids=None, attention_mask_d=None):
        q_embs, q_all_layer_hidden, _ = self.query_emb(self.model_type+'_query', query_ids, attention_mask_q)
        d_embs, d_all_layer_hidden, mask_doc = self.body_emb(self.model_type+'_doc', doc_ids, attention_mask_d)
        return q_embs, d_embs, q_all_layer_hidden, d_all_layer_hidden, mask_doc

class config_(object):
    def __init__(self, hidden_size, hidden_act):
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act

class Reranker(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size):
        super(Reranker, self).__init__()
        self.encoder = encoder
        self.qa_classifier = nn.Linear(hidden_size, 1)
        init_weights([self.qa_classifier])

    def forward(self, input_ids: T, attention_mask: T):
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        N, M, L = input_ids.size()
        relevance_logits, attention_map, rank_logits_all_layer = self._forward(input_ids.view(N * M, L),attention_mask.view(N * M, L))
        return relevance_logits.view(N, M), attention_map, rank_logits_all_layer.permute([1,0,2]).view(rank_logits_all_layer.shape[1], N, M)

    def _forward(self, input_ids, attention_mask):
        sequence_output, _, _, attention_map, all_layer_hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        rank_logits = self.qa_classifier(sequence_output[:, 0, :])
        rank_logits_all_layer = self.qa_classifier(all_layer_hidden[:, :, 0, :])

        return rank_logits, attention_map, rank_logits_all_layer

def init_weights(modules):
    for module in modules:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
