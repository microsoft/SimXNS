import os
import warnings

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from transformers import BertModel, BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig, \
    RobertaModel, ElectraForMaskedLM, ElectraModel, ElectraForMaskedLM
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertPreTrainingHeads, BertLayer
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling, MaskedLMOutput
from transformers.models.roberta.modeling_roberta import RobertaLayer

from arguments import DataTrainingArguments, ModelArguments, CoCondenserPreTrainingArguments
from transformers import TrainingArguments
import logging

logger = logging.getLogger(__name__)


class CondenserForPretraining(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        train_args: TrainingArguments
    ):
        super(CondenserForPretraining, self).__init__()
        self.lm = bert
        self.c_head = nn.ModuleList(
            [BertLayer(bert.config) for _ in range(model_args.n_head_layers)]
        )
        self.query_head = nn.ModuleList(
            [BertLayer(bert.config) for _ in range(model_args.n_head_layers)]
        )
        self.gpt_head = nn.ModuleList(
            [BertLayer(bert.config) for _ in range(model_args.n_head_layers)]
        )
        self.next_head = nn.ModuleList(
            [BertLayer(bert.config) for _ in range(model_args.n_head_layers)]
        )
        self.overlap_head = nn.ModuleList(
            [BertLayer(bert.config) for _ in range(model_args.n_head_layers)]
        )
        #self.c_head.apply(self.lm._init_weights)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

    def forward(self, model_input, labels):
        lm_out: MaskedLMOutput = self.lm(
            input_ids=model_input['input_ids'],
            attention_mask=model_input['attention_mask'],
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        cls_hiddens = lm_out.hidden_states[-1][:, :1]

        # decoder with itself
        skip_hiddens = self.lm.bert.embeddings(input_ids = model_input['decoder_input_ids']) #lm_out.hidden_states[self.model_args.skip_from]
        hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)
        attention_mask = self.lm.get_extended_attention_mask(
            model_input['attention_mask'],
            model_input['attention_mask'].shape,
            model_input['attention_mask'].device
        )
        for layer in self.c_head:
            layer_out = layer(
                hiddens,
                attention_mask,
            )
            hiddens = layer_out[0]
        loss = self.mlm_loss(hiddens, model_input['decoder_labels'])

        # decoder with query
        query_skip_hiddens = self.lm.bert.embeddings(input_ids = model_input['query_input_ids']) #lm_out.hidden_states[self.model_args.skip_from]
        query_hiddens = torch.cat([cls_hiddens, query_skip_hiddens[:, 1:]], dim=1)
        query_attention_mask = self.lm.get_extended_attention_mask(
            model_input['query_attention_mask'],
            model_input['query_attention_mask'].shape,
            model_input['query_attention_mask'].device
        )
        for layer in self.query_head:
            query_layer_out = layer(
                query_hiddens,
                query_attention_mask,
            )
            query_hiddens = query_layer_out[0]
        query_loss = self.mlm_loss(query_hiddens, model_input['query_labels'])

        # decoder with query
        gpt_skip_hiddens = self.lm.bert.embeddings(input_ids = model_input['gpt_input_ids']) #lm_out.hidden_states[self.model_args.skip_from]
        gpt_hiddens = torch.cat([cls_hiddens, gpt_skip_hiddens[:, 1:]], dim=1)
        gpt_attention_mask = self.lm.get_extended_attention_mask(
            model_input['gpt_attention_mask'],
            model_input['gpt_attention_mask'].shape,
            model_input['gpt_attention_mask'].device
        )
        for layer in self.gpt_head:
            gpt_layer_out = layer(
                gpt_hiddens,
                gpt_attention_mask,
            )
            gpt_hiddens = gpt_layer_out[0]
        gpt_loss = self.mlm_loss(gpt_hiddens, model_input['gpt_labels'])

        # next encoder-decoder
        next_encoder_lm_out: MaskedLMOutput = self.lm(
            input_ids=model_input['next_encoder_input_ids'],
            attention_mask=model_input['next_encoder_attention_mask'],
            labels=model_input['next_encoder_labels'],
            output_hidden_states=True,
            return_dict=True
        )
        next_decoder_cls_hiddens = next_encoder_lm_out.hidden_states[-1][:, :1]

        # decoder with itself
        next_decoder_skip_hiddens = self.lm.bert.embeddings(input_ids=model_input['next_decoder_input_ids'])  # lm_out.hidden_states[self.model_args.skip_from]
        next_decoder_hiddens = torch.cat([next_decoder_cls_hiddens, next_decoder_skip_hiddens[:, 1:]], dim=1)
        next_decoder_attention_mask = self.lm.get_extended_attention_mask(
            model_input['next_decoder_attention_mask'],
            model_input['next_decoder_attention_mask'].shape,
            model_input['next_decoder_attention_mask'].device
        )
        for layer in self.next_head:
            next_decoder_layer_out = layer(
                next_decoder_hiddens,
                next_decoder_attention_mask,
            )
            next_decoder_hiddens = next_decoder_layer_out[0]
        next_loss = self.mlm_loss(next_decoder_hiddens, model_input['next_decoder_labels'])

        # overlap encoder-decoder
        overlap_encoder_lm_out: MaskedLMOutput = self.lm(
            input_ids=model_input['overlap_encoder_input_ids'],
            attention_mask=model_input['attention_mask'],
            labels=model_input['overlap_encoder_labels'],
            output_hidden_states=True,
            return_dict=True
        )
        overlap_decoder_cls_hiddens = overlap_encoder_lm_out.hidden_states[-1][:, :1]

        # decoder with itself
        overlap_decoder_skip_hiddens = self.lm.bert.embeddings(input_ids=model_input['overlap_decoder_input_ids'])  # lm_out.hidden_states[self.model_args.skip_from]
        overlap_decoder_hiddens = torch.cat([overlap_decoder_cls_hiddens, overlap_decoder_skip_hiddens[:, 1:]], dim=1)
        for layer in self.overlap_head:
            overlap_decoder_layer_out = layer(
                overlap_decoder_hiddens,
                attention_mask,
            )
            overlap_decoder_hiddens = overlap_decoder_layer_out[0]
        overlap_loss = self.mlm_loss(overlap_decoder_hiddens, model_input['overlap_decoder_labels'])

        final_loss = loss + query_loss + gpt_loss + next_loss + overlap_loss + lm_out.loss + next_encoder_lm_out.loss + overlap_encoder_lm_out.loss

        return final_loss


    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss


    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataTrainingArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args, data_args, train_args)
        path = args[0]
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = model.load_state_dict(model_dict, strict=False)
        return model

    @classmethod
    def from_config(
            cls,
            config: PretrainedConfig,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: TrainingArguments,
    ):
        hf_model = AutoModelForMaskedLM.from_config(config)
        model = cls(hf_model, model_args, data_args, train_args)

        return model

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('lm')]
        warnings.warn(f'omiting {len(hf_weight_keys)} transformer weights')
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))

class ELECTRACondenserForPretraining(nn.Module):
    def __init__(
        self,
        discriminator: ElectraModel,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        train_args: TrainingArguments
    ):
        super(ELECTRACondenserForPretraining, self).__init__()
        self.dis = discriminator
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.c_head = nn.ModuleList(
            [BertLayer(bert_config) for _ in range(model_args.n_head_layers)]
        )
        self.next_head = nn.ModuleList(
            [BertLayer(bert_config) for _ in range(model_args.n_head_layers)]
        )
        self.overlap_head = nn.ModuleList(
            [BertLayer(bert_config) for _ in range(model_args.n_head_layers)]
        )
        #self.c_head.apply(self.lm._init_weights)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

    def forward(self, model_input, labels):
        lm_out_dis = self.dis(input_ids=model_input['input_ids'], attention_mask=model_input['attention_mask'], labels=labels,
                              output_hidden_states=True, return_dict=True)
        cls_hiddens = lm_out_dis.hidden_states[-1][:, :1]
        skip_hiddens = self.dis.electra.embeddings(input_ids = model_input['decoder_input_ids']) #lm_out.hidden_states[self.model_args.skip_from]
        hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)
        attention_mask = self.dis.get_extended_attention_mask(
            model_input['attention_mask'],
            model_input['attention_mask'].shape,
            model_input['attention_mask'].device
        )
        for layer in self.c_head:
            layer_out = layer(
                hiddens,
                attention_mask,
            )
            hiddens = layer_out[0]
        loss = self.mlm_loss(hiddens, model_input['decoder_labels'])

        next_skip_hiddens = self.dis.electra.embeddings(input_ids=model_input['next_input_ids'])  # lm_out.hidden_states[self.model_args.skip_from]
        next_hiddens = torch.cat([cls_hiddens, next_skip_hiddens[:, 1:]], dim=1)
        next_attention_mask = self.dis.get_extended_attention_mask(
            model_input['next_attention_mask'],
            model_input['next_attention_mask'].shape,
            model_input['next_attention_mask'].device
        )
        for layer in self.next_head:
            next_layer_out = layer(
                next_hiddens,
                next_attention_mask,
            )
            next_hiddens = next_layer_out[0]
        next_loss = self.mlm_loss(next_hiddens, model_input['next_labels'])

        overlap_skip_hiddens = self.dis.electra.embeddings(input_ids=model_input['overlap_input_ids'])  # lm_out.hidden_states[self.model_args.skip_from]
        overlap_hiddens = torch.cat([cls_hiddens, overlap_skip_hiddens[:, 1:]], dim=1)
        overlap_attention_mask = self.dis.get_extended_attention_mask(
            model_input['overlap_attention_mask'],
            model_input['overlap_attention_mask'].shape,
            model_input['overlap_attention_mask'].device
        )
        for layer in self.overlap_head:
            overlap_layer_out = layer(
                overlap_hiddens,
                overlap_attention_mask,
            )
            overlap_hiddens = overlap_layer_out[0]
        overlap_loss = self.mlm_loss(overlap_hiddens, model_input['overlap_labels'])

        loss = next_loss + overlap_loss + loss + lm_out_dis.loss

        return loss

    def contrastive_loss(self, hiddens1, hiddens2):
        # hiddens: [batch, hidden]
        batch, hidden_size = hiddens1.size()
        z1 = hiddens1.reshape(-1, hidden_size)
        z2 = hiddens2.reshape(-1, hidden_size)

        if dist.is_initialized():
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        scores = torch.matmul(z1, torch.transpose(z2, 0, 1)) / self.model_args.temp

        softmax_scores = F.log_softmax(scores, dim=1)
        con_labels = torch.arange(scores.size(0), device=softmax_scores.device).long()
        # torch.eye(batch*num, device=softmax_scores.device, dtype=torch.long)

        loss = F.nll_loss(softmax_scores, con_labels, reduction="mean")
        return loss

    def mlm_loss(self, hiddens, labels):
        prediction_scores = self.dis.generator_predictions(hiddens)
        prediction_scores = self.dis.generator_lm_head(prediction_scores)

        masked_lm_loss = self.cross_entropy(
            prediction_scores.view(-1, self.dis.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss


    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataTrainingArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        dis_model = ElectraForMaskedLM.from_pretrained(*args, **kwargs)
        dis_model.generator_lm_head.weight = dis_model.electra.embeddings.word_embeddings.weight

        model = cls(dis_model, model_args, data_args, train_args)
        path = args[0]
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            model.load_state_dict(model_dict, strict=False)
        return model

    def save_pretrained(self, output_dir: str):
        self.dis.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('dis') or k.startswith('gen')]
        warnings.warn(f'omiting {len(hf_weight_keys)} transformer weights')
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))


class RobertaCondenserForPretraining(CondenserForPretraining):
    def __init__(
            self,
            roberta: RobertaModel,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: TrainingArguments
    ):
        super(CondenserForPretraining, self).__init__()
        self.lm = roberta
        self.c_head = nn.ModuleList(
            [RobertaLayer(roberta.config) for _ in range(model_args.n_head_layers)]
        )
        self.c_head.apply(self.lm._init_weights)
        # self.mlm_head = BertOnlyMLMHead(bert.config)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.lm_head(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss

class CoCondenserForPretraining(CondenserForPretraining):
    def __init__(
            self,
            bert: BertModel,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: CoCondenserPreTrainingArguments
    ):
        super(CoCondenserForPretraining, self).__init__(bert, model_args, data_args, train_args)

        effective_bsz = train_args.per_device_train_batch_size * self._world_size() * 2
        target = torch.arange(effective_bsz, dtype=torch.long).view(-1, 2).flip([1]).flatten().contiguous()

        self.register_buffer(
            'co_target', target
        )

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.train_args.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: Tensor):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt

    def forward(self, model_input, labels, grad_cache: Tensor = None, chunk_offset: int = None):
        attention_mask = self.lm.get_extended_attention_mask(
            model_input['attention_mask'],
            model_input['attention_mask'].shape,
            model_input['attention_mask'].device
        )

        lm_out: MaskedLMOutput = self.lm(
            **model_input,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )

        cls_hiddens = lm_out.hidden_states[-1][:, :1]
        if self.train_args.local_rank > -1 and grad_cache is None:
            co_cls_hiddens = self.gather_tensors(cls_hiddens.squeeze().contiguous())[0]
        else:
            co_cls_hiddens = cls_hiddens.squeeze()

        skip_hiddens = lm_out.hidden_states[self.model_args.skip_from]
        hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)

        for layer in self.c_head:
            layer_out = layer(
                hiddens,
                attention_mask,
            )
            hiddens = layer_out[0]

        loss = self.mlm_loss(hiddens, labels)
        if self.model_args.late_mlm:
            loss += lm_out.loss

        if grad_cache is None:
            co_loss = self.compute_contrastive_loss(co_cls_hiddens)
            return loss + co_loss
        else:
            loss = loss * (float(hiddens.size(0)) / self.train_args.per_device_train_batch_size)
            cached_grads = grad_cache[chunk_offset: chunk_offset + co_cls_hiddens.size(0)]
            surrogate = torch.dot(cached_grads.flatten(), co_cls_hiddens.flatten())
            return loss, surrogate

    @staticmethod
    def _world_size():
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1

    def compute_contrastive_loss(self, co_cls_hiddens):
        similarities = torch.matmul(co_cls_hiddens, co_cls_hiddens.transpose(0, 1))
        similarities.fill_diagonal_(float('-inf'))
        co_loss = F.cross_entropy(similarities, self.co_target) * self._world_size()
        return co_loss
