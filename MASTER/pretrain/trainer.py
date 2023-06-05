import os
from contextlib import nullcontext

from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import ElectraForMaskedLM
import torch
import torch.distributed as dist
from torch import nn, Tensor
from torch.cuda.amp import autocast
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption
try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False

import logging

logger = logging.getLogger(__name__)


class CondenserPreTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(CondenserPreTrainer, self).__init__(*args, **kwargs)

    def in_batch_nearest(self, cls_hiddens):
        # [batch, hidden]
        dot_map = torch.matmul(cls_hiddens, cls_hiddens.transpose(0, 1))    #[batch, batch]
        nearest_ids = torch.argmax(dot_map * (1 - torch.eye(dot_map.size(0), device=dot_map.device)), -1)
        nearest_cls_hiddens = cls_hiddens[nearest_ids]
        return nearest_cls_hiddens

    def generate_replaced_ids(self, model_input):
        with torch.no_grad():
            lm_out = self.model.dis(
                    input_ids=model_input['input_ids'],
                    attention_mask=model_input['attention_mask'],
                    output_hidden_states=True, return_dict=True
                )
            cls_hiddens = lm_out.hidden_states[-1][:, 0]
            shuffled_cls_hiddens = self.in_batch_nearest(cls_hiddens)
            #print(shuffled_cls_hiddens.size())
            skip_hiddens = self.model.dis.electra.embeddings(input_ids=model_input['decoder_input_ids'])  # lm_out.hidden_states[self.model_args.skip_from]

            hiddens = torch.cat([shuffled_cls_hiddens.unsqueeze(1), skip_hiddens[:, 1:]], dim=1)
            attention_mask = self.model.dis.get_extended_attention_mask(
                model_input['decoder_attention_mask'],
                model_input['decoder_attention_mask'].shape,
                model_input['decoder_attention_mask'].device
            )
            for layer in self.model.c_head:
                layer_out = layer(
                    hiddens,
                    attention_mask,
                )
                hiddens = layer_out[0]
            prediction_scores = self.model.dis.generator_predictions(hiddens)
            prediction_scores = self.model.dis.generator_lm_head(prediction_scores)

            predicted_ids = torch.argmax(prediction_scores, -1)
            replaced_decoder_input_ids = torch.where(model_input['decoder_input_ids']!=103, model_input['decoder_input_ids'], predicted_ids)

            decoder_lm_out = self.model.dis(
                input_ids=model_input['decoder_input_ids'],
                attention_mask=model_input['decoder_attention_mask'],
                output_hidden_states=True, return_dict=True
            )
            decoder_cls_hiddens = decoder_lm_out.hidden_states[-1][:, 0]
            decoder_shuffled_cls_hiddens = self.in_batch_nearest(decoder_cls_hiddens)
            # print(shuffled_cls_hiddens.size())
            decoder_skip_hiddens = self.model.dis.electra.embeddings(input_ids=model_input['input_ids'])  # lm_out.hidden_states[self.model_args.skip_from]

            decoder_hiddens = torch.cat([decoder_shuffled_cls_hiddens.unsqueeze(1), decoder_skip_hiddens[:, 1:]], dim=1)
            decoder_attention_mask = self.model.dis.get_extended_attention_mask(
                model_input['attention_mask'],
                model_input['attention_mask'].shape,
                model_input['attention_mask'].device
            )
            for layer in self.model.c_head:
                decoder_layer_out = layer(
                    decoder_hiddens,
                    decoder_attention_mask,
                )
                decoder_hiddens = decoder_layer_out[0]
            decoder_prediction_scores = self.model.dis.generator_predictions(decoder_hiddens)
            decoder_prediction_scores = self.model.dis.generator_lm_head(decoder_prediction_scores)

            decoder_predicted_ids = torch.argmax(decoder_prediction_scores, -1)
            replaced_input_ids = torch.where(model_input['input_ids'] != 103, model_input['input_ids'], decoder_predicted_ids)

        return replaced_decoder_input_ids, replaced_input_ids

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        # we are not going to do this in this
        # as collator will be generating new columns
        return dataset

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.args.warmup_ratio > 0:
            self.args.warmup_steps = num_training_steps * self.args.warmup_ratio

        super().create_optimizer_and_scheduler(num_training_steps)

    def compute_loss(self, model, inputs):
        labels = inputs.pop('labels')
        loss = model(inputs, labels)
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop('labels')

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16:
                with autocast():
                    outputs = model(inputs, labels)
            else:
                outputs = model(inputs, labels)

            loss = outputs

        return (loss, None, None)


class CoCondenserPretrainer(CondenserPreTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        super(CondenserPreTrainer, self).__init__(*args, **kwargs)

        if self.args.cache_chunk_size != -1:
            if not _grad_cache_available:
                raise ValueError(
                    'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')
            self.gc = GradCache(
                models=[self.model.lm],
                chunk_sizes=self.args.cache_chunk_size,
                loss_fn=self.model.compute_contrastive_loss,
                get_rep_fn=lambda x: x.hidden_states[-1][:, 0],
                fp16=self.args.fp16,
                scaler=self.scaler
            )

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.args.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: Tensor):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt

    def compute_loss(self, model, inputs, grad_cache=None, chunk_offset=None):
        labels = inputs.pop('labels')
        return model(inputs, labels, grad_cache=grad_cache, chunk_offset=chunk_offset)

    def split_tensor_dict(self, td: Dict[str, Tensor]):
        keys = list(td.keys())
        chunked_tensors = [td[k].split(self.args.cache_chunk_size) for k in keys]
        return [dict(zip(keys, tt)) for tt in zip(*chunked_tensors)]

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if self.args.cache_chunk_size == -1:
            return super(CoCondenserPretrainer, self).training_step(model, inputs)

        model.train()

        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop('labels')

        # Construct the gradient cache
        chunked_inputs = self.split_tensor_dict(inputs)
        for c in chunked_inputs:
            c['output_hidden_states'] = True
        cls_hiddens, rnd_states = self.gc.forward_no_grad(self.model.lm, chunked_inputs)
        if self.args.local_rank > -1:
            cls_hiddens = self.gather_tensors(cls_hiddens.contiguous())[0]
        grad_cache, total_loss = self.gc.build_cache(cls_hiddens)
        grad_cache = grad_cache[0]
        if self.args.local_rank > -1:
            total_loss = total_loss / dist.get_world_size()

        inputs['labels'] = labels
        chunked_inputs = self.split_tensor_dict(inputs)

        # Compute the full loss with cached gradients
        for local_chunk_id, chunk in enumerate(chunked_inputs):
            device_offset = max(0, self.args.local_rank) * self.args.per_device_train_batch_size * 2
            local_offset = local_chunk_id * self.args.cache_chunk_size
            chunk_offset = device_offset + local_offset
            with rnd_states[local_chunk_id]:
                if self.use_amp:
                    with autocast():
                        lm_loss, surrogate = self.compute_loss(model, chunk, grad_cache, chunk_offset)
                else:
                    lm_loss, surrogate = self.compute_loss(model, chunk, grad_cache, chunk_offset)

            if self.args.gradient_accumulation_steps > 1:
                raise ValueError

            ddp_no_sync = self.args.local_rank > -1 and (local_chunk_id + 1 < len(chunked_inputs))
            with model.no_sync() if ddp_no_sync else nullcontext():
                if self.use_amp:
                    (self.scaler.scale(lm_loss) + surrogate).backward()
                elif self.use_apex:
                    raise ValueError
                elif self.deepspeed:
                    raise ValueError
                else:
                    (lm_loss + surrogate).backward()
            total_loss += lm_loss
        return total_loss.detach()
