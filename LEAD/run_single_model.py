import sys
sys.path += ['../']
sys.path += ['../../']
import logging
import os
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd()))))
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
from transformers import (
    get_linear_schedule_with_warmup,
)
import time
import transformers
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

import collections
from dataset import TraditionDataset

retrieverBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "q_ids",
        "q_attn_mask",
        "c_ids",
        "c_attn_mask",
        "c_q_mapping",
        "is_positive",
    ],
)

from util import get_loss_dual, _save_checkpoint, _load_saved_state, load_model, \
                 set_env, get_arguments, get_loss_cross, set_seed, \
                 is_first_worker, load_states_from_checkpoint, get_optimizer, evaluate_dev

def train(args, model, tokenizer):
    """ Train the model """
    global loss
    logger.info("Training/evaluation parameters %s", args)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)  # nll loss for query
    optimizer = get_optimizer(args, model, weight_decay=args.weight_decay)

    unique_identifier = str(time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) + '_' + str(time.localtime().tm_hour)+ '_' + str(time.localtime().tm_min) + '_' + str(args.dataset)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        if ('distilbert' in args.model_type):
            state = False
        else:
            state = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=state,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Max steps = %d", args.max_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Hard negative number per query = %d", args.num_hard_negatives)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )

    model.zero_grad()
    model.train()
    set_seed(args)  # Added here for reproductibility

    warm_up_ratio = 0.1
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(warm_up_ratio * args.max_steps), num_training_steps=args.max_steps
    )

    global_step = 0

    if args.load_continue_train_path is not None:
        saved_states = load_states_from_checkpoint(args.load_continue_train_path)
        global_step = _load_saved_state(model, optimizer, scheduler, saved_states)

    if args.reset_global_step:
        global_step = 0
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

    # Dataset
    train_dataset = TraditionDataset(args, args.train_file, tokenizer, num_hard_negatives=args.num_hard_negatives,
                                     max_seq_length=args.max_doc_length, max_q_length=args.max_query_length,
                                     shuffle_positives=args.shuffle_positives)
    train_sample = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sample,
                                  collate_fn=train_dataset.get_collate_fn,
                                  batch_size=args.train_batch_size, num_workers=12)

    tr_loss = 0
    while global_step < args.max_steps:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            if ('cross_encoder' in args.model_type):
                batch_reranker = batch['reranker']
                inputs_reranker = {"input_ids": batch_reranker[0].long().to(args.device), "attention_mask": batch_reranker[1].long().to(args.device)}
                output_reranker, _, _ = model(**inputs_reranker)
                loss = get_loss_cross(args, output_reranker)

            elif ('dual_encoder' in args.model_type):
                batch_retriever = batch['retriever']
                inputs_retriever = {"query_ids": batch_retriever[0].long().to(args.device),
                                    "attention_mask_q": batch_retriever[1].long().to(args.device),
                                    "doc_ids": batch_retriever[2].long().to(args.device),
                                    "attention_mask_d": batch_retriever[3].long().to(args.device)}
                local_q_vector, local_ctx_vectors, _, _, _ = model(**inputs_retriever)
                loss = get_loss_dual(args, local_q_vector, local_ctx_vectors, inputs_retriever['attention_mask_d'],
                                     reduction='mean')

            elif ('distilbert' in args.model_type):
                batch_retriever = batch['retriever']
                inputs_retriever = {"query_ids": batch_retriever[0].long().to(args.device),
                                    "attention_mask_q": batch_retriever[1].long().to(args.device),
                                    "doc_ids": batch_retriever[2].long().to(args.device),
                                    "attention_mask_d": batch_retriever[3].long().to(args.device)}
                local_q_vector, local_ctx_vectors, _, _, _ = model(**inputs_retriever)
                loss = get_loss_dual(args, local_q_vector, local_ctx_vectors, inputs_retriever['attention_mask_d'],
                                     reduction='mean')

            elif ('colbert' in args.model_type):
                batch_retriever = batch['retriever']
                inputs_retriever = {"query_ids": batch_retriever[4].long().to(args.device),
                                    "attention_mask_q": batch_retriever[5].long().to(args.device),
                                    "doc_ids": batch_retriever[6].long().to(args.device),
                                    "attention_mask_d": batch_retriever[7].long().to(args.device)}
                local_q_vector, local_ctx_vectors, _, _, _ = model(**inputs_retriever)
                loss = get_loss_dual(args, local_q_vector, local_ctx_vectors, inputs_retriever['attention_mask_d'],
                                     reduction='mean')

            loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            epoch_iterator.set_postfix(loss=loss.item())
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    logger.info(" Evaluation Start ")
                    logger.info(" Evaluation Done ")
                    if is_first_worker():
                        if ('cross_encoder' in args.model_type):
                            mode = 'ce'
                        elif ('dual_encoder' in args.model_type):
                            mode = 'de'
                        elif ('distilbert' in args.model_type):
                            mode = 'db'
                        elif ('colbert' in args.model_type):
                            mode = 'col'
                        _save_checkpoint(args, model, optimizer, scheduler, global_step, unique_identifier, mode)

                if global_step >= args.max_steps:
                    break
    return global_step


def main():
    # Preparation
    args = get_arguments()
    set_env(args)
    tokenizer, model = load_model(args)

    # Logger
    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    log_path = os.path.join(args.output_dir, 'log.txt')
    handler = logging.FileHandler(log_path, 'w', 'utf-8')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    print(logger)

    # Training
    global_step = train(args, model, tokenizer)
    logger.info(" global_step = %s", global_step)


if __name__ == "__main__":
    main()
