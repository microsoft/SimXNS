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

logger = logging.getLogger(__name__)

import collections
from dataset import TraditionDataset
import transformers
transformers.logging.set_verbosity_error()
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

from util import _save_checkpoint, _load_saved_state, load_model, \
                 set_env, get_arguments, fwd_pass, set_seed, \
                 is_first_worker, load_states_from_checkpoint, get_optimizer, \
                 distill_loss, evaluate_dev, select_layer

def train(args, model_de, model_db, model_col, model_ce, tokenizer):
    """ Train the model """
    logger.info("Training/evaluation parameters %s", args)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)  # nll loss for query

    optimizer_dual, optimizer_db, optimizer_col, optimizer_ce = None, None, None, None
    if args.distill_de and args.train_de:
        optimizer_de = get_optimizer(args, model_de, weight_decay=args.weight_decay)
    if args.distill_db and args.train_db:
        optimizer_db = get_optimizer(args, model_db, weight_decay=args.weight_decay)
    if args.distill_col and args.train_col:
        optimizer_col = get_optimizer(args, model_col, weight_decay=args.weight_decay)
    if args.distill_ce and args.train_ce:
        optimizer_ce = get_optimizer(args, model_ce, weight_decay=args.weight_decay)

    unique_identifier = str(time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) + '_' + str(time.localtime().tm_hour) \
                        + '_' + str(time.localtime().tm_min) + '_' + str(args.dataset)+ '_distill'

    args.model_type = 'distill'

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        if args.distill_de:
            model_de = torch.nn.parallel.DistributedDataParallel(model_de, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)
        if args.distill_db:
            model_db = torch.nn.parallel.DistributedDataParallel(model_db, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)
        if args.distill_col:
            model_col = torch.nn.parallel.DistributedDataParallel(model_col, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)
        if args.distill_ce:
            model_ce = torch.nn.parallel.DistributedDataParallel(model_ce, device_ids=[args.rank], output_device=args.rank, find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Max steps = %d", args.max_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Hard negative number per query = %d", args.num_hard_negatives)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))

    if args.distill_de:
        if args.train_de:
            model_de.zero_grad()
            model_de.train()
        else:
            model_de.eval()

    if args.distill_db:
        if args.train_db:
            model_db.zero_grad()
            model_db.train()
        else:
            model_db.eval()

    if args.distill_col:
        if args.train_col:
            model_col.zero_grad()
            model_col.train()
        else:
            model_col.eval()

    if args.distill_ce:
        if args.train_ce:
            model_ce.zero_grad()
            model_ce.train()
        else:
            model_ce.eval()

    set_seed(args)  # Added here for reproductibility
    warm_up_ratio = args.warm_up_ratio
    global_step = 0

    if args.distill_de and args.train_de:
        scheduler_de = get_linear_schedule_with_warmup(optimizer_de, num_warmup_steps=int(warm_up_ratio * args.max_steps), num_training_steps=args.max_steps)
        if args.load_continue_train:
            saved_states = load_states_from_checkpoint(args.distill_de_path)
            global_step = _load_saved_state(model_de, optimizer_de, scheduler_de, saved_states)

    if args.distill_db and args.train_db:
        scheduler_db = get_linear_schedule_with_warmup(optimizer_db, num_warmup_steps=int(warm_up_ratio * args.max_steps), num_training_steps=args.max_steps)
        if args.load_continue_train:
            saved_states = load_states_from_checkpoint(args.distill_db_path)
            global_step = _load_saved_state(model_db, optimizer_db, scheduler_db, saved_states)

    if args.distill_col and args.train_col:
        scheduler_col = get_linear_schedule_with_warmup(optimizer_col, num_warmup_steps=int(warm_up_ratio * args.max_steps), num_training_steps=args.max_steps)
        if args.load_continue_train:
            saved_states = load_states_from_checkpoint(args.distill_col_path)
            global_step = _load_saved_state(model_col, optimizer_col, scheduler_col, saved_states)

    if args.distill_ce and args.train_ce:
        scheduler_ce = get_linear_schedule_with_warmup(optimizer_ce, num_warmup_steps=int(warm_up_ratio * args.max_steps), num_training_steps=args.max_steps)
        if args.load_continue_train:
            saved_states = load_states_from_checkpoint(args.distill_ce_path)
            global_step = _load_saved_state(model_ce, optimizer_ce, scheduler_ce, saved_states)

    # Dataset
    train_dataset = TraditionDataset(args, args.train_file, tokenizer, num_hard_negatives=args.num_hard_negatives,
                                     max_seq_length=args.max_doc_length, max_q_length=args.max_query_length,
                                     shuffle_positives=args.shuffle_positives)
    train_sample = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    ## when using DDP, each process should hold train_batch_size samples,so the real batch size should equal to train_batch_size * gpu_num
    train_dataloader = DataLoader(train_dataset, sampler=train_sample, collate_fn=train_dataset.get_collate_fn, batch_size=args.train_batch_size, num_workers=2)

    tr_loss = 0.0

    ##  Initialize layer selection
    selected_index_list_db, selected_index_list_teacher = select_layer(args)

    while global_step < args.max_steps:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            ## distilbert
            q_embs_db, d_embs_db, d_embs_db_dr, q_all_layer_hidden_db, d_all_layer_hidden_db = None, None, None, None, None
            batch_retriever = batch['retriever']
            if args.distill_db:
                model_db.train()
                inputs_retriever_db = {"query_ids": batch_retriever[0].long().to(args.device),
                                    "attention_mask_q": batch_retriever[1].long().to(args.device),
                                    "doc_ids": batch_retriever[2].long().to(args.device),
                                    "attention_mask_d": batch_retriever[3].long().to(args.device),
                                    }
                q_embs_db, d_embs_db, q_all_layer_hidden_db, d_all_layer_hidden_db, doc_mask_db = model_db(**inputs_retriever_db)

            ## dual encoder
            q_embs_dual, d_embs_dual, d_embs_dual_dr, q_all_layer_hidden_dual, d_all_layer_hidden_dual = None, None, None, None, None
            if args.distill_de:
                if args.train_de:
                    model_de.train()
                else:
                    model_de.eval()
                inputs_retriever_dual = {"query_ids": batch_retriever[0].long().to(args.device),
                                    "attention_mask_q": batch_retriever[1].long().to(args.device),
                                    "doc_ids": batch_retriever[2].long().to(args.device),
                                    "attention_mask_d": batch_retriever[3].long().to(args.device),
                                    }
                q_embs_dual, d_embs_dual, q_all_layer_hidden_dual, d_all_layer_hidden_dual, doc_mask_de = model_de(**inputs_retriever_dual)

            ## colbert
            q_embs_col, d_embs_col, q_all_layer_hidden_col, d_all_layer_hidden_col, doc_mask_col = None, None, None, None, None
            inputs_retriever_col = {'attention_mask_d': None}
            if args.distill_col:
                if args.train_col:
                    model_col.train()
                else:
                    model_col.eval()
                inputs_retriever_col = {"query_ids": batch_retriever[4].long().to(args.device),
                                         "attention_mask_q": batch_retriever[5].long().to(args.device),
                                         "doc_ids": batch_retriever[6].long().to(args.device),
                                         "attention_mask_d": batch_retriever[7].long().to(args.device)
                                         }
                q_embs_col, d_embs_col, q_all_layer_hidden_col, d_all_layer_hidden_col, doc_mask_col= model_col(**inputs_retriever_col)

            ## cross encoder
            output_reranker, attention_map, all_layer_score_ce = None, None, None
            if args.distill_ce:
                if args.train_ce:
                    model_ce.train()
                else:
                    model_ce.eval()
                batch_reranker = batch['reranker']
                inputs_reranker = {"input_ids": batch_reranker[0].long().to(args.device), "attention_mask": batch_reranker[1].long().to(args.device)}
                output_reranker, attention_map, all_layer_score_ce = model_ce(**inputs_reranker)

            ## distillation loss
            loss, loss_dict = distill_loss(args, q_embs_col, q_embs_dual, q_embs_db, d_embs_col, d_embs_dual, d_embs_db, doc_mask_col,
                                           selected_index_list_db, selected_index_list_teacher, attention_map,
                                           q_all_layer_hidden_dual, d_all_layer_hidden_dual, q_all_layer_hidden_db, d_all_layer_hidden_db, q_all_layer_hidden_col, d_all_layer_hidden_col,
                                           all_layer_score_ce, inputs_retriever_col['attention_mask_d'], output_reranker, batch, reduction='mean')
            loss = loss / args.gradient_accumulation_steps
            del batch

            all_optimizer = []
            if args.distill_de and args.train_de:
                all_optimizer.append(optimizer_de)
            if args.distill_db and args.train_db:
                all_optimizer.append(optimizer_db)
            if args.distill_col and args.train_col:
                all_optimizer.append(optimizer_col)
            if args.distill_ce and args.train_ce:
                all_optimizer.append(optimizer_ce)

            loss.backward()
            epoch_iterator.set_postfix(loss=loss.item())
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.distill_de and args.train_de:
                    torch.nn.utils.clip_grad_norm_(model_de.parameters(), args.max_grad_norm)
                if args.distill_db and args.train_db:
                    torch.nn.utils.clip_grad_norm_(model_db.parameters(), args.max_grad_norm)
                if args.distill_col and args.train_col:
                    torch.nn.utils.clip_grad_norm_(model_col.parameters(), args.max_grad_norm)
                if args.distill_ce and args.train_ce:
                    torch.nn.utils.clip_grad_norm_(model_ce.parameters(), args.max_grad_norm)

                if args.distill_de and args.train_de:
                    optimizer_de.step()
                    scheduler_de.step()
                    optimizer_de.zero_grad()
                if args.distill_db and args.train_db:
                    optimizer_db.step()
                    scheduler_db.step()
                    optimizer_db.zero_grad()
                if args.distill_col and args.train_col:
                    optimizer_col.step()
                    scheduler_col.step()
                    optimizer_col.zero_grad()
                if args.distill_ce and args.train_ce:
                    optimizer_ce.step()
                    scheduler_ce.step()
                    optimizer_ce.zero_grad()

                global_step += 1

                if global_step % args.save_steps == 0:
                    selected_index_list_db, selected_index_list_teacher = select_layer(args)
                    logger.info(" Saving Start ")
                    if is_first_worker():
                        if args.distill_de and args.train_de:
                            _save_checkpoint(args, model_de, optimizer_de, scheduler_de, global_step, unique_identifier,'de')
                        if args.distill_db and args.train_db:
                            _save_checkpoint(args, model_db, optimizer_db, scheduler_db, global_step, unique_identifier, 'db')
                        if args.distill_col and args.train_col:
                            _save_checkpoint(args, model_col, optimizer_col, scheduler_col, global_step, unique_identifier, 'col')
                        if args.distill_ce and args.train_ce:
                            _save_checkpoint(args, model_ce, optimizer_ce, scheduler_ce, global_step, unique_identifier, 'ce')
                    logger.info(" Evaluation Done ")

                if global_step >= args.max_steps:
                    break

    return global_step


def main():
    # Preparation
    args = get_arguments()
    set_env(args)

    model_de, model_db, model_col, model_ce = None, None, None, None
    ## distilbert
    if args.distill_db:
        tmp = args.pretrained_model_name
        if args.model_type == 'distilbert':
            args.pretrained_model_name = 'distilbert-base-uncased'
        elif args.model_type == 'dual_encoder':
            args.pretrained_model_name = 'master'
            # args.pretrained_model_name = 'nghuyong/ernie-2.0-base-en'
            # args.pretrained_model_name = 'Luyu/co-condenser-marco'

        tokenizer, model_db = load_model(args)
        if args.distill_db_path is not None:
            saved_state = load_states_from_checkpoint(args.distill_db_path)
            model_db.load_state_dict(saved_state.model_dict, strict=False)
        args.pretrained_model_name = tmp

    ## dual_encoder
    if args.distill_de:
        args.model_type = 'dual_encoder'
        tokenizer, model_de = load_model(args)
        if args.distill_de_path is not None:
            saved_state = load_states_from_checkpoint(args.distill_de_path)
            model_de.load_state_dict(saved_state.model_dict, strict=False)

    ## colbert
    if args.distill_col:
        args.model_type = 'colbert'
        tokenizer, model_col = load_model(args)
        if args.distill_col_path is not None:
            saved_state = load_states_from_checkpoint(args.distill_col_path)
            model_col.load_state_dict(saved_state.model_dict, strict=False)

    ## ce
    if args.distill_ce:
        args.model_type = 'cross_encoder'
        tokenizer, model_ce = load_model(args)
        if args.distill_ce_path is not None:
            saved_state = load_states_from_checkpoint(args.distill_ce_path)
            model_ce.load_state_dict(saved_state.model_dict, strict=False)

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
    global_step = train(args, model_de, model_db, model_col, model_ce, tokenizer)
    logger.info(" global_step = %s", global_step)


if __name__ == "__main__":
    main()
