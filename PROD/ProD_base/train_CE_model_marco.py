from os.path import join
import sys

sys.path += ['../']
import argparse
import glob
import json
import logging
import os
import random
import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
#  
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from model.models import HFBertEncoder,Reranker
from utils.lamb import Lamb
import random
import transformers
from transformers import (
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_processors as processors
import copy
from torch import nn
import pickle

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import pandas as pd

logger = logging.getLogger(__name__)
from utils.util import (
    set_seed,
    is_first_worker,
    TraditionDataset,GetTrainingDataProcessingFn,GetDevDataProcessingFn_ZH,dev_batcher
)
from utils.dpr_utils import (
    load_states_from_checkpoint,
    get_model_obj,
    CheckpointState,
    get_optimizer,
)
import collections
from utils.marco_until import (
    Rocketqa_v2Dataset
)


def get_bert_reranker_components(args,**kwargs):
    encoder = HFBertEncoder.init_encoder(
        args,
    )
    hidden_size = encoder.config.hidden_size
    reranker = Reranker(encoder, hidden_size)

    return  reranker


def train(args, model, tokenizer):
    """ Train the model """
    logger.info("Training/evaluation parameters %s", args)
    tb_writer = None
    if is_first_worker():
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)  # nll loss for query
    optimizer = get_optimizer(args, model, weight_decay=args.weight_decay)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        # from apex.parallel import DistributedDataParallel as DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False,
        )
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Max steps = %d", args.max_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    tr_loss = 0.0
    model.zero_grad()
    model.train()
    set_seed(args)  # Added here for reproductibility
    iter_count = 0

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )
    global_step = 0

    train_dataset = Rocketqa_v2Dataset(args.origin_data_dir, tokenizer, num_hard_negatives=args.number_neg,
                                       max_seq_length=args.max_seq_length, corpus_path=args.corpus_path)
    train_sample = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sample,
                                  collate_fn=Rocketqa_v2Dataset.get_collate_fn(args),
                                  batch_size=args.train_batch_size, num_workers=10)

    # load last ckpt
    if args.output_dir is not None:
        checkpoint_files = []
        if os.path.exists(args.output_dir):
            for item in os.scandir(args.output_dir):
                if item.is_file():
                    if "checkpoint" in item.path:
                        checkpoint_files.append(item.path)
            if len(checkpoint_files) != 0:
                checkpoint_files.sort(key=lambda f: int(f.split('checkpoint-')[1]), reverse=True)
                logger.info("***** load " + checkpoint_files[0] + " *****")
                saved_state = load_states_from_checkpoint(checkpoint_files[0])
                global_step = _load_saved_state(model, optimizer, scheduler, saved_state)
            else:
                logger.info("***** there are no checkpoint in" + args.output_dir + " *****")

    tr_contr_loss = 0
    tr_classfi_loss = 0
    # validate_rank = evaluate_dev(args, model, tokenizer)
    while global_step < args.max_steps:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        if args.num_epoch != 0 and iter_count > args.num_epoch:
            break
        # train_dataset = load_stream_dataset(args)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            loss, relative_loss, classfi_loss = fwd_pass(args, model, batch, optimizer)

            loss = loss/args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            
            tr_loss += loss.item()
            tr_contr_loss += relative_loss
            tr_classfi_loss += classfi_loss
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = tr_loss / args.logging_steps
                    classfi_loss_scalar = tr_classfi_loss/args.logging_steps
                    contr_loss_scalar = tr_contr_loss/args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logs["classfi_loss"] = classfi_loss_scalar
                    logs["contr_loss"] = contr_loss_scalar
                    tr_loss = 0
                    tr_classfi_loss = 0
                    tr_contr_loss = 0
                    if is_first_worker():
                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        logger.info(json.dumps({**logs, **{"step": global_step}}))

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # if global_step > 500000:
                    #     validate_rank = evaluate_dev(args, model, tokenizer)
                    # else:
                    #     validate_rank = evaluate_dev(args, model, tokenizer)
                    if is_first_worker():
                        _save_checkpoint(args, model, optimizer, scheduler, global_step)
                        # tb_writer.add_scalar("dev_nll_loss/dev_avg_rank", validate_rank[0], global_step)
                if global_step >= args.max_steps:
                    break
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        tb_writer.close()
    return global_step


def fwd_pass(args, model, batch,optimizer):
    batch_reranker = tuple(t.to(args.device) for t in batch['reranker'])
    inputs_reranker = {"input_ids": batch_reranker[0].long(), "attention_mask": batch_reranker[1].long()}
    output_reranker = model(**inputs_reranker)
    binary_logits,relevance_logits,_ = output_reranker
    
    relevance_target = torch.zeros(relevance_logits.size(0), dtype=torch.long).to(args.device)
    loss_fct = torch.nn.CrossEntropyLoss()
    relative_loss = loss_fct(relevance_logits,relevance_target)
    
    binary_logits = binary_logits.view(-1,2)
    classfi_target = torch.ones(binary_logits.size(0), dtype=torch.long).to(args.device)
    classfi_target[::2] = 0
    classfi_loss = loss_fct(binary_logits,classfi_target)

    loss = relative_loss + 0*classfi_loss

    return loss,relative_loss.item(),classfi_loss.item()

    
def sum_main(x, opt):
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x

def evaluate_dev(args, model, tokenizer):
    dev_dataset = TraditionDataset(args.origin_data_dir_dev,tokenizer,num_hard_negatives = args.number_neg,is_training=False)
    dev_sample = RandomSampler(dev_dataset) if args.local_rank == -1 else DistributedSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sample,
                        collate_fn=TraditionDataset.get_collate_fn(args),
                        batch_size=args.train_batch_size,num_workers=10,shuffle=False)
    correct_predictions_count_all = 0
    example_num = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader):
            batch_reranker = tuple(t.to(args.device) for t in batch['reranker'])
            inputs_reranker = {"input_ids": batch_reranker[0].long(), "attention_mask": batch_reranker[1].long()}
            output_reranker = model(**inputs_reranker)
            binary_logits,relevance_logits,_ =output_reranker
            relevance_target = torch.zeros(relevance_logits.size(0), dtype=torch.long).to(args.device)
            loss_fct = torch.nn.CrossEntropyLoss()
            relative_loss = loss_fct(relevance_logits,relevance_target)
            total_loss+=relative_loss
            max_score, max_idxs = torch.max(relevance_logits, 1)
            correct_predictions_count = (max_idxs == 0).sum()
            correct_predictions_count_all+=correct_predictions_count
            example_num +=  batch['reranker'][1].size(0)
    example_num = torch.tensor(1).to(relevance_logits)*example_num
    total_loss = torch.tensor(1).to(relevance_logits)*total_loss
    correct_predictions_count_all = torch.tensor(1).to(relevance_logits)*correct_predictions_count_all
    correct_predictions_count_all = sum_main(correct_predictions_count_all,args)
    example_num = sum_main(example_num,args)
    total_loss = sum_main(total_loss,args)
    total_loss = total_loss / i
    correct_ratio = float(correct_predictions_count_all / example_num)
    logger.info('NLL Validation: loss = %f. correct prediction ratio  %d/%d ~  %f', total_loss,
                correct_predictions_count_all.item(),
                example_num.item(),
                correct_ratio
                )

    model.train()
    return total_loss, correct_ratio



def _save_checkpoint(args, model, optimizer, scheduler, step: int) -> str:
    offset = step
    epoch = 0
    model_to_save = get_model_obj(model)
    cp = os.path.join(args.output_dir, 'checkpoint-' + str(offset))

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
    optimizer.load_state_dict(saved_state.optimizer_dict)
    scheduler.load_state_dict(saved_state.scheduler_dict)
    return step


def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=False,
        help="The input data dir. Should contain the cached passage and query files",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list:",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--num_epoch",
        default=0,
        type=int,
        help="Number of epoch to train, if specified will use training data instead of ann",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--num_hidden_layers",
        default=12,
        type=int,
        help="Number of floors of each tower in the two tower model",
    )
    parser.add_argument(
        "--corpus_path",
        default="/colab_space/fanshuai/KDmarco/coCondenser-marco/marco",
        type=str,
        help="passage path",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument("--triplet", default=False, action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )

    parser.add_argument(
        "--optimizer",
        default="adamW",
        type=str,
        help="Optimizer - lamb or adamW",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=300000,
        type=int,
        help="If > 0: set total number of training steps to perform",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
            "--gradient_checkpointing",
            default=False,
            action="store_true",
        )
    parser.add_argument(
            "--contr_loss",
            default=False,
            action="store_true",
        )
    parser.add_argument(
            "--origin_data_dir",
            default=None,
            type=str,
        )
    parser.add_argument(
            "--origin_data_dir_dev",
            default=None,
            type=str,
        )
    # ----------------- ANN HyperParam ------------------

    parser.add_argument(
        "--load_optimizer_scheduler",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--single_warmup",
        default=True,
        action="store_true",
        help="use single or re-warmup",
    )

    parser.add_argument("--adv_data_path",
                        type=str,
                        default=None,
                        help="adv_data_path", )

    parser.add_argument("--ann_data_path",
                        type=str,
                        default=None,
                        help="adv_data_path", )
    # ----------------- End of Doc Ranking HyperParam ------------------
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--number_neg", type=int, default=20, help="For distant debugging.")
    parser.add_argument("--number_easy_neg", type=int, default=0, help="For distant debugging.")
    args = parser.parse_args()

    return args


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
   
    tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True)
    model = get_bert_reranker_components(args)
        
    if args.model_name_or_path is not None:
        saved_state = load_states_from_checkpoint(args.model_name_or_path)
        model.load_state_dict(saved_state.model_dict,strict=False)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    return tokenizer, model


def main():
    args = get_arguments()
    set_env(args)
    tokenizer, model = load_model(args)

    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    log_path = os.path.join(args.output_dir, 'log.txt')
    handler = logging.FileHandler(log_path, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    print(logger)

    global_step = train(args, model, tokenizer)
    logger.info(" global_step = %s", global_step)

    if args.local_rank != -1:
        dist.barrier()
    


if __name__ == "__main__":
    main()
    logger.info(" train done")