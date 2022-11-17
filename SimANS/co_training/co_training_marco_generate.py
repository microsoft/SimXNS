import sys

sys.path += ['../']
import argparse
import logging
import os
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import torch.distributed as dist
from torch import nn
from model.models import BiBertEncoder
from co_training_generate import RenewTools
import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    BertTokenizer,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
from utils.util import (
    set_seed,
    is_first_worker,
)
from utils.dpr_utils import (
    load_states_from_checkpoint,
    CheckpointState
)
import collections

studentBatch = collections.namedtuple(
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



def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
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
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
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
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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
        "--distill_loss",
        default=False,
        action="store_true",
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
    parser.add_argument(
        "--fix_embedding",
        default=False,
        action="store_true",
        help="use single or re-warmup",
    )
    parser.add_argument(
        "--continue_train",
        default=False,
        action="store_true",
        help="use single or re-warmup",
    )
    parser.add_argument(
        "--adv_loss_alpha",
        default=0.3,
        type=float,
        help="use single or re-warmup",
    )

    parser.add_argument("--adv_lambda", default=0., type=float)
    parser.add_argument("--adv_steps", default=3, type=int)
    # ----------------- End of Doc Ranking HyperParam ------------------
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--test_qa_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--train_qa_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--dev_qa_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--passage_path", type=str, default="", help="For distant debugging.")
    parser.add_argument("--iteration_step", default=80, type=int)
    parser.add_argument("--iteration_reranker_step", default=40, type=int)
    parser.add_argument("--temperature_distill", default=3, type=float)

    parser.add_argument("--scale_simmila", default=False, action="store_true")
    parser.add_argument("--teacher_learning_rate", default=0, type=float)
    parser.add_argument("--load_cache", default=False, action="store_true")
    parser.add_argument("--ann_dir", type=str, default="", help="For distant debugging.")

    parser.add_argument("--global_step", type=int, default=0, help="For distant debugging.")
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
    model = BiBertEncoder(args)
    if args.model_name_or_path is not None:
        saved_state = load_states_from_checkpoint(args.model_name_or_path)
        model.load_state_dict(saved_state.model_dict, strict=False)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )
    return tokenizer, model


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model


def get_new_dataset(args, model, global_step, renew_tools):
    model_path = os.path.join(args.output_dir, 'checkpoint-' + str(global_step))
    saved_state = load_states_from_checkpoint(model_path)
    model.load_state_dict(saved_state.model_dict, strict=False)
    model_to_load = get_model_obj(model)
    model_to_load.load_state_dict(saved_state.model_dict)
    model.eval()
    # model.to(args.device)
    logger.info(" model_path = %s", model_path)
    with torch.no_grad():
        passage_embedding, passage_embedding_id = renew_tools.get_passage_embedding(args, model)
        torch.distributed.barrier()
        if is_first_worker():
            train_q, train_q_embed, train_q_embed2id = renew_tools.get_question_embedding(args,
                                                                                          model, args.train_qa_path,
                                                                                          mode='train')
            dev_q, dev_q_embed, dev_q_embed2id = renew_tools.get_question_embedding(args,
                                                                                    model, args.dev_qa_path, mode='dev')

            gpu_index_flat, passage_embedding2id = renew_tools.get_new_faiss_index(args, passage_embedding,
                                                                                   passage_embedding_id)
            data_dir = os.path.abspath(os.path.dirname(args.train_qa_path))
            ground_truth_path = os.path.join(data_dir, 'qrels.train.tsv')
            renew_tools.get_question_topk(train_q, train_q_embed, train_q_embed2id, ground_truth_path,
                                          gpu_index_flat, passage_embedding2id,
                                          mode='train', step_num=global_step)
            data_dir = os.path.abspath(os.path.dirname(args.dev_qa_path))
            ground_truth_path = os.path.join(data_dir, 'qrels.dev.tsv')
            renew_tools.get_question_topk(dev_q, dev_q_embed, dev_q_embed2id, ground_truth_path,
                                          gpu_index_flat, passage_embedding2id,
                                          mode='dev', step_num=global_step)


def main():
    args = get_arguments()
    set_env(args)
    basic_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(basic_format)
    log_path = os.path.join(args.output_dir, 'log.txt')
    # sh = logging.StreamHandler()
    handler = logging.FileHandler(log_path, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.addHandler(sh)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    print(logger)
    tokenizer, model = load_model(args)
    # load passage
    temp_slice_dir = os.path.join(args.ann_dir, 'temp')
    # temp_slice_dir='/mnt/data/denseIR/ckpt/run_de_model_ict_ernie_marco/80k'
    passages_title_path = os.path.join(args.passage_path, 'para.title.txt')
    passages_ctx_path = os.path.join(args.passage_path, 'para.txt')
    renew_tools = RenewTools(passages_title_path=passages_title_path,
                             passages_ctx_path=passages_ctx_path, tokenizer=tokenizer,
                             output_dir=args.ann_dir, temp_dir=temp_slice_dir)
    dist.barrier()
    global_step = args.global_step
    if global_step > args.max_steps:
        pass
    else:
        get_new_dataset(args, model, global_step, renew_tools)
    # dist.barrier()
    logger.info(" global_step = %s", global_step)

    # if args.local_rank != -1:
    #     dist.barrier()


if __name__ == "__main__":
    main()
