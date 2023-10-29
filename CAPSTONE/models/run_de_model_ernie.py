import argparse
import json
import logging
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
import torch.nn.functional as F
from transformers import (
    BertTokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import collections
from torch.utils.tensorboard import SummaryWriter
import sys
pyfile_path = os.path.abspath(__file__) 
pyfile_dir = os.path.dirname(os.path.dirname(pyfile_path)) # equals to the path '../'
sys.path.append(pyfile_dir)

from models.modules import BiBertEncoder, calculate_dual_encoder_cont_loss
from utils.util import (
    set_seed,
    is_first_worker,
    TraditionDataset,
    get_optimizer,
    sum_main
)
from utils.dpr_utils import (
    load_states_from_checkpoint,
    load_states_from_checkpoint_ict,
    get_model_obj,
    _save_checkpoint,
    _load_saved_state
)

from co_training_generate_new_train_wiki import RenewTools

# logger = logging.getLogger(__name__)
logger = logging.getLogger("__main__")
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


def train(args, model, tokenizer):
    """ Train the model """
    logger.info("Training/evaluation parameters %s", args)
    tb_writer = SummaryWriter(log_dir=args.log_dir) if is_first_worker() else None
    optimizer = get_optimizer(args.optimizer, model, weight_decay=args.weight_decay, lr=args.learning_rate, adam_epsilon=args.adam_epsilon)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        from apex.parallel import DistributedDataParallel as DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Max steps = %d", args.max_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.per_device_train_batch_size * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    tr_loss = 0.0
    model.zero_grad()
    model.train()
    iter_count = 0
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )
    global_step = 0
    train_dataset = TraditionDataset(args.train_file, tokenizer, 
                                    num_hard_negatives=args.number_neg,
                                    is_training=True,
                                    passages_path=args.passage_path, 
                                    max_seq_length=args.max_seq_length, 
                                    max_query_length=args.max_query_length, 
                                    shuffle_positives=args.shuffle_positives,
                                    expand_doc_w_query=args.expand_doc_w_query, 
                                    expand_corpus=args.expand_corpus, 
                                    top_k_query=args.top_k_query, 
                                    append=args.append, 
                                    gold_query_prob=args.gold_query_prob, 
                                    select_generated_query=args.select_generated_query,
                                    metric=args.metric,
                                    query_path=args.query_path, 
                                    delimiter=args.delimiter, 
                                    n_sample=args.n_sample,
                                    total_part=args.total_part, 
                                    filter_threshold=args.filter_threshold)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  collate_fn=TraditionDataset.get_collate_fn(args),
                                  batch_size=args.per_device_train_batch_size, num_workers=5 )
    dev_dataset = TraditionDataset( args.validation_file, tokenizer,
                                    num_hard_negatives = args.number_neg,
                                    is_training=False,
                                    passages_path=args.passage_path, 
                                    max_seq_length=args.max_seq_length, 
                                    max_query_length=args.max_query_length,
                                    shuffle_positives=False,
                                    expand_doc_w_query=args.expand_doc_w_query, 
                                    expand_corpus=args.expand_corpus, 
                                    top_k_query=args.top_k_query, 
                                    append=args.append, 
                                    gold_query_prob=args.gold_query_prob, 
                                    select_generated_query=args.select_generated_query,
                                    metric=args.metric,
                                    query_path=args.query_path, 
                                    delimiter=args.delimiter, 
                                    n_sample=args.n_sample,
                                    total_part=args.total_part,
                                    filter_threshold=args.filter_threshold,
                                    psg_id_2_query_dict=train_dataset.psg_id_2_query_dict)
    dev_sampler = SequentialSampler(dev_dataset) if args.local_rank == -1 else DistributedSampler(dev_dataset, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler,
                                collate_fn=TraditionDataset.get_collate_fn(args),
                                batch_size=args.per_device_eval_batch_size, num_workers=5, shuffle=False)

    validate_loss, validate_acc = evaluate_dev(args, model, tokenizer, dev_dataloader)
    model.train()
    gradual_step = 0
    while global_step < args.max_steps:
        iter_count += 1
        if args.num_train_epochs >0 and iter_count > args.num_train_epochs:
            break
        # if args.n_gpu>1:
        # shuffle the data for each epoch
        if  'gradual' in args.select_generated_query:
            gradual_step += 1
            logger.info(f'gradual_step: {gradual_step}.')
            train_sampler.set_epoch(iter_count)
            status = train_dataset.reset_select_generated_query(global_step, args.max_steps, args.total_part, args.select_generated_query)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch_retriever = batch['retriever']
            inputs_retriever = {"query_ids": batch_retriever[0].long().to(args.device),
                                "attention_mask_q": batch_retriever[1].long().to(args.device),
                                "input_ids_a": batch_retriever[2].long().to(args.device),
                                "attention_mask_a": batch_retriever[3].long().to(args.device)}
            local_positive_idxs = batch_retriever[4]
            local_q_vector, local_ctx_vectors = model(**inputs_retriever)

            loss, is_correct = calculate_dual_encoder_cont_loss(args.local_rank, local_q_vector, local_ctx_vectors, local_positive_idxs)
            loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

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
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["train_learning_rate"] = learning_rate_scalar
                    logs["train_loss"] = loss_scalar
                    tr_loss = 0
                    if is_first_worker():
                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        logger.info(json.dumps({**logs, **{"step": global_step}}))

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    validate_loss, validate_acc = evaluate_dev(args, model, tokenizer, dev_dataloader)
                    model.train()
                    if is_first_worker():
                        if global_step >= args.max_steps-4*args.save_steps:
                            _save_checkpoint(args, model, optimizer, scheduler, global_step)
                        tb_writer.add_scalar("dev_nll_loss", validate_loss, global_step)
                        tb_writer.add_scalar("dev_acc", validate_acc, global_step)
                if global_step >= args.max_steps:
                    break
            if  'gradual' in args.select_generated_query:
                new_status = train_dataset.reset_select_generated_query(global_step, args.max_steps, args.total_part, args.select_generated_query)
                if new_status!= status:
                    status = new_status # re-initialize the dataloader
                    break
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        tb_writer.close()
    return global_step




def evaluate(model, tokenizer, args):
    logger.info("Training/evaluation parameters %s", args)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    model.eval()
    dev_dataset = TraditionDataset( args.validation_file, tokenizer,
                                    num_hard_negatives = args.number_neg,
                                    is_training=False,
                                    passages_path=args.passage_path, 
                                    max_seq_length=args.max_seq_length, 
                                    max_query_length=args.max_query_length,
                                    shuffle_positives=False,
                                    expand_doc_w_query=args.expand_doc_w_query, 
                                    expand_corpus=args.expand_corpus, 
                                    top_k_query=args.top_k_query, 
                                    append=args.append, 
                                    gold_query_prob=args.gold_query_prob, 
                                    select_generated_query=args.select_generated_query,
                                    metric=args.metric,
                                    query_path=args.query_path, 
                                    delimiter=args.delimiter, 
                                    n_sample=args.n_sample,
                                    total_part=args.total_part, 
                                    filter_threshold=args.filter_threshold)
    dev_sampler = SequentialSampler(dev_dataset) if args.local_rank == -1 else DistributedSampler(dev_dataset, shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler,
                                collate_fn=TraditionDataset.get_collate_fn(args),
                                batch_size=args.per_device_eval_batch_size, num_workers=5, shuffle=False)

    validate_loss, validate_acc = evaluate_dev(args, model, tokenizer, dev_dataloader)
    return validate_loss, validate_acc


def evaluate_dev(args, model, tokenizer, dev_dataloader):
    """
    Evaluate the model on the validation set.
    """
    if len(dev_dataloader.dataset) ==0:
        return 0, 0
    correct_predictions_count_all = 0
    example_num = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dev_dataloader):
            batch_retriever = batch['retriever']
            inputs_retriever = {"query_ids": batch_retriever[0].long().to(args.device),
                                "attention_mask_q": batch_retriever[1].long().to(args.device),
                                "input_ids_a": batch_retriever[2].long().to(args.device),
                                "attention_mask_a": batch_retriever[3].long().to(args.device)}
            local_q_vector, local_ctx_vectors = model(**inputs_retriever)
            question_num = local_q_vector.size(0)
            retriever_local_ctx_vectors = local_ctx_vectors.reshape(question_num, local_ctx_vectors.size(0) // question_num, -1)

            relevance_logits = torch.einsum("bh,bdh->bd", [local_q_vector, retriever_local_ctx_vectors])
            relevance_target = torch.zeros(relevance_logits.size(0), dtype=torch.long).to(args.device)
            loss_fct = torch.nn.CrossEntropyLoss()
            relative_loss = loss_fct(relevance_logits, relevance_target)
            batch_size = relevance_target.shape[0]
            total_loss += relative_loss*batch_size
            max_score, max_idxs = torch.max(relevance_logits, 1)
            correct_predictions_count = (max_idxs == 0).sum()
            correct_predictions_count_all+=correct_predictions_count
            example_num += batch_size
    example_num = torch.tensor(1).to(args.device)*example_num
    total_loss = torch.tensor(1).to(args.device)*total_loss
    correct_predictions_count_all = torch.tensor(1).to(args.device)*correct_predictions_count_all
    correct_predictions_count_all = sum_main(correct_predictions_count_all,args)
    example_num = sum_main(example_num,args)
    total_loss = sum_main(total_loss,args)
    ave_loss = (total_loss / example_num).item()

    correct_ratio = float(correct_predictions_count_all / example_num)
    logger.info('Validation: loss = %f. correct prediction ratio  %d/%d ~  %f', ave_loss,
                correct_predictions_count_all.item(),
                example_num.item(),
                correct_ratio
                )

    model.train()
    return ave_loss, correct_ratio

def predict(model, tokenizer, args):
    def get_new_dataset(args, model, global_step, renew_tools):
        model.eval()
        with torch.no_grad():
            
            K = renew_tools.compute_passage_embedding(args, model)
            torch.distributed.barrier()
            if is_first_worker():
                if args.evaluate_beir:
                    embedding_and_id_generator = renew_tools.get_passage_embedding(args, K)
                    question_embeddings, question_ids = renew_tools.get_question_embeddings(args, renew_tools.questions, model)
                    renew_tools.get_question_topk(None, None, question_embeddings, question_ids, None, False, None,
                                                  embedding_and_id_generator,
                                                  mode='test', step_num=global_step)   

                # evaluate the TREC-19, 20 test set.
                elif args.evaluate_trec:
                    assert args.test_qa_path is not None and args.load_cache
                    embedding_and_id_generator = renew_tools.get_passage_embedding(args, K)
                    test_q, test_a, test_q_embed, test_q_embed2id, test_q_ids, is_msmarco_data = renew_tools.get_question_embedding(
                        args, model, args.test_qa_path, mode='test')
                    renew_tools.get_question_topk(test_q,test_a,test_q_embed, test_q_embed2id, test_q_ids, is_msmarco_data, None,
                                                embedding_and_id_generator,
                                                mode='test', step_num=global_step, 
                                                evaluate_trec=True,
                                                query_positive_id_path=args.query_positive_id_path,
                                                prefix=args.prefix) 
                else: # ms-marco, nq, tq
                    if args.dev_qa_path is not None:
                        embedding_and_id_generator = renew_tools.get_passage_embedding(args, K)
                        dev_q, dev_a, dev_q_embed, dev_q_embed2id, dev_q_ids, is_msmarco_data = renew_tools.get_question_embedding(
                            args, model, args.dev_qa_path, mode='dev')
                        renew_tools.get_question_topk(dev_q, dev_a, dev_q_embed, dev_q_embed2id, dev_q_ids, is_msmarco_data, args.validation_file,
                                                      embedding_and_id_generator,
                                                      mode='dev', step_num=global_step)
                        args.load_cache = True

                    if args.train_qa_path is not None: 
                        embedding_and_id_generator = renew_tools.get_passage_embedding(args, K)
                        train_q, train_a, train_q_embed, train_q_embed2id, train_q_ids, is_msmarco_data = renew_tools.get_question_embedding(
                            args, model, args.train_qa_path, mode='train')
                        renew_tools.get_question_topk(train_q,train_a,train_q_embed, train_q_embed2id, train_q_ids, is_msmarco_data, args.train_file,
                                                    embedding_and_id_generator,
                                                    mode='train', step_num=global_step)

                    if args.test_qa_path is not None:
                        embedding_and_id_generator = renew_tools.get_passage_embedding(args, K)
                        test_q, test_a, test_q_embed, test_q_embed2id, test_q_ids, is_msmarco_data = renew_tools.get_question_embedding(
                            args, model, args.test_qa_path, mode='test')
                        renew_tools.get_question_topk(test_q,test_a,test_q_embed, test_q_embed2id, test_q_ids, is_msmarco_data, None,
                                                      embedding_and_id_generator,
                                                      mode='test', step_num=global_step)   


                     

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=args.fp16_opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
    model.eval()
    temp_slice_dir = os.path.join(args.output_dir, 'temp')
    renew_tools = RenewTools(args, passages_path=args.passage_path, tokenizer=tokenizer,
                             output_dir=args.output_dir, temp_dir=temp_slice_dir, 
                             expand_doc_w_query=args.expand_doc_w_query, 
                             expand_corpus=args.expand_corpus,
                             top_k_query=args.top_k_query,
                             append=args.append, 
                             query_path=args.query_path, 
                             delimiter=args.delimiter, 
                             evaluate_beir=args.evaluate_beir,
                             beir_data_path=args.beir_data_path
                             )
    
    if args.model_name_or_path is not None:
        global_step = int(args.model_name_or_path.strip().split('/checkpoint-')[1])
    else:
        global_step = 0
    get_new_dataset(args, model, global_step, renew_tools)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction.")
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
        "--model_name_or_path_ict",
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
        "--num_train_epochs",
        default=0,
        type=int,
        help="Number of epoch to train, if specified will use training data instead of ann",
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length",
        default=32,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )
    parser.add_argument(
        "--per_device_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for predition.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--optimizer",
        default="adamW",
        type=str,
        help="Optimizer - lamb or adamW",
    )
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument( "--max_steps", default=20000, type=int, 
                        help="If > 0: set total number of training steps to perform",
                        )
    parser.add_argument("--warmup_steps", default=2000, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")

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
            "--train_file",
            default=None,
            type=str,
        )
    parser.add_argument(
            "--validation_file",
            default=None,
            type=str,
        )
    

    parser.add_argument(
            "--fix_embedding",
            default=False,
            action="store_true",
            help="use single or re-warmup",
        )

    parser.add_argument(
            "--shuffle_positives",
            default=False,
            action="store_true",
            help="use single or re-warmup")

    parser.add_argument("--number_neg", type=int, default=1, help="number of negative docs for each query.")
    parser.add_argument("--delimiter", default='blank', type=str, choices=['blank', 'sep'])
    
    # ----------------- End of Doc Ranking HyperParam ------------------
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    # ------------------- Prediction HyperParam  -----------------------
    parser.add_argument("--test_qa_path", type=str, default=None, help="The path the test question-answer file.")
    parser.add_argument("--train_qa_path", type=str, default=None, help="The path the train question-answer file.")
    parser.add_argument("--dev_qa_path", type=str, default=None, help="The path the validation question-answer file.")
    parser.add_argument("--passage_path", type=str, default="", help="The path of the passage file.")
    parser.add_argument("--load_cache", default=False, action="store_true", help='Whether load the cached passage embeddings.')
    
    # ------------------- Doc2query HyperParam  -----------------------
    parser.add_argument("--expand_doc_w_query", default=False, action="store_true", help="whether expand documents with queries.")
    parser.add_argument("--expand_corpus", default=False, action="store_true", 
                        help="If true, during training, add one query to the doc; "
                             "During inference, each doc will have top_k_query views. Note you can only specify one mode.") 
    
    parser.add_argument("--top_k_query", default=1, type=int, help="The number of queries append or prepend to the document.")
    parser.add_argument("--append", default=False, action="store_true", help="whether append or prepend the queries to the doc.")
    parser.add_argument("--gold_query_prob", default=0, type=float, 
                        help="The prob of using gold query, 1 - gold_query_prob is the prob of using the generated query."
                             "(only effective when expand_corpus=True)")
    parser.add_argument("--select_generated_query", default='random', type=str,  
                        help="How to select the generated query. " 
                             "gold means using the gold query."
                             "first means selects the first query from the generated query list;"
                             "random means randomly selects a query from the generated query list;"
                             "top-k first computes the metric score between the gold and generated query and selects one from the top-k list;"
                             'k-th first computes the metric score between the gold and generated query and selects the k-th;'
                             'k-part first computes the metric score between the gold and generated query and selects the k-th part (The default total part is 10);'
                             "bottom-k first computes the metric score between the gold and generated query and selects one from the bottom-k list;"
                             "gradual-gold/gradual means gradually increase the relevance of the query used during training;"
                             "batch-uniform means select the i-th query equally (i.e. n 1-th, n 2-th, n 3-th ... );"
                             "batch-uniform-gold means select the i-th query equally (i.e. n gold, n 1-th, n 2-th, n 3-th ... );"
                             "(only effective when expand_corpus=True and gold_query_prob=0).")  
    parser.add_argument("--filter_threshold", default=1, type=float)
    parser.add_argument("--total_part", default=10, type=int)
    parser.add_argument("--metric", default='rouge-l', type=str, choices=['rouge-l', 'bleu', 'meteor'],
                        help="The metric used to compute the score between the gold and generated query. "
                             "(only effective when select_generated_query is not random.)")
    parser.add_argument("--query_path", default=None, type=str, help="Path of the generated queries.")

    # parameters only used in small version
    parser.add_argument('--n_sample', type=int, default=0, help='number of selected negative examples.')
    

    # parameters for evaluate trec test data
    parser.add_argument("--evaluate_trec",
                        action="store_true",
                        help="Whether to evaluate the trec test data, otherwise evaluate msmarco.")
    parser.add_argument("--query_positive_id_path", default=None, type=str, help="Path of the query_positive_id_path.")
    parser.add_argument("--prefix", default=None, type=str, help="")

    # parameters for evaluate trec test data
    parser.add_argument("--evaluate_beir",
                        action="store_true",
                        help="Whether to evaluate the beir datasets, otherwise evaluate msmarco.")
    parser.add_argument("--beir_data_path", default=None, type=str, help="path of a dataset in beir")


    args = parser.parse_args()
    if args.per_device_eval_batch_size<1:
        args.per_device_eval_batch_size = args.per_device_train_batch_size
    
    # check hyper-parameters
    if args.expand_doc_w_query + args.expand_corpus == 2:
        raise ValueError('Note you can only specify one mode.')
    
    if args.expand_corpus:
        if args.do_eval or args.do_train:
            assert args.top_k_query == 1

    if args.gold_query_prob>0:
        assert args.expand_corpus
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
        args.n_gpu = torch.distributed.get_world_size()
    args.device = device
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16
                   )

    # Set seed
    set_seed(args)

    # Create output directory if needed
    if is_first_worker() and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.local_rank != -1:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will create the output dir.


    basic_format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
    formatter = logging.Formatter(basic_format)
    log_path = os.path.join(args.output_dir, 'log.txt')
    # sh = logging.StreamHandler()
    handler = logging.FileHandler(log_path, 'a', 'utf-8')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.addHandler(sh)
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    print(logger)

def load_model(args, local_rank=None, device=None, model_name_or_path_ict=None, model_name_or_path=None, 
               model_type=None, fix_embedding=None):
    # Load pretrained model and tokenizer
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    tokenizer = AutoTokenizer.from_pretrained(model_type, do_lower_case=True)
    model = BiBertEncoder(args, model_type=model_type)

    if model_name_or_path_ict is not None:
        saved_state = load_states_from_checkpoint_ict(model_name_or_path_ict)
        model.load_state_dict(saved_state)
    if model_name_or_path is not None:
        saved_state = load_states_from_checkpoint(model_name_or_path)
        model.load_state_dict(saved_state.model_dict, strict=False)

    if fix_embedding:
        word_embedding = model.ctx_model.get_input_embeddings()
        word_embedding.requires_grad = False

    if local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)
    return tokenizer, model


def main():
    args = get_arguments()
    set_env(args)
    logger.info("args:\n%s", '\n'.join([f'    {arg}={getattr(args, arg)}'  for arg in vars(args)]))
    tokenizer, model = load_model(args, local_rank=args.local_rank, device=args.device, 
                                    model_name_or_path_ict=args.model_name_or_path_ict, 
                                    model_name_or_path=args.model_name_or_path, 
                                    model_type=args.model_type, fix_embedding=args.fix_embedding)
     
    if args.delimiter =='sep':
        args.delimiter = tokenizer.sep_token
    elif args.delimiter =='blank':
        args.delimiter = ' '
    else:
        raise ValueError()
    assert args.do_train + args.do_eval + args.do_predict ==1, 'You can only specify one mode.'

    if args.do_train:
        logger.info("*** Train ***")
        global_step = train(args, model, tokenizer)
        logger.info(" global_step = %s", global_step)

    if args.do_eval:
        logger.info("*** Evaluate ***") 
        evaluate(model, tokenizer, args)

    if args.do_predict:
        logger.info("*** Predict ***") 
        predict(model, tokenizer, args)

if __name__ == "__main__":
    main()
