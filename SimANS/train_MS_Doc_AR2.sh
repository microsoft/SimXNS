EXP_NAME=co_training_MS_MARCO_Doc_SimANS
Iteration_step=5000
Iteration_reranker_step=1000
MAX_STEPS=40000

# for global_step in `seq 0 2000 $MAX_STEPS`; do echo $global_step; done;
for global_step in `seq 0 $Iteration_step $MAX_STEPS`;
do
    python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9539 Doc_training/co_training_doc_train.py \
    --model_type=ckpt/MS-Doc/adore-star \
    --model_name_or_path=ckpt/MS-Doc/checkpoint-20000 \
    --max_seq_length=512 --per_gpu_train_batch_size=32 --gradient_accumulation_steps=1 \
    --number_neg=15 --learning_rate=5e-6 \
    --teacher_model_type=roberta-base \
    --teacher_model_path=ckpt/MS-Doc/checkpoint-reranker20000 \
    --teacher_learning_rate=1e-6 \
    --output_dir=ckpt/$EXP_NAME \
    --log_dir=tensorboard/logs/$EXP_NAME \
    --origin_data_dir=data/MS-Doc/train_ce_0.tsv \
    --train_qa_path=data/MS-Doc/msmarco-doctrain-queries.tsv \
    --passage_path=data/MS-Doc \
    --logging_steps=100 --save_steps=5000 --max_steps=$MAX_STEPS \
    --gradient_checkpointing --distill_loss \
    --iteration_step=$Iteration_step \
    --iteration_reranker_step=$Iteration_reranker_step \
    --temperature_distill=1 --ann_dir=ckpt/$EXP_NAME/temp --adv_lambda 1 --global_step=$global_step

    g_global_step=`expr $global_step + $Iteration_step`
    python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9539 Doc_training/co_training_doc_generate.py \
    --model_type=ckpt/MS-Doc/adore-star \
    --max_seq_length=512 \
    --output_dir=ckpt/$EXP_NAME \
    --log_dir=tensorboard/logs/$EXP_NAME \
    --train_qa_path=data/MS-Doc/msmarco-doctrain-queries.tsv \
    --dev_qa_path=data/MS-Doc/msmarco-docdev-queries.tsv \
    --passage_path=data/MS-Doc \
    --max_steps=$MAX_STEPS \
    --gradient_checkpointing \
    --ann_dir=ckpt/$EXP_NAME/temp --global_step=$g_global_step
done
