EXP_NAME=run_de_ms_MASTER_all_1   # de means dual encoder.
DATA_DIR=/kun_data/marco/
OUT_DIR=output/$EXP_NAME
TB_DIR=tensorboard_log/$EXP_NAME    # tensorboard log path

for epoch in 80000
do

# Fine-tune with BM25 negatives using CKPT from epoch as initialization
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=69539 \
./MS/run_de_model.py \
--model_type=/kun_data/Austerlitz/MT5_Gen/ckpt/SIMLM/checkpoint-$epoch \
--origin_data_dir=$DATA_DIR/train_stage1.tsv \
--origin_data_dir_dev=$DATA_DIR/dev.query.txt \
--passage_path=/kun_data/marco \
--dataset=MS-MARCO \
--max_seq_length=128 --per_gpu_train_batch_size=8 --gradient_accumulation_steps=1 \
--learning_rate=5e-6 --output_dir $OUT_DIR \
--warmup_steps 1000 --logging_steps 200 --save_steps 5000 --max_steps 30000 \
--log_dir $TB_DIR \
--number_neg 31 --fp16

# Evaluation 3h totally
for CKPT_NUM in 20000 25000 30000
do
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=19539 \
./MS/inference_de.py \
--model_type=bert-base-uncased \
--eval_model_dir=$OUT_DIR/checkpoint-$CKPT_NUM \
--output_dir=$OUT_DIR/$CKPT_NUM \
--train_qa_path=/kun_data/marco/train.query.txt \
--test_qa_path=/kun_data/marco/dev.query.txt \
--max_seq_length=128 --per_gpu_eval_batch_size=1024 \
--passage_path=/kun_data/marco \
--dataset=MS-MARCO \
--fp16
done

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=19539 \
./MS/inference_de.py \
--model_type=bert-base-uncased \
--eval_model_dir=$OUT_DIR/checkpoint-25000 \
--output_dir=$OUT_DIR/25000 \
--train_qa_path=/kun_data/marco/train.query.txt \
--test_qa_path=/kun_data/marco/dev.query.txt \
--max_seq_length=128 --per_gpu_eval_batch_size=1024 \
--passage_path=/kun_data/marco \
--dataset=MS-MARCO \
--fp16 --write_hardneg=True

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=69539 \
./MS/run_de_model.py \
--model_type=/kun_data/Austerlitz/MT5_Gen/ckpt/SIMLM/checkpoint-$epoch \
--origin_data_dir=$OUT_DIR/25000/train_ce_hardneg.tsv \
--origin_data_dir_dev=$DATA_DIR/dev.query.txt \
--passage_path=/kun_data/marco \
--dataset=MS-MARCO \
--max_seq_length=128 --per_gpu_train_batch_size=8 --gradient_accumulation_steps=1 \
--learning_rate=5e-6 --output_dir $OUT_DIR \
--warmup_steps 1000 --logging_steps 200 --save_steps 5000 --max_steps 40000 \
--log_dir $TB_DIR \
--number_neg 31 --fp16

# Evaluation 3h totally
for CKPT_NUM in 25000 30000 35000 40000
do
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=19539 \
./MS/inference_de.py \
--model_type=bert-base-uncased \
--eval_model_dir=$OUT_DIR/checkpoint-$CKPT_NUM \
--output_dir=$OUT_DIR/$CKPT_NUM \
--train_qa_path=/kun_data/marco/train.query.txt \
--test_qa_path=/kun_data/marco/dev.query.txt \
--max_seq_length=128 --per_gpu_eval_batch_size=1024 \
--passage_path=/kun_data/marco \
--dataset=MS-MARCO \
--fp16
done

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=19539 \
./MS/inference_de.py \
--model_type=bert-base-uncased \
--eval_model_dir=$OUT_DIR/checkpoint-40000 \
--output_dir=$OUT_DIR/40000 \
--train_qa_path=/kun_data/marco/train.query.txt \
--test_qa_path=/kun_data/marco/dev.query.txt \
--max_seq_length=128 --per_gpu_eval_batch_size=1024 \
--passage_path=/kun_data/marco \
--dataset=MS-MARCO \
--fp16 --write_hardneg=True

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=19539 \
./MS/run_ce_model_ele.py \
--model_type=google/electra-base-discriminator --max_seq_length=192 \
--per_gpu_train_batch_size=1 --gradient_accumulation_steps=8 \
--number_neg=63 --learning_rate=1e-5 \
--output_dir=$OUT_DIR \
--origin_data_dir=$OUT_DIR/32000/train_ce_hardneg.tsv \
--origin_data_dir_dev=$DATA_DIR/dev.query.txt \
--passage_path=/kun_data/marco \
--dataset=MS-MARCO \
--warmup_steps=2000 --logging_steps=500 --save_steps=8000 \
--max_steps=33000 --log_dir=$TB_DIR

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9539 MS/co_training_model_ele.py \
    --model_type=/kun_data/Austerlitz/MT5_Gen/ckpt/SIMLM/checkpoint-$epoch \
    --max_seq_length=128 --per_gpu_train_batch_size=16 --gradient_accumulation_steps=4 \
    --number_neg=41 --learning_rate=5e-6 \
    --reranker_model_type=google/electra-base-discriminator \
    --reranker_model_path=output/run_de_ms_MASTER_all_1/checkpoint-ce-24000 \
    --output_dir=$OUT_DIR \
    --log_dir=$TB_DIR \
    --origin_data_dir=output/run_de_ms_MASTER_all_1/32000/train_ce_hardneg.tsv \
    --origin_data_dir_dev=$DATA_DIR/dev.query.txt \
    --passage_path=/kun_data/marco \
    --dataset=MS-MARCO \
    --warmup_steps 5000 --logging_steps 500 --save_steps 5000 --max_steps 25000 \
    --gradient_checkpointing --normal_loss \
    --temperature_normal=1

for CKPT_NUM in 10000 15000 20000 25000
do
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=19539 \
./MS/inference_de.py \
--model_type=bert-base-uncased \
--eval_model_dir=$OUT_DIR/checkpoint-$CKPT_NUM \
--output_dir=$OUT_DIR/$CKPT_NUM \
--train_qa_path=/kun_data/marco/train.query.txt \
--test_qa_path=/kun_data/marco/dev.query.txt \
--max_seq_length=128 --per_gpu_eval_batch_size=1024 \
--passage_path=/kun_data/marco \
--dataset=MS-MARCO \
--fp16
done

done
