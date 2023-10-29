EXP_NAME=run_de_tq_MT5_Gen_hardneg_KD   # de means dual encoder.
DATA_DIR=/kun_data/DR/ARR/data/
OUT_DIR=output/$EXP_NAME
TB_DIR=tensorboard_log/$EXP_NAME    # tensorboard log path

for epoch in 80000
do
# Fine-tune with BM25 negatives using CKPT from epoch as initialization
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=69539 \
./wiki/run_de_model.py \
--model_type=/kun_data/Austerlitz/MT5_Gen/ckpt/Wiki/checkpoint-$epoch \
--origin_data_dir=$DATA_DIR/biencoder-trivia-train.json \
--origin_data_dir_dev=$DATA_DIR/biencoder-trivia-dev.json \
--max_seq_length=128 --per_gpu_train_batch_size=32 --gradient_accumulation_steps=1 \
--learning_rate=1e-5 --output_dir $OUT_DIR \
--warmup_steps 8000 --logging_steps 100 --save_steps 10000 --max_steps 80000 \
--log_dir $TB_DIR \
--number_neg 1 --fp16

# Evaluation 3h totally
for CKPT_NUM in 40000 50000 60000 70000 80000
do
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=19539 \
./wiki/inference_de.py \
--model_type=bert-base-uncased \
--eval_model_dir=$OUT_DIR/checkpoint-$CKPT_NUM \
--output_dir=$OUT_DIR/$CKPT_NUM \
--test_qa_path=$DATA_DIR/trivia-test.qa.csv \
--train_qa_path=$DATA_DIR/trivia-train.qa.csv \
--dev_qa_path=$DATA_DIR/trivia-dev.qa.csv \
--max_seq_length=128 --per_gpu_eval_batch_size=1024 \
--passage_path=$DATA_DIR/psgs_w100.tsv \
--fp16
done

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=19539 \
./wiki/inference_de.py \
--model_type=bert-base-uncased \
--eval_model_dir=$OUT_DIR/checkpoint-50000 \
--output_dir=$OUT_DIR/$CKPT_NUM \
--test_qa_path=$DATA_DIR/trivia-test.qa.csv \
--train_qa_path=$DATA_DIR/trivia-train.qa.csv \
--dev_qa_path=$DATA_DIR/trivia-dev.qa.csv \
--golden_train_qa_path=$DATA_DIR/biencoder-trivia-train.json \
--golden_dev_qa_path=$DATA_DIR/biencoder-trivia-dev.json \
--max_seq_length=128 --per_gpu_eval_batch_size=1024 \
--passage_path=$DATA_DIR/psgs_w100.tsv \
--fp16 --write_hardneg=True

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=69539 \
./wiki/run_de_model.py \
--model_type=/kun_data/Austerlitz/MT5_Gen/ckpt/Wiki/checkpoint-$epoch \
--origin_data_dir=$OUT_DIR/$CKPT_NUM/train_ce_hardneg.json \
--origin_data_dir_dev=$OUT_DIR/$CKPT_NUM/dev_ce_hardneg.json \
--max_seq_length=128 --per_gpu_train_batch_size=8 --gradient_accumulation_steps=1 \
--learning_rate=5e-6 --output_dir $OUT_DIR \
--warmup_steps 8000 --logging_steps 100 --save_steps 10000 --max_steps 80000 \
--log_dir $TB_DIR \
--number_neg 1 --fp16

# Evaluation 3h totally
for CKPT_NUM in 40000 50000 60000 70000 80000
do
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=19539 \
./wiki/inference_de.py \
--model_type=bert-base-uncased \
--eval_model_dir=$OUT_DIR/checkpoint-$CKPT_NUM \
--output_dir=$OUT_DIR/$CKPT_NUM \
--test_qa_path=$DATA_DIR/trivia-test.qa.csv \
--train_qa_path=$DATA_DIR/trivia-train.qa.csv \
--dev_qa_path=$DATA_DIR/trivia-dev.qa.csv \
--max_seq_length=128 --per_gpu_eval_batch_size=1024 \
--passage_path=$DATA_DIR/psgs_w100.tsv \
--fp16
done

done