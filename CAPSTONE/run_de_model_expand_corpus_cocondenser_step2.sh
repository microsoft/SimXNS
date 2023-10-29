echo "start de model warmup"

MAX_SEQ_LEN=144
MAX_Q_LEN=32
DATA_DIR=./reformed_marco
QUERY_PATH=./docTTTTTquery_full/doc2query_merge.tsv # each doc is paired with 80 generated quries.
CORPUS_NAME=corpus
CKPT_NUM=25000

MODEL_TYPE=Luyu/co-condenser-marco
MODEL=cocondenser

number_neg=31
total_part=4
select_generated_query=gradual
delimiter=sep
gold_query_prob=0

# path to the mined hard negatives
train_file=checkpoints_full_query/run_de_marco_144_32_t5_append_sep_gqp0_cocondenser_neg31_gradual_3_part/20000_corpus_top_k_query_10/train_20000_0_query.json
validation_file=checkpoints_full_query/run_de_marco_144_32_t5_append_sep_gqp0_cocondenser_neg31_gradual_3_part/20000_corpus_top_k_query_10/dev_20000_0_query.json

EXP_NAME=run_de_marco_${MAX_SEQ_LEN}_${MAX_Q_LEN}_t5_append_${delimiter}_gqp${gold_query_prob}_${MODEL}_neg${number_neg}_${select_generated_query}_${total_part}_part_step2_selfdata_shuffle_positives_F
OUT_DIR=checkpoints_full_query/$EXP_NAME
TB_DIR=tensorboard_log_full_query/$EXP_NAME    # tensorboard log path
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9589 \
    ./models/run_de_model_ernie.py \
    --model_type $MODEL_TYPE  \
    --train_file $train_file \
    --validation_file $validation_file \
    --max_seq_length $MAX_SEQ_LEN --max_query_length $MAX_Q_LEN \
    --per_device_train_batch_size=8 --gradient_accumulation_steps=1 \
    --number_neg $number_neg --learning_rate 5e-6 \
    --output_dir $OUT_DIR \
    --warmup_steps 2500 --logging_steps 100 --save_steps 1000 --max_steps 25000 \
    --log_dir $TB_DIR \
    --passage_path=$DATA_DIR/${CORPUS_NAME}.tsv \
    --fp16 --do_train \
    --expand_corpus --query_path $QUERY_PATH \
    --top_k_query 1   --append --delimiter $delimiter \
    --gold_query_prob $gold_query_prob --select_generated_query $select_generated_query --total_part $total_part    
    
# 2. Evaluate retriever and generate hard topk
echo "start de model inference"
for k in 10
do
    python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9536 \
        ./models/run_de_model_ernie.py \
        --model_type $MODEL_TYPE  \
        --model_name_or_path $OUT_DIR/checkpoint-$CKPT_NUM \
        --output_dir ${OUT_DIR}/${CKPT_NUM}_${CORPUS_NAME}_top_k_query_${k} \
        --train_file $DATA_DIR/biencoder-marco-train.json \
        --validation_file $DATA_DIR/biencoder-marco-dev.json \
        --train_qa_path $DATA_DIR/marco-train.qa.csv \
        --dev_qa_path $DATA_DIR/marco-dev.qa.csv \
        --max_seq_length $MAX_SEQ_LEN --max_query_length $MAX_Q_LEN --per_device_eval_batch_size 1024 \
        --passage_path $DATA_DIR/${CORPUS_NAME}.tsv \
        --fp16 --do_predict \
        --expand_corpus --query_path $QUERY_PATH \
        --top_k_query $k  --append --delimiter $delimiter

    # TREC 19 and 20 
    for year in 19 20
    do
        python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9535 \
            ./models/run_de_model_ernie.py \
            --model_type $MODEL_TYPE \
            --model_name_or_path $OUT_DIR/checkpoint-$CKPT_NUM \
            --output_dir ${OUT_DIR}/${CKPT_NUM}_${CORPUS_NAME}_top_k_query_${k} \
            --max_seq_length $MAX_SEQ_LEN --max_query_length $MAX_Q_LEN --per_device_eval_batch_size 512 \
            --passage_path $DATA_DIR/${CORPUS_NAME}.tsv \
            --evaluate_trec --prefix trec${year}_test \
            --test_qa_path ./trec_${year}/test20${year}.qa.csv \
            --query_positive_id_path ./trec_${year}/20${year}qrels-pass.txt \
            --fp16 --do_predict \
            --expand_corpus --query_path $QUERY_PATH \
            --top_k_query $k  --append --delimiter $delimiter \
            --load_cache 
    done
done

