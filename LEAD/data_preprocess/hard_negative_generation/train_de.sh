#!/bin/bash

MODEL_TYPE=dual_encoder
DATASET=$1
MASTER_PORT=$2
MAX_DOC_LENGTH=$3
MAX_QUERY_LENGTH=$4
TRAIN_BATCH_SIZE=$5
NUM_NEGATIVES=$6

BASE_DATA_DIR=data_preprocess/${DATASET}
BASE_OUTPUT_DIR=model_output/${MODEL_TYPE}
TRAIN_FILE=${BASE_DATA_DIR}/biencoder-${DATASET}-train.json
PASSAGE_PATH=${BASE_DATA_DIR}/psgs_w100.tsv

TOKENIZER_NAME=bert-base-uncased
PRETRAINED_MODEL_NAME=Luyu/co-condenser-marco
MAX_STEPS=20000
LOGGING_STEPS=10
SAVE_STEPS=1000
LR=2e-5
GRADIENT_ACCUMULATION_STEPS=1
######################################## Training ########################################
echo "****************begin Train****************"
cd ../
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=${MASTER_PORT} \
    ./run_single_model.py \
    --model_type=${MODEL_TYPE} \
    --pretrained_model_name=${PRETRAINED_MODEL_NAME} \
    --tokenizer_name=${TOKENIZER_NAME} \
    --dataset=${DATASET} \
    --train_file=${TRAIN_FILE} \
    --passage_path=${PASSAGE_PATH} \
    --max_doc_length=${MAX_DOC_LENGTH} --max_query_length=${MAX_QUERY_LENGTH} \
    --per_gpu_train_batch_size=${TRAIN_BATCH_SIZE} --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate=${LR}  --num_hard_negatives ${NUM_NEGATIVES} \
    --share_weight \
    --logging_steps ${LOGGING_STEPS} --save_steps ${SAVE_STEPS} --max_steps ${MAX_STEPS} --seed 888 \
    --output_dir=${BASE_OUTPUT_DIR}/models

echo "****************End Train****************"
