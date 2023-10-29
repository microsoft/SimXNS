#!/bin/bash

MODEL_TYPE=distilbert
DATASET=$1
MASTER_PORT=$2
DISTILL_LAYER_NUM=$3
MAX_DOC_LENGTH=$4
MAX_QUERY_LENGTH=$5
TRAIN_BATCH_SIZE=$6
NUM_NEGATIVES=$7
WARM_RATIO=$8
DE_MODEL_PATH=$9
DB_MODEL_PATH=${10}


BASE_DATA_DIR=data_preprocess/${DATASET}
BASE_OUTPUT_DIR=model_output/${MODEL_TYPE}
TRAIN_FILE=${BASE_DATA_DIR}/biencoder-${DATASET}-train-hard.json
PASSAGE_PATH=${BASE_DATA_DIR}/psgs_w100.tsv
DISTILL_DE_PATH=model_output/dual_encoder/models/${DE_MODEL_PATH}                         # The initialized parameter of ce
DISTILL_DB_PATH=model_output/distilbert/models/${DB_MODEL_PATH}                       # The initialized parameter of db


TOKENIZER_NAME=bert-base-uncased
PRETRAINED_MODEL_NAME=Luyu/co-condenser-marco
DISTILL_PARA_DB=1                   # The distillation parameter of loss_db
DISTILL_PARA_DE=1                   # The distillation parameter of loss_de
DISTILL_PARA_DE_DB_DIS=1
DISTILL_PARA_DE_DB_LAYER_SCORE=1
GRADIENT_ACCUMULATION_STEPS=1
SAVE_STEPS=10
TEMPERATURE=1
LAYER_TEMPERATURE=10
MAX_STEPS=50000
LOGGING_STEPS=10
LR=5e-5

cd ..
######################################## Training ########################################
echo "****************begin Train****************"
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=${MASTER_PORT} \
    ./run_LEAD.py \
    --model_type=${MODEL_TYPE} \
    --pretrained_model_name=${PRETRAINED_MODEL_NAME} \
    --tokenizer_name=${TOKENIZER_NAME} \
    --train_file=${TRAIN_FILE} \
    --passage_path=${PASSAGE_PATH} \
    --dataset=${DATASET} \
    --max_doc_length=${MAX_DOC_LENGTH} --max_query_length=${MAX_QUERY_LENGTH} \
    --per_gpu_train_batch_size=${TRAIN_BATCH_SIZE} --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate=${LR}  --num_hard_negatives ${NUM_NEGATIVES} \
    --share_weight \
    --logging_steps ${LOGGING_STEPS} --save_steps ${SAVE_STEPS} --max_steps ${MAX_STEPS} --seed 888 \
    --temperature ${TEMPERATURE} \
    --layer_temperature ${LAYER_TEMPERATURE} \
    --output_dir=${BASE_OUTPUT_DIR}/models \
    --distill_para_de=${DISTILL_PARA_DE} \
    --distill_para_db=${DISTILL_PARA_DB} \
    --distill_para_de_db_dis=${DISTILL_PARA_DE_DB_DIS} \
    --distill_para_de_db_layer_score=${DISTILL_PARA_DE_DB_LAYER_SCORE} \
    --distill_de --train_de \
    --distill_de_path=${DISTILL_DE_PATH} \
    --distill_db --train_db \
    --distill_db_path=${DISTILL_DB_PATH} \
    --distill_de_db_layer_score \
    --disitll_layer_num=${DISTILL_LAYER_NUM} \
    --layer_selection_random \
    --layer_score_reweight \
    --warm_up_ratio=${WARM_RATIO}

echo "****************End Train****************"