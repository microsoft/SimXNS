#!/bin/bash

DATASET=$1
MASTER_PORT=$2
MODEL_PATH=$3
CKPT_NAME=$4
MAX_DOC_LENGTH=$5
MAX_QUERY_LENGTH=$6

MODEL_TYPE=dual_encoder
BASE_DATA_DIR=data_preprocess/${DATASET}
BASE_OUTPUT_DIR=model_output/${MODEL_TYPE}

PASSAGE_PATH=${BASE_DATA_DIR}/psgs_w100.tsv

if [ ${DATASET} == 'mspas' ];
then
  TRAIN_FILE=${BASE_DATA_DIR}/biencoder-mspas-train-full.json
else
  TRAIN_FILE=${BASE_DATA_DIR}/biencoder-${DATASET}-train.json
fi

TOKENIZER_NAME=bert-base-uncased
PRETRAINED_MODEL_NAME=Luyu/co-condenser-marco

EVAL_BATCHSIZE=256

####################################### Multi Evaluation ########################################
echo "****************begin Retrieve****************"
cd ..

python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=${MASTER_PORT} retrieve_hard_negative.py \
  --model_type=${MODEL_TYPE} \
  --dataset ${DATASET} \
  --tokenizer_name=${TOKENIZER_NAME} \
  --pretrained_model_name=${PRETRAINED_MODEL_NAME} \
  --eval_model_dir=${BASE_OUTPUT_DIR}/models/${MODEL_PATH}/de-checkpoint-${CKPT_NAME} \
  --output_dir=${BASE_DATA_DIR} \
  --train_file=${TRAIN_FILE} \
  --passage_path=${PASSAGE_PATH} \
  --max_query_length=${MAX_QUERY_LENGTH} \
  --max_doc_length=${MAX_DOC_LENGTH} \
  --per_gpu_eval_batch_size $EVAL_BATCHSIZE

echo "****************End Retrieve****************"