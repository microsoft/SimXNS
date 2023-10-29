#!/bin/bash

MODEL_TYPE=dual_encoder
DATASET=$1
MASTER_PORT=$2
MODEL_PATH=$3
FIRST_STEPS=$4
EVAL_STEPS=$5
MAX_STEPS=$6
MAX_DOC_LENGTH=$7
MAX_QUERY_LENGTH=$8

BASE_DATA_DIR=data_preprocess/${DATASET}
BASE_OUTPUT_DIR=model_output/${MODEL_TYPE}
PASSAGE_PATH=${BASE_DATA_DIR}/psgs_w100.tsv
TEST_FILE=${BASE_DATA_DIR}/${DATASET}-test.qa.csv

TOKENIZER_NAME=bert-base-uncased
PRETRAINED_MODEL_NAME=master
#PRETRAINED_MODEL_NAME=Luyu/co-condenser-marco
EVAL_BATCHSIZE=256

##remember to change 'de-checkpoint-${CKPT_NAME}' to 'db-checkpoint-${CKPT_NAME}' when evaluating distillation result
####################################### Multi Evaluation ########################################
echo "****************begin Evaluate****************"
cd ..
for ITER in $(seq 0 $((($MAX_STEPS - $FIRST_STEPS)/ $EVAL_STEPS)))
do
CKPT_NAME=$(($FIRST_STEPS + $ITER * $EVAL_STEPS))
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=${MASTER_PORT} inference_de.py \
  --model_type=${MODEL_TYPE} \
  --tokenizer_name=${TOKENIZER_NAME} \
  --pretrained_model_name=${PRETRAINED_MODEL_NAME} \
  --eval_model_dir=${BASE_OUTPUT_DIR}/models/${MODEL_PATH}/de-checkpoint-${CKPT_NAME} \
  --output_dir=${BASE_OUTPUT_DIR}/eval/${MODEL_PATH}/de-checkpoint-${CKPT_NAME} \
  --test_file=${TEST_FILE} \
  --passage_path=${PASSAGE_PATH} \
  --max_doc_length=${MAX_DOC_LENGTH} --max_query_length=${MAX_QUERY_LENGTH} \
  --per_gpu_eval_batch_size $EVAL_BATCHSIZE
done

echo "****************End Evaluate****************"
