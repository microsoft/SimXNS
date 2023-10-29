# PROD： Progressive Distillation for Dense Retrieval

This repo provides the code and models for our WWW2023 paper [***PROD: Progressive Distillation for Dense Retrieval***](https://arxiv.org/abs/2209.13335).
In the paper, we propose a novel distillation framework for dense retrieval, which consists of a teacher progressive distillation and a data progressive distillation to gradually improve the student.

<div align=center><img src="image\framework.jpg" width = "450" height = 300/></div>

PROD can be used to distill small dense retriever models with any number of layers n, such as n = 6, n=4, n=2. The student model can be initialized directly with the parameters of the first n layers of `ERNIE-2.0-BASE`. The current codebase supports complete PROD training and testing process, we will continue to improve this repo in the future.

## Dependencies：

- python>=3.6
- torch>=1.7.1
- datasets>=1.12.1
- transformers>=4.9.2 (Huggingface)
- faiss == 1.7.2
- huggingface-hub>=0.0.19
- pytrec-eval == 0.5

## Resources

### Data

All the datasets we use are widely-used and open-source. For convenience, you can quickly obtain the MARCO-Passage dataset and MARCO-Doc dataset for training through `MarcoPas_Data.sh` and `MarcoDoc_Data.sh`:

```shell
bash MarcoPas_Data.sh 
bash MarcoDoc_Data.sh
```

### Model

PROD method can greatly improve the retrieval performance of the 6-layer dense retriever model, which is comparable to the retrieval performance of most current 12-layer dense retriever models. We provide 6-layer dense retriever models training with PROD on  Natural Questions , MARCO-Passage, and MARCO-Document. 

**Natural Questions:**

| Model            | Recall@1 | Recall@5 | Recall@20 | Recall@100 |
| ---------------- | :--------: | :--------: | :---------: | :----------: |
| 6-layer          | 46.0     | 67.8     | 78.9      | 86.2       |
| [6-layer PROD](https://drive.google.com/file/d/1_ym1CnagaszfJ4bk7Ek9oRxTJpzcOJmj/view?usp=sharing) | **57.6** | **75.6** | **84.7**  | **89.6**   |

------

**MARCO-Passage:**

| Model            | MRR@10   | Recall@5 | Recall@20 | Recall@50 | Recall@1k |
| ---------------- | :--------: | :--------: | :---------: | :---------: | :---------: |
| 6-layer          | 31.7     | 47.5     | 69.0      | 80.0      | 96.2      |
| [6-layer PROD](https://drive.google.com/file/d/1NbqO5JqI3qkYDlPlb7ROW7KQHTbGOItL/view?usp=sharing) | **39.3** | **56.4** | **78.1**  | **87.0**  | **98.4**  |

------

**MARCO-Document:**

| Model            | MRR@10   | Recall@5 | Recall@20 | Recall@100 |
| ---------------- | :--------: | :--------: | :---------: | :----------: |
| 6-layer          | 34.0     | 52.1     | 75.6      | 90.4       |
| [6-layer PROD](https://drive.google.com/file/d/1MLFcx81TmRPWHNuHL34OgGyQq1YTlsQO/view?usp=sharing) | **42.8** | **62.4** | **83.9** | **93.3**   |

## Quick Training Example for MARCO-Passage

Next, we will show how to train the complete PROD on MARCO-Passage(`marco`). If you want to train PROD on another dataset, just modify the data set name at the end of the `py` file, for example, `nq` for Natural Questions, `marcodoc` for MARCO Document.

PROD method is mainly divided into three stages:

1. Warm up
2. Teacher Progressive Distillation
3. Data Progressive Distillation

------

### Warm up

First, we train a 6/12 layer dual-encoder model with origin data:

```shell
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9505 \
./ProD_base/train_DE_model_marco.py \
--model_type="nghuyong/ernie-2.0-base-en" \
--origin_data_dir=./data/marco/marco_train.json \
--corpus_path=./data/marco \
--max_seq_length=144 --per_gpu_train_batch_size=16 --gradient_accumulation_steps=1 \
--output_dir  ../result/DE_6layer \
--log_dir ../log/DE_6layer \
--warmup_steps 4000 --logging_steps 100 --save_steps 1000 --max_steps 40000 --learning_rate=2e-5 \
--number_neg 1 --num_hidden_layers 6(12)
```

Next, use the trained 12 layer dual-encoder to flash MARCO data:

```shell
# inference
python -u -m torch.distributed.launch --nproc_per_node=4 --master_port=9560 \
./ProD_base/inference_DE_marco.py \
--model_type="nghuyong/ernie-2.0-base-en" \
--train_qa_path=./marco/train.query.txt \
--test_qa_path=./marco/dev.query.txt \
--ground_truth_path=./marco/qrels.dev.tsv \
--train_ground_truth_path=./marco/qrels.train.tsv \
--max_seq_length=144 --per_gpu_eval_batch_size=1024 --top_k=1000 \
--eval_model_dir=../result/DE_12layer/checkpoint-40000 \
--output_dir=../result/DE_12layer/40000 \
--passage_path=./marco \
--num_hidden_layers 12


# generate
GROUND_TRUE_FILE="./marco/qrels.train.tsv"
QUERY_FILE="./marco/train.query.txt"
RESULT_FILE="../result/DE_12layer/40000/train_result_dict_list.pkl"
OUTPUT_DIR="./marco/marco_train_flash1.json"
NEG_NUM=100

python ./ProD_KD/utils/preprae_ce_marco_train.py $GROUND_TRUE_FILE $QUERY_FILE $RESULT_FILE $OUTPUT_DIR $NEG_NUM
```

Use `marco_train_flash1.json` to continue training the 6-layer/12-layer dual-encoder to get the strong baseline

```shell
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9525 \
./ProD_base/train_DE_model_marco.py \
--model_type="nghuyong/ernie-2.0-base-en" \
--origin_data_dir=./marco/marco_train_flash1.json \
--corpus_path=./marco \
--max_seq_length=144 --per_gpu_train_batch_size=16 --gradient_accumulation_steps=1 \
--output_dir  ../result/DE_6layer_f1_cont \
--log_dir ../log/DE_6layer_f1_cont \
--model_name_or_path=../result/DE_6layer/checkpoint-40000 \
--warmup_steps 1000 --logging_steps 100 --save_steps 1000 --max_steps 10000 --learning_rate=2e-5 \
--number_neg 1 --num_hidden_layers 6(12)
```

This step can also be replaced by training a 6-layer dual-encoder model from scratch with the refresh data. The experiment shows that the performance of continue train is better on the MARCO dataset.

------

### Teacher Progressive Distillation

#### dual-encoder distillation

In this step, we use a large(12-layer) dual-encoder to distill a small(6-layer) dual-encoder

```shell
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9536 \
./ProD_KD/run_progressive_distill_marco.py \
--teacher_model_type="nghuyong/ernie-2.0-en" \
--model_type="nghuyong/ernie-2.0-base-en" \
--origin_data_dir=./marco/marco_train_flash1.json \
--corpus_path=./marco \
--max_seq_length=144 --per_gpu_train_batch_size=16 --gradient_accumulation_steps=1 \
--warmup_steps 4000 --logging_steps 100 --save_steps 1000 --max_steps 40000 --learning_rate=5e-5 \
--number_neg 1 \
--output_dir  ../result/12DEt6DE_distill \
--log_dir ../log/12DEt6DE_distill \
--teacher_num_hidden_layers 12 --student_num_hidden_layers 6 \
--teacher_model_path="../result/DE_12layer_f1_cont/checkpoint-10000" \
--student_model_path="../result/DE_6layer_f1_cont/checkpoint-10000" \
--KD_type="KD_softmax" --CE_WEIGHT 0.1 --KD_WEIGHT 0.9 --TEMPERATURE 4.0
```

After the distillation is completed, we use 12-layers dual-encoder to refresh the data

```shell
# inference
python -u -m torch.distributed.launch --nproc_per_node=4 --master_port=9560 \
./ProD_base/inference_DE_marco.py \
--model_type="nghuyong/ernie-2.0-base-en" \
--train_qa_path=./marco/train.query.txt \
--test_qa_path=./marco/dev.query.txt \
--ground_truth_path=./marco/qrels.dev.tsv \
--train_ground_truth_path=./marco/qrels.train.tsv \
--max_seq_length=144 --per_gpu_eval_batch_size=1024 --top_k=1000 \
--eval_model_dir=../result/DE_12layer_f1_cont/checkpoint-40000 \
--output_dir=../result/DE_12layer_f1_cont/40000 \
--passage_path=./marco \
--num_hidden_layers 12


# generate
GROUND_TRUE_FILE="./marco/qrels.train.tsv"
QUERY_FILE="./marco/train.query.txt"
RESULT_FILE="../result/DE_12layer_f1_cont/40000/train_result_dict_list.pkl"
OUTPUT_DIR="./marco/marco_train_flash2.json"
NEG_NUM=100

python ./ProD_KD/utils/preprae_ce_marco_train.py $GROUND_TRUE_FILE $QUERY_FILE $RESULT_FILE $OUTPUT_DIR $NEG_NUM
```

#### 12-layer cross-encoder distillation

In this step, we use `marco_train_flash2.json`  to train a 12 layer cross-encoder：

```shell
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9546 \
./ProD_base/train_CE_model_marco.py \
--model_type="nghuyong/ernie-2.0-base-en" --max_seq_length=160 \
--per_gpu_train_batch_size=4 --gradient_accumulation_steps=8 \
--number_neg=15 --learning_rate=1e-5 \
--origin_data_dir=./marco/marco_train_flash2.json
--corpus_path=./marco \
--output_dir=../result/Ranker_12layer \
--log_dir=../log/Ranker_12layer \
--warmup_steps=1000 --logging_steps=100 --save_steps=500 \
--max_steps=6000 --num_hidden_layers 12
```

Next, we use the 12-layers cross-encoder to continue distill 6-layers student model:

```shell
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9447 \
./ProD_KD/run_progressive_distill_marco.py \
--teacher_model_type="nghuyong/ernie-2.0-en" \
--model_type="nghuyong/ernie-2.0-base-en" \
--origin_data_dir=./marco/marco_train_flash2.json \
--corpus_path=./marco \
--max_seq_length=144 --per_gpu_train_batch_size=8 --gradient_accumulation_steps=1 \
--warmup_steps 4000 --logging_steps 100 --save_steps 1000 --max_steps 40000 --learning_rate=5e-5 \
--number_neg 15 --neg_type="random" --open_LwF \
--output_dir  ../result/12CEt6DE_distill \
--log_dir ../log/12CEt6DE_distill \
--teacher_num_hidden_layers 12 --student_num_hidden_layers 6 \
--teacher_model_path="../result/Ranker_12layer/checkpoint-6000" \
--student_model_path="../result/12DEt6DE_distill/checkpoint-40000" \
--KD_type="KD_softmax" --CE_WEIGHT 0.1 --KD_WEIGHT 0.9 --TEMPERATURE 4.0 --LwF_WEIGHT 1.0 \
--teacher_type="cross_encoder" --model_class="dual_encoder"
```

At the end of this step, we used student model after distillation to refresh the data:

```shell
# inference
python -u -m torch.distributed.launch --nproc_per_node=4 --master_port=9560 \
./ProD_base/inference_DE_marco.py \
--model_type="nghuyong/ernie-2.0-base-en" \
--train_qa_path=./marco/train.query.txt \
--test_qa_path=./marco/dev.query.txt \
--ground_truth_path=./marco/qrels.dev.tsv \
--train_ground_truth_path=./marco/qrels.train.tsv \
--max_seq_length=144 --per_gpu_eval_batch_size=1024 --top_k=1000 \
--eval_model_dir=../result/12CEt6DE_distill/checkpoint-40000 \
--output_dir=../result/12CEt6DE_distill/40000 \
--passage_path=./marco \
--num_hidden_layers 6


# generate
GROUND_TRUE_FILE="./marco/qrels.train.tsv"
QUERY_FILE="./marco/train.query.txt"
RESULT_FILE="../result/12CEt6DE_distill/40000/train_result_dict_list.pkl"
OUTPUT_DIR="./marco/marco_train_flash3.json"
NEG_NUM=100

python ./ProD_KD/utils/preprae_ce_marco_train.py $GROUND_TRUE_FILE $QUERY_FILE $RESULT_FILE $OUTPUT_DIR $NEG_NUM
```

#### 24-layer cross-encoder distillation

Similar to the previous step, we first trained a 24 cross-encoder

```shell
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9921 \
./ProD_base/train_CE_model_marco.py \
--model_type="nghuyong/ernie-2.0-large-en" --max_seq_length=160 \
--per_gpu_train_batch_size=4 --gradient_accumulation_steps=8 \
--number_neg=15 --learning_rate=5e-6 \
--origin_data_dir=./marco/marco_train_flash3.json \
--corpus_path=./marco \
--output_dir=../result/Ranker_24layer \
--log_dir=../log/Ranker_24layer \
--warmup_steps=600 --logging_steps=100 --save_steps=500 \
--max_steps=6000 --num_hidden_layers 24
```

Next, we use the 24-layers cross-encoder to continue distill 6-layers student model:

```shell
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9446 \
./ProD_KD/run_progressive_distill_marco.py \
--teacher_model_type="nghuyong/ernie-2.0-large-en" \
--model_type="nghuyong/ernie-2.0-base-en" \
--origin_data_dir=./marco/marco_train_flash3.json \
--corpus_path=./marco \
--max_seq_length=144 --per_gpu_train_batch_size=8 --gradient_accumulation_steps=1 \
--warmup_steps 4000 --logging_steps 100 --save_steps 1000 --max_steps 40000 --learning_rate=5e-5 \
--number_neg 15 --neg_type="random" --open_LwF \
--output_dir  ../result/24CEt6DE_distill \
--log_dir ../log/24CEt6DE_distill \
--teacher_num_hidden_layers 24 --student_num_hidden_layers 6 \
--teacher_model_path="../result/Ranker_24layer/checkpoint-6000" \
--student_model_path="../result/12CEt6DE_distill/checkpoint-40000" \
--KD_type="KD_softmax" --CE_WEIGHT 0.1 --KD_WEIGHT 0.9 --TEMPERATURE 4.0 --LwF_WEIGHT 1.0 \
--teacher_type="cross_encoder" --model_class="dual_encoder"
```

At the end of this step, we used student model after distillation to refresh the data:

```shell
# inference
python -u -m torch.distributed.launch --nproc_per_node=4 --master_port=9560 \
./ProD_base/inference_DE_marco.py \
--model_type="nghuyong/ernie-2.0-base-en" \
--train_qa_path=./marco/train.query.txt \
--test_qa_path=./marco/dev.query.txt \
--ground_truth_path=./marco/qrels.dev.tsv \
--train_ground_truth_path=./marco/qrels.train.tsv \
--max_seq_length=144 --per_gpu_eval_batch_size=1024 --top_k=1000 \
--eval_model_dir=../result/24CEt6DE_distill/checkpoint-40000 \
--output_dir=../result/24CEt6DE_distill/40000 \
--passage_path=./marco \
--num_hidden_layers 6


# generate
GROUND_TRUE_FILE="./marco/qrels.train.tsv"
QUERY_FILE="./marco/train.query.txt"
RESULT_FILE="../result/24CEt6DE_distill/40000/train_result_dict_list.pkl"
OUTPUT_DIR="./marco/marco_train_flash4.json"
NEG_NUM=100

python ./ProD_KD/utils/preprae_ce_marco_train.py $GROUND_TRUE_FILE $QUERY_FILE $RESULT_FILE $OUTPUT_DIR $NEG_NUM
```

------

### Data Progressive Distillation

In this step, you can retrain a 24-layers cross-encoder with `marco_train_flash4.json`, or use the previous 24-layers cross-encoder，Use reranker to rerank the result of student model on train data：

```shell
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9238 \
./ProD_base/rerank_train_eval_marco.py \
--model_type="nghuyong/ernie-2.0-large-en" \
--model_name_or_path../result/Ranker_24layer/checkpoint-6000 \
--result_data_path=../result/24CEt6DE_distill/40000/train_result_dict_list.pkl \
--query_path=./marco/train.query.txt \
--corpus_path=./marco \
--real_data_path=./marco/qrels.train.tsv \
--output_dir=../result/Ranker_24layer/6000rank \
--log_dir=../result/Ranker_24layer/6000rank \
--per_gpu_train_batch_size=20 --max_seq_length=160 \
--num_hidden_layers 24
```

If the dataset is large, this step will take a lot of time, it will further simplify the runtime in future releases. You can increase `per_gpu_train_batch_size` or GPU number to speed up inference if your machine is powerful enough. Or for simple testing, you can also choose a 12-layers cross-encoder, and the experimental results will not differ too much.

Next, we select the data that teachers do well but students do not, and separate them:

```shell
STUDENT_RESULT_FILE="../result/24CEt6DE_distill/40000/train_result_dict_list.pkl"
TEACHER_RESULT_FILE="../result/Ranker_24layer/6000rank/reranker_train_result_dict.pkl"
DATA_FILE="./marco/marco_train_flash4.json"
OUTPUT_DIR="./marco/CE24_top2t15_better.json"
GROUND_TRUE_FILE="./marco/qrels.train.tsv"

python ./ProD_KD/utils/dataset_division_marco.py $STUDENT_RESULT_FILE $TEACHER_RESULT_FILE $DATA_FILE $OUTPUT_DIR $GROUND_TRUE_FILE
```

Finally, we performed a further distillation training using the data obtained above and added LWF distillation stabilization.

It should be noted that the number of epoch in this step is determined according to the amount of separated data,  Recommended training: 3-4 epochs

```shell
python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9462 \
./ProD_KD/run_progressive_distill_marco.py \
--teacher_model_type="nghuyong/ernie-2.0-large-en" \
--model_type="nghuyong/ernie-2.0-base-en" \
--origin_data_dir=./marco/CE24_top2t15_better.json \
--corpus_path=./marco \
--max_seq_length=144 --per_gpu_train_batch_size=8 --gradient_accumulation_steps=1 \
--warmup_steps 200 --logging_steps 100 --save_steps 500 --max_steps 2000 --learning_rate=1e-5 \
--number_neg 15 --neg_type="descend" --open_LwF \
--output_dir  ../result/24CEt6DE_hardcont_distill_LwF \
--log_dir ../log/24CEt6DE_hardcont_distill_LwF \
--teacher_num_hidden_layers 24 --student_num_hidden_layers 6 \
--teacher_model_path="../result/Ranker_24layer/checkpoint-6000" \
--student_model_path="../result/24CEt6DE_distill/checkpoint-40000" \
--KD_type="KD_softmax" --CE_WEIGHT 0.1 --KD_WEIGHT 0.9 --TEMPERATURE 4.0 --LwF_WEIGHT 1.0 \
--teacher_type="cross_encoder" --model_class="dual_encoder"
```

Finally, generally take the last checkpoints for testing

```shell
# inference
python -u -m torch.distributed.launch --nproc_per_node=4 --master_port=9560 \
./ProD_base/inference_DE_marco.py \
--model_type="nghuyong/ernie-2.0-base-en" \
--train_qa_path=./marco/train.query.txt \
--test_qa_path=./marco/dev.query.txt \
--ground_truth_path=./marco/qrels.dev.tsv \
--train_ground_truth_path=./marco/qrels.train.tsv \
--max_seq_length=144 --per_gpu_eval_batch_size=1024 --top_k=1000 \
--eval_model_dir=../result/24CEt6DE_hardcont_distill_LwF/checkpoint-2000 \
--output_dir=../result/24CEt6DE_hardcont_distill_LwF/2000 \
--passage_path=./marco \
--num_hidden_layers 6
```

## 📜 Citation

Please cite our paper if you use [PROD](https://arxiv.org/abs/2209.13335) in your work:
```bibtex
@article{lin2023prod,
   title={PROD: Progressive Distillation for Dense Retrieval},
   author={Zhenghao Lin, Yeyun Gong, Xiao Liu, Hang Zhang, Chen Lin, Anlei Dong, Jian Jiao, Jingwen Lu, Daxin Jiang, Rangan Majumder, Nan Duan},
   booktitle = {{WWW}},
   year={2023}
}
```