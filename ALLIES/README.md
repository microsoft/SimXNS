# ALLIES

The code for our paper [ALLIES: Prompting Large Language Model with Beam Search](https://arxiv.org/abs/2305.14766).

![model](assets/model.jpg)

## Dataset

### NQ

dataset/nq-test.jsonl

### TriviaQA

dataset/tqa-test.jsonl

### WebQ

dataset/webq-test.jsonl


## Released Resources

We release the preprocessed data and trained ckpts in [Azure Blob](https://msranlciropen.blob.core.windows.net/simxns/ALLIES/).
Here we also provide the file list under this URL:
<details>
<summary><b>Click here to see the file list.</b></summary>
<pre><code>INFO: nq/de-checkpoint-10000/passage_embedding.pb;  Content Length: 60.13 GiB
INFO: nq/de-checkpoint-10000/passage_embedding2id.pb;  Content Length: 160.33 MiB
INFO: webq/de-checkpoint-400/passage_embedding.pb;  Content Length: 60.13 GiB
INFO: webq/de-checkpoint-400/passage_embedding2id.pb;  Content Length: 160.33 MiB
INFO: tq/de-checkpoint-10000/passage_embedding.pb;  Content Length: 60.13 GiB
INFO: tq/de-checkpoint-10000/passage_embedding2id.pb;  Content Length: 160.33 MiB</code></pre>
</details>

To download the files, please refer to [HOW_TO_DOWNLOAD](https://github.com/microsoft/SimXNS/tree/main/HOW_TO_DOWNLOAD.md).



## Run

### Directly Answer

```
python main.py --dataset $dataset --task answer_without_retrieval  --apikey $ID
```

### Answer with retrieval

```
python main.py --dataset $dataset --task answer_with_retrieval --topK $retrieval_num  --apikey $ID
```

### GenRead

```
python main.py --dataset $dataset --task genread --apikey $ID
```

### Allies

```
##GENREAD
python main.py --dataset $dataset --task ALLIES --retrieval_type generate --beam_size $beam_size --beam_Depth $beam_depth --ask_question_num $ask_question_num --apikey $ID

##Retrieval
python main.py --dataset $dataset --task ALLIES --topK $retrieval_num --retrieval_type retrieve --beam_size $beam_size --beam_Depth $beam_depth --ask_question_num $ask_question_num --apikey $ID
```

## Parameters

- $dataset: Dataset for testing
- $ID: The key for API
- $beam_size: Beam size
- $beam_depth: Beam depth
- $ask_question_num: Ask question number
- $retrieval_num: Retrieval doc num
