# SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval

This repository contains the code for our EMNLP2022 paper [***SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval***](https://arxiv.org/abs/2210.11773).


## 🚀 Overview

We propose ***SimANS***, a simple, general and flexible ambiguous negatives sampling method for dense text retrieval. It can be easily applied to various dense retrieval methods.

The key code of our SimANS is the implementation of the sampling distribution as:

$$p_{i} \propto \exp{(-a\cdot(s(q,d_{i})-s(q,\tilde{d}^{+})-b)^{2})}, \forall~d_{i} \in \hat{\mathcal{D}}^{-}$$

```python
def SimANS(pos_pair, neg_pairs_list):
    pos_id, pos_score = int(pos_pair[0]), float(pos_pair[1])
    neg_candidates, neg_scores = [], []
    for pair in neg_pairs_list:
        neg_id, neg_score = pair
        neg_score = math.exp(-(neg_score - pos_score) ** 2 * self.tau)
        neg_candidates.append(neg_id)
        neg_scores.append(neg_score)
    return random.choices(neg_candidates, weights=neg_scores, k=num_hard_negatives)
```

Here we show the main results on [MS MARCO](https://microsoft.github.io/msmarco/), [Natural Questions](https://ai.google.com/research/NaturalQuestions/) and [TriviaQA](http://nlp.cs.washington.edu/triviaqa/). This method outperformes the state-of-the-art methods.
![SimANS Main Result](figs/simans_main_result.jpg)

This method has been applied in [Microsoft Bing](https://www.bing.com/), and we also show the results on the industry dataset.
<!-- ![SimANS Industry Result](figs/simans_industry_result.jpg) -->
<div align=center> <img src="figs/simans_industry_result.jpg" width = 45%/> </div>

Please find more details in the paper.


## Released Resources

We release the preprocessed data and trained ckpts in [Azure Blob](https://msranlciropen.blob.core.windows.net/simxns/SimANS/).
Here we also provide the file list under this URL:
<details>
<summary><b>Click here to see the file list.</b></summary>
<pre><code>INFO: best_simans_ckpt.zip;  Content Length: 7.74 GiB
INFO: best_simans_ckpt/MS-Doc/checkpoint-25000;  Content Length: 1.39 GiB
INFO: best_simans_ckpt/MS-Doc/log.txt;  Content Length: 78.32 KiB
INFO: best_simans_ckpt/MS-Pas/checkpoint-20000;  Content Length: 2.45 GiB
INFO: best_simans_ckpt/MS-Pas/log.txt;  Content Length: 82.74 KiB
INFO: best_simans_ckpt/NQ/checkpoint-30000;  Content Length: 2.45 GiB
INFO: best_simans_ckpt/NQ/log.txt;  Content Length: 298.44 KiB
INFO: best_simans_ckpt/TQ/checkpoint-10000;  Content Length: 2.45 GiB
INFO: best_simans_ckpt/TQ/log.txt;  Content Length: 99.44 KiB
INFO: ckpt.zip;  Content Length: 19.63 GiB
INFO: ckpt/MS-Doc/adore-star/config.json;  Content Length: 1.37 KiB
INFO: ckpt/MS-Doc/adore-star/pytorch_model.bin;  Content Length: 480.09 MiB
INFO: ckpt/MS-Doc/checkpoint-20000;  Content Length: 1.39 GiB
INFO: ckpt/MS-Doc/checkpoint-reranker20000;  Content Length: 1.39 GiB
INFO: ckpt/MS-Pas/checkpoint-20000;  Content Length: 2.45 GiB
INFO: ckpt/MS-Pas/checkpoint-reranker20000;  Content Length: 3.75 GiB
INFO: ckpt/NQ/checkpoint-reranker26000;  Content Length: 3.75 GiB
INFO: ckpt/NQ/nq_fintinue.pkl;  Content Length: 2.45 GiB
INFO: ckpt/TQ/checkpoint-reranker34000;  Content Length: 3.75 GiB
INFO: ckpt/TQ/triviaqa_fintinue.pkl;  Content Length: 2.45 GiB
INFO: data.zip;  Content Length: 18.43 GiB
INFO: data/MS-Doc/dev_ce_0.tsv;  Content Length: 15.97 MiB
INFO: data/MS-Doc/msmarco-docdev-qrels.tsv;  Content Length: 105.74 KiB
INFO: data/MS-Doc/msmarco-docdev-queries.tsv;  Content Length: 215.14 KiB
INFO: data/MS-Doc/msmarco-docs.tsv;  Content Length: 21.32 GiB
INFO: data/MS-Doc/msmarco-doctrain-qrels.tsv;  Content Length: 7.19 MiB
INFO: data/MS-Doc/msmarco-doctrain-queries.tsv;  Content Length: 14.76 MiB
INFO: data/MS-Doc/train_ce_0.tsv;  Content Length: 1.13 GiB
INFO: data/MS-Pas/dev.query.txt;  Content Length: 283.39 KiB
INFO: data/MS-Pas/para.title.txt;  Content Length: 280.76 MiB
INFO: data/MS-Pas/para.txt;  Content Length: 2.85 GiB
INFO: data/MS-Pas/qrels.dev.tsv;  Content Length: 110.89 KiB
INFO: data/MS-Pas/qrels.train.addition.tsv;  Content Length: 5.19 MiB
INFO: data/MS-Pas/qrels.train.tsv;  Content Length: 7.56 MiB
INFO: data/MS-Pas/train.query.txt;  Content Length: 19.79 MiB
INFO: data/MS-Pas/train_ce_0.tsv;  Content Length: 1.68 GiB
INFO: data/NQ/dev_ce_0.json;  Content Length: 632.98 MiB
INFO: data/NQ/nq-dev.qa.csv;  Content Length: 605.48 KiB
INFO: data/NQ/nq-test.qa.csv;  Content Length: 289.99 KiB
INFO: data/NQ/nq-train.qa.csv;  Content Length: 5.36 MiB
INFO: data/NQ/train_ce_0.json;  Content Length: 5.59 GiB
INFO: data/TQ/dev_ce_0.json;  Content Length: 646.60 MiB
INFO: data/TQ/train_ce_0.json;  Content Length: 5.62 GiB
INFO: data/TQ/trivia-dev.qa.csv;  Content Length: 3.03 MiB
INFO: data/TQ/trivia-test.qa.csv;  Content Length: 3.91 MiB
INFO: data/TQ/trivia-train.qa.csv;  Content Length: 26.67 MiB
INFO: data/psgs_w100.tsv;  Content Length: 12.76 GiB</code></pre>
</details>

To download the files, please refer to [HOW_TO_DOWNLOAD](https://github.com/microsoft/SimXNS/tree/main/HOW_TO_DOWNLOAD.md).


## 🙋 How to Use

**⚙️ Environment Setting**

To faithfully reproduce our results, please use the correct `1.7.1` pytorch version corresponding to your platforms/CUDA versions according to [Released Packages by pytorch](https://anaconda.org/pytorch/pytorch), and install faiss successfully for evaluation.

We list our command to prepare the experimental environment as follows:
```bash
conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch
conda install faiss-gpu cudatoolkit=11.0 -c pytorch
conda install transformers
pip install tqdm
pip install tensorboardX
pip install lmdb
pip install datasets
pip install wandb
pip install sklearn
pip install boto3
```

**💾 Data and Initial Checkpoint**

We list the necessary data for training on MS-Pas/MS-Doc/NQ/TQ [here](https://msranlciropen.blob.core.windows.net/simxns/SimANS/data.zip). You can download the compressed file directly or with [Microsoft's AzCopy CLI tool](https://learn.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy) and put the content in `./data`.
If you are working on MS-Pas, you will need this [additional file](https://msranlciropen.blob.core.windows.net/simxns/SimANS/data/MS-Pas/qrels.train.addition.tsv) for training.

In our approach, we require to use the checkpoint from AR2 for initialization. We release them [here](https://msranlciropen.blob.core.windows.net/simxns/SimANS/ckpt.zip). You can download the all-in-one compressed file and put the content in `./ckpt`.


**📋 Training Scripts**

We provide the training scripts using SimANS on SOTA AR2 model for MS-MARCO-Passage/Document Retrieval, NQ and TQ datasets, and have set up the best hyperparameters for training. You can run it to automatically finish the training and evaluation.
```bash
bash train_MS_Pas_AR2.sh
bash train_MS_Doc_AR2.sh
bash train_NQ_AR2.sh
bash train_TQ_AR2.sh
```

For results in the paper, we use 8 * A100 GPUs with CUDA 11. Using different types of devices or different versions of CUDA/other softwares may lead to different performance.

**⚽ Best SimANS Checkpoint**

For better reproducing our experimental results, we also release all the checkpoint of our approach [here](https://msranlciropen.blob.core.windows.net/simxns/SimANS/best_simans_ckpt.zip). You can download the compressed file and reuse the content for evaluation.


## 📜 Citation

Please cite our paper if you use [SimANS](https://arxiv.org/abs/2210.11773) in your work:
```bibtex
@article{zhou2022simans,
   title={SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval},
   author={Kun Zhou, Yeyun Gong, Xiao Liu, Wayne Xin Zhao, Yelong Shen, Anlei Dong, Jingwen Lu, Rangan Majumder, Ji-Rong Wen, Nan Duan and Weizhu Chen},
   booktitle = {{EMNLP}},
   year={2022}
}
```
