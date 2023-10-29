# CAPSTONE
This repo provides the code and models in [*CAPSTONE*](https://arxiv.org/abs/2212.09114).
In the paper, we propose CAPSTONE, a curriculum sampling for dense retrieval with document expansion, to bridge the gap between training and inference for dual-cross-encoder.


# Requirements
pip install datasets==2.4.0  
pip install rouge_score==0.1.2  
pip install nltk 
pip install transformers==4.21.1 
<!-- sudo chown -R $USER /opt/conda -->
conda install -y -c pytorch faiss-gpu==1.7.1  
pip install tensorboard  
pip install pytrec-eval  

# Get Data 
Download the cleaned corpus hosted by RocketQA team, generate BM25 negatives for MS-MARCO. Then, download [TREC-Deep-Learning-2019](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019)(TREC DL 19) and [TREC-Deep-Learning-2020](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020)(TREC DL 20). 

```
bash get_msmarco.sh
cd ..
bash get_trec.sh
python preprocess/preprocess_msmarco.py --data_dir ./marco --output_path ./reformed_marco
python preprocess/preprocess_trec.py
```

Download the generated queries for MS-MARCO and merge duplicated queries. Note the query file is around 19GB. 
```
bash get_doc2query_marco.sh
python preprocess/merge_query.py
```

# Train CAPSTONE on MS-MARCO

Train CAPSTONE on MS-MARCO for two stages and initialize the retriever with coCondenser at each stage.
At the first training stage, the hard negatives are sampled from the official BM25 hard negatives, but at the second training stage, the hard negatives are sampled from the mined hard negatives.

Evaluate CAPSTONE on the MS-MARCO development set, TREC DL 19 and 20 test sets. 
```bash
bash run_de_model_expand_corpus_cocondenser.sh
bash run_de_model_expand_corpus_cocondenser_step2.sh
```
If you want to evaluate CAPSTONE on BEIR benchmark, you should download [BEIR](https://github.com/beir-cellar/beir) datasets and generate queries for BEIR. 
```bash
bash generate_query.sh
```

# Well-trained Checkpoints 
| Model           |  Download link
|----------------------|--------|
| The checkpoint for CAPSTONE trained on MS-MARCO at the first step| [\[link\]](https://drive.google.com/file/d/1QTsHQV8BQDJmxGD--fr4hmYdjpizE0zq/view?usp=sharing)  | 
| The checkpoint for CAPSTONE trained on MS-MARCO at the second step| [\[link\]](https://drive.google.com/file/d/1tssOGIRwwXpn2yg4StiRC3B3u0_UqzLG/view?usp=sharing)  | 





# Citation
If you want to use this code in your research, please cite our [paper](https://arxiv.org/abs/2212.09114):
```bash

@inproceedings{he-CAPSTONE,
    title = "CAPSTONE: Curriculum Sampling for Dense Retrieval with Document Expansion",
    author = "Xingwei He and Yeyun Gong and A-Long Jin and Hang Zhang and Anlei Dong and Jian Jiao and Siu Ming Yiu and Nan Duan",
    booktitle = "Proceedings of EMNLP",
    year = "2023",
}
