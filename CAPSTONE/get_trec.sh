# this shell is from  https://github.com/NLPCode/tevatron/blob/main/examples/coCondenser-marco/get_data.sh
SCRIPT_DIR=$PWD

#https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019
mkdir trec_19
cd trec_19
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip msmarco-test2019-queries.tsv.gz
rm msmarco-test2019-queries.tsv.gz
wget --no-check-certificate https://trec.nist.gov/data/deep/2019qrels-pass.txt

cd ..

#https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020
mkdir trec_20
cd trec_20
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz
gunzip msmarco-test2020-queries.tsv.gz
rm msmarco-test2020-queries.tsv.gz
wget --no-check-certificate https://trec.nist.gov/data/deep/2020qrels-pass.txt