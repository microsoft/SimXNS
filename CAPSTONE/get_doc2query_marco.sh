
# get the generated query from https://github.com/castorini/docTTTTTquery 
mkdir tmp
mkdir docTTTTTquery_full
cd tmp
wget https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/predicted_queries_topk_sampling.zip

unzip predicted_queries_topk_sampling.zip

for i in $(seq -f "%03g" 0 17); do
    echo "Processing chunk $i"
    paste predicted_queries_topk_sample???.txt${i}-1004000 \
    > predicted_queries_topk.txt${i}-1004000
done

cat predicted_queries_topk.txt???-1004000 > doc2query.tsv
mv doc2query.tsv ../docTTTTTquery_full
cd ..
rm -rf tmp
