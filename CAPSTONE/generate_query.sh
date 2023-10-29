# generate queries for beir data
for dataset in cqadupstack/android cqadupstack/english cqadupstack/gaming cqadupstack/gis cqadupstack/mathematica cqadupstack/physics cqadupstack/programmers cqadupstack/stats cqadupstack/tex cqadupstack/unix cqadupstack/webmasters cqadupstack/wordpress \
                quora robust04 trec-news nq signal1m dbpedia-entity \
                nfcorpus scifact arguana scidocs fiqa trec-covid webis-touche2020 \
                bioasq hotpotqa fever climate-fever
do 
    python -u -m torch.distributed.launch --nproc_per_node=8 --master_port=9589 \
    ./models/generate_query.py --dataset $dataset
done
