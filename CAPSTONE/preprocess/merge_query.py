import json
query_filename='./docTTTTTquery_full/doc2query.tsv'

# reform
min_q=80
with open(query_filename, 'r', encoding='utf-8') as fr, open('doc2query_merge.tsv', 'w', encoding='utf-8') as fw:
    for psg_id, line in enumerate(fr):
        example = line.strip().split('\t')
        example = list(set([e.strip() for e in example]))
        min_q=min(min_q, len(example))
        output_data = '\t'.join([str(psg_id)]+example)
        #output_data = {'passage_id': str(psg_id), 'generated_queries': example}
        fw.write(output_data+'\n')
    print(min_q) 

