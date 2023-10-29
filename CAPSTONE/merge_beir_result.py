"""
this file is used to merge the beir results of each dataset.
"""
import argparse
import os
import json
corpus_list = ['trec-covid', 'bioasq', 'nfcorpus', 'nq', 'hotpotqa', 'fiqa', 'signal1m', 'trec-news', 'robust04', 'arguana', 
            'webis-touche2020', 'cqadupstack', 'quora', 'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact' ] 
corpus_list2 = ["cqadupstack/android", "cqadupstack/english", "cqadupstack/gaming", "cqadupstack/gis", "cqadupstack/mathematica", "cqadupstack/physics", 
                "cqadupstack/programmers", "cqadupstack/stats", "cqadupstack/tex", "cqadupstack/unix", "cqadupstack/webmasters", "cqadupstack/wordpress"]
                                   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beir_data_path", default=None, type=str, help="path of a dataset in beir")
    parser.add_argument("--checkpoint_num", default=20000, type=int)
    args = parser.parse_args()
    print(os.path.join(args.beir_data_path, f'test_eval_result{args.checkpoint_num}.json'))
    with open(os.path.join(args.beir_data_path, f'test_eval_result{args.checkpoint_num}.txt'), 'w') as fw:
        fw.write("NDCG@10\n")
        # fw.write('K=10\n')
        # total_value = 0
        # for corpus in corpus_list:
        #     if corpus=='cqadupstack':
        #         total = 0
        #         for subcorpus in corpus_list2:
        #             filename = os.path.join(args.beir_data_path, f'{subcorpus}', f'test_eval_result{args.checkpoint_num}_0_query.json')
        #             with open(filename, 'r') as fr:
        #                 results = json.load(fr)
        #                 total += float(results['NDCG@10'])
        #         value = total/len(corpus_list2)
        #         total_value += value
        #         value = str(value)

        #     else:
        #         filename = os.path.join(args.beir_data_path, f'{corpus}', f'test_eval_result{args.checkpoint_num}_0_query.json')
        #         if os.path.exists(filename):
        #             with open(filename, 'r') as fr:
        #                 results = json.load(fr)
        #                 value = results['NDCG@10']
        #         else:
        #             value ='0'
        #             print(f'{corpus} no results.')
        #         total_value += float(value)
        #     fw.write(f'{corpus}: {value:.3}\n')
        # fw.write(f'Average: {total_value/len(corpus_list):.3}\n')

        # fw.write('K=5\n')
        # total_value = 0
        # for corpus in corpus_list:
        #     if corpus=='cqadupstack':
        #         total = 0
        #         for subcorpus in corpus_list2:
        #             # filename = os.path.join(args.beir_data_path, f'{subcorpus}', f'test_eval_result{args.checkpoint_num}.json')
        #             filename = os.path.join(args.beir_data_path, f'{subcorpus}', f'test_eval_result{args.checkpoint_num}_0_5_query.json')
        #             with open(filename, 'r') as fr:
        #                 results = json.load(fr)
        #                 total += float(results['NDCG@10'])
        #         value = total/len(corpus_list2)
        #         total_value += value
        #         value = str(value)

        #     else:
        #         # filename = os.path.join(args.beir_data_path, f'{corpus}', f'test_eval_result{args.checkpoint_num}.json')
        #         filename = os.path.join(args.beir_data_path, f'{corpus}', f'test_eval_result{args.checkpoint_num}_0_5_query.json')
        #         if os.path.exists(filename):
        #             with open(filename, 'r') as fr:
        #                 results = json.load(fr)
        #                 value = results['NDCG@10']
        #         else:
        #             value ='0'
        #             print(f'{corpus} no results.')
        #         total_value += float(value)
        #     fw.write(f'{corpus}: {value:.3}\n')
        # fw.write(f'Average: {total_value/len(corpus_list):.3}\n')


        fw.write('base\n')
        total_value = 0
        for corpus in corpus_list:
            if corpus=='cqadupstack':
                total = 0
                for subcorpus in corpus_list2:
                    # filename = os.path.join(args.beir_data_path, f'{subcorpus}', f'test_eval_result{args.checkpoint_num}.json')
                    filename = os.path.join(args.beir_data_path, f'{subcorpus}', f'test_eval_result{args.checkpoint_num}.json')
                    with open(filename, 'r') as fr:
                        results = json.load(fr)
                        total += float(results['NDCG@10'])
                value = total/len(corpus_list2)
                total_value += value
                value = str(value)

            else:
                # filename = os.path.join(args.beir_data_path, f'{corpus}', f'test_eval_result{args.checkpoint_num}.json')
                filename = os.path.join(args.beir_data_path, f'{corpus}', f'test_eval_result{args.checkpoint_num}.json')
                if os.path.exists(filename):
                    with open(filename, 'r') as fr:
                        results = json.load(fr)
                        value = results['NDCG@10']
                else:
                    value ='0'
                    print(f'{corpus} no results.')
                total_value += float(value)
            fw.write(f'{corpus}: {value:.3}\n')
        fw.write(f'Average: {total_value/len(corpus_list):.3}\n')
