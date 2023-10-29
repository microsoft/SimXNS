import argparse
from utils import *
from retrieval_utils import *
from tools import *
import wandb
import time
import tqdm
from tqdm import trange


def genread(args, inlines, wandb, log_file_1, log_file_2):
    all_token_count = 0
    exact_match_count = 0
    all_output = []
    f1_score_all = []
    
    for i in trange(len(inlines)):
        start_time = time.time()
        output, token_count = gen_background(args, inlines[i]['question'])
        all_token_count += token_count
        output, token_count = answer_with_gen(args, inlines[i]['question'], output, log_file_2)
        all_token_count += token_count
        answer = inlines[i]['answer']
        all_output.append(output)

        if ems(output, answer):
            exact_match_count += 1
        end_time = time.time()
        f1_score_tmp = [f1_score(normalize_answer(output), normalize_answer(answer[index])) for index in range(len(answer))]
        f1_score_all.append(np.max(f1_score_tmp))

        wandb.log({'step': i, 'Exact Match': exact_match_count / (i + 1), 'Exact Match Count': exact_match_count,
                   'F1 Score': np.mean(f1_score_all), 'All Token Number': all_token_count, 'Query Per Minute': 60 / (end_time - start_time),
                   'Token Per Minute': 60 / (end_time - start_time) * token_count})
        write(log_file_1, f'{i}: {output}')

    final_em = round(exact_match_count / len(inlines), 4)
    final_f1 = round(np.mean(f1_score_all), 4)

    wandb.log({'Final Exact Match': final_em, 'Final F1 Score': final_f1, 'Final Exact Match Count': exact_match_count})
    write(log_file_1, f'EM: {final_em}')

    return final_em, exact_match_count, final_f1


def answer_without_retrieval(args, inlines, wandb, log_file_1):
    all_token_count = 0
    exact_match_count = 0
    all_output = []
    f1_score_all = []

    for i in trange(len(inlines)):
        start_time = time.time()
        output, token_count = directly_answer(args, inlines[i]['question'])
        
        all_token_count += token_count

        answer = inlines[i]['answer']
        all_output.append(output)
        
        if ems(output, answer): 
            exact_match_count += 1
        end_time = time.time()

        f1_score_tmp = [f1_score(normalize_answer(output), normalize_answer(answer[index])) for index in range(len(answer))]
        f1_score_all.append(np.max(f1_score_tmp))

        wandb.log({'step': i, 'Exact Match': exact_match_count / (i + 1), 'Exact Match Count': exact_match_count,
                   'F1 Score': np.mean(f1_score_all), 'All Token Number': all_token_count, 'Query Per Minute': 60 / (end_time - start_time),
                   'Token Per Minute': 60 / (end_time - start_time) * token_count})
        write(log_file_1, f'{i}: {output}')

    final_em = round(exact_match_count / len(inlines), 4)
    final_f1 = round(np.mean(f1_score_all), 4)

    wandb.log({'Final Exact Match': final_em, 'Final F1 Score': final_f1, 'Final Exact Match Count': exact_match_count})
    write(log_file_1, f'EM: {final_em}')

    return final_em, exact_match_count, final_f1


def answer_with_retrieval(args, inlines, wandb, log_file_1, log_file_2):
    tokenizer, model, cpu_index, passage_embedding2id, passages = prepare_dr(args)

    all_token_count = 0
    exact_match_count = 0
    all_output = []
    f1_score_all = []

    for i in trange(len(inlines)):
        start_time = time.time()
        output, token_count = retrieve_then_answer(args, inlines[i]['question'], args.topK, tokenizer, 
                                                   model, cpu_index, passage_embedding2id, passages, log_file_2)

        all_token_count += token_count
        answer = inlines[i]['answer']
        all_output.append(output)
        
        if ems(output, answer): 
            exact_match_count += 1
        end_time = time.time()

        f1_score_tmp = [f1_score(normalize_answer(output), normalize_answer(answer[index])) for index in range(len(answer))]
        f1_score_all.append(np.max(f1_score_tmp))

        wandb.log({'step': i, 'Exact Match': exact_match_count / (i + 1), 'Exact Match Count': exact_match_count,
                   'F1 Score': np.mean(f1_score_all), 'All Token Number': all_token_count, 'Query Per Minute': 60 / (end_time - start_time),
                   'Token Per Minute': 60 / (end_time - start_time) * token_count})
        write(log_file_1, f'{i}: {output}')

    final_em = round(exact_match_count / len(inlines), 4)
    final_f1 = round(np.mean(f1_score_all), 4)

    wandb.log({'Final Exact Match': final_em, 'Final F1 Score': final_f1, 'Final Exact Match Count': exact_match_count})
    write(log_file_1, f'EM: {final_em}')

    return final_em, exact_match_count, final_f1


def ALLIES(args, inlines, wandb, log_file_1, log_file_2):
    tokenizer, model, cpu_index, passage_embedding2id, passages = prepare_dr(args)

    f1_score_all = []
    exact_match_count = 0
    all_record = {'all_token_count':0, 'all_api_times':0, 'all_retrieval_times':0, \
                  'directly_answer_token_count':0,'directly_cal_score_token_count':0, \
                  'gen_background_token_count':0, 'summary_token_count':0, 'answer_with_evidence_token_count':0, 
                  'cal_score_with_evidence_token_count':0, 'expand_question_token_count':0}
    
    for i in trange(len(inlines)):
        start_time = time.time()
        try:
            output, _, record = ALLIES_model(args, inlines[i]['question'], tokenizer, model, cpu_index,
                                                     passage_embedding2id, passages, log_file_2, i)
            for key in all_record:
                all_record[key] += record[key]
        except:
            output = 'None'
            record = {'all_token_count': 0}

        answer = inlines[i]['answer']
        if ems(output, answer): 
            exact_match_count += 1
        end_time = time.time()

        f1_score_tmp = [f1_score(normalize_answer(output), normalize_answer(answer[index])) for index in range(len(answer))]
        f1_score_all.append(np.max(f1_score_tmp))

        log_dict = {'step': i, 'Exact Match': exact_match_count / (i + 1), 'Exact Match Count': exact_match_count,
                   'F1 Score': np.mean(f1_score_all), 'Query Per Minute': 60 / (end_time - start_time),
                   'Token Per Minute': 60 / (end_time - start_time) * record['all_token_count']}
        log_dict.update(all_record)
        wandb.log(log_dict)
        write(log_file_1, f'{i}: {output}')

    final_em = round(exact_match_count / len(inlines), 4)
    final_f1 = round(np.mean(f1_score_all), 4)

    wandb.log({'Final Exact Match': final_em, 'Final F1 Score': final_f1, 'Final Exact Match Count': exact_match_count})
    write(log_file_1, f'EM: {final_em}')

    return final_em, exact_match_count, final_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True, help="dataset name: [nq, tqa, webq, wizard, fever, fm2]")
    parser.add_argument("--data_path", default='', type=str, required=False)
    parser.add_argument("--dr_path", default='', type=str, required=False)
    parser.add_argument("--passage_embdding_path", default='', type=str, required=False)
    parser.add_argument("--task", default=None, type=str, required=True, help="task name")
    parser.add_argument("--split", default='test', type=str, required=False)
    parser.add_argument("--unique_identifier", default='', type=str, required=False)
    parser.add_argument("--topK", default=0, type=int, required=False, help="Retrieve topK documents")
    parser.add_argument("--beam_size", default=0, type=int, required=False, help="beam_size")
    parser.add_argument("--beam_Depth", default=0, type=int, required=False, help="beam_Depth")
    parser.add_argument("--ask_question_num", default=0, type=int, required=False, help="ask_question_num")
    parser.add_argument("--threshold", default=0.8, type=float, required=False)
    parser.add_argument("--retrieval_type", default='', type=str, required=False)
    parser.add_argument("--apikey", default=0, type=int, required=False, help="API_ID")
    parser.add_argument("--device", default=0, type=int, required=False, help="device id")
    parser.add_argument("--summary", default='no', type=str, required=False)
    parser.add_argument("--wandb_key", default='gpt', type=str, required=False)
    args = parser.parse_args()

    wandb.login(key=args.wandb_key)
    wandb.init(project='ALLIES', entity="Your Username")
    unique_identifier = str(time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) + '_' + str(time.localtime().tm_hour)+ '_' + \
                        str(time.localtime().tm_min) + '_' + str(args.dataset) + '_' + str(args.task)
    wandb.run.name = unique_identifier
    wandb.config.update(args)
    args.unique_identifier = unique_identifier
    args.save_file = f'{args.data_path}/result/{args.dataset}/{args.task}/{unique_identifier}'
    print_args(args)
    set_openai(args)


    inputfile = f'{args.data_path}/indatasets/{args.dataset}/{args.dataset}-{args.split}.jsonl'
    inlines = readfiles(inputfile)
    create_directory(f'{args.data_path}/result/{args.dataset}/{args.task}')
    log_file_1 = open(f'{args.data_path}/result/{args.dataset}/{args.task}/{unique_identifier}.txt', 'w') 
    log_file_2 = open(f'{args.data_path}/result/{args.dataset}/{args.task}/{unique_identifier}_prompt.txt', 'w') 

    if args.task == 'answer_without_retrieval':
        final_em, exact_match_count, final_f1 = answer_without_retrieval(args, inlines, wandb, log_file_1)
    elif args.task == 'answer_with_retrieval':
        final_em, exact_match_count, final_f1 = answer_with_retrieval(args, inlines, wandb, log_file_1, log_file_2)
    elif args.task == 'ALLIES':
        final_em, exact_match_count, final_f1 = ALLIES(args, inlines, wandb, log_file_1, log_file_2)
    elif args.task == 'genread':
        final_em, exact_match_count, final_f1 = genread(args, inlines, wandb, log_file_1, log_file_2)

    print(f'Final EM: {final_em}')
    print(f'Final F1: {final_f1}')
    print(f'Final EMC: {exact_match_count}')