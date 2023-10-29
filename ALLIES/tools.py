import argparse
from utils import *
from retrieval_utils import *
from tools import *
import wandb
import time
import tqdm
import re

output_debug = False


def extract_floats_from_string(input_string):
    pattern = r"[-+]?\d*\.\d+|\d+"
    floats = re.findall(pattern, input_string)
    return [float(num) for num in floats]


def gen_background(args, original_question):
    prompt = "Generate a short background document from Wikipedia to answer the given question. \n\n {query} \n\n"
    prompt = prompt.replace('{query}', original_question)
    output = run_inference_openai(prompt)
    answer = output[0][0]
    token_count = output[1]

    if output_debug:
        print('---------------------------------')
        print('----------gen_background----------')
        print(prompt)
        print(output)
        print('---------------------------------')
        print()
    return answer, token_count


def answer_with_gen(args, original_question, gen_doc, log_file_2):
    prompt = "Refer to the passage below and answer the following question with just one entity. \n\n Passage: {background} \n\n Question: {query} \n\n The answer is"
    prompt = prompt.replace('{background}', gen_doc).replace('{query}', original_question)
    write(log_file_2, f'{prompt}')
    output = run_inference_openai(prompt)
    answer = output[0][0]
    token_count = output[1]

    if output_debug:
        print('---------------------------------')
        print('----------answer_with_gen----------')
        print(prompt)
        print(output)
        print('---------------------------------')
        print()
    return answer, token_count


def directly_answer(args, original_question):
    prompt = '''Given a question: {query} 
Answer the question with just one entity
'''
    prompt = prompt.replace('{query}', original_question)
    output = run_inference_openai(prompt)
    answer = output[0][0]
    token_count = output[1]
    if output_debug:
        print('---------------------------------')
        print('----------directly_answer----------')
        print(prompt)
        print(output)
        print('---------------------------------')
        print()
    return answer, token_count

def retrieve_then_answer(args, original_question, topK, tokenizer, model, cpu_index, passage_embedding2id, passages, log_file_2):
    prompt = "Refer to the passage below and answer the following question with just one entity. \n\n Passage: {background} \n\n Question: {query} \n\n The answer is"
    relevant_doc = ' '.join(retrieval(original_question, topK, tokenizer, model, cpu_index, passage_embedding2id, passages, args))
    prompt = prompt.replace('{background}', relevant_doc).replace('{query}', original_question)
    write(log_file_2, f'{prompt}')
    output = run_inference_openai(prompt)
    answer = output[0][0]
    token_count = output[1]
    if output_debug:
        print('---------------------------------')
        print('----------retrieve_then_answer----------')
        print(prompt)
        print(output)
        print('---------------------------------')
        print()
    return answer, token_count


def answer_with_evidence(args, original_question, query_list, evidence_list):
    prompt = '''Given following query-evidence pair: 
{doc}
Please refer to the query-evidence pair above and based on your own knowledge, answer the following question with just one entity. 
Question: {query} 
You Must give one answer.
If the answer involves multiple entities, directly output the first one.
The answer is
'''
    query_evidence_pair = '\n'.join(
        [f'Query: {query_list[i]}. Evidence: {evidence_list[i]}' for i in range(len(query_list))])
    prompt = prompt.replace('{query}', original_question).replace('{doc}', query_evidence_pair)
    output = run_inference_openai(prompt)
    answer = output[0][0]
    token_count = output[1]
    if output_debug:
        print('---------------------------------')
        print('----------answer_with_evidence----------')
        print(prompt)
        print(output)
        print('---------------------------------')
        print()
    return answer, token_count


def directly_cal_score(args, original_question, answer):
    prompt = '''Given the question: "{query}" and the candidate answer: "{answer}",
Utilizing your own reasoning ability, assess the probability that the candidate answer is the true answer.
Please provide a number between 0 and 1 as the output, following the guidelines below:
If the probability is between 0 and 0.3, it signifies that the model has substantial evidence to suggest it is an incorrect answer.
If the probability is between 0.3 and 0.5, it suggests that the model leans towards considering it an incorrect answer, but lacks concrete evidence.
If the probability is between 0.5 and 0.7, it indicates that the model leans towards considering it a correct answer, but lacks concrete evidence.
If the probability is greater than 0.7, it signifies that the model has substantial evidence to suggest it is the correct answer.
If the candidate answer doesn't provide clear solution to the question, the probability should be 0. 
Your output is a number with no extra string.
The score is:
'''
    prompt = prompt.replace('{query}', original_question).replace('{answer}', answer)

    output = run_inference_ms(prompt)
    token_count = output[1]
    score = float(output[0][0].split(' ')[0])

    if output_debug:
        print('---------------------------------')
        print('----------directly_cal_score----------')
        print(prompt)
        print(output)
        print('---------------------------------')
        print()
    return score, token_count


def cal_score_with_evidence(args, original_question, answer, query_list, evidence_list):
    prompt = '''Given the question: "{query}" and the candidate answer: "{answer}",
refer to the query-evidence pair below and utilize your own reasoning ability to assess the probability that the candidate answer is the true answer.
Query-evidence pair: 
{doc} 
Please provide a number between 0 and 1 as the output, following the guidelines below:
If the probability is between 0 and 0.3, it signifies that the model has substantial evidence to suggest it is an incorrect answer.
If the probability is between 0.3 and 0.5, it suggests that the model leans towards considering it an incorrect answer, but lacks concrete evidence.
If the probability is between 0.5 and 0.7, it indicates that the model leans towards considering it a correct answer, but lacks concrete evidence.
If the probability is greater than 0.7, it signifies that the model has substantial evidence to suggest it is the correct answer.
If the candidate answer doesn't provide clear solution to the question, the probability should be 0. 
Your output is a number with no extra string.
The score is:
'''
    query_evidence_pair = '\n'.join(
        [f'Query: {query_list[i]}. Evidence: {evidence_list[i]}' for i in range(len(query_list))])
    prompt = prompt.replace('{query}', original_question).replace('{answer}', answer).replace('{doc}',
                                                                                              query_evidence_pair)
    output = run_inference_ms(prompt)
    token_count = output[1]
    score = float(output[0][0].split(' ')[0])

    if output_debug:
        print('---------------------------------')
        print('----------cal_score_with_evidence----------')
        print(prompt)
        print(output)
        print('---------------------------------')
        print()
    return score, token_count


def expand_question(args, original_question, query_list, evidence_list):
    prompt_3 = '''Given the question: {query}, please generate some questions that can help answer the given question with the following constraints:
1. You should output no more than ''' + str(args.ask_question_num) + ''' questions
2. You should directly output the ranked sub-questions based on the importance
3. The generated questions should be diverse and focus on different aspects of the given question.
4. You should output in the following format:
    Ranked Questions:
    1. [Question 1]
…
'''
    prompt_4 = '''Given the question: {query} and following query-evidence pair:
{doc}.
please generate some questions that can help answer the given question with the following constraints:
1. You should output no more than ''' + str(args.ask_question_num) + ''' questions.
2. You should directly output the ranked sub-questions based on the importance.
3. The generated questions should be diverse and focus on different aspects of the given question.
4. You should output in the following format:
    Ranked Questions:
    1. [Question 1]
…
    '''
    if len(query_list) == 0:
        prompt = prompt_3.replace('{query}', original_question)
    else:
        query_evidence_pair = '\n'.join(
            [f'Query: {query_list[i]}. Evidence: {evidence_list[i]}' for i in range(len(query_list))])
        prompt = prompt_4.replace('{query}', original_question).replace('{doc}', query_evidence_pair)

    output = run_inference_openai(prompt)
    token_count = output[1]
    extended_questions = [query[3:] for query in output[0][0].split('\n')[1:]]

    if output_debug:
        print('---------------------------------')
        print('----------expand_question----------')
        print(prompt)
        print(output)
        print('---------------------------------')
        print()
    return extended_questions, token_count


def summary(args, original_question, relevant_doc):
    prompt = '''
Given the original question: "{query}" and the provided document:
"{doc}"
You must follow the requirements:
If the evidence provide information about the question:
output the factual information from the evidence that is relevant to the question
else:
try to output related information of the question based on your knowledge.
'''
    prompt = prompt.replace('{query}', original_question).replace('{doc}', relevant_doc)
    output = run_inference_ms(prompt)
    sumary_doc = output[0][0]
    token_count = output[1]
    if output_debug:
        print('---------------------------------')
        print('----------summary----------')
        print(prompt)
        print(output)
        print('---------------------------------')
        print()
    return sumary_doc, token_count


def ALLIES_model(args, original_question, tokenizer, model, cpu_index, passage_embedding2id, passages, log_file_2, index):
    all_token_count, all_api_times, all_retrieval_times, directly_answer_token_count, \
    directly_cal_score_token_count, gen_background_token_count, summary_token_count, \
    answer_with_evidence_token_count, cal_score_with_evidence_token_count, expand_question_token_count = 0,0,0,0,0,0,0,0,0,0
   
    ## 第一个节点
    ## 直接回答
    answer, token_count = directly_answer(args, original_question)
    # answer = normalize_answer(answer.replace('Answer', ''))
    all_token_count += token_count
    directly_answer_token_count += token_count
    all_api_times += 1
    ## 直接打分
    score, token_count = directly_cal_score(args, original_question, answer)
    all_token_count += token_count
    directly_cal_score_token_count += token_count
    all_api_times += 1
    S_previous = [[original_question, [], [], answer, score]]

    ## 第二个节点
    ## 将original query的相关文档召回
    if args.retrieval_type =='retrieve':
        relevant_doc = ' '.join(retrieval(original_question, args.topK, tokenizer, model, cpu_index, passage_embedding2id, passages, args))
        all_retrieval_times += 1
    elif args.retrieval_type =='generate':
        relevant_doc, token_count = gen_background(args, original_question)
        gen_background_token_count += token_count
        all_token_count += token_count
        all_api_times += 1

    ## 总结该相关文档
    if args.summary == 'yes':
        summary_doc, token_count = summary(args, original_question, relevant_doc)
        all_token_count += token_count
        summary_token_count += token_count
        all_api_times += 1
    else:
        summary_doc = relevant_doc

    ## 尝试基于evidence回答
    answer, token_count = answer_with_evidence(args, original_question, [original_question], [summary_doc])
    # answer = normalize_answer(answer.replace('Answer', ''))
    all_token_count += token_count
    answer_with_evidence_token_count += token_count
    all_api_times += 1
    # 基于问题、evidence、答案以及自身的知识，打分
    score, token_count = cal_score_with_evidence(args, original_question, answer, [original_question], [summary_doc])
    all_token_count += token_count
    cal_score_with_evidence_token_count += token_count
    all_api_times += 1

    S_previous.append([original_question, [original_question], [summary_doc], answer, score])

    for _ in range(args.beam_Depth):
        S_current = []
        for elem in S_previous:
            ## 扩展问题
            extended_questions, token_count = expand_question(args, elem[0], elem[1], elem[2])
            expand_question_token_count += token_count
            all_token_count += token_count
            all_api_times += 1

            ## 针对每个问题
            for question in extended_questions:
                ## 召回doc
                if args.retrieval_type =='retrieve':
                    relevant_doc = ' '.join(retrieval(question, args.topK, tokenizer, model, cpu_index, passage_embedding2id, passages, args))
                    all_retrieval_times += 1
                elif args.retrieval_type =='generate':
                    relevant_doc, token_count = gen_background(args, question)
                    gen_background_token_count += token_count
                    all_token_count += token_count
                    all_api_times += 1

                ## 总结之前的evidence
                if args.summary == 'yes':
                    summary_doc, token_count = summary(args, original_question, relevant_doc)
                    summary_token_count += token_count
                    all_token_count += token_count
                    all_api_times += 1
                else:
                    summary_doc = relevant_doc

                ## 尝试基于evidence回答
                answer, token_count = answer_with_evidence(args, original_question, elem[1] + [question], elem[2] + [summary_doc])
                # answer = normalize_answer(answer.replace('Answer', ''))
                answer_with_evidence_token_count += token_count
                all_token_count += token_count
                all_api_times += 1

                # 基于问题、evidence、答案以及自身的知识，打分
                score, token_count = cal_score_with_evidence(args, original_question, answer, elem[1] + [question],
                                                             elem[2] + [summary_doc])
                cal_score_with_evidence_token_count += token_count
                all_token_count += token_count
                all_api_times += 1

                ## 加入当前队列中
                new_elem = [original_question, elem[1] + [question], elem[2] + [summary_doc], answer, score]
                S_current.append(new_elem)

        ## 只保留最重要的几个
        S_current.sort(key=lambda x: x[4], reverse=True)
        S_previous = S_current[:args.beam_size]

        ## 判断中间的一个打分是否足够高，如果很高，直接输出答案
        for elem in S_previous:
            if elem[4] >= args.threshold:
                result = {'all_token_count':all_token_count, 'all_api_times':all_api_times, 'all_retrieval_times':all_retrieval_times, \
                        'directly_answer_token_count':directly_answer_token_count,'directly_cal_score_token_count':directly_cal_score_token_count, \
                        'gen_background_token_count':gen_background_token_count, 'summary_token_count':summary_token_count, 'answer_with_evidence_token_count':answer_with_evidence_token_count, 
                        'cal_score_with_evidence_token_count':cal_score_with_evidence_token_count, 'expand_question_token_count':expand_question_token_count}
                write(log_file_2, f'{index}: {S_current}')
                return elem[-2], S_current, result

    S_current.sort(key=lambda x: x[4], reverse=True)
    write(log_file_2, f'{index}: {S_current}')
    final_answer = S_current[0][-2]
    
    result = {'all_token_count':all_token_count, 'all_api_times':all_api_times, 'all_retrieval_times':all_retrieval_times, \
            'directly_answer_token_count':directly_answer_token_count,'directly_cal_score_token_count':directly_cal_score_token_count, \
            'gen_background_token_count':gen_background_token_count, 'summary_token_count':summary_token_count, 'answer_with_evidence_token_count':answer_with_evidence_token_count, 
            'cal_score_with_evidence_token_count':cal_score_with_evidence_token_count, 'expand_question_token_count':expand_question_token_count}
    
    return final_answer, S_current, result