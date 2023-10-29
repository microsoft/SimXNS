from dataclasses import dataclass
import torch
import numpy as np
from collections import Counter
from dataclasses import dataclass
from datasets import  load_metric
from rouge_score import rouge_scorer

import nltk
import random
from typing import Optional, List

class compute_metric:
    def __init__(self):
        self.rouge_metric = load_metric('rouge')
        self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types =  ["rougeL"], use_stemmer=True)
        self.bleu_scorer = load_metric('bleu')
        self.meteor_scorer = load_metric('meteor')

    def postprocess_text_bleu(self, preds, labels):
        preds = [nltk.word_tokenize(pred) for pred in preds]
        labels = [nltk.word_tokenize(label) for label in labels]
        
        return preds, labels

    def __call__(self, preds, labels):
        # preds, labels = eval_preds
        result = {}
        # Some simple post-processing
        preds_bleu, labels_bleu = self.postprocess_text_bleu(preds, labels)
        result['rougeL'] = self.rouge_metric.compute(predictions=preds, references=labels, use_stemmer=True)['rougeL'].mid.fmeasure * 100

        result['bleu_4'] = self.bleu_scorer.compute(predictions=preds_bleu, references=[[l] for l in labels_bleu], max_order=4)['bleu'] * 100

        result['meteor'] = self.meteor_scorer.compute(predictions=preds_bleu, references=labels_bleu)['meteor'] * 100
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def compute_rouge(self, preds, labels):
        result = {}
        # Some simple post-processing
        preds_bleu, labels_bleu = self.postprocess_text_bleu(preds, labels)
        result['rougeL'] = self.rouge_metric.compute(predictions=preds, references=labels, use_stemmer=True)['rougeL'].mid.fmeasure * 100
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def get_candidates(self, targets: List[str], preds: List[str], num_cand:int, num_cand_picked:int, strategy:str, gold_as_positive=False):
        """
        args:
            targets: list of targets for each sample
            preds: list of predictions, length == len(targets) * num_cand
            num_cand: number of candidates
            num_cand_picked: number of returned indices per sample 
            strategy: how to select num_cand_picked negatives from the preds
            gold_as_positive: use the gold or use the one of preds with the highest reward as the positive.
        returns:
            indices: Torch tensor, (B * (C-1), ), since the positive candidate is not generated from the generator
            candiates: candidates, with the length of len(targets) * num_cand_picked
            NOTE: We should always keep the positive sequences in the first candidate for each sample
        """
        preds_bleu, targets_bleu = self.postprocess_text_bleu(preds, targets)
        # preds_meteor = [' '.join(pred) for pred in preds_bleu]
        # targets_meteor = [' '.join(label) for label in targets_bleu]
        # print(targets_meteor)

        indices = []
        candidates = []
        rewards = []
        for i,t in enumerate(targets):
            scores = []
            ps = preds[i * num_cand: (i+1)*num_cand]
            ps_bleu = preds_bleu[i * num_cand: (i+1)*num_cand]
            for j,p in enumerate(ps):
                if len(ps_bleu[j]) == 0:
                    rouge_score = 0
                    bleu_score = 0
                    meteor_score = 0
                else:
                    # rouge_score = self.rouge_metric.compute(predictions=[p], references=[t], use_stemmer=True)['rougeL'].mid.fmeasure
                    bleu_score = self.bleu_scorer.compute(predictions = [ps_bleu[j]], references = [[targets_bleu[i]]], max_order = 4)['bleu']
                    meteor_score = self.meteor_scorer.compute(predictions = [ps_bleu[j]], references = [targets_bleu[i]])['meteor']
                # reward = rouge_score + bleu_score + meteor_score
                reward = bleu_score + meteor_score
                scores.append((j + i * num_cand, reward , p))

            scores = sorted(scores, key = lambda x: x[1], reverse=True)
            
            if gold_as_positive:
                # rouge_score = self.rouge_metric.compute(predictions = [t], references= [t], use_stemmer=True)['rougeL'].mid.fmeasure
                bleu_score = self.bleu_scorer.compute(predictions = [targets_bleu[i]], references = [[targets_bleu[i]]], max_order = 4)['bleu']
                meteor_score = self.meteor_scorer.compute(predictions = [targets_bleu[i]], references = [targets_bleu[i]])['meteor']
                # reward = rouge_score + bleu_score + meteor_score
                reward = bleu_score + meteor_score
                idx_this = []
                cand_this = [t]
                rewards_this = [reward]
                max_num = num_cand_picked - 1
            else:
                idx_this = [scores[0][0]] # the first as pos
                cand_this = [scores[0][2]]
                rewards_this = [scores[0][1]]
                scores = scores[1:]
                max_num = num_cand_picked - 1

            if strategy == 'random':
                s_for_pick = random.sample(scores, max_num)
                idx_this +=  [s[0] for s in s_for_pick]
                cand_this +=  [s[2] for s in s_for_pick]
                rewards_this += [s[1] for s in s_for_pick]
            else:
                if strategy == 'top':
                    idx_this +=  [s[0] for s in scores[:max_num]]
                    cand_this +=  [s[2] for s in scores[:max_num]]
                    rewards_this += [s[1] for s in scores[:max_num]]
                elif strategy == 'bottom':
                    idx_this +=  [s[0] for s in scores[-max_num:]]
                    cand_this +=  [s[2] for s in scores[-max_num:]]
                    rewards_this += [s[1] for s in scores[-max_num:]]
                elif strategy == 'top-bottom':
                    n_top = max_num // 2
                    n_bottom = max_num - n_top
                    idx_this +=  [s[0] for s in scores[:n_top]]
                    cand_this += [s[2] for s in scores[:n_top]]
                    idx_this +=  [s[0] for s in scores[-n_bottom:]]
                    cand_this += [s[2] for s in scores[-n_bottom:]]
                    rewards_this += [s[1] for s in scores[:n_top]]
                    rewards_this += [s[1] for s in scores[-n_bottom:]]

            indices += idx_this
            candidates += cand_this
            rewards.append(rewards_this)
        return candidates, torch.FloatTensor(rewards)
        # return torch.LongTensor(indices), candidates, torch.FloatTensor(rewards)

    def compute_metric_score(self, target:str, preds:List[str], metric:str):
        score_list = []
        preds_bleu, targets_bleu = self.postprocess_text_bleu(preds, [target])

        for i, pred in enumerate(preds_bleu):
            if metric=='rouge-l':
                score = self.rouge_scorer.score(target=target, prediction=preds[i])['rougeL'].fmeasure
                # score = self.rouge_metric.compute(predictions=[preds[i]], references=[target], use_stemmer=True)['rougeL'].mid.fmeasure
            elif metric =='bleu':
                score = self.bleu_scorer.compute(predictions = [preds_bleu[i]], references = [[targets_bleu[0]]], max_order = 4)['bleu']
            elif metric=='meteor':
                score = self.meteor_scorer.compute(predictions = [preds_bleu[i]], references = [targets_bleu[0]])['meteor']
            else:
                raise ValueError()
            score_list.append(score)
        return score_list


