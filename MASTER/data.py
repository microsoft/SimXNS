import random
from dataclasses import dataclass
from typing import List, Dict
import copy
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask


@dataclass
class CondenserCollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    decoder_mlm_probability: float = 0.5
    frequency_dict: dict = None

    def __post_init__(self):
        super(CondenserCollator, self).__post_init__()

        from transformers import BertTokenizer, BertTokenizerFast
        from transformers import RobertaTokenizer, RobertaTokenizerFast
        if isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            self.whole_word_cand_indexes = self._whole_word_cand_indexes_bert
        elif isinstance(self.tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
            self.whole_word_cand_indexes = self. _whole_word_cand_indexes_roberta
        else:
            raise NotImplementedError(f'{type(self.tokenizer)} collator not supported yet')

        self.specials = self.tokenizer.all_special_tokens

    def _whole_word_cand_indexes_bert(self, input_tokens: List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        return cand_indexes

    def _whole_word_cand_indexes_bert_keyword(self, input_tokens: List[str]):
        cand_indexes = []
        cand_tokens = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
                cand_tokens[-1]+=token
            else:
                cand_indexes.append([i])
                cand_tokens.append(token)
        return cand_indexes, [1/self.frequency_dict[ele] if ele in self.frequency_dict else 1 for ele in cand_tokens]

    def _whole_word_cand_indexes_roberta(self, input_tokens: List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                raise ValueError('We expect only raw input for roberta for current implementation')

            if i == 0:
                cand_indexes.append([0])
            elif not token.startswith('\u0120'):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        return cand_indexes

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = self._whole_word_cand_indexes_bert(input_tokens)

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _whole_word_mask_dual(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = self._whole_word_cand_indexes_bert(input_tokens)

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * 0.5))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        dual_mask_labels = [0 if i in covered_indexes else 1 for i in range(len(input_tokens))]
        return mask_labels, dual_mask_labels

    def _whole_word_mask_decoder(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = self._whole_word_cand_indexes_bert(input_tokens)

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.decoder_mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _whole_word_mask_decoder_keyword(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes, cand_probs = self._whole_word_cand_indexes_bert_keyword(input_tokens)
        assert len(cand_indexes) == len(cand_probs)

        num_to_predict = min(max_predictions, max(1, int(round(len([token for token in input_tokens if token not in self.specials]) * self.decoder_mlm_probability))))
        masked_lms = []
        covered_indexes = set()

        new_cand_indexes = copy.deepcopy(cand_indexes)
        new_cand_probs = copy.deepcopy(cand_probs)
        while len(masked_lms) < num_to_predict:
            if len(new_cand_probs)==0:
                break
            selected_index_set = random.choices(new_cand_indexes, weights=new_cand_probs, k=10)

            for index_set in selected_index_set:
                if len(masked_lms) >= num_to_predict:
                    break
                # If adding a whole-word mask would exceed the maximum number of
                # predictions, then just skip this candidate.
                #if len(masked_lms) + len(index_set) > num_to_predict:
                #    continue
                is_any_index_covered = False
                for index in index_set:
                    if index in covered_indexes:
                        is_any_index_covered = True
                        break
                if is_any_index_covered:
                    continue
                for index in index_set:
                    covered_indexes.add(index)
                    masked_lms.append(index)

            new_cand_probs_1 = []
            new_cand_indexes_1 = []
            for cand_index, cand_prob in zip(new_cand_indexes, new_cand_probs):
                if cand_index[0] not in covered_indexes:
                    new_cand_indexes_1.append(cand_index)
                    new_cand_probs_1.append(cand_prob)

            new_cand_probs, new_cand_indexes = new_cand_probs_1, new_cand_indexes_1

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _truncate(self, example: List[int]):
        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)
        if len(example) <= tgt_len:
            return example
        trunc = len(example) - tgt_len
        trunc_left = random.randint(0, trunc)
        trunc_right = trunc - trunc_left

        truncated = example[trunc_left:]
        if trunc_right > 0:
            truncated = truncated[:-trunc_right]

        if not len(truncated) == tgt_len:
            print(len(example), len(truncated), trunc_left, trunc_right, tgt_len, flush=True)
            raise ValueError
        return truncated

    def _pad(self, seq, val=0):
        tgt_len = self.max_seq_length
        assert len(seq) <= tgt_len
        return seq + [val for _ in range(tgt_len - len(seq))]

    def __call__(self, examples: List[Dict[str, List[int]]]):
        encoded_examples = []
        masks = []
        next_encoder_encoded_examples = []
        next_encoder_masks = []
        next_decoder_encoded_examples = []
        next_decoder_masks = []
        query_encoded_examples = []
        query_masks = []
        gpt_encoded_examples = []
        gpt_masks = []

        mlm_masks = []
        decoder_mlm_masks = []
        next_encoder_mlm_masks = []
        next_decoder_mlm_masks = []
        overlap_encoder_mlm_masks = []
        overlap_decoder_mlm_masks = []
        query_mlm_masks = []
        gpt_mlm_masks = []

        for e in examples:
            e_trunc = self._truncate(e['text'])
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]
            mlm_mask = self._whole_word_mask(tokens)
            mlm_mask = self._pad([0] + mlm_mask)
            mlm_masks.append(mlm_mask)

            decoder_mlm_mask = self._whole_word_mask_decoder_keyword(tokens)
            decoder_mlm_mask = self._pad([0] + decoder_mlm_mask)
            decoder_mlm_masks.append(decoder_mlm_mask)

            long_query = []
            for query in e['queries']:
                long_query.extend(query+[102])
            long_query = self._truncate(long_query)
            query_tokens = [self.tokenizer._convert_id_to_token(tid) for tid in long_query]
            query_mlm_mask = self._whole_word_mask_decoder(query_tokens)
            query_mlm_mask = self._pad([0] + query_mlm_mask)
            query_mlm_masks.append(query_mlm_mask)

            gpt_e_trunc = self._truncate(e['next'][0])
            gpt_tokens = [self.tokenizer._convert_id_to_token(tid) for tid in gpt_e_trunc]
            if len(gpt_tokens)==0:
                gpt_e_trunc = e_trunc
                gpt_tokens = tokens
            gpt_mlm_mask = self._whole_word_mask_decoder(gpt_tokens)
            gpt_mlm_mask = self._pad([0] + gpt_mlm_mask)
            gpt_mlm_masks.append(gpt_mlm_mask)

            pat_id = len(tokens)//2
            next_encoder_e_trunc = e_trunc[:pat_id]
            next_encoder_tokens = tokens[:pat_id]
            next_encoder_mlm_mask = self._whole_word_mask(next_encoder_tokens)
            next_encoder_mlm_mask = self._pad([0] + next_encoder_mlm_mask)
            next_encoder_mlm_masks.append(next_encoder_mlm_mask)

            next_decoder_e_trunc = e_trunc[pat_id:]
            next_decoder_tokens = tokens[pat_id:]
            next_decoder_mlm_mask = self._whole_word_mask_decoder_keyword(next_decoder_tokens)
            next_decoder_mlm_mask = self._pad([0] + next_decoder_mlm_mask)
            next_decoder_mlm_masks.append(next_decoder_mlm_mask)

            overlap_encoder_mlm_mask, overlap_decoder_mlm_mask = self._whole_word_mask_dual(tokens)
            overlap_encoder_mlm_mask = self._pad([0] + overlap_encoder_mlm_mask)
            overlap_encoder_mlm_masks.append(overlap_encoder_mlm_mask)

            overlap_decoder_mlm_mask = self._pad([0] + overlap_decoder_mlm_mask)
            overlap_decoder_mlm_masks.append(overlap_decoder_mlm_mask)

            encoded = self.tokenizer.encode_plus(
                e_trunc,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            masks.append(encoded['attention_mask'])
            encoded_examples.append(encoded['input_ids'])

            encoded_query = self.tokenizer.encode_plus(
                long_query,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            query_masks.append(encoded_query['attention_mask'])
            query_encoded_examples.append(encoded_query['input_ids'])

            encoded_gpt = self.tokenizer.encode_plus(
                gpt_e_trunc,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            gpt_masks.append(encoded_gpt['attention_mask'])
            gpt_encoded_examples.append(encoded_gpt['input_ids'])

            next_encoder_encoded = self.tokenizer.encode_plus(
                next_encoder_e_trunc,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            next_encoder_masks.append(next_encoder_encoded['attention_mask'])
            next_encoder_encoded_examples.append(next_encoder_encoded['input_ids'])

            next_decoder_encoded = self.tokenizer.encode_plus(
                next_decoder_e_trunc,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            next_decoder_masks.append(next_decoder_encoded['attention_mask'])
            next_decoder_encoded_examples.append(next_decoder_encoded['input_ids'])

        inputs, labels = self.torch_mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(mlm_masks, dtype=torch.long)
        )

        decoder_inputs, decoder_labels = self.torch_mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(decoder_mlm_masks, dtype=torch.long)
        )

        query_inputs, query_labels = self.torch_mask_tokens(
            torch.tensor(query_encoded_examples, dtype=torch.long),
            torch.tensor(query_mlm_masks, dtype=torch.long)
        )

        gpt_inputs, gpt_labels = self.torch_mask_tokens(
            torch.tensor(gpt_encoded_examples, dtype=torch.long),
            torch.tensor(gpt_mlm_masks, dtype=torch.long)
        )

        next_encoder_inputs, next_encoder_labels = self.torch_mask_tokens(
            torch.tensor(next_encoder_encoded_examples, dtype=torch.long),
            torch.tensor(next_encoder_mlm_masks, dtype=torch.long)
        )

        next_decoder_inputs, next_decoder_labels = self.torch_mask_tokens(
            torch.tensor(next_decoder_encoded_examples, dtype=torch.long),
            torch.tensor(next_decoder_mlm_masks, dtype=torch.long)
        )

        overlap_encoder_inputs, overlap_encoder_labels = self.torch_mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(overlap_encoder_mlm_masks, dtype=torch.long)
        )

        overlap_decoder_inputs, overlap_decoder_labels = self.torch_mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(overlap_decoder_mlm_masks, dtype=torch.long)
        )
        batch = {
            "input_ids": inputs,
            "labels": labels,
            "decoder_input_ids": decoder_inputs,
            "decoder_labels": decoder_labels,
            "query_input_ids": query_inputs,
            "query_labels": query_labels,
            "gpt_input_ids": gpt_inputs,
            "gpt_labels": gpt_labels,
            "next_encoder_input_ids": next_encoder_inputs,
            "next_encoder_labels": next_encoder_labels,
            "next_decoder_input_ids": next_decoder_inputs,
            "next_decoder_labels": next_decoder_labels,
            "overlap_encoder_input_ids": overlap_encoder_inputs,
            "overlap_encoder_labels": overlap_encoder_labels,
            "overlap_decoder_input_ids": overlap_decoder_inputs,
            "overlap_decoder_labels": overlap_decoder_labels,
            "attention_mask": torch.tensor(masks),
            "query_attention_mask": torch.tensor(query_masks),
            "gpt_attention_mask": torch.tensor(gpt_masks),
            "next_encoder_attention_mask": torch.tensor(next_encoder_masks),
            "next_decoder_attention_mask": torch.tensor(next_decoder_masks),
        }

        return batch


@dataclass
class CoCondenserCollator(CondenserCollator):
    def __call__(self, examples):
        examples = sum(examples, [])
        examples = [{'text': e} for e in examples]

        return super(CoCondenserCollator, self).__call__(examples)


class CoCondenserDataset(Dataset):
    def __init__(self, dataset, data_args):
        self.dataset = dataset
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        spans = self.dataset[item]['spans']
        return random.sample(spans, 2)
