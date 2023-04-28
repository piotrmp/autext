from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import defaultdict
import re

from features.feature_generator import FeatureGenerator
from tqdm.auto import tqdm

fixed_len = 1024
BATCH_SIZE = 16
#eps = 1e-40

class WordFrequency(FeatureGenerator):
    def __init__(self, device, local_device, language):
        self.device = device
        self.local_device = local_device
        self.language = language

        if language == 'en':
            self.tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            word_freq_matrix_df = pd.read_csv(f"resources/{language}/word_freq_matrix.tsv.gz", compression='gzip', header=0, sep='\t', quotechar='"')
            word_freq_dict = word_freq_matrix_df[word_freq_matrix_df.word.notnull()].set_index('word').freq.to_dict()
            self.word_freq_dict = defaultdict(lambda: 1, word_freq_dict)


        elif language == 'es':
            self.tokenizer = GPT2TokenizerFast.from_pretrained("PlanTL-GOB-ES/gpt2-base-bne")
            self.tokenizer.pad_token = '?'

            word_freq_matrix_df = pd.read_csv(f"resources/{language}/word_freq_matrix.tsv.gz", compression='gzip', header=0, sep='\t', quotechar='"')
            word_freq_dict = word_freq_matrix_df[word_freq_matrix_df.word.notnull()].set_index('word').freq.to_dict()
            self.word_freq_dict = defaultdict(lambda: 1, word_freq_dict)

    

    def word_features(self, sentences):
        FEATURE_NUM = 1
        results = np.zeros((len(sentences), fixed_len, FEATURE_NUM))

        print("Computing word frequency")
            
        for i, sentence in tqdm(enumerate(sentences)):
            
            encoded = self.tokenizer(sentence, return_offsets_mapping=True)

            # get word ids and positions
            desired_output = []
            for word_id in encoded.word_ids():
                if word_id is not None:
                    start, end = encoded.word_to_tokens(word_id)
                    if start == end - 1:
                        tokens = [start]
                    else:
                        tokens = [start, end-1]
                    if len(desired_output) == 0 or desired_output[-1] != tokens:
                        desired_output.append(tokens)

            # regroup subwords into words and calculate word frequency
            pos_subpiece = [sentence[p1:p2] for p1, p2 in encoded['offset_mapping']]
            sentence_parts = []
            sentence_freq = []
            for pos in desired_output:
                if len(pos) == 1:
                    w = pos_subpiece[pos[0]].strip()
                    sentence_parts.append(w)
                    w_f = self.word_freq_dict.get(w, 1)
                    sentence_freq.append(np.log(w_f))
                if len(pos) == 2:
                    w = "".join(pos_subpiece[pos[0]:pos[1]]).strip()
                    sentence_parts.append(w)
                    w_f = self.word_freq_dict.get(w, 1)
                    sentence_freq.append(np.log(w_f))

            freq_all = [sentence_freq[pos] if pos != None else 1 for pos in encoded.word_ids()]

            # replace values in results array
            for j,freq in enumerate(freq_all[:fixed_len]):
                results[i][j] = freq

        return results
