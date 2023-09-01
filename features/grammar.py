from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import language_tool_python
import pandas as pd

from features.feature_generator import FeatureGenerator
from tqdm.auto import tqdm

fixed_len = 128
BATCH_SIZE = 16
#eps = 1e-40

class GrammarFeatures(FeatureGenerator):
    def __init__(self, device, local_device, language):
        self.device = device
        self.local_device = local_device
        self.language = language
        self.difference_window = 5
        print("Initiating grammar check...")
        #static_grammar_df = pd.read_csv(f"resources/static_grammar_check.tsv.gz", compression='gzip', header=0, sep='\t', quotechar='"')
        #self.grammar_check_dict = static_grammar_df.set_index('sentence').sentence_checked.to_dict()

        if language == 'en':
            self.tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # import grammar checker
            #self.grammar_checker = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
            self.grammar_checker = language_tool_python.LanguageTool('en', remote_server='http://localhost:8010')

        elif language == 'es':
            self.tokenizer = GPT2TokenizerFast.from_pretrained("PlanTL-GOB-ES/gpt2-base-bne")
            self.tokenizer.pad_token = '?'
            # import grammar checker
            # self.grammar_checker = language_tool_python.LanguageToolPublicAPI('es')
            self.grammar_checker = language_tool_python.LanguageTool('es', remote_server='http://localhost:8010')
    
    def word_features(self, sentences):
        FEATURE_NUM = 1
        results = np.zeros((len(sentences), fixed_len, FEATURE_NUM))
        #for i_m, model_id in enumerate(self.models):
        print("Computing grammar errors...")
        progress_bar = tqdm(range(len(sentences)), ascii=True)
        for i, sentence in enumerate(sentences):
            # check grammar
            try:
                #sentence_checked = self.grammar_check_dict[sentence]
                sentence_checked = self.grammar_checker.correct(sentence)
            except:
                sentence_checked = sentence

            sentence_tokenized = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(sentence))
            sentence_checked_tokenized = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(sentence_checked))

            diff = self.difference_window
            sentence_results = [word in sentence_checked_tokenized[max(0,i-diff):min(len(sentence_tokenized),i+diff+1)] for i, word in enumerate(sentence_tokenized)]

            # replace values in results array
            for j,grammar_value in enumerate(sentence_results[:fixed_len]):
                results[i][j] = int(grammar_value)
            progress_bar.update(1)
            
        return results
