from transformers import GPT2TokenizerFast
from tokenizers.pre_tokenizers import Whitespace

from features.feature_generator import FeatureGenerator

model_id = "gpt2"


class MultiToken(FeatureGenerator):
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.pre_tokenizer = Whitespace()
    
    def features(self, sentences):
        encodings = self.tokenizer(sentences)
        pre_tokens = [self.pre_tokenizer.pre_tokenize_str(sentence) for sentence in sentences]
        output = []
        for encs, prets in zip(encodings['input_ids'], pre_tokens):
            output.append([len(encs) / len(prets)])
        return output
