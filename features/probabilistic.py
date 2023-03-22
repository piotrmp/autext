from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

from features.feature_generator import FeatureGenerator
from tqdm.auto import tqdm

fixed_len = 1024
BATCH_SIZE = 16


class ProbabilisticFeatures(FeatureGenerator):
    def __init__(self, device, local_device, language):
        self.device = device
        self.local_device = local_device
        self.language = language
        if language == 'en':
            self.models = ["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large"]
        elif language == 'es':
            self.models = ["PlanTL-GOB-ES/gpt2-base-bne", "PlanTL-GOB-ES/gpt2-large-bne"]
    
    def word_features(self, sentences):
        results = np.zeros((len(sentences), fixed_len, 3 * len(self.models)))
        for i_m, model_id in enumerate(self.models):
            print("Computing perplexity using " + model_id)
            if self.language == 'en':
                model = GPT2LMHeadModel.from_pretrained(model_id)
                tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
                tokenizer.pad_token = tokenizer.eos_token
            elif self.language == 'es':
                model = AutoModelForCausalLM.from_pretrained(model_id, is_decoder=True)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                tokenizer.pad_token = '?'
            model.to(self.device)
            batches = [sentences[i:i + BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)]
            progress_bar = tqdm(range(len(batches)), ascii=True)
            with torch.no_grad():
                for i_b, batch in enumerate(batches):
                    # Warning: tokenising for each model, but assuming all tokenisers produce aligned outputs
                    encodings = tokenizer(batch, padding=True, truncation=True, max_length=fixed_len,
                                          return_tensors="pt")
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    target_ids = encodings['input_ids'].clone()
                    outputs = model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
                    logits = outputs['logits']
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = target_ids[..., 1:].contiguous()
                    probs = torch.nn.functional.softmax(shift_logits, dim=-1)
                    probs_seen = torch.gather(probs, 2, shift_labels.reshape(shift_labels.shape + (1,))).reshape(
                        shift_labels.shape)
                    log_probs_seen = torch.log(probs_seen).to(self.local_device).numpy()
                    greedy = torch.argmax(probs, -1)
                    probs_greedy = torch.gather(probs, 2, greedy.reshape(greedy.shape + (1,))).reshape(
                        greedy.shape)
                    log_probs_greedy = torch.log(probs_greedy).to(self.local_device).numpy()
                    mask = np.array([x[1:].to(self.local_device).numpy() for x in encodings['attention_mask']])
                    results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1],
                    i_m * 3] = log_probs_seen * mask
                    results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1],
                    i_m * 3 + 1] = log_probs_greedy * mask
                    results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1], i_m * 3 + 2] = mask
                    progress_bar.update(1)
        return results
