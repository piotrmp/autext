from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import numpy as np

from features.feature_generator import FeatureGenerator
from tqdm.auto import tqdm

fixed_len = 1024
BATCH_SIZE = 16


class ProbabilisticFeatures(FeatureGenerator):
    def __init__(self, device, local_device):
        self.device = device
        self.local_device = local_device
        self.models = ["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large"][:2]
    
    def features(self, sentences):
        results = [[] for sentence in sentences]
        for model_id in self.models:
            print("Computing perplexity using " + model_id)
            model = GPT2LMHeadModel.from_pretrained(model_id)
            tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
            model.to(self.device)
            case_counter = 0
            batches = [sentences[i:i + BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)]
            progress_bar = tqdm(range(len(batches)), ascii=True)
            with torch.no_grad():
                for batch in batches:
                    encodings = tokenizer(batch, padding=True, truncation=True, max_length=fixed_len,
                                          return_tensors="pt")
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    target_ids = encodings['input_ids'].clone()
                    outputs = model(encodings['input_ids'], attention_mask=encodings['attention_mask'],
                                    labels=target_ids)
                    logits = outputs['logits']
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = target_ids[..., 1:].contiguous()
                    probs = torch.nn.functional.softmax(shift_logits, dim=-1)
                    probs_seen = torch.gather(probs, 2, shift_labels.reshape(shift_labels.shape + (1,))).reshape(
                        shift_labels.shape)
                    greedy = torch.argmax(probs, -1)
                    probs_greedy = torch.gather(probs, 2, greedy.reshape(greedy.shape + (1,))).reshape(
                        greedy.shape)
                    log_quotient = torch.log(probs_seen / probs_greedy)
                    for i in range(shift_logits.shape[0]):
                        # Mask to select only non-padding words
                        mask = encodings['attention_mask'][i][1:].to(torch.bool)
                        # Perplexity computed from probability of observed tokens
                        probs_seen_here = torch.masked_select(probs_seen[i], mask).to(self.local_device).numpy()
                        # probs_seen_here = probs_seen[i].to(self.local_device).numpy()
                        if len(probs_seen_here) == 0:
                            perplexity = 0
                        else:
                            perplexity = np.exp(-np.mean(np.log(probs_seen_here)))
                        results[case_counter].append(perplexity)
                        # Log-quotient of probabilities observed vs greedy choice
                        log_quotient_here = torch.masked_select(log_quotient[i], mask).to(self.local_device).numpy()
                        if len(log_quotient_here) == 0:
                            mean = 0.0
                            std = 0.0
                        else:
                            mean = np.mean(log_quotient_here)
                            std = np.std(log_quotient_here)
                        results[case_counter].append(mean)
                        results[case_counter].append(std)
                        # Fin.
                        case_counter = case_counter + 1
                    progress_bar.update(1)
        return results
    
    def word_features(self, sentences):
        results = np.zeros((len(sentences), fixed_len, 3 * len(self.models)))
        for i_m, model_id in enumerate(self.models):
            print("Computing perplexity using " + model_id)
            model = GPT2LMHeadModel.from_pretrained(model_id)
            tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
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
                    outputs = model(encodings['input_ids'], attention_mask=encodings['attention_mask'],
                                    labels=target_ids)
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
                    results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1], i_m * 3] = log_probs_seen * mask
                    results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1], i_m * 3 + 1] = log_probs_greedy * mask
                    results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1], i_m * 3 + 2] = mask
                    progress_bar.update(1)
        return results
