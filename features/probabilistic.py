import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM

from features.feature_generator import FeatureGenerator

fixed_len = 1024
BATCH_SIZE = 16
eps = 1e-40


class ProbabilisticFeatures(FeatureGenerator):
    def __init__(self, device, local_device, language):
        self.device = device
        self.local_device = local_device
        self.language = language
        if language == 'en':
            self.models = ["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        elif language == 'es':
            self.models = ["PlanTL-GOB-ES/gpt2-base-bne", "PlanTL-GOB-ES/gpt2-large-bne"]  # eq to gpt2 and gpt2-large
    
    def word_features(self, sentences):
        FEATURE_NUM = 3
        results = np.zeros((len(sentences), fixed_len, FEATURE_NUM * len(self.models) + 1))
        reference_tokenisations = {}
        for i_m, model_id in enumerate(self.models):
            print("Computing perplexity using " + model_id)
            if not model_id.startswith('PlanTL-GOB-ES'):
                # Not Spanish model
                model = GPT2LMHeadModel.from_pretrained(model_id)
                tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Spanish model
                model = AutoModelForCausalLM.from_pretrained(model_id, is_decoder=True)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                tokenizer.pad_token = '?'
            model.to(self.device)
            batches = [sentences[i:i + BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)]
            progress_bar = tqdm(range(len(batches)), ascii=True)
            # Random numbers instead of model
            results[:, :, (i_m * FEATURE_NUM):((i_m+1) * FEATURE_NUM )] = np.random.random((len(sentences), fixed_len, FEATURE_NUM))
            continue
            ##
            with torch.no_grad():
                for i_b, batch in enumerate(batches):
                    # Warning: tokenising for each model, but assuming all tokenisers produce aligned outputs
                    encodings = tokenizer(batch, padding=True, truncation=True, max_length=fixed_len,
                                          return_offsets_mapping=True, return_tensors="pt")
                    if i_m == 0:
                        reference_tokenisations[i_b] = encodings['offset_mapping'].numpy()
                    token_mapping = self.compute_mapping(reference_tokenisations[i_b],
                                                         encodings['offset_mapping'].numpy())
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    if self.language == 'es':
                        # For some reason too high word ids are generated by tokeniser
                        encodings['input_ids'][encodings['input_ids'] >= 50261] = 0
                    target_ids = encodings['input_ids'].clone()
                    outputs = model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
                    logits = outputs['logits']
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = target_ids[..., 1:].contiguous()
                    probs = torch.nn.functional.softmax(shift_logits, dim=-1)
                    probs_seen = torch.gather(probs, 2, shift_labels.reshape(shift_labels.shape + (1,))).reshape(
                        shift_labels.shape)
                    greedy = torch.argmax(probs, -1)
                    probs_greedy = torch.gather(probs, 2, greedy.reshape(greedy.shape + (1,))).reshape(
                        greedy.shape)
                    # Prepare the vectors for output
                    log_probs_seen = torch.log(probs_seen).to(self.local_device).numpy()
                    log_probs_seen = np.concatenate((np.array([[0]] * len(batch)), log_probs_seen), axis=1)
                    log_probs_greedy = torch.log(probs_greedy).to(self.local_device).numpy()
                    log_probs_greedy = np.concatenate((np.array([[0]] * len(batch)), log_probs_greedy), axis=1)
                    entropy = torch.sum(torch.log2(probs + eps) * (-probs), dim=-1).to(self.local_device).numpy()
                    entropy = np.concatenate((np.array([[0]] * len(batch)), entropy), axis=1)
                    # Store the mask data
                    mask = np.array([x.to(self.local_device).numpy() for x in encodings['attention_mask']])
                    if i_m == 0:
                        results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1], 0] = mask
                    # Store the data
                    results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1],
                    i_m * FEATURE_NUM + 1] = log_probs_seen * mask
                    results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1],
                    i_m * FEATURE_NUM + 2] = log_probs_greedy * mask
                    results[(i_b * BATCH_SIZE):(i_b * BATCH_SIZE + len(batch)), :mask.shape[1],
                    i_m * FEATURE_NUM + 3] = entropy * mask
                    progress_bar.update(1)
        return results
    
    def compute_mapping(self, reference_mapping, new_mapping):
        if np.array_equal(reference_mapping, new_mapping):
            return None
        else:
            results = []
            for i_b in range(len(reference_mapping)):
                result = {}
                ref = reference_mapping[i_b]
                new = new_mapping[i_b]
                i_r = 0
                for i_n in range(len(new)):
                    while i_r < len(ref) - 1 and ref[i_r][1] <= new[i_n][0]:
                        i_r = i_r + 1
                    if ref[i_r][1] > new[i_n][0] and ref[i_r][0] < new[i_n][1]:
                        result[i_n] = i_r
                        #i_r = i_r + 1
                results.append(result)
            return results
