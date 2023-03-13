from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import numpy as np

from features.feature_generator import FeatureGenerator
from tqdm.auto import tqdm

fixed_len = 1024
BATCH_SIZE = 16


class Perplexity(FeatureGenerator):
    def __init__(self, device, local_device):
        self.device = device
        self.local_device = local_device
    
    def features(self, sentences):
        results = [[] for sentence in sentences]
        for model_id in ["distilgpt2", "gpt2", "gpt2-medium"]:
            print("Computing perplexity using "+model_id)
            model = GPT2LMHeadModel.from_pretrained(model_id)
            tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
            model.to(self.device)
            celoss = CrossEntropyLoss(ignore_index=tokenizer.encode(tokenizer.eos_token)[0])
            counter = 0
            batches = [sentences[i:i + BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)]
            progress_bar = tqdm(range(len(batches)), ascii=True)
            for batch in batches:
                encodings = tokenizer(batch, padding=True, truncation=True, max_length=fixed_len, return_tensors="pt")
                encodings = {k: v.to(self.device) for k, v in encodings.items()}
                target_ids = encodings['input_ids'].clone()
                with torch.no_grad():
                    outputs = model(encodings['input_ids'], attention_mask=encodings['attention_mask'],
                                    labels=target_ids)
                logits = outputs['logits']
                shift_logits = logits[..., :-1, :]
                shift_labels = target_ids[..., 1:]
                for i in range(shift_logits.shape[0]):
                    loss = celoss(shift_logits[i], shift_labels[i]).to(self.local_device).numpy()
                    loss = float(loss)
                    results[counter].append(np.exp(loss))
                    counter = counter + 1
                progress_bar.update(1)
        return results
