from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

# TODO generalise to load other models
model_id = "gpt2"
fixed_len = 1024


class Perplexity():
    def __init__(self, device, local_device):
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.device = device
        self.local_device = local_device
        max_length = self.model.config.n_positions
        assert (max_length >= fixed_len)
    
    def perplexity(self, sentence):
        #TODO make it process several sentences at a time
        encodings = self.tokenizer(sentence, truncation=True, max_length=fixed_len, return_tensors="pt")
        
        input_ids = encodings.input_ids.to(self.device)
        target_ids = input_ids.clone()
        with torch.no_grad():
            # TODO obtain per-token probs rather than total loss by using logits
            outputs = self.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss.to(self.local_device)
        
        ppl = float(torch.exp(neg_log_likelihood).numpy())
        return ppl
