from torch.nn import Module, LSTM, Linear, LogSoftmax, NLLLoss
import torch
from transformers import RobertaModel


class MutantRoBERTa(Module):
    
    def __init__(self, task, local_device, roberta_variant):
        super(MutantRoBERTa, self).__init__()
        self.llm = RobertaModel.from_pretrained(roberta_variant)
        linear_size = self.llm.config.hidden_size
        self.linear_layer = Linear(linear_size, 2 if task == 'subtask_1' else 6)
        self.softmax_layer = LogSoftmax(1)
        self.loss_fn = NLLLoss()
        self.local_device = local_device
    
    def forward(self, x_sequence_features, x_input_ids, x_attention_mask):
        pure_embeddings = self.llm.embeddings.word_embeddings(x_input_ids)
        embeddings_part1 = x_sequence_features[:, :pure_embeddings.shape[1], :] * 0.01
        embeddings_part2 = pure_embeddings[:, :, x_sequence_features.shape[2]:]
        mutated_embeddings = torch.cat((embeddings_part1, embeddings_part2), dim=2)
        roberta_output = self.llm(attention_mask = x_attention_mask, inputs_embeds = mutated_embeddings)['pooler_output']
        scores = self.linear_layer(roberta_output)
        logprobabilities = self.softmax_layer(scores)
        return logprobabilities
    
    def compute_loss(self, pred, true):
        output = self.loss_fn(pred, true)
        return output

    def postprocessing(self, Y, argmax=True):
        if argmax:
            decisions = Y.argmax(1).to(self.local_device).numpy()
        else:
            decisions = Y.to(self.local_device).numpy()
        return decisions

    def freeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = False

    def unfreeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = True


