from torch.nn import Module, LSTM, Linear, LogSoftmax, NLLLoss
import torch
from transformers import RobertaModel


class HybridBiLSTMRoBERTa(Module):
    
    def __init__(self, seq_feature_len, task, local_device, roberta_variant, disable_sequence):
        super(HybridBiLSTMRoBERTa, self).__init__()
        self.llm = RobertaModel.from_pretrained(roberta_variant)
        linear_size = self.llm.config.hidden_size
        if not disable_sequence:
            self.lstm_layer = LSTM(input_size=seq_feature_len, hidden_size=64, batch_first=True, bidirectional=True)
            linear_size += self.lstm_layer.hidden_size * 2
        self.linear_layer = Linear(linear_size, 2 if task == 'subtask_1' else 6)
        self.softmax_layer = LogSoftmax(1)
        self.loss_fn = NLLLoss()
        self.local_device = local_device
        self.disable_sequence = disable_sequence
    
    def forward(self, x_sequence_features, x_input_ids, x_attention_mask):
        roberta_output = self.llm(x_input_ids, x_attention_mask)['pooler_output']
        if not self.disable_sequence:
            _, (hidden_state, _) = self.lstm_layer(x_sequence_features)
            transposed = torch.transpose(hidden_state, 0, 1)
            reshaped = torch.reshape(transposed, (transposed.shape[0], -1))
            concatenated = torch.cat((reshaped, roberta_output), -1)
        else:
            concatenated = roberta_output
        scores = self.linear_layer(concatenated)
        logprobabilities = self.softmax_layer(scores)
        return logprobabilities
    
    def compute_loss(self, pred, true):
        output = self.loss_fn(pred, true)
        return output
    
    def postprocessing(self, Y):
        decisions = Y.argmax(1).to(self.local_device).numpy()
        return decisions

    def freeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = False

    def unfreeze_llm(self):
        for param in self.llm.parameters():
            param.requires_grad = True


