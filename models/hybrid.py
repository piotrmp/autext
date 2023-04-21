from torch.nn import Module, LSTM, Linear, LogSoftmax, NLLLoss
import torch
from transformers import RobertaModel


class HybridBiLSTMRoBERTa(Module):
    
    def __init__(self, seq_feature_len, task, local_device):
        super(HybridBiLSTMRoBERTa, self).__init__()
        self.lstm_layer = LSTM(input_size=seq_feature_len, hidden_size=64, batch_first=True, bidirectional=True)
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.linear_layer = Linear(self.lstm_layer.hidden_size * 2 + self.roberta.config.hidden_size,
                                   2 if task == 'subtask_1' else 6)
        self.softmax_layer = LogSoftmax(1)
        self.loss_fn = NLLLoss()
        self.local_device = local_device
    
    def forward(self, x_sequence_features, x_input_ids, x_attention_mask):
        _, (hidden_state, _) = self.lstm_layer(x_sequence_features)
        transposed = torch.transpose(hidden_state, 0, 1)
        reshaped = torch.reshape(transposed, (transposed.shape[0], -1))
        roberta_output = self.roberta(x_input_ids, x_attention_mask)['pooler_output']
        concatenated = torch.cat((reshaped, roberta_output), -1)
        scores = self.linear_layer(concatenated)
        logprobabilities = self.softmax_layer(scores)
        return logprobabilities
    
    def compute_loss(self, pred, true):
        output = self.loss_fn(pred, true)
        return output
    
    def postprocessing(self, Y):
        decisions = Y.argmax(1).to(self.local_device).numpy()
        return decisions

