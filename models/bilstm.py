from tqdm.auto import tqdm
from torch.nn import Module, Embedding, LSTM, Linear, LogSoftmax, NLLLoss
import torch
import numpy as np

class BiLSTM(Module):
    
    def __init__(self, feature_len, task, local_device):
        super(BiLSTM, self).__init__()
        self.lstm_layer = LSTM(input_size=feature_len, hidden_size=64, batch_first=True, bidirectional=True)
        self.linear_layer = Linear(self.lstm_layer.hidden_size * 2, 2 if task == 'subtask_1' else 6)
        self.softmax_layer = LogSoftmax(1)
        self.loss_fn = NLLLoss()
        self.local_device = local_device
    
    def forward(self, x, _unused1, _unused2):
        _, (hidden_state, _) = self.lstm_layer(x)
        transposed = torch.transpose(hidden_state, 0, 1)
        reshaped = torch.reshape(transposed, (transposed.shape[0], -1))
        scores = self.linear_layer(reshaped)
        logprobabilities = self.softmax_layer(scores)
        return logprobabilities
    
    def compute_loss(self, pred, true):
        output = self.loss_fn(pred, true)
        return output
    
    def postprocessing(self, Y):
        decisions = Y.argmax(1).to(self.local_device).numpy()
        return decisions
