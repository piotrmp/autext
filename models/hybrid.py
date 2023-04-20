from tqdm.auto import tqdm
from torch.nn import Module, Embedding, LSTM, Linear, LogSoftmax, NLLLoss
import torch
import numpy as np
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


def train_loop(dataloader, model, optimizer, device, local_device, skip_visual=False):
    print("Training...")
    model.train()
    progress_bar = tqdm(range(len(dataloader)), ascii=True, disable=skip_visual)
    losses = []
    for X, Y in dataloader:
        X = X.to(device)
        Y = Y.to(device)
        pred = model(X)
        loss = model.compute_loss(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().to(local_device).numpy())
        progress_bar.update(1)
    print('Train loss: ' + str(np.mean(losses)))


def eval_loop(dataloader, model, device, local_device, skip_visual=False):
    print("Evaluating...")
    model.eval()
    progress_bar = tqdm(range(len(dataloader)), ascii=True, disable=skip_visual)
    correct = 0
    size = 0
    TPs = 0
    FPs = 0
    FNs = 0
    preds = []
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            pred = model.postprocessing(model(X))
            preds.append(pred)
            Y = Y.numpy()
            eq = np.equal(Y, pred)
            size += len(eq)
            correct += sum(eq)
            TPs += sum(np.logical_and(np.equal(Y, 1.0), np.equal(pred, 1.0)))
            FPs += sum(np.logical_and(np.equal(Y, 0.0), np.equal(pred, 1.0)))
            FNs += sum(np.logical_and(np.equal(Y, 1.0), np.equal(pred, 0.0)))
            progress_bar.update(1)
    print('Accuracy: ' + str(correct / size))
    # print('Binary F1: ' + str(2 * TPs / (2 * TPs + FPs + FNs)))
    preds = np.concatenate(preds)
    return (preds)
