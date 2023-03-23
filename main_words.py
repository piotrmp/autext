import torch, pathlib, random, sys
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Module, Embedding, LSTM, Linear, LogSoftmax, NLLLoss
from torch.optim import Adam

from features.probabilistic import ProbabilisticFeatures

random.seed(10)

language = 'en'
task = 'subtask_2'
if len(sys.argv) == 3:
    language = sys.argv[1]
    task = sys.argv[2]

print("Loading data...")
train_text = []
test_text = []
train_Y = []
test_Y = []
path = pathlib.Path.home() / 'data' / 'autext' / 'data' / task / language / 'train.tsv'
for i, line in enumerate(open(path)):
    parts = line.strip().split('\t')
    sentence = parts[1]
    if sentence == 'text':
        continue
    Y = None
    if task == 'subtask_1':
        if parts[2] == 'generated':
            Y = 1
        elif parts[2] == 'human':
            Y = 0
    if task == 'subtask_2':
        if parts[2] == 'A':
            Y = 0
        elif parts[2] == 'B':
            Y = 1
        elif parts[2] == 'C':
            Y = 2
        elif parts[2] == 'D':
            Y = 3
        elif parts[2] == 'E':
            Y = 4
        elif parts[2] == 'F':
            Y = 5
    if random.random() < 0.2:
        test_text.append(sentence)
        test_Y.append(Y)
    else:
        train_text.append(sentence)
        train_Y.append(Y)
    if i > 1000:
        break

train_Y = np.array(train_Y)
test_Y = np.array(test_Y)

print("Loaded data with " + str(len(train_Y)) + " training instances and " + str(len(test_Y)) + " test instances.")

# Preparing feature generators
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
local_device = torch.device('cpu')

perp = ProbabilisticFeatures(device, local_device, language)
feature_generators = [perp]

print("Generating features...")
train_X = []
test_X = []
train_text = train_text
test_text = test_text
for feature_generator in feature_generators:
    train_X.append(np.array(feature_generator.word_features(train_text)))
    test_X.append(np.array(feature_generator.word_features(test_text)))

train_X = np.concatenate(train_X, axis=2)
test_X = np.concatenate(test_X, axis=2)

print("Building a model...")
BATCH_SIZE = 16
train_dataset = TensorDataset(torch.tensor(train_X).float(), torch.tensor(np.array(train_Y)).long())
test_dataset = TensorDataset(torch.tensor(test_X).float(), torch.tensor(np.array(test_Y)).long())
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)


class BiLSTM(Module):
    
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm_layer = LSTM(input_size=train_X.shape[2], hidden_size=16, batch_first=True, bidirectional=True)
        self.linear_layer = Linear(self.lstm_layer.hidden_size * 2, 2 if task == 'subtask_1' else 6)
        self.softmax_layer = LogSoftmax(1)
        self.loss_fn = NLLLoss()
    
    def forward(self, x):
        _, (hidden_state, _) = self.lstm_layer(x)
        transposed = torch.transpose(hidden_state, 0, 1)
        reshaped = torch.reshape(transposed, (transposed.shape[0], -1))
        scores = self.linear_layer(reshaped)
        logprobabilities = self.softmax_layer(scores)
        return logprobabilities
    
    def compute_loss(self, pred, true):
        output = self.loss_fn(pred, true)
        return output
    
    @staticmethod
    def postprocessing(Y):
        decisions = Y.argmax(1).to(local_device).numpy()
        return decisions


def train_loop(dataloader, model, optimizer, device, skip_visual=False):
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


def eval_loop(dataloader, model, device, skip_visual=False):
    print("Evaluating...")
    model.eval()
    progress_bar = tqdm(range(len(dataloader)), ascii=True, disable=skip_visual)
    correct = 0
    size = 0
    TPs = 0
    FPs = 0
    FNs = 0
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            pred = model.postprocessing(model(X))
            Y = Y.numpy()
            eq = np.equal(Y, pred)
            size += len(eq)
            correct += sum(eq)
            TPs += sum(np.logical_and(np.equal(Y, 1.0), np.equal(pred, 1.0)))
            FPs += sum(np.logical_and(np.equal(Y, 0.0), np.equal(pred, 1.0)))
            FNs += sum(np.logical_and(np.equal(Y, 1.0), np.equal(pred, 0.0)))
            progress_bar.update(1)
    print('Accuracy: ' + str(correct / size))
    print('Binary F1: ' + str(2 * TPs / (2 * TPs + FPs + FNs)))

model = BiLSTM().to(device)
print("Preparing training")
model = model.to(device)
learning_rate = 1e-3
optimizer = Adam(model.parameters(), lr=learning_rate)
skip_visual = False
eval_loop(test_loader, model, device, skip_visual)
for epoch in range(10):
    print("EPOCH " + str(epoch + 1))
    train_loop(train_loader, model, optimizer, device, skip_visual)
    eval_loop(test_loader, model, device, skip_visual)
