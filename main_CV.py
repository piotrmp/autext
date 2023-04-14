import torch, pathlib, random, sys
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from models.bilstm import train_loop, eval_loop, BiLSTM

from features.probabilistic import ProbabilisticFeatures

random.seed(10)

language = 'en'
task = 'subtask_1'
if len(sys.argv) == 3:
    language = sys.argv[1]
    task = sys.argv[2]

print("Loading data...")
all_text = []
all_Y = []
all_folds = []
path = pathlib.Path.home() / 'data' / 'autext' / 'data' / task / language / 'train.tsv'
path_CV = pathlib.Path.home() / 'data' / 'autext' / 'data' / task / language / 'train_topic.tsv'
for i, (line, line_CV) in enumerate(zip(open(path), open(path_CV))):
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
    all_text.append(sentence)
    all_Y.append(Y)
    all_folds.append(int(line_CV.strip().split('\t')[1]))
    if i > 1000:
        break

all_Y = np.array(all_Y)
all_folds = np.array(all_folds)

print("Loaded data with " + str(len(all_Y)) + " instances.")

# Preparing feature generators
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
local_device = torch.device('cpu')

perp = ProbabilisticFeatures(device, local_device, language)
feature_generators = [perp]

print("Generating features...")
all_X = []
for feature_generator in feature_generators:
    all_X.append(np.array(feature_generator.word_features(all_text)))

all_X = np.concatenate(all_X, axis=2)

result = np.empty(all_Y.shape)
for fold in np.unique(all_folds):
    print("Working on fold " + str(fold))
    train_X = all_X[all_folds != fold]
    train_Y = all_Y[all_folds != fold]
    test_X = all_X[all_folds == fold]
    test_Y = all_Y[all_folds == fold]
    
    print("Building a model...")
    BATCH_SIZE = 16
    train_dataset = TensorDataset(torch.tensor(train_X).float(), torch.tensor(np.array(train_Y)).long())
    test_dataset = TensorDataset(torch.tensor(test_X).float(), torch.tensor(np.array(test_Y)).long())
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)
    
    model = BiLSTM(all_X.shape[2], task, local_device).to(device)
    print("Preparing training")
    model = model.to(device)
    learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)
    skip_visual = False
    pred = eval_loop(test_loader, model, device, local_device, skip_visual)
    for epoch in range(10):
        print("EPOCH " + str(epoch + 1))
        train_loop(train_loader, model, optimizer, device, local_device, skip_visual)
        pred = eval_loop(test_loader, model, device, local_device, skip_visual)
    result[all_folds == fold] = pred

overall_accuracy = np.mean(result == all_Y)
print('Total accuracy: ' + str(overall_accuracy))
