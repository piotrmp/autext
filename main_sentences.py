import torch, pathlib, random, sys
import numpy as np
from sklearn.linear_model import LogisticRegression

from features.multitoken import MultiToken
from features.probabilistic import ProbabilisticFeatures

random.seed(10)

language = 'en'
task = 'subtask_1'
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
    #if i > 1000:
    #    break

train_Y = np.array(train_Y)
test_Y = np.array(test_Y)

print("Loaded data with " + str(len(train_Y)) + " training instances and " + str(len(test_Y)) + " test instances.")

# Preparing feature generators
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
local_device = torch.device('cpu')

perp = ProbabilisticFeatures(device, local_device)
mult = MultiToken()
feature_generators = [perp, mult]

print("Generating features...")
train_X = []
test_X = []

for feature_generator in feature_generators:
    train_X.append(np.array(feature_generator.features(train_text)))
    test_X.append(np.array(feature_generator.features(test_text)))

train_X = np.concatenate(train_X, axis=1)
test_X = np.concatenate(test_X, axis=1)

print("Building a model...")
model = LogisticRegression(max_iter=10000).fit(train_X, train_Y)

print("Evaluating...")
predictions = model.predict(test_X)
print('Accuracy: ' + str(np.mean(predictions == test_Y)))