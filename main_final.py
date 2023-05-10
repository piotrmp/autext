import pathlib
import random
import sys

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset, DataLoader
from transformers import RobertaTokenizer

from features.grammar import GrammarFeatures
from features.probabilistic import ProbabilisticFeatures, fixed_len
from features.word_frequency import WordFrequency
from models.bilstm import BiLSTM
from models.hybrid import HybridBiLSTMRoBERTa
from models.training import eval_loop, train_loop

random.seed(10)
torch.manual_seed(10)
np.random.seed(0)

language = 'en'
task = 'subtask_1'
if len(sys.argv) == 3:
    language = sys.argv[1]
    task = sys.argv[2]

model_type = 'BiLSTM'
# model_type = 'Hybrid'
disable_sequence = False

if language == 'en':
    roberta_variant = "roberta-base"
elif language == 'es':
    roberta_variant = 'PlanTL-GOB-ES/roberta-base-bne'

print("Loading data...")
traindev_folds = {}
for line in open(pathlib.Path.home() / 'data' / 'autext' / 'data' / task / language / 'train_5folds.tsv'):
    parts = line.strip().split('\t')
    if len(parts) == 2:
        traindev_folds[parts[0]] = parts[1]

train_text = []
train_Y = []
train_ids = []
dev_text = []
dev_Y = []
dev_ids = []
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
    if traindev_folds[parts[0]] == '0':
        dev_text.append(sentence)
        dev_Y.append(Y)
        dev_ids.append(parts[0])
    else:
        train_text.append(sentence)
        train_Y.append(Y)
        train_ids.append(parts[0])
    #if i > 1000:
    #    break

test_text = []
test_ids = []
testpath = pathlib.Path.home() / 'data' / 'autext' / 'data' / task / language / 'test.tsv'
for i, line in enumerate(open(testpath)):
    parts = line.strip().split('\t')
    sentence = parts[1]
    if sentence == 'text':
        continue
    test_text.append(sentence)
    test_ids.append(parts[0])
    #if i > 1000:
    #    break

train_Y = np.array(train_Y)
dev_Y = np.array(dev_Y)

print("Loaded data with " + str(len(train_Y) + len(dev_Y)) + " instances.")

# Preparing feature generators
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
local_device = torch.device('cpu')

perp = ProbabilisticFeatures(device, local_device, language, disable_sequence)
#gram = GrammarFeatures(device, local_device, language)
#freq = WordFrequency(device, local_device, language)
feature_generators = [perp]

# print("Generating text derivations...")
# text_derivator = TextDerivator(language, device, path.parent / 'train-derived.tsv')
# derivations = text_derivator.derive(all_text)

print("Generating sequence features...")
train_X = []
dev_X = []
test_X = []
for in_text, out_X in zip([train_text, dev_text, test_text], [train_X, dev_X, test_X]):
    for feature_generator in feature_generators:
        out_X.append(np.array(feature_generator.word_features(in_text)))
train_X = np.concatenate(train_X, axis=2)
dev_X = np.concatenate(dev_X, axis=2)
test_X = np.concatenate(test_X, axis=2)

print("Tokenising text for RoBERTa...")
tokenizer = RobertaTokenizer.from_pretrained(roberta_variant)
train_encodings = tokenizer(train_text, padding=True, truncation=True, max_length=fixed_len, return_tensors="pt")
dev_encodings = tokenizer(dev_text, padding=True, truncation=True, max_length=fixed_len, return_tensors="pt")
test_encodings = tokenizer(test_text, padding=True, truncation=True, max_length=fixed_len, return_tensors="pt")

# CUDA memory cleaning
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(str(torch.cuda.get_device_properties(0).total_memory))
    print(str(torch.cuda.memory_reserved(0)))
    print(str(torch.cuda.memory_allocated(0)))

train_input_ids = train_encodings['input_ids']
train_attention_mask = train_encodings['attention_mask']

dev_input_ids = dev_encodings['input_ids']
dev_attention_mask = dev_encodings['attention_mask']

test_input_ids = test_encodings['input_ids']
test_attention_mask = test_encodings['attention_mask']

print("Building a model...")
BATCH_SIZE = 16
train_dataset = TensorDataset(torch.tensor(train_X).float(), train_input_ids, train_attention_mask,
                              torch.tensor(np.array(train_Y)).long())
dev_dataset = TensorDataset(torch.tensor(dev_X).float(), dev_input_ids, dev_attention_mask,
                            torch.tensor(np.array(dev_Y)).long())
test_dataset = TensorDataset(torch.tensor(test_X).float(), test_input_ids, test_attention_mask,
                             torch.tensor(np.zeros(len(test_text))).long())
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

if model_type == 'BiLSTM':
    model = BiLSTM(train_X.shape[2], task, local_device).to(device)
elif model_type == 'Hybrid':
    model = HybridBiLSTMRoBERTa(train_X.shape[2], task, local_device, roberta_variant, disable_sequence).to(device)

print("Preparing training")
model = model.to(device)
learning_rate = 1e-3
optimizer = Adam(model.parameters(), lr=learning_rate)
milestones = [5] if model_type == 'Hybrid' else []
scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.02)
skip_visual = False
stats_file = open(pathlib.Path.home() / 'data' / 'autext' / 'out' / (task + '_' + language + '_stats.tsv'),
                  'w')
stats_file.write('epoch\ttrain_F1\tdev_F1\n')

eval_loop(dev_loader, model, device, local_device, skip_visual)
for epoch in range(20):
    print("EPOCH " + str(epoch + 1))
    if model_type == 'Hybrid':
        if epoch < 5:
            model.freeze_llm()
        else:
            model.unfreeze_llm()
    train_f1 = train_loop(train_loader, model, optimizer, scheduler, device, local_device, skip_visual)
    dev_f1 = eval_loop(dev_loader, model, device, local_device, skip_visual, test=False)
    test_preds, test_probs = eval_loop(test_loader, model, device, local_device, skip_visual, test=True)
    
    stats_file.write(str(epoch + 1) + '\t' + str(train_f1) + '\t' + str(dev_f1) + '\n')
    
    with open(pathlib.Path.home() / 'data' / 'autext' / 'out' / (
            task + '_' + language + '_preds_' + str(epoch + 1) + '.tsv'), 'w') as f:
        f.write('id\tlabel\n')
        for test_id, pred in zip(test_ids, test_preds):
            label = ['human', 'generated'][pred] if task == 'subtask_1' else ['A', 'B', 'C', 'D', 'E', 'F'][pred]
            f.write(test_id + '\t' + label + '\n')
    
    with open(pathlib.Path.home() / 'data' / 'autext' / 'out' / (
            task + '_' + language + '_probs_test_' + str(epoch + 1) + '.tsv'), 'w') as f:
        f.write('id\t' + '\t'.join(
            ['human', 'generated'] if task == 'subtask_1' else ['A', 'B', 'C', 'D', 'E', 'F']) + '\n')
        for test_id, prob in zip(test_ids, test_probs):
            f.write(test_id + '\t' + '\t'.join([str(x) for x in prob]) + '\n')
    
    _, dev_probs = eval_loop(dev_loader, model, device, local_device, skip_visual, test=True)
    _, train_probs = eval_loop(train_loader, model, device, local_device, skip_visual, test=True)
    with open(pathlib.Path.home() / 'data' / 'autext' / 'out' / (
            task + '_' + language + '_probs_traindev_' + str(epoch + 1) + '.tsv'), 'w') as f:
        f.write('id\t' + '\t'.join(
            ['human', 'generated'] if task == 'subtask_1' else ['A', 'B', 'C', 'D', 'E', 'F']) + '\n')
        for test_id, prob in zip(train_ids + dev_ids, [x for x in train_probs] + [x for x in dev_probs]):
            f.write(test_id + '\t' + '\t'.join([str(x) for x in prob]) + '\n')

stats_file.close()
print("The end!")
