import numpy as np
import pathlib
import random
import sys
import torch
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset, DataLoader
from transformers import RobertaTokenizer

from features.grammar import GrammarFeatures
from features.probabilistic import ProbabilisticFeatures, fixed_len
from features.word_frequency import WordFrequency
from models.bilstm import BiLSTM
from models.hybrid import HybridBiLSTMRoBERTa
from models.mutant import MutantRoBERTa
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
# model_type = 'Mutant'
disable_sequence = False

if language == 'en':
    roberta_variant = "roberta-base"
elif language == 'es':
    roberta_variant = 'PlanTL-GOB-ES/roberta-base-bne'

print("Loading data...")
all_text = []
all_Y = []
all_folds = []
all_ids = []
path = pathlib.Path.home() / 'data' / 'autext' / 'data' / task / language / 'train.tsv'
path_CV = pathlib.Path.home() / 'data' / 'autext' / 'data' / task / language / 'train_topic.tsv'
for i, (line, line_CV) in enumerate(zip(open(path), open(path_CV))):
    parts = line.strip().split('\t')
    sentence = parts[1]
    if sentence == 'text':
        continue
    all_ids.append(parts[0])
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
    #if i > 1000:
    #    break

all_Y = np.array(all_Y)
all_folds = np.array(all_folds)

print("Loaded data with " + str(len(all_Y)) + " instances.")

# Preparing feature generators
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
local_device = torch.device('cpu')

perp = ProbabilisticFeatures(device, local_device, language, disable_sequence)
# gram = GrammarFeatures(device, local_device, language)
# freq = WordFrequency(device, local_device, language)
feature_generators = [perp]

# print("Generating text derivations...")
# text_derivator = TextDerivator(language, device, path.parent / 'train-derived.tsv')
# derivations = text_derivator.derive(all_text)

print("Generating sequence features...")
all_X = []
for text_variant in [all_text]:  # , derivations]:
    for feature_generator in feature_generators:
        all_X.append(np.array(feature_generator.word_features(text_variant)))
all_X = np.concatenate(all_X, axis=2)

print("Tokenising text for RoBERTa...")
tokenizer = RobertaTokenizer.from_pretrained(roberta_variant)
all_encodings = tokenizer(all_text, padding=True, truncation=True, max_length=fixed_len, return_tensors="pt")

# CUDA memory cleaning
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(str(torch.cuda.get_device_properties(0).total_memory))
    print(str(torch.cuda.memory_reserved(0)))
    print(str(torch.cuda.memory_allocated(0)))

result = np.empty(all_Y.shape)
partial_f1s = []

BATCH_SIZE = 16
for fold in np.unique(all_folds):
    print("Working on fold " + str(fold))
    train_X = all_X[all_folds != fold]
    train_input_ids = all_encodings['input_ids'][all_folds != fold]
    train_attention_mask = all_encodings['attention_mask'][all_folds != fold]
    train_Y = all_Y[all_folds != fold]
    train_ids = np.array(all_ids)[all_folds != fold]
    
    test_X = all_X[all_folds == fold]
    test_input_ids = all_encodings['input_ids'][all_folds == fold]
    test_attention_mask = all_encodings['attention_mask'][all_folds == fold]
    test_Y = all_Y[all_folds == fold]
    test_ids = np.array(all_ids)[all_folds == fold]
    
    print("Building a model...")
    train_dataset = TensorDataset(torch.tensor(train_X).float(), train_input_ids, train_attention_mask,
                                  torch.tensor(np.array(train_Y)).long())
    test_dataset = TensorDataset(torch.tensor(test_X).float(), test_input_ids, test_attention_mask,
                                 torch.tensor(np.array(test_Y)).long())
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)
    if model_type == 'BiLSTM':
        model = BiLSTM(all_X.shape[2], task, local_device).to(device)
    elif model_type == 'Hybrid':
        model = HybridBiLSTMRoBERTa(all_X.shape[2], task, local_device, roberta_variant, disable_sequence).to(device)
    elif model_type == 'Mutant':
        model = MutantRoBERTa(task, local_device, roberta_variant).to(device)
    print("Preparing training")
    model = model.to(device)
    learning_rate = 2e-05 if model_type == 'Mutant' else 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)
    milestones = [5] if model_type == 'Hybrid' else []
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.02)
    skip_visual = False
    eval_loop(test_loader, model, device, local_device, skip_visual)

    stats_file = open(pathlib.Path.home() / 'data' / 'autext' / 'out' / (task + '_' + language + '_F'+str(fold)+'_stats.tsv'),
                      'w')
    stats_file.write('epoch\ttrain_F1\tdev_F1\n')
    for epoch in range(20):
        print("EPOCH " + str(epoch + 1))
        if model_type == 'Hybrid':
            if epoch < 5:
                model.freeze_llm()
            else:
                model.unfreeze_llm()
        train_f1 = train_loop(train_loader, model, optimizer, scheduler, device, local_device, skip_visual)
        dev_f1 = eval_loop(test_loader, model, device, local_device, skip_visual)

        stats_file.write(str(epoch + 1) + '\t' + str(train_f1) + '\t' + str(dev_f1) + '\n')

        _, test_probs = eval_loop(test_loader, model, device, local_device, skip_visual, test=True)
        _, train_probs = eval_loop(train_loader, model, device, local_device, skip_visual, test=True)
        with open(pathlib.Path.home() / 'data' / 'autext' / 'out' / (
                task + '_' + language + '_F'+str(fold)+'_E' + str(epoch + 1) + '.tsv'), 'w') as f:
            f.write('id\t' + '\t'.join(
                ['human', 'generated'] if task == 'subtask_1' else ['A', 'B', 'C', 'D', 'E', 'F']) + '\n')
            for case_id, prob in zip(np.concatenate((train_ids,test_ids)), [x for x in train_probs] + [x for x in test_probs]):
                f.write(case_id + '\t' + '\t'.join([str(x) for x in prob]) + '\n')

    stats_file.close()
    pred, _ = eval_loop(test_loader, model, device, local_device, skip_visual, test=True)
    result[all_folds == fold] = pred
    partial_f1s.append(f1_score(y_true=all_Y[all_folds == fold], y_pred=pred, average="macro"))

overall_accuracy = np.mean(result == all_Y)
print('Total accuracy: ' + str(overall_accuracy))
f1_macro = f1_score(y_true=all_Y, y_pred=result, average="macro")
print('Partial F1 scores:')
for x in partial_f1s:
    print('\t' + str(x))
print('Total F1 score: ' + str(f1_macro))
print('Variance: ' + str(np.var(partial_f1s)))