from sklearn.metrics import f1_score

tested_file = '/Users/piotr/data/autext/out-bilstm-probs2/subtask_1_en_probs_traindev_18.tsv'
pred_labels= {}
for line in open(tested_file):
    if line.startswith('id'):
        continue
    parts = line.strip().split('\t')
    pred_labels[parts[0]] = ('human' if float(parts[1])>float(parts[2]) else 'generated')

gold_file = '/Users/piotr/data/autext/data/subtask_1/en/train.tsv'
gold_ids = []
gold_labels = {}
for line in open(gold_file):
    if line.startswith('id'):
        continue
    parts = line.strip().split('\t')
    gold_labels[parts[0]]=parts[2]

true_Y = []
pred_Y = []
for line in open('/Users/piotr/data/autext/data/subtask_1/en/train_5folds.tsv'):
    parts = line.strip().split('\t')
    if len(parts)==2 and parts[1]=='0' and parts[0] in pred_labels:
        true_Y.append(gold_labels[parts[0]])
        pred_Y.append(pred_labels[parts[0]])

f1=f1_score(y_true=true_Y, y_pred=pred_Y, average="macro")
print("FIn "+str(f1))

