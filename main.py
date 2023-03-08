import torch, pathlib

from perplexity import Perplexity

# Based on https://huggingface.co/docs/transformers/perplexity

# TODO generalise to other devices
device = torch.device("mps")
local_device = torch.device('cpu')

perp = Perplexity(device, local_device)

gen_ppls = []
hum_ppls = []
path = pathlib.Path.home() / 'Downloads' / 'train.tsv'
for line in open(path):
    parts = line.strip().split('\t')
    sentence = parts[1]
    if sentence == 'text':
        continue
    ppl = perp.perplexity(sentence)
    if parts[2] == 'generated':
        gen_ppls.append(ppl)
    elif parts[2] == 'human':
        hum_ppls.append(ppl)
    if len(gen_ppls) + len(hum_ppls) > 1000:
        break

print(sum(gen_ppls) / len(gen_ppls))
print(sum(hum_ppls) / len(hum_ppls))
