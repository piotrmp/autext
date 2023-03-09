import torch, pathlib, numpy
from matplotlib import pyplot

from perplexity import Perplexity

# Based on https://huggingface.co/docs/transformers/perplexity

# TODO generalise to other devices
device = torch.device("mps")
local_device = torch.device('cpu')

perp = Perplexity(device, local_device)

gen_ppls = []
hum_ppls = []
path = pathlib.Path.home() / 'Downloads' / 'train.tsv'
for i, line in enumerate(open(path)):
    parts = line.strip().split('\t')
    sentence = parts[1]
    if sentence == 'text':
        continue
    ppl = perp.perplexity(sentence)
    if parts[2] == 'generated':
        gen_ppls.append(ppl)
    elif parts[2] == 'human':
        hum_ppls.append(ppl)
    if i > 1000:
        break
    else:
        print(i)

bins = numpy.linspace(0, 200, 200)
pyplot.hist(gen_ppls, bins, alpha=0.5, label='Automatic')
pyplot.hist(hum_ppls, bins, alpha=0.5, label='Human')
pyplot.legend(loc='upper right')
pyplot.show()
