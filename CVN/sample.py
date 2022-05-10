import os

import numpy as np

from utils.dataset import dataset
from utils.parser import parse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Setup this trial's config
config = parse()

# Generate a list of files to use for training
fluxswap = [os.path.join(config.dataset,'FHC_fluxswap',f) for f in os.listdir(os.path.join(config.dataset,'FHC_fluxswap'))]
nonswap = [os.path.join(config.dataset,'FHC_nonswap',f) for f in os.listdir(os.path.join(config.dataset,'FHC_nonswap'))]
cosmics = [os.path.join(config.dataset,'cosmics',f) for f in os.listdir(os.path.join(config.dataset,'cosmics'))]
files = fluxswap+nonswap+cosmics
data = dataset(config, files, run_info=False)

label = []
for n,i in enumerate(data.ids):
    label.append(data.load_label(i))

n , bins, _ = plt.hist(label, bins=5, range=(0,5))
plt.savefig('sample.png')

print(n[0])
print(n[1])
print(n[2])
print(n[3])

print('Cosmic Fraction: ', str(n[3]/np.sum(n)))
print('Numu weight: ', np.max(n)/n[0])
print('Nue weight:  ', np.max(n)/n[1])
print('NC weight:   ', np.max(n)/n[2])
print('Cos weight:  ', np.max(n)/n[3])
