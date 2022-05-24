import os
import time
import h5py

from utils.dataset import dataset
from utils.parser import parse
from random import sample
import numpy as np

# import matplotlib.pyplot as plt
#################################################################################

# Setup this trial's config
config = parse()

files = ['/wclustre/nova/users/rafaelma/dataset_test.h5']

test = dataset(config, files)


from utils.model import model

# Initialize the model
kModel = model(config)

runs = []
subruns = []
cycles = []
evts = []
sls = []

probs = []
labels = []

t0 = time.time()

for n, i in enumerate(test.ids):
    if n % 50 == 0:
        print(str(n) + ' events processed in ' + str(time.time() - t0))

    prop = test.load_prop(i)

    runs.append(prop['run'])
    subruns.append(prop['subrun'])
    cycles.append(prop['cyc'])
    evts.append(prop['evt'])

    pm = test.load_pm(i)
    pm = pm.reshape((2, 100, 80, 1))
    pm = np.concatenate((pm[0].reshape((100, 80)), pm[1].reshape((100, 80))), axis=1)
    pm = np.stack((pm,) * 3, axis=-1)
    p = kModel.predict(pm)
    probs.append(p)
    labels.append(prop['label'])

# Save in a file for later plotting
hf = h5py.File(os.path.join(config.out_directory, 'predictions_' + config.name + '.h5'), 'w')
hf.create_dataset('runs', data=runs, compression='gzip')
hf.create_dataset('subruns', data=subruns, compression='gzip')
hf.create_dataset('cycles', data=cycles, compression='gzip')
hf.create_dataset('evts', data=evts, compression='gzip')

hf.create_dataset('probs', data=probs, compression='gzip')
hf.create_dataset('labels', data=labels, compression='gzip')
hf.close()
