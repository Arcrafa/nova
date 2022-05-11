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

# Generate a list of files to use for training
electron = [os.path.join(config.dataset,'ND-Single-Electron',f) for f in os.listdir(os.path.join(config.dataset,'ND-Single-Electron'))]
muon = [os.path.join(config.dataset,'ND-Single-Muon',f) for f in os.listdir(os.path.join(config.dataset,'ND-Single-Muon'))]
piminus = [os.path.join(config.dataset,'ND-Single-PiMinus',f) for f in os.listdir(os.path.join(config.dataset,'ND-Single-PiMinus'))]
files = electron+muon+piminus

data = dataset(config, files, run_info=False)
train, test = data.split(frac=0.8)

from utils.model import model
# Initialize the model
kModel = model(config)

probs = []
labels = []
mc_cosmics = []
vtx_stop=[]
pms=[]
t0 = time.time()

for n, i in enumerate(test.ids):
    if n % 50 == 0:
        print(str(n) + ' events processed in ' + str(time.time() - t0))

    prop = test.load_prop(i)
    pm = test.load_pm(i)
    pms.append(pm)
    pm = pm.reshape((2, 100, 80, 1))
    pm = np.concatenate((pm[0].reshape((100, 80)), pm[1].reshape((100, 80))), axis=1)
    pm = np.stack((pm,) * 3, axis=-1)
    p = kModel.predict(pm)

    probs.append(p)

    labels.append(np.array(prop['label']).astype(int))
    mc_cosmics.append(test.load_mc_cosmic(i))
    vtx_stop.append(test.load_vtx_stop(i))

data = {'muon': labels.count(0), 'electron': labels.count(1), 'piminus': labels.count(2)}
print(data)

# Save in a file for later plotting
hf = h5py.File(os.path.join(config.out_directory, 'predictions_' + config.name + '.h5'), 'w')
hf.create_dataset('probs', data=probs, compression='gzip')
hf.create_dataset('labels', data=np.array(labels).astype(int), compression='gzip')
hf.create_dataset('rec.mc.cosmic', data=mc_cosmics, compression='gzip')
hf.create_dataset('vtx_stop', data=vtx_stop, compression='gzip')
hf.create_dataset('pm', data=pms, compression='gzip')
hf.close()
