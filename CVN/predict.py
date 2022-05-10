import os
import time
import h5py

from utils.model import model
from utils.dataset import dataset
from utils.parser import parse

#################################################################################

# Setup this trial's config
config = parse()

# Generate a list of files to use for predicting
fluxswap = [os.path.join(config.dataset,'FHC_fluxswap',f) for f in os.listdir(os.path.join(config.dataset,'FHC_fluxswap'))]
nonswap = [os.path.join(config.dataset,'FHC_nonswap',f) for f in os.listdir(os.path.join(config.dataset,'FHC_nonswap'))]
cosmics = [os.path.join(config.dataset,'cosmics',f) for f in os.listdir(os.path.join(config.dataset,'cosmics'))]
files = fluxswap+nonswap+cosmics

data = dataset(config, files, run_info=False)
train, test = data.split()

# Initialize the model
kModel = model(config)

probs  = []
labels = []

t0 = time.time()
for n,i in enumerate(test.ids):
    if n % 50 == 0:
        print(str(n)+' events processed in '+str(time.time()-t0))
        
    prop = data.load_prop(i)
    pm = data.load_pm(i)
    p = kModel.predict(pm)
    probs.append(p)
    labels.append(prop['label'])

# Save in a file for later plotting
hf = h5py.File(os.path.join(config.out_directory,'predictions_'+config.name+'.h5'),'w')
hf.create_dataset('probs',data=probs, compression='gzip')
hf.create_dataset('labels',data=labels, compression='gzip')
hf.close()
