import os
import h5py

from utils.dataset import dataset
from utils.parser import parse
import numpy as np
import time
import utils.generator as gen

from random import sample
#################################################################################
ti = time.time()
# Setup this trial's config
config = parse()


# Setup this trial's config
config = parse()

# Generate a list of files to use for training
files = [os.path.join(config.dataset,f) for f in os.listdir(config.dataset)]
data = dataset(config, files, run_info=True)
train, test = data.split()
from utils.model import model
print('creando el modelo')
# Initialize the model
kModel = model(config)

# GO GO GO
print('entrenamiento inicial')
kModel.train(train, test)


del train
kModel=model(config,model_file=os.path.join(config.out_directory,
                                           'weights_'+config.name+'_best.h5'))
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
    labels.append(prop['label'])
    pm = test.load_pm(i)
    pm = pm.reshape((2, 100, 80, 1))
    pm = np.concatenate((pm[0].reshape((100, 80)), pm[1].reshape((100, 80))), axis=1)
    pm = np.stack((pm,) * 3, axis=-1)
    p = kModel.predict(pm)
    probs.append(p)


# Save in a file for later plotting
hf = h5py.File(os.path.join(config.out_directory, 'predictions_' + config.name + '.h5'), 'w')
hf.create_dataset('runs', data=runs, compression='gzip')
hf.create_dataset('subruns', data=subruns, compression='gzip')
hf.create_dataset('cycles', data=cycles, compression='gzip')
hf.create_dataset('evts', data=evts, compression='gzip')

hf.create_dataset('probs', data=probs, compression='gzip')
hf.create_dataset('labels', data=labels, compression='gzip')
hf.close()
