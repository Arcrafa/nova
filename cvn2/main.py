import os
import h5py

from utils.dataset import dataset
from utils.parser import parse
import numpy as np
import time


from random import sample
#################################################################################

# Setup this trial's config
config = parse()

# Generate a list of files to use for training
electron = [os.path.join(config.dataset,'ND-Single-Electron',f) for f in os.listdir(os.path.join(config.dataset,'ND-Single-Electron'))]
muon = [os.path.join(config.dataset,'ND-Single-Muon',f) for f in os.listdir(os.path.join(config.dataset,'ND-Single-Muon'))]
piminus = [os.path.join(config.dataset,'ND-Single-PiMinus',f) for f in os.listdir(os.path.join(config.dataset,'ND-Single-PiMinus'))]
files = electron+muon+piminus
print('cargando el dataset')
data = dataset(config, files, run_info=False)
train, test = data.split()



from utils.model import model
print('creando el modelo')
# Initialize the model
kModel = model(config)

# GO GO GO
print('entrenamiento inicial')
kModel.train(train, test)


probs = []
labels = []
mc_cosmics = []
vtx_stop=[]
pms=[]
t0 = time.time()

for n, prop in enumerate(test.props):
    if n % 50 == 0:
        print(str(n) + ' events processed in ' + str(time.time() - t0))

    #prop = data.load_prop(i)
    pm = data.load_pm(n)
    pms.append(pm)
    pm = pm.reshape((2, 100, 80, 1))
    pm = np.concatenate((pm[0].reshape((100, 80)), pm[1].reshape((100, 80))), axis=1)
    pm = np.stack((pm,) * 3, axis=-1)
    p = kModel.predict(pm)

    probs.append(p)

    labels.append(np.array(prop['label']).astype(int))
    mc_cosmics.append(data.load_mc_cosmic(n))
    vtx_stop.append(data.load_vtx_stop(n))

data = {'muon': labels.count(13), 'electron': labels.count(11), 'piminus': labels.count(-211)}
print(data)

# Save in a file for later plotting
hf = h5py.File(os.path.join(config.out_directory, 'predictions_' + config.name + '.h5'), 'w')
hf.create_dataset('probs', data=probs, compression='gzip')
hf.create_dataset('labels', data=np.array(labels).astype(int), compression='gzip')
hf.create_dataset('rec.mc.cosmic', data=mc_cosmics, compression='gzip')
hf.create_dataset('vtx_stop', data=vtx_stop, compression='gzip')
hf.create_dataset('pm', data=pms, compression='gzip')
hf.close()


