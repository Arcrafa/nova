import os

from utils.model import model
from utils.dataset import dataset
from utils.parser import parse

from random import sample
#################################################################################

# Setup this trial's config
config = parse()
num_samples=900
# Generate a list of files to use for training
electron = [os.path.join(config.dataset,'ND-Single-Electron',f) for f in sample(os.listdir(os.path.join(config.dataset,'ND-Single-Electron')),num_samples)]
muon = [os.path.join(config.dataset,'ND-Single-Muon',f) for f in sample(os.listdir(os.path.join(config.dataset,'ND-Single-Muon')),num_samples)]
piminus = [os.path.join(config.dataset,'ND-Single-PiMinus',f) for f in sample(os.listdir(os.path.join(config.dataset,'ND-Single-PiMinus')),num_samples)]
files = electron+muon+piminus
print('cargando el dataset')
data = dataset(config, files, run_info=False)
train, test = data.split()


print('creando el modelo')
# Initialize the model
kModel = model(config)

# GO GO GO
print('entrenamiento inicial')
kModel.train(train, test)

