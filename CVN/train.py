import os

from utils.model import model
from utils.dataset import dataset
from utils.parser import parse

#################################################################################

# Setup this trial's config
config = parse()

# Generate a list of files to use for training
fluxswap = [os.path.join(config.dataset,'FHC_fluxswap',f) for f in os.listdir(os.path.join(config.dataset,'FHC_fluxswap'))]
nonswap = [os.path.join(config.dataset,'FHC_nonswap',f) for f in os.listdir(os.path.join(config.dataset,'FHC_nonswap'))]
cosmics = [os.path.join(config.dataset,'cosmics',f) for f in os.listdir(os.path.join(config.dataset,'cosmics'))]
files = fluxswap+nonswap+cosmics

data = dataset(config, files, run_info=False)
train, test = data.split()

# Initialize the model
kModel = model(config)

# GO GO GO
kModel.train(train, test)
