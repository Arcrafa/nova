import os

from keras.utils import plot_model

from utils.model import model
from utils.parser import parse

#################################################################################

# Setup this trial's config
config = parse()

# Initialize the model
kModel = model(config).keras_model

kModel.summary()

plot_model(kModel, to_file='model.png')
