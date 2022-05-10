import os
import h5py
import numpy as np
from utils.parser import parse
# import matplotlib.pyplot as plt
#################################################################################

# Setup this trial's config
config = parse()

# Generate a list of files to use for training
electron = [os.path.join(config.dataset,'ND-Single-Electron',f) for f in os.listdir(os.path.join(config.dataset,'ND-Single-Electron'))]
muon = [os.path.join(config.dataset,'ND-Single-Muon',f) for f in os.listdir(os.path.join(config.dataset,'ND-Single-Muon'))]
piminus = [os.path.join(config.dataset,'ND-Single-PiMinus',f) for f in os.listdir(os.path.join(config.dataset,'ND-Single-PiMinus'))]
files = electron+muon+piminus
#contar muones
count_m=len(muon)
print('numero de archivos de muon ',count_m)
sum_m=0
for f in muon:
    h5 = h5py.File(f, 'r')
    sum_m+=len(h5['rec.mc.cosmic']['pdg'])

print('numero de registros de muones ',sum_m)
print('promedio de muones por archivo ',sum_m/count_m)
# contar electrones
count_e=len(electron)
print('numero de archivos de electrones ',count_e)
sum_e=0
for f in electron:
    h5 = h5py.File(f, 'r')
    sum_e+=len(h5['rec.mc.cosmic']['pdg'])

print('numero de registros de electrones ',sum_e)
print('promedio de electrones por archivo ',sum_e/count_e)
#contar piones

count_p=len(piminus)
print('numero de archivos de piones ',count_e)
sum_p=0
for f in piminus:
    h5 = h5py.File(f, 'r')
    sum_p+=len(h5['rec.mc.cosmic']['pdg'])

print('numero de registros de piones ',sum_p)
print('promedio de piones por archivo ',sum_p/count_p)