import os
import h5py
import numpy as np
import time
import h5py
import os
import pandas as pd
import numpy as np
from imblearn.under_sampling import NearMiss
from PandAna import *
#################################################################################
ti = time.time()
# Setup this trial's config


# Generate a list of files to use for training
electron = [os.path.join('/wclustre/novapro/R19-11-18-Prod5_fullset/','ND-Single-Electron',f) for f in os.listdir(os.path.join('/wclustre/novapro/R19-11-18-Prod5_fullset/','ND-Single-Electron'))]
muon = [os.path.join('/wclustre/novapro/R19-11-18-Prod5_fullset/','ND-Single-Muon',f) for f in os.listdir(os.path.join('/wclustre/novapro/R19-11-18-Prod5_fullset/','ND-Single-Muon'))]
piminus = [os.path.join('/wclustre/novapro/R19-11-18-Prod5_fullset/','ND-Single-PiMinus',f) for f in os.listdir(os.path.join('/wclustre/novapro/R19-11-18-Prod5_fullset/','ND-Single-PiMinus'))]
files = electron + muon + piminus

tables = loader(files)


# Containment
def kContain(tables):
    df = tables['rec.mc.cosmic']
    return (df['vtx.x'] > -180) & (df['vtx.x'] < 180) & (df['vtx.y'] > -180) &   (df['vtx.y'] < 180) & (df['vtx.z'] > 50) & (df['vtx.z'] < 1200) &  (df['stop.x'] > -180) & (df['stop.x'] < 180) & (df['stop.y'] > -180) & (df['stop.y'] < 180) &  (df['stop.z'] > 30) &  (df['stop.z'] < 700)


kContain = Cut(kContain)


def kMap(tables):
    return tables['rec.training.cvnmaps']['cvnmap']


def kCosmic(tables):
    return tables['rec.mc.cosmic'][
        ['E', 'azimuth', 'eff', 'enter.x', 'enter.y', 'enter.z', 'exit.x', 'exit.y', 'exit.z', 'nhitslc', 'nhittot',
         'p.E', 'p.px', 'p.py', 'p.pz', 'pdg', 'penter.E', 'penter.px', 'penter.py', 'penter.pz', 'rec.mc.cosmic_idx',
         'stop.x', 'stop.y', 'stop.z', 'time', 'visE', 'visEinslc', 'vtx.x', 'vtx.y', 'vtx.z', 'zenith']]


kCosmic = Var(kCosmic)

specMap = spectrum(tables, kContain, kMap)
specCosmic = spectrum(tables, kContain, kCosmic)
# GO GO GO
tables.Go()

dfCosmics = specCosmic.df().reset_index()
pdg = dfCosmics['pdg']
dfCosmics = dfCosmics.drop(['pdg'], axis=1)

us = NearMiss()

dfCosmics, pdg = us.fit_resample(dfCosmics, pdg)
dfCosmics = pd.concat([dfCosmics, pdg], axis=1, join='inner')

df = pd.merge(dfCosmics, specMap.df().reset_index(), on=['run', 'subrun', 'cycle', 'evt', 'subevt'], how='inner')

def save(df,file):
    hf = h5py.File(file, 'w')
    hf.create_dataset('cvnmap', data=np.stack(df['cvnmap']), compression='gzip')
    df = df.drop(['cvnmap'], axis=1)
    for col in df.columns:
        hf.create_dataset(col, data=df[col], compression='gzip')

    hf.close()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop(['pdg'], axis=1), df['pdg'], test_size=0.10, random_state=42)
df_train=pd.concat([X_train,y_train], axis=1, join='inner').reset_index()
df_test=pd.concat([X_test,y_test], axis=1, join='inner').reset_index()
save(df_train,'./wclustre/nova/users/rafaelma/dataset.h5')
save(df_test,'./wclustre/nova/users/rafaelma/dataset_test.h5')