import os
import sys
import random

from PandAna import *


# Containment
def kContain(tables):
    df = tables['rec.mc.cosmic']
    return (df['vtx.x'] > -180) & (df['vtx.x'] < 180) & (df['vtx.y'] > -180) & (df['vtx.y'] < 180) & (
            df['vtx.z'] > 50) & (df['vtx.z'] < 1200) & (df['stop.x'] > -180) & (df['stop.x'] < 180) & (
                   df['stop.y'] > -180) & (df['stop.y'] < 180) & (df['stop.z'] > 30) & (df['stop.z'] < 700)


kContain = Cut(kContain)


def kMap(tables):
    return tables['rec.training.cvnmaps']['cvnmap']


def kCosmic(tables):
    return tables['rec.mc.cosmic'][
        ['E', 'azimuth', 'eff', 'enter.x', 'enter.y', 'enter.z', 'exit.x', 'exit.y', 'exit.z', 'nhitslc', 'nhittot',
         'p.E', 'p.px', 'p.py', 'p.pz', 'pdg', 'penter.E', 'penter.px', 'penter.py', 'penter.pz', 'rec.mc.cosmic_idx',
         'stop.x', 'stop.y', 'stop.z', 'time', 'visE', 'visEinslc', 'vtx.x', 'vtx.y', 'vtx.z', 'zenith']]


kCosmic = Var(kCosmic)

if __name__ == '__main__':
    # Miniprod 5 h5s
    indir = sys.argv[1]
    outdir = sys.argv[2]

    # Generate a list of files to use for training
    print('Change files in ' + indir + ' to training files in ' + outdir)
    files = [f for f in os.listdir(indir) if 'h5caf.h5' in f]
    files = random.sample(files, len(files))
    print('There are ' + str(len(files)) + ' files.')

    # One file at a time to avoid problems with loading a bunch of pixel maps in memory
    for i, f in enumerate(files):

        # Definte the output name and don't recreate it
        outname = '{0}_TrainData{1}'.format(f[:-9], f[-9:])
        if os.path.exists(os.path.join(outdir, outname)):
            continue

        tables = loader([os.path.join(indir, f)])

        specMap = spectrum(tables, kContain, kMap)
        specCosmic = spectrum(tables, kContain, kCosmic)
        # GO GO GO
        tables.Go()

        # Don't save an empty file
        if specCosmic.entries() == 0 or specMap.entries() == 0:
            print(str(i) + ': File ' + f + ' is empty.')
            continue

        df = pd.merge(specCosmic.df(), specMap.df(), on=['run', 'subrun', 'cycle', 'evt', 'subevt'], how='inner').reset_index()
        df=df.sample(50)
        hf = h5py.File(os.path.join(outdir, outname), 'w')
        hf.create_dataset('cvnmap', data=np.stack(df['cvnmap']), compression='gzip')
        df = df.drop(['cvnmap'], axis=1)
        for col in df.columns:
            hf.create_dataset(col, data=df[col], compression='gzip')

        hf.close()
