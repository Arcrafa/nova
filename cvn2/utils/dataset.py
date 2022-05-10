import sys
import h5py
import random
import numpy as np

from sklearn.model_selection import train_test_split

class dataset():
    def __init__(self, config, files=None, run_info=False):
        self._config = config
        self._props = []

        # Loop over each file and count the number of pixel maps present
        for f in files or [0]:
            if f == 0:
                break

            h5 = h5py.File(f,'r')
            labs = h5[self._config.lab_location][:]

            runs    = h5['run'][:]
            subruns = h5['subrun'][:]
            cycs    = h5['cycle'][:]
            evts    = h5['event'][:]
            sls     = h5['slice'][:]
                
            # Cut nutaus and limit cosmics
            labs = self.remove_taus(labs)
            labs = self.downsample_cosmics(labs, self._config.cosmic_fraction)

            # store the file location and index within file for each
            for n, (lab,run,subrun,cyc,evt,sl) in enumerate(zip(labs, runs, subruns, cycs, evts, sls)):
                if lab > 900: continue
                # the run info can sometimes be memory consuming, only load if needed
                lab = self.interaction_to_nu(lab)
                if run_info:
                    self._props.append({'file': f, 'idx': n, 'label': lab,
                                        'run': run, 'subrun': subrun, 'cyc': cyc, 'evt': evt, 'sl': sl})
                else:
                    self._props.append({'file': f, 'idx': n, 'label': lab})

            h5.close()
        else:
            self.prepare()
    def _vtx(self, labels,vtx_x,vtx_y,vtx_z):
        labels[(labels>=8) & (labels<=11)] = 999
        return labels
    # Remove all tau labels from the sample
    def remove_taus(self, labels):
        labels[(labels>=8) & (labels<=11)] = 999
        return labels

    # Remove cosmics til they are <frac> of the total sample
    def downsample_cosmics(self, labels, frac):
        if frac < 0:
            return labels
        ncosmics = labels[labels==15].shape[0]
        nsamples = labels[labels!=999].shape[0]
        if ncosmics <= frac*nsamples:
            return labels
        ndel = int(np.floor((ncosmics - frac*nsamples) / (1 - frac)))
        delcosmics = np.sort(random.sample(list(np.where(labels==15)[0]), ndel))
        labels[delcosmics] = 999
        return labels

    def interaction_to_nu(self, lab):
        if lab >= 0 and lab <= 3:
            return 0
        elif lab >= 4 and lab <= 7 or lab == 12:
            return 1
        elif lab >= 8 and lab <= 11:
            sys.exit('There should not be any taus at this point. Abort!')
        elif lab == 13:
            return 2
        else:
            return 3

    # add external data properties to this dataset
    def add_props(self, props):
        self._props += props
        self.prepare()

    # set the available ids and shuffle
    def prepare(self):
        n = len(self._props)
        self.ids = np.arange(n)
        print('Loaded '+str(n)+' events')
        np.random.shuffle(self.ids)

    # split into two datasets of size frac and 1-frac
    def split(self, frac = 0.1):
        print('Splitting into '+str(1-frac)+' train and '+str(frac)+' eval...')
        train,test = train_test_split(self._props, test_size=frac)

        d1 = dataset(self._config)
        d1.add_props(train)

        d2 = dataset(self._config)
        d2.add_props(test)

        return d1, d2

    # load a given data property
    def load_prop(self, i):
        return self._props[i]

    # load a pixel map from an hdf5
    def load_pm(self, i):
        prop = self.load_prop(i)

        f = h5py.File(prop['file'],'r')
        pm = f[self._config.map_location][prop['idx']]
        f.close()

        return pm

    # load a label from an hdf5
    def load_label(self, i):
        return self.load_prop(i)['label']
