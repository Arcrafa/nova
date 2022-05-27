import h5py
import numpy as np
from sklearn.model_selection import train_test_split


class dataset():
    def __init__(self, config, files=None, run_info=False):
        self._config = config
        self._props = []

        print('cargando archivos')
        # Loop over each file and count the number of pixel maps present
        for f in files or [0]:
            if f == 0:
                break

            h5 = h5py.File(f, 'r')
            labs = h5['pdg'][:]

            runs = h5['run'][:]
            subruns = h5['subrun'][:]
            cycs = h5['cycle'][:]
            evts = h5['evt'][:]

            # store the file location and index within file for each
            for n, (lab, run, subrun, cyc, evt) in enumerate(
                    zip(labs, runs, subruns, cycs, evts)):

                # the run info can sometimes be memory consuming, only load if needed
                lab = self.parse_label(lab)
                if run_info:
                    self._props.append({'file': f, 'idx': n, 'label': lab,
                                        'run': run, 'subrun': subrun, 'cyc': cyc, 'evt': evt})
                else:
                    self._props.append({'file': f, 'idx': n, 'label': lab})

            h5.close()
        else:
            self.prepare()

    # Remove cosmics til they are <frac> of the total sample

    def parse_label(self, lab):
        if lab == 13: return 0
        if lab == 11: return 1
        if lab == -211: return 2

    # add external data properties to this dataset
    def add_props(self, props):
        self._props += props
        self.prepare()

    # set the available ids and shuffle
    def prepare(self):
        n = len(self._props)
        self.ids = np.arange(n)
        print('Loaded ' + str(n) + ' events')
        np.random.shuffle(self.ids)

    # split into two datasets of size frac and 1-frac
    def split(self, frac=0.1):
        print('Splitting into ' + str(1 - frac) + ' train and ' + str(frac) + ' eval...')
        train, test = train_test_split(self._props, test_size=frac)

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

        f = h5py.File(prop['file'], 'r')
        cvnmaps = f['cvnmap']
        pm = cvnmaps[prop['idx']]
        f.close()

        return pm

    # load a label from an hdf5
    def load_label(self, i):
        return self.load_prop(i)['label']
