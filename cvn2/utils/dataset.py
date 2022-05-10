import sys
import h5py
import random
import numpy as np
from tqdm import tqdm
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
            labs = h5['rec.mc.cosmic']['pdg']

            cvnmaps = h5['rec.training.cvnmaps']
            # labs = [labels.index(f[f.find('single_') + 7:-9])] * len(cvnmaps['evt'][:])

            runs = cvnmaps['run'][:]
            subruns = cvnmaps['subrun'][:]
            cycs = cvnmaps['cycle'][:]
            evts = cvnmaps['evt'][:]

            vtxs_x = h5['rec.mc.cosmic']['vtx.x']
            vtxs_y = h5['rec.mc.cosmic']['vtx.y']
            vtxs_z = h5['rec.mc.cosmic']['vtx.z']

            stops_x = h5['rec.mc.cosmic']['stop.x']
            stops_y = h5['rec.mc.cosmic']['stop.y']
            stops_z = h5['rec.mc.cosmic']['stop.z']

            # store the file location and index within file for each
            for n, (lab, run, subrun, cyc, evt, vtx_x, vtx_y, vtx_z, stop_x, stop_y, stop_z) in enumerate(
                    zip(labs, runs, subruns, cycs, evts, vtxs_x, vtxs_y, vtxs_z, stops_x, stops_y, stops_z)):

                if stop_x < -180: continue
                if stop_x > 180: continue
                if stop_y < -180: continue
                if stop_y > 180: continue
                if stop_z < 50: continue
                if stop_z > 1200: continue
                if vtx_x < -180: continue
                if vtx_x > 180: continue
                if vtx_y < -180: continue
                if vtx_y > 180: continue
                if vtx_z < 30: continue
                if vtx_z > 700: continue
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

    def remove_taus(self, labels):
        labels[(labels >= 8) & (labels <= 11)] = 999
        return labels

    # Remove cosmics til they are <frac> of the total sample
    def downsample_cosmics(self, labels, frac):
        if frac < 0:
            return labels
        ncosmics = labels[labels == 15].shape[0]
        nsamples = labels[labels != 999].shape[0]
        if ncosmics <= frac * nsamples:
            return labels
        ndel = int(np.floor((ncosmics - frac * nsamples) / (1 - frac)))
        delcosmics = np.sort(random.sample(list(np.where(labels == 15)[0]), ndel))
        labels[delcosmics] = 999
        return labels

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
        muones = list(filter(lambda x: x['label'] == 0, self._props))
        electrones = random.sample(list(filter(lambda x: x['label'] == 1, self._props)), len(muones))
        piones = random.sample(list(filter(lambda x: x['label'] == 2, self._props)), len(electrones))

        train_m, test_m = train_test_split(muones, test_size=frac)
        train_e, test_e = train_test_split(electrones, test_size=frac)
        train_p, test_p = train_test_split(piones, test_size=frac)

        d1 = dataset(self._config)
        d1.add_props(train_m)
        d1.add_props(train_e)
        d1.add_props(train_p)

        d2 = dataset(self._config)
        d2.add_props(test_m)
        d2.add_props(test_e)
        d2.add_props(test_p)

        return d1, d2

    # load a given data property
    def load_prop(self, i):
        return self._props[i]

    # load a pixel map from an hdf5
    def load_pm(self, i):
        prop = self.load_prop(i)

        f = h5py.File(prop['file'], 'r')
        cvnmaps = f['rec.training.cvnmaps']['cvnmap']
        pm = cvnmaps[prop['idx']]
        f.close()

        return pm

    # load a label from an hdf5
    def load_label(self, i):
        return self.load_prop(i)['label']

    def load_mc_cosmic(self, i):
        prop = self.load_prop(i)

        f = h5py.File(prop['file'], 'r')
        p_pz = (f['rec.mc.cosmic']['p.pz'])[prop['idx']]
        p_E = (f['rec.mc.cosmic']['p.E'])[prop['idx']]
        nhitslc = (f['rec.mc.cosmic']['nhitslc'])[prop['idx']]
        f.close()
        return p_pz, p_E, nhitslc

    def load_vtx_stop(self, i):
        prop = self.load_prop(i)

        f = h5py.File(prop['file'], 'r')
        vtx_x = (f['rec.mc.cosmic']['vtx.x'])[prop['idx']]
        vtx_y = (f['rec.mc.cosmic']['vtx.y'])[prop['idx']]
        vtx_z = (f['rec.mc.cosmic']['vtx.z'])[prop['idx']]

        stop_x = (f['rec.mc.cosmic']['stop.x'])[prop['idx']]
        stop_y = (f['rec.mc.cosmic']['stop.y'])[prop['idx']]
        stop_z = (f['rec.mc.cosmic']['stop.z'])[prop['idx']]

        f.close()
        return np.array((vtx_x, vtx_y, vtx_z, stop_x, stop_y, stop_z))
