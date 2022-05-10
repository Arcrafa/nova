import sys
import h5py
import random
import numpy as np

from sklearn.model_selection import train_test_split


class dataset():
    def __init__(self, config, files=None, run_info=False):
        self._config = config
        self.props = []
        # Loop over each file and count the number of pixel maps present
        for f in files or [0]:
            if f == 0:
                break
            h5 = h5py.File(f, 'r')
            cvnmaps = h5['rec.training.cvnmaps']
            #labs = [labels.index(f[f.find('single_') + 7:-9])] * len(cvnmaps['evt'][:])

            runs = cvnmaps['run'][:]
            subruns = cvnmaps['subrun'][:]
            cycs = cvnmaps['cycle'][:]
            evts = cvnmaps['evt'][:]
            labs= h5['rec.mc.cosmic']['pdg']
            # Cuts
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

                if run_info:
                    self.props.append({'file': f, 'idx': n, 'label': lab,
                                        'run': run, 'subrun': subrun, 'cyc': cyc, 'evt': evt})
                else:
                    self.props.append({'file': f, 'idx': n, 'label': lab})

            h5.close()
        else:
            self.prepare()

    # add external data properties to this dataset
    def add_props(self, props):
        self.props += props
        self.prepare()

    # set the available ids and shuffle
    def prepare(self):
        n = len(self.props)
        self.ids = np.arange(n)
        print('Loaded ' + str(n) + ' events')
        np.random.shuffle(self.ids)


    # split into two datasets of size frac and 1-frac
    def split(self, frac=0.2):
        print('Splitting into ' + str(1 - frac) + ' train and ' + str(frac) + ' eval...')
        muones=list(filter(lambda x: x['label']==0, self.props))
        electrones = random.sample(list(filter(lambda x: x['label'] == 1, self.props)), len(muones))
        piones = random.sample(list(filter(lambda x: x['label'] == 2, self.props)), len(electrones))

        self.props=[]

        self.add_props(muones)
        self.add_props(electrones)
        self.add_props(piones)

        train_m, test_m = train_test_split(muones, test_size=frac)
        train_e, test_e = train_test_split(electrones, test_size=frac)
        train_p, test_p = train_test_split(piones, test_size=frac)

        d1 = dataset(self._config)
        d1.add_props(train_m)
        d1.add_props(train_e)
        d1.add_props(train_p)
        random.shuffle(d1.props)

        d2 = dataset(self._config)
        d2.add_props(test_m)
        d2.add_props(test_e)
        d2.add_props(test_p)
        random.shuffle(d2.props)

        return d1, d2

    # load a given data property
    def load_prop(self, i):
        return self.props[i]

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
        label =self.load_prop(i)['label']
        if label==13: return 0
        if label == 11: return 1
        if label == -211: return 2
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
