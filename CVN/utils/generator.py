import numpy as np
from keras.utils import to_categorical

def generator(config, dataset, augment=False):
    def raw_gen():
        this = dataset
        num_classes = config.num_classes
        scale_dev = config.scale_dev
        aug = augment

        while True:
            np.random.shuffle(this.ids)

            for idx in this.ids:
                pm = this.load_pm(idx)
                lab = this.load_label(idx)

                # Augmentation
                if aug:
                    pm = pm*np.random.normal(1, scale_dev)
                
                yield pm, to_categorical(lab, num_classes=num_classes)

    return raw_gen
