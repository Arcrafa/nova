import os
import re
import numpy as np

import keras
from keras.optimizers import SGD
from keras import backend as K
from tensorflow.python.framework import graph_util
import tensorflow as tf
import pandas as pd

import utils.architecture as ac
import utils.generator as gen

class model():
    def __init__(self, config,model_file=None):
        self._config = config
        self.keras_model = ac.architecture(config)

        if self._config.model_file:
            self.load_weights(self._config.model_file)
        else:
            self.epoch = 0
        if (model_file is not None):
            self.load_weights(model_file)
    # Load weights from a file
    def load_weights(self, path):
        print('Loading model weights from '+path+'...')
        self.keras_model.load_weights(path)
        m = re.search('(\d\d\d).h5', path)
        if m:
            self.epoch = int(m.group(1))
            print('Loaded weights at epoch '+str(self.epoch))
        else:
            self.epoch = 0
            print('Epoch could not be determined. Ignoring...')

    # Setup the data generators and compile the model then fit
    def train(self, train_data, eval_data):
        train_gen = gen.generator(self._config, train_data, augment=True)
        eval_gen  = gen.generator(self._config, eval_data, augment=False)

        train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float32, tf.int32),
                        output_shapes=(tf.TensorShape([16000]),
                        tf.TensorShape([self._config.num_classes]))).batch(self._config.batch_size)
        eval_dataset = tf.data.Dataset.from_generator(eval_gen, output_types=(tf.float32, tf.int32),
                        output_shapes=(tf.TensorShape([16000]),
                        tf.TensorShape([self._config.num_classes]))).batch(self._config.batch_size)

        opt = SGD(lr=self._config.learning_rate, momentum=self._config.momentum, 
                  nesterov=self._config.nesterov)

        self.keras_model.compile(loss='categorical_crossentropy', metrics=['acc'],optimizer=opt)

        # callbacks for controlling the model while training
        callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(self._config.out_directory, 
                                                     'weights_'+self._config.name+'_{epoch:03d}.h5'),
                                                     save_best_only=True, save_weights_only=True),
                     keras.callbacks.ModelCheckpoint(os.path.join(self._config.out_directory,
                                                     'weights_'+self._config.name+'_{epoch:03d}.h5'),
                                                     save_freq=self._config.checkpoint_period, 
                                                     save_weights_only=True),
                     keras.callbacks.ReduceLROnPlateau(factor=self._config.learning_rate_factor, verbose=1, 
                                                       patience=self._config.learning_rate_patience),
                     keras.callbacks.EarlyStopping(patience=self._config.early_stopping_patience, 
                                                   restore_best_weights=True)]

        # GO GO GO
        history=self.keras_model.fit(x=train_dataset,
                             steps_per_epoch=self._config.train_iterations,
                             validation_data=eval_dataset,
                             validation_steps=self._config.eval_iterations,
                             epochs=self._config.epochs,
                             callbacks=callbacks,
                             verbose=2,
                             workers=2,
                             max_queue_size=16,
                             use_multiprocessing=True,
                             initial_epoch=self.epoch)

        #self.keras_model.save(os.path.join(self._config.out_directory,
        #                                   'weights_'+self._config.name+'_best.h5'))

        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(history.history)

        # save to json:
        hist_json_file = self._config.name+'history.json'
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)

    def predict(self, pm):
        p = self.keras_model.predict(np.array([pm]))
        return p[0]
    
    def save_model(self):
        K.set_learning_phase(0)
        model = self.keras_model

        num_output = self._config.num_classes

        pred_node_names = ['output_out']
        pred = [tf.identity(model.outputs[i], name = pred_node_names[i])
                for i in range(num_output)]

        sess = K.get_session()
        od_graph_def = graph_util.convert_variables_to_constants(sess,
                                                         sess.graph.as_graph_def(),
                                                         pred_node_names)

        frozen_graph_path = 'model_'+self._config.name+'.pb'
        with tf.gfile.GFile(frozen_graph_path, 'wb') as f:
            f.write(od_graph_def.SerializeToString())
