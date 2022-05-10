import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers

"""
def architecture(config):
    # pixel maps stored as (16000,), reshape to each view
    input = KL.Input(shape=(100,160), name='input')
    joined = KL.Reshape((2,100,80,1), name='reshape_input')(input)
    joined = KL.Lambda(lambda x: tf.unstack(x,axis=1), name='split')(joined)

    # a modified version of mobilenet v2
    output = mynet_graph(joined[0], joined[1])

    # 1 output probability for WS
    output = KL.Dense(config.num_classes, activation='softmax', name='out')(output)

    model = Model(inputs=input, outputs=output)

    return model

"""


def architecture(config):
    inputs = layers.Input(shape=(100, 160, 3), name='input')
    # x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = True

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    # top_dropout_rate = 0.2
    x = layers.Dense(15, activation="relu", name="dense_pred0")(x)

    x = layers.Dense(10, activation="relu", name="dense_pred1")(x)

    x = layers.Dense(5, activation="relu", name="dense_pred2")(x)

    outputs = layers.Dense(config.num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")

    return model
