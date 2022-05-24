import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers

def architecture(config):
    inputs = layers.Input(shape=(100, 160, 3), name='input')
    # x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable  =True

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    # top_dropout_rate = 0.2
    #x = layers.Dense(15, activation="relu", name="dense_pred0")(x)

    x = layers.Dense(10, activation="relu", name="dense_pred1")(x)

    x = layers.Dense(5, activation="relu", name="dense_pred2")(x)

    outputs = layers.Dense(config.num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")

    return model
