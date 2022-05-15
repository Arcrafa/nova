import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers

def architecture(config):
    # pixel maps stored as (16000,), reshape to each view
    input = layers.Input(shape=(16000,), name='input')
    joined = layers.Reshape((2, 100, 80), name='reshape_input')(input)
    joined = layers.Lambda(lambda x: tf.unstack(x, axis=1), name='split')(joined)

    output = mynet_graph(joined[0], joined[1])

    output = layers.Dense(config.num_classes, activation="softmax", name="pred")(output)

    # Compile
    output = tf.keras.Model(input, output, name="EfficientNet")

    return output

def mynet_graph(input_x, input_y):
    def subnet(input,name):

        input=layers.Lambda(lambda x: tf.stack((x,) * 3, axis=-1))(input)
        model = EfficientNetB0(include_top=False, input_tensor=input, weights="imagenet")
        # Freeze the pretrained weights
        model.trainable = True
        for layer in model.layers:
            layer._name = layer.name + str(name)
        return model.output

    x = subnet(input_x,name='_view_x')
    y = subnet(input_y,name='_view_y')

    merge = layers.Concatenate()([x, y])

    # Rebuild top
    #merge = layers.AveragePooling2D(pool_size=2, padding='same')(merge)
    #merge = layers.BatchNormalization()(merge)
    #merge = layers.AveragePooling2D(pool_size=2, padding='same')(merge)
    #merge = layers.BatchNormalization()(merge)

    merge = layers.GlobalAveragePooling2D(name="avg_pool")(merge)
    merge = layers.BatchNormalization()(merge)

    #merge = layers.Dense(20, activation="relu", name="dense_pred-1")(merge)

    merge = layers.Dense(15, activation="relu", name="dense_pred0")(merge)

    merge = layers.Dense(10, activation="relu", name="dense_pred1")(merge)

    merge = layers.Dense(5, activation="relu", name="dense_pred2")(merge)

    return merge