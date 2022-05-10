import keras
import keras.backend as K
from keras.models import Model
import keras.layers as KL
import tensorflow as tf

def architecture(config):
    # pixel maps stored as (16000,), reshape to each view
    input = KL.Input(shape=(16000,), name='input')
    joined = KL.Reshape((2,100,80,1), name='reshape_input')(input)
    joined = KL.Lambda(lambda x: tf.unstack(x,axis=1), name='split')(joined)
    
    # a modified version of mobilenet v2
    output = mynet_graph(joined[0], joined[1])
    
    # 1 output probability for WS
    output = KL.Dense(config.num_classes, activation='softmax', name='out')(output)
        
    model = Model(inputs=input, outputs=output)
    
    return model

def my_relu6(x):
    return K.relu(x, max_value=6)

def my_hard_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def my_return_activation(x, nl):
    if nl == 'HS':
        x = KL.Activation(my_hard_swish)(x)
    if nl == 'RE':
        x = KL.Activation(my_relu6)(x)
    return x

def my_squeeze(inputs, ratio=16):
    input_channels = int(inputs.shape[-1])

    x = KL.GlobalAveragePooling2D()(inputs)
    x = KL.Dense(input_channels//ratio, activation='relu')(x)
    x = KL.Dense(input_channels, activation='hard_sigmoid')(x)
    x = KL.Reshape((1, 1, input_channels))(x)
    x = KL.Multiply()([inputs, x])

    return x

def my_conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1), nl='RE'):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = KL.Conv2D(filters, kernel,
                  padding='same',
                  use_bias=False,
                  strides=strides)(inputs)
    x = KL.BatchNormalization(axis=channel_axis)(x)
    x = my_return_activation(x, nl)

    return x

def my_bottleneck(inputs, filters, kernel, t, s, squeeze, nl='RE'):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    channels = K.int_shape(inputs)[channel_axis]
    tchannel = channels * t
    r = s == 1 and channels == filters

    x = my_conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

    x = KL.DepthwiseConv2D(kernel,
                           strides=(s, s),
                           depth_multiplier=1,
                           padding='same')(x)
    x = KL.BatchNormalization(axis=channel_axis)(x)
    x = my_return_activation(x, nl)

    x = KL.Conv2D(filters,
                  (1, 1),
                  strides=(1, 1),
                  padding='same')(x)
    x = KL.BatchNormalization(axis=channel_axis)(x)

    if squeeze:
        x = my_squeeze(x)

    # Identity shortcut
    if r:
        x = KL.add([x, inputs])
    # Convolution shortcut
    else:
        shortcut = KL.Conv2D(filters,
                             (1, 1),
                             strides=(s, s),
                             padding='same')(inputs)
        shortcut = KL.BatchNormalization(axis=channel_axis)(shortcut)
        x = KL.add([x, shortcut])

    return x

def my_inverted_residual_block(inputs, filters, kernel, s, t, n, squeeze, nl):
    x = my_bottleneck(inputs, filters, kernel, t, s, squeeze, nl)

    for i in range(1, n):
        x = my_bottleneck(x, filters, kernel, t, 1, squeeze, nl)

    return x

def mynet_graph(input_x, input_y):
    def subnet(x):
        x = my_conv_block(x, 32, (5, 5), strides=(2, 2), nl='HS')
        x = my_inverted_residual_block(x, 16, (3, 3), s=1, t=2, n=1, squeeze=False, nl='RE')
        x = KL.AveragePooling2D(pool_size=2, padding='same')(x)
        x = my_inverted_residual_block(x, 24, (3, 3), s=1, t=6, n=2, squeeze=False, nl='RE')
        return x

    x = subnet(input_x)
    y = subnet(input_y)

    merge = KL.Maximum()([x, y])

    merge = KL.AveragePooling2D(pool_size=2, padding='same')(merge)
    merge = my_inverted_residual_block(merge, 32,  (3, 3), s=1, t=6, n=3, squeeze=False, nl='RE')
    merge = KL.AveragePooling2D(pool_size=2, padding='same')(merge)
    merge = my_inverted_residual_block(merge, 48,  (3, 3), s=1, t=6, n=4, squeeze=False,  nl='RE')
    merge = my_inverted_residual_block(merge, 64,  (3, 3), s=1, t=6, n=3, squeeze=True,  nl='HS')
    merge = KL.AveragePooling2D(pool_size=2, padding='same')(merge)
    merge = my_inverted_residual_block(merge, 96, (3, 3), s=1, t=6, n=3, squeeze=True,  nl='HS')
    merge = my_inverted_residual_block(merge, 160, (3, 3), s=1, t=6, n=1, squeeze=True,  nl='HS')

    merge = KL.GlobalAveragePooling2D()(merge)
    merge = KL.Dense(1024)(merge)
    merge = my_return_activation(merge, 'HS')
    merge = KL.Dropout(rate=0.3)(merge)

    return merge
