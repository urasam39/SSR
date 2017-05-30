# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://openreview.net/pdf?id=S1gNakBFx)

Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling1D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.topology import get_source_inputs
from keras.utils.visualize_util import plot

def down_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1 = filters
    bn_axis = 2
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv1D(filters1, kernel_size, name=conv_name_base + '2a', subsample_length=2, border_mode='same')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    return x


def up_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1 = filters
    bn_axis = 2
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv1D(filters1, kernel_size, name=conv_name_base + '2a', border_mode='same')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = UpSampling1D(length=2)(x)

    return x


def ResNet50(input_tensor=None, input_shape=None):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    bn_axis = 3

    x = Conv1D(8, 9, name='res2a_branch1a', subsample_length=2, border_mode='same', input_shape=(None, 1, 6000))(img_input)
    x = BatchNormalization(axis=2, name='bn1a_branch2a')(x)
    x1 = Activation('relu')(x)
    #x1 = down_block(img_input, 9, 8, stage=1, block='a') 
    x2 = down_block(x1, 9, 10, stage=2, block='a') 
    x3 = down_block(x2, 9, 12, stage=3, block='a') 
    x = down_block(x3, 9, 14, stage=4, block='a') 
    
    x = up_block(x, 9, 12, stage=4, block='b')
    x = layers.merge([x, x3], mode='sum')
    x = up_block(x, 9, 10, stage=3, block='b')
    x = layers.merge([x, x2], mode='sum')
    x = up_block(x, 9, 8, stage=2, block='b')
    x = layers.merge([x, x1], mode='sum')
    x = up_block(x, 9, 1, stage=1, block='b')

    x = layers.merge([x, img_input], mode='sum')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnetaudio')

    return model
