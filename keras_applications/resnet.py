"""ResNet models for Keras.

# Reference paper

- [Deep Residual Learning for Image Recognition]
  (https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)

# Reference implementations

- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/resnets.py)
- [Caffe ResNet]
  (https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from . import get_keras_submodule

backend = get_keras_submodule('backend')
layers = get_keras_submodule('layers')
models = get_keras_submodule('models')
keras_utils = get_keras_submodule('utils')

from . import imagenet_utils
from .imagenet_utils import decode_predictions
from .imagenet_utils import _obtain_input_shape
from .resnet_common import ResNet


def block(x, filters, kernel_size=3, stride=1,
          conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):
    def stack_fn(x):
        x = stack(x, 64, 3, stride1=1, name='conv2')
        x = stack(x, 128, 4, name='conv3')
        x = stack(x, 256, 6, name='conv4')
        x = stack(x, 512, 3, name='conv5')
        return x
    return ResNet(stack_fn, False, True, 'resnet50',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes)


def ResNet101(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000):
    def stack_fn(x):
        x = stack(x, 64, 3, stride1=1, name='conv2')
        x = stack(x, 128, 4, name='conv3')
        x = stack(x, 256, 23, name='conv4')
        x = stack(x, 512, 3, name='conv5')
        return x
    return ResNet(stack_fn, False, True, 'resnet101',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes)


def ResNet152(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000):
    def stack_fn(x):
        x = stack(x, 64, 3, stride1=1, name='conv2')
        x = stack(x, 128, 8, name='conv3')
        x = stack(x, 256, 36, name='conv4')
        x = stack(x, 512, 3, name='conv5')
        return x
    return ResNet(stack_fn, False, True, 'resnet152',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes)


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='caffe')


setattr(ResNet50, '__doc__', ResNet.__doc__)
setattr(ResNet101, '__doc__', ResNet.__doc__)
setattr(ResNet152, '__doc__', ResNet.__doc__)
