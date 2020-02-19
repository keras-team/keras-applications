"""MobileNet v3 models for Keras.

The following table describes the performance of MobileNets:
------------------------------------------------------------------------
MACs stands for Multiply Adds

| Classification Checkpoint| MACs(M)| Parameters(M)| Top1 Accuracy| Pixel1 CPU(ms)|

| [mobilenet_v3_large_1.0_224]              | 217 | 5.4 | 75.2 | 51.2 |
| [mobilenet_v3_large_0.75_224]             | 155 | 4.0 | 73.3 | 39.8 |
| [mobilenet_v3_large_minimalistic_1.0_224] | 209 | 3.9 | 72.3 | 44.1 |
| [mobilenet_v3_small_1.0_224]              | 66  | 2.9 | 67.5 | 15.8 |
| [mobilenet_v3_small_0.75_224]             | 44  | 2.4 | 65.4 | 12.8 |
| [mobilenet_v3_small_minimalistic_1.0_224] | 65  | 2.0 | 61.9 | 12.2 |

The weights for all 6 models are obtained and
translated from the Tensorflow checkpoints
from TensorFlow checkpoints found [here]
(https://github.com/tensorflow/models/tree/master/research/
slim/nets/mobilenet/README.md).

# Reference

This file contains building code for MobileNetV3, based on
[Searching for MobileNetV3]
(https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

from . import get_submodules_from_kwargs
from . import imagenet_utils
from .imagenet_utils import _obtain_input_shape
from .imagenet_utils import decode_predictions

BASE_WEIGHT_PATH = ('https://github.com/DrSlink/mobilenet_v3_keras/'
                    'releases/download/v1.0/')

backend = None
layers = None
models = None
keras_utils = None


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([layers.Activation(hard_sigmoid)(x), x])


def _activation(x, name='relu'):
    if name == 'relu':
        return layers.ReLU()(x)
    elif name == 'hardswish':
        return hard_swish(x)


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/
# slim/nets/mobilenet/mobilenet.py


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobileNetV3(input_shape=None,
                alpha=1.0,
                model_type='large',
                minimalistic=False,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                classes=1000,
                pooling=None,
                dropout_rate=0.2,
                regularizer=None,
                **kwargs):
    """Instantiates the MobileNetV3 architecture.

    # Arguments
        input_shape: optional shape tuple, to be specified if you would
            like to use a model with an input img resolution that is not
            (224, 224, 3).
            It should have exactly 3 inputs channels (224, 224, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
            E.g. `(160, 160, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
            depth multiplier in the MobileNetV3 paper, but the name is kept for
            consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                are used at each layer.
        model_type: MobileNetV3 is defined as two models: large and small. These
        models are targeted at high and low resource use cases respectively.
        minimalistic: In addition to large and small models this module also contains
            so-called minimalistic models, these models have the same per-layer
            dimensions characteristic as MobilenetV3 however, they don't utilize any
            of the advanced blocks (squeeze-and-excite units, hard-swish, and 5x5
            convolutions). While these models are less efficient on CPU, they are
            much more performant on GPU/DSP.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        dropout_rate: fraction of the input units to drop on the last layer
        regularizer: wich regularizer to use on each conv2d layer (in paper l2(1e-5))
    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid model type, argument for `weights`,
            or invalid input shape when weights='imagenet'
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(
                    keras_utils.get_source_inputs(input_tensor))
            except ValueError:
                raise ValueError('input_tensor: ', input_tensor,
                                 'is not type input_tensor')
        if is_input_t_tensor:
            if backend.image_data_format == 'channels_first':
                if backend.int_shape(input_tensor)[1] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape,
                                     'and input_tensor: ', input_tensor,
                                     'do not meet the same shape requirements')
            else:
                if backend.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape,
                                     'and input_tensor: ', input_tensor,
                                     'do not meet the same shape requirements')
        else:
            raise ValueError('input_tensor specified: ', input_tensor,
                             'is not a keras tensor')

    # If input_shape is None, infer shape from input_tensor
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError('input_tensor: ', input_tensor,
                             'is type: ', type(input_tensor),
                             'which is not a valid type')

        if backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == 'channels_first':
                rows = backend.int_shape(input_tensor)[2]
                cols = backend.int_shape(input_tensor)[3]
                input_shape = (3, cols, rows)
            else:
                rows = backend.int_shape(input_tensor)[1]
                cols = backend.int_shape(input_tensor)[2]
                input_shape = (cols, rows, 3)
    # If input_shape is None and input_tensor is None using standart shape
    if input_shape is None and input_tensor is None:
        input_shape = (None, None, 3)

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if rows and cols and (rows < 32 or cols < 32):
        raise ValueError('Input size must be at least 32x32; got `input_shape=' +
                         str(input_shape) + '`')
    if weights == 'imagenet':
        if minimalistic is False and alpha not in [0.75, 1.0] \
                or minimalistic is True and alpha != 1.0:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of `0.75`, `1.0` for non minimalistic'
                             ' or `1.0` for minimalistic only.')

        if rows != cols or rows != 224:
            warnings.warn('`input_shape` is undefined or non-square, '
                          'or `rows` is not 224.'
                          ' Weights for input shape (224, 224) will be'
                          ' loaded as the default.')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    if minimalistic:
        kernel = 3
        activation = 'relu'
        se_ratio = None
    else:
        kernel = 5
        activation = 'hardswish'
        se_ratio = 0.25

    x = layers.Conv2D(16,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='same',
                      kernel_regularizer=regularizer,
                      use_bias=False,
                      name='Conv')(img_input)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='Conv/BatchNorm')(x)
    x = _activation(x, name=activation)

    if model_type == 'small':
        #   (inputs, expansion, alpha, out_ch, kernel_size, stride, se_ratio,
        #   activation, regularizer, block_id)
        x = _inverted_res_block(x, 1, 16, alpha, 3, 2, se_ratio,
                                'relu', regularizer, 0)
        x = _inverted_res_block(x, 4.5, 24, alpha, 3, 2, None,
                                'relu', regularizer, 1)
        x = _inverted_res_block(x, 3.66, 24, alpha, 3, 1, None,
                                'relu', regularizer, 2)
        x = _inverted_res_block(x, 4, 40, alpha, kernel, 2, se_ratio,
                                activation, regularizer, 3)
        x = _inverted_res_block(x, 6, 40, alpha, kernel, 1, se_ratio,
                                activation, regularizer, 4)
        x = _inverted_res_block(x, 6, 40, alpha, kernel, 1, se_ratio,
                                activation, regularizer, 5)
        x = _inverted_res_block(x, 3, 48, alpha, kernel, 1, se_ratio,
                                activation, regularizer, 6)
        x = _inverted_res_block(x, 3, 48, alpha, kernel, 1, se_ratio,
                                activation, regularizer, 7)
        x = _inverted_res_block(x, 6, 96, alpha, kernel, 2, se_ratio,
                                activation, regularizer, 8)
        x = _inverted_res_block(x, 6, 96, alpha, kernel, 1, se_ratio,
                                activation, regularizer, 9)
        x = _inverted_res_block(x, 6, 96, alpha, kernel, 1, se_ratio,
                                activation, regularizer, 10)
        last_conv_ch = _make_divisible(576 * alpha, 8)
        last_point_ch = 1024
    elif model_type == 'large':
        x = _inverted_res_block(x, 1, 16, alpha, 3, 1, None,
                                'relu', regularizer, 0)
        x = _inverted_res_block(x, 4, 24, alpha, 3, 2, None,
                                'relu', regularizer, 1)
        x = _inverted_res_block(x, 3, 24, alpha, 3, 1, None,
                                'relu', regularizer, 2)
        x = _inverted_res_block(x, 3, 40, alpha, kernel, 2, se_ratio,
                                'relu', regularizer, 3)
        x = _inverted_res_block(x, 3, 40, alpha, kernel, 1, se_ratio,
                                'relu', regularizer, 4)
        x = _inverted_res_block(x, 3, 40, alpha, kernel, 1, se_ratio,
                                'relu', regularizer, 5)
        x = _inverted_res_block(x, 6, 80, alpha, 3, 2, None,
                                activation, regularizer, 6)
        x = _inverted_res_block(x, 2.5, 80, alpha, 3, 1, None,
                                activation, regularizer, 7)
        x = _inverted_res_block(x, 2.3, 80, alpha, 3, 1, None,
                                activation, regularizer, 8)
        x = _inverted_res_block(x, 2.3, 80, alpha, 3, 1, None,
                                activation, regularizer, 9)
        x = _inverted_res_block(x, 6, 112, alpha, 3, 1, se_ratio,
                                activation, regularizer, 10)
        x = _inverted_res_block(x, 6, 112, alpha, 3, 1, se_ratio,
                                activation, regularizer, 11)
        x = _inverted_res_block(x, 6, 160, alpha, kernel, 2, se_ratio,
                                activation, regularizer, 12)
        x = _inverted_res_block(x, 6, 160, alpha, kernel, 1, se_ratio,
                                activation, regularizer, 13)
        x = _inverted_res_block(x, 6, 160, alpha, kernel, 1, se_ratio,
                                activation, regularizer, 14)
        last_conv_ch = _make_divisible(960 * alpha, 8)
        last_point_ch = 1280
    else:
        raise ValueError('Invalid model type')

    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_point_ch = _make_divisible(last_point_ch * alpha, 8)

    x = layers.Conv2D(last_conv_ch,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      kernel_regularizer=regularizer,
                      use_bias=False,
                      name='Conv_1')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='Conv_1/BatchNorm')(x)
    x = _activation(x, name=activation)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape((1, 1, last_conv_ch))(x)
        x = layers.Conv2D(last_point_ch,
                          kernel_size=1,
                          padding='same',
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          name='Conv_2')(x)
        x = _activation(x, name=activation)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(classes,
                          kernel_size=1,
                          padding='same',
                          use_bias=True,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          name='Logits')(x)
        x = layers.Flatten()(x)
        x = layers.Softmax(name='Predictions/Softmax')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='MobilenetV3')

    # Load weights.
    if weights == 'imagenet':
        model_name = 'weights_mobilenet_v3_' + model_type
        if minimalistic:
            model_name += '_minimalistic'
        model_name += '_224_' + str(alpha) + '_float'
        if not include_top:
            model_name += '_no_top'
        model_name += '.h5'
        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = keras_utils.get_file(model_name, weight_path,
                                            cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def _inverted_res_block(inputs, expansion, alpha, out_ch, kernel_size, stride,
                        se_ratio, activation, regularizer, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    out_channels = _make_divisible(out_ch * alpha, 8)
    exp_size = _make_divisible(in_channels * expansion, 8)
    x = inputs
    prefix = 'expanded_conv/'
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(exp_size,
                          kernel_size=1,
                          padding='same',
                          kernel_regularizer=regularizer,
                          use_bias=False,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + 'expand/BatchNorm')(x)
        x = _activation(x, activation)

    x = layers.DepthwiseConv2D(kernel_size,
                               strides=stride,
                               padding='same',
                               dilation_rate=1,
                               depthwise_regularizer=regularizer,
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name=prefix + 'depthwise/BatchNorm')(x)
    x = _activation(x, activation)

    if se_ratio:
        reduced_ch = _make_divisible(exp_size * se_ratio, 8)
        y = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(x)
        y = layers.Reshape((1, 1, exp_size))(y)
        y = layers.Conv2D(reduced_ch,
                          kernel_size=1,
                          padding='same',
                          kernel_regularizer=regularizer,
                          use_bias=True,
                          name=prefix + 'squeeze_excite/Conv')(y)
        y = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(y)
        y = layers.Conv2D(exp_size,
                          kernel_size=1,
                          padding='same',
                          kernel_regularizer=regularizer,
                          use_bias=True,
                          name=prefix + 'squeeze_excite/Conv_1')(y)
        y = layers.Activation(hard_sigmoid)(y)
        if backend.backend() == 'theano':
            # For the Theano backend, we have to explicitly make
            # the excitation weights broadcastable.
            y = layers.Lambda(
                lambda br: backend.pattern_broadcast(br, [True, True, True, False]),
                output_shape=lambda input_shape: input_shape,
                name=prefix + 'squeeze_excite/broadcast')(y)
        x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([y, x])

    x = layers.Conv2D(out_channels,
                      kernel_size=1,
                      padding='same',
                      kernel_regularizer=regularizer,
                      use_bias=False,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name=prefix + 'project/BatchNorm')(x)

    if in_channels == out_channels and stride == 1:
        x = layers.Add(name=prefix + 'Add')([inputs, x])
    return x
