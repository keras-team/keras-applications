"""UNet model for Keras

# Reference

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](
    https://arxiv.org/abs/1505.04597)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from . import get_keras_submodule

backend = get_keras_submodule('backend')
engine = get_keras_submodule('engine')
layers = get_keras_submodule('layers')
models = get_keras_submodule('models')


def double_conv3x3(input_tensor,
                   filters,
                   activation,
                   padding,
                   block_name,
                   conv_index=1):
    """ Apply successively two convolution layers

    # Arguments
        input_tensor: input tensor
        filters: integer, number of filters on convolution layers
        activation: activation function name on convolution layers
        padding: padding to use on convolution layers
        block: current block name
        conv_index: current index of the convolution,
            to build the right name for layer

    # Returns
        Output tensor for the block
    """
    x = layers.Conv2D(filters, (3, 3),
                      activation=activation,
                      padding=padding,
                      name=block_name + '_conv' + str(conv_index)
                      )(input_tensor)
    conv_index += 1
    x = layers.Conv2D(filters, (3, 3),
                      activation=activation,
                      padding=padding,
                      name=block_name + '_conv' + str(conv_index))(x)

    return x


def double_conv3x3_maxpool(input_tensor, filters, activation, padding, block_name):
    """ Apply successively two convolution layers and a max-pooling

    # Arguments
        input_tensor: input tensor
        filters: integer, number of filters on convolution layers
        activation: activation function name on convolution layers
        padding: padding to use on convolution layers
        block: current block name

    # Returns
        Output tensor for the block
    """
    conv = double_conv3x3(input_tensor, filters, activation, padding, block_name)
    x = layers.MaxPooling2D((2, 2),
                            strides=(2, 2),
                            name=block_name + '_pool')(conv)

    return x, conv


def UNet(input_tensor=None,
         input_shape=None,
         padding='valid',
         weights=None):
    """ Instanciates the UNet architecture.

    Optionally loads weights pre-trained on Biomedical Image segmentation dataset ?

    # Arguments
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        padding: padding to use. Original paper uses valid padding which implies
            cropping, a same padding keep the image size between input and output.
        weights: Path to pretrained weights
    """
    depth_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Block 1
    x, conv1 = double_conv3x3_maxpool(img_input, 64,
                                      activation='relu',
                                      padding=padding,
                                      block_name='block1')

    # Block 2
    x, conv2 = double_conv3x3_maxpool(x, 128,
                                      activation='relu',
                                      padding=padding,
                                      block_name='block2')

    # Block 3
    x, conv3 = double_conv3x3_maxpool(x, 256,
                                      activation='relu',
                                      padding=padding,
                                      block_name='block3')

    # Block 4
    x, conv4 = double_conv3x3_maxpool(x, 512,
                                      activation='relu',
                                      padding=padding,
                                      block_name='block4')

    # Block 5 - lowest resolution
    x = double_conv3x3(x, 1024,
                       activation='relu',
                       padding=padding,
                       block_name='block5')

    # If padding is valid, it is necessary to crop the outputs of convolutions
    # as seen in the original paper
    if padding == 'valid':
        conv1 = layers.Cropping2D(cropping=((88, 88), (88, 88)))(conv1)
        conv2 = layers.Cropping2D(cropping=((40, 40), (40, 40)))(conv2)
        conv3 = layers.Cropping2D(cropping=((16, 16), (16, 16)))(conv3)
        conv4 = layers.Cropping2D(cropping=((4, 4), (4, 4)))(conv4)

    # Block 6
    x = layers.UpSampling2D((2, 2), name='block6_upsample')(x)
    conv6 = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block6_conv1')(x)

    x = layers.Concatenate(axis=depth_axis, name='block6_concat')([conv4, conv6])
    x = double_conv3x3(x, 512,
                       activation='relu', padding=padding,
                       block_name='block6', conv_index=2)

    # Block 7
    x = layers.UpSampling2D((2, 2), name='block7_upsample')(x)
    conv7 = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block7_conv1')(x)
    x = layers.Concatenate(axis=depth_axis, name='block7_concat')([conv3, conv7])
    x = double_conv3x3(x, 256,
                       activation='relu', padding=padding,
                       block_name='block7', conv_index=2)

    # Block 8
    x = layers.UpSampling2D((2, 2), name='block8_upsample')(x)
    conv8 = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block8_conv1')(x)
    x = layers.Concatenate(axis=depth_axis, name='block8_concat')([conv2, conv8])
    x = double_conv3x3(x, 128,
                       activation='relu', padding=padding,
                       block_name='block8', conv_index=2)

    # Block 9
    x = layers.UpSampling2D((2, 2), name='block9_upsample')(x)
    conv9 = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block9_conv1')(x)
    x = layers.Concatenate(axis=depth_axis, name='block9_concat')([conv1, conv9])
    x = double_conv3x3(x, 64,
                       activation='relu', padding=padding,
                       block_name='block9', conv_index=2)

    x = layers.Conv2D(2, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block10_conv1')(x)

    model = models.Model(inputs, x, name='unet')

    if weights is not None:
        model.load_weights(weights)

    return model
