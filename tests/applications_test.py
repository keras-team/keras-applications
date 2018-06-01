import pytest
import random
import os
from multiprocessing import Process, Queue

import keras
# TODO: remove the few lines below once the Keras release
# is configured to use keras_applications
import keras_applications
keras_applications.set_keras_submodules(
    backend=keras.backend,
    engine=keras.engine,
    layers=keras.layers,
    models=keras.models,
    utils=keras.utils)

from keras_applications import densenet
from keras_applications import inception_resnet_v2
from keras_applications import inception_v3
from keras_applications import mobilenet
from keras_applications import mobilenet_v2
from keras_applications import nasnet
from keras_applications import resnet50
from keras_applications import vgg16
from keras_applications import vgg19
from keras_applications import xception

from keras.utils.test_utils import keras_test
from keras import backend


pytestmark = pytest.mark.skipif(
    os.environ.get('CORE_CHANGED', 'True') == 'False' and
    os.environ.get('APP_CHANGED', 'True') == 'False',
    reason='Runs only when the relevant files have been modified.')


MOBILENET_LIST = [(mobilenet.MobileNet, 1024),
                  (mobilenet_v2.MobileNetV2, 1280)]
DENSENET_LIST = [(densenet.DenseNet121, 1024),
                 (densenet.DenseNet169, 1664),
                 (densenet.DenseNet201, 1920)]
NASNET_LIST = [(nasnet.NASNetMobile, 1056),
               (nasnet.NASNetLarge, 4032)]


@keras_test
def _test_application_basic(app, last_dim=1000):
    model = app(weights=None)
    assert model.output_shape == (None, last_dim)


@keras_test
def _test_application_notop(app, last_dim):
    model = app(weights=None, include_top=False)
    assert model.output_shape == (None, None, None, last_dim)


@keras_test
def _test_application_variable_input_channels(app, last_dim):
    if backend.image_data_format() == 'channels_first':
        input_shape = (1, None, None)
    else:
        input_shape = (None, None, 1)
    model = app(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, last_dim)

    if backend.image_data_format() == 'channels_first':
        input_shape = (4, None, None)
    else:
        input_shape = (None, None, 4)
    model = app(weights=None, include_top=False, input_shape=input_shape)
    assert model.output_shape == (None, None, None, last_dim)


@keras_test
def _test_app_pooling(app, last_dim):
    model = app(weights=None,
                include_top=False,
                pooling=random.choice(['avg', 'max']))
    assert model.output_shape == (None, last_dim)


def test_resnet50():
    app = resnet50.ResNet50
    last_dim = 2048
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_vgg():
    app = random.choice([vgg16.VGG16, vgg19.VGG19])
    last_dim = 512
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


@pytest.mark.skipif((backend.backend() != 'tensorflow'),
                    reason='Xception supported only on TensorFlow')
def test_xception():
    app = xception.Xception
    last_dim = 2048
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_inceptionv3():
    app = inception_v3.InceptionV3
    last_dim = 2048
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_inceptionresnetv2():
    app = inception_resnet_v2.InceptionResNetV2
    last_dim = 1536
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_mobilenet():
    app, last_dim = random.choice(MOBILENET_LIST)
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_densenet():
    app, last_dim = random.choice(DENSENET_LIST)
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


@pytest.mark.skipif((backend.backend() != 'tensorflow'),
                    reason='NASNets are supported only on TensorFlow')
def test_nasnet():
    app, last_dim = random.choice(NASNET_LIST)
    _test_application_basic(app)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


if __name__ == '__main__':
    pytest.main([__file__])
