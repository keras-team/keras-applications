import pytest
import random
import six
import numpy as np

import keras_applications
from keras.applications import densenet
from keras.applications import inception_resnet_v2
from keras.applications import inception_v3
from keras.applications import mobilenet
try:
    from keras.applications import mobilenet_v2
except ImportError:
    from keras.applications import mobilenetv2 as mobilenet_v2
from keras.applications import nasnet
from keras.applications import resnet50
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import xception
from keras.preprocessing import image
from keras import backend
from keras import layers
from keras import models
from keras import utils

from multiprocessing import Process, Queue


def keras_modules_injection(base_fun):

    def wrapper(*args, **kwargs):
        if hasattr(keras_applications, 'get_submodules_from_kwargs'):
            kwargs['backend'] = backend
            kwargs['layers'] = layers
            kwargs['models'] = models
            kwargs['utils'] = utils
        return base_fun(*args, **kwargs)
    return wrapper


for (name, module) in [('resnet', keras_applications.resnet),
                       ('resnet_v2', keras_applications.resnet_v2),
                       ('resnext', keras_applications.resnext)]:
    module.decode_predictions = keras_modules_injection(module.decode_predictions)
    module.preprocess_input = keras_modules_injection(module.preprocess_input)
    for app in dir(module):
        if app[0].isupper():
            setattr(module, app, keras_modules_injection(getattr(module, app)))
    setattr(keras_applications, name, module)


RESNET_LIST = [keras_applications.resnet.ResNet50,
               keras_applications.resnet.ResNet101,
               keras_applications.resnet.ResNet152]
RESNETV2_LIST = [keras_applications.resnet_v2.ResNet50V2,
                 keras_applications.resnet_v2.ResNet101V2,
                 keras_applications.resnet_v2.ResNet152V2]
RESNEXT_LIST = [keras_applications.resnext.ResNeXt50,
                keras_applications.resnext.ResNeXt101]
MOBILENET_LIST = [(mobilenet.MobileNet, mobilenet, 1024),
                  (mobilenet_v2.MobileNetV2, mobilenet_v2, 1280)]
DENSENET_LIST = [(densenet.DenseNet121, 1024),
                 (densenet.DenseNet169, 1664),
                 (densenet.DenseNet201, 1920)]
NASNET_LIST = [(nasnet.NASNetMobile, 1056),
               (nasnet.NASNetLarge, 4032)]


def keras_test(func):
    """Function wrapper to clean up after TensorFlow tests.
    # Arguments
        func: test function to clean up after.
    # Returns
        A function wrapping the input function.
    """
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        if backend.backend() == 'tensorflow' or backend.backend() == 'cntk':
            backend.clear_session()
        return output
    return wrapper


def _get_elephant(target_size):
    # For models that don't include a Flatten step,
    # the default is to accept variable-size inputs
    # even when loading ImageNet weights (since it is possible).
    # In this case, default to 299x299.
    if target_size[0] is None:
        target_size = (299, 299)
    img = image.load_img('tests/data/elephant.jpg',
                         target_size=tuple(target_size))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def _get_output_shape(model_fn, preprocess_input=None):
    if backend.backend() == 'cntk':
        # Create model in a subprocess so that
        # the memory consumed by InceptionResNetV2 will be
        # released back to the system after this test
        # (to deal with OOM error on CNTK backend).
        # TODO: remove the use of multiprocessing from these tests
        # once a memory clearing mechanism
        # is implemented in the CNTK backend.
        def target(queue):
            model = model_fn()
            if preprocess_input is None:
                queue.put(model.output_shape)
            else:
                x = _get_elephant(model.input_shape[1:3])
                x = preprocess_input(x)
                queue.put((model.output_shape, model.predict(x)))
        queue = Queue()
        p = Process(target=target, args=(queue,))
        p.start()
        p.join()
        # The error in a subprocess won't propagate
        # to the main process, so we check if the model
        # is successfully created by checking if the output shape
        # has been put into the queue
        assert not queue.empty(), 'Model creation failed.'
        return queue.get_nowait()
    else:
        model = model_fn()
        if preprocess_input is None:
            return model.output_shape
        else:
            x = _get_elephant(model.input_shape[1:3])
            x = preprocess_input(x)
            return (model.output_shape, model.predict(x))


@keras_test
def _test_application_basic(app, last_dim=1000, module=None):
    if module is None:
        output_shape = _get_output_shape(lambda: app(weights=None))
        assert output_shape == (None, None, None, last_dim)
    else:
        output_shape, preds = _get_output_shape(
            lambda: app(weights='imagenet'), module.preprocess_input)
        assert output_shape == (None, last_dim)

        names = [p[1] for p in module.decode_predictions(preds)[0]]
        # Test correct label is in top 3 (weak correctness test).
        assert 'African_elephant' in names[:3]


@keras_test
def _test_application_notop(app, last_dim):
    output_shape = _get_output_shape(
        lambda: app(weights=None, include_top=False))
    assert output_shape == (None, None, None, last_dim)


@keras_test
def _test_application_variable_input_channels(app, last_dim):
    if backend.image_data_format() == 'channels_first':
        input_shape = (1, None, None)
    else:
        input_shape = (None, None, 1)
    output_shape = _get_output_shape(
        lambda: app(weights=None, include_top=False, input_shape=input_shape))
    assert output_shape == (None, None, None, last_dim)

    if backend.image_data_format() == 'channels_first':
        input_shape = (4, None, None)
    else:
        input_shape = (None, None, 4)
    output_shape = _get_output_shape(
        lambda: app(weights=None, include_top=False, input_shape=input_shape))
    assert output_shape == (None, None, None, last_dim)


@keras_test
def _test_app_pooling(app, last_dim):
    output_shape = _get_output_shape(
        lambda: app(weights=None,
                    include_top=False,
                    pooling=random.choice(['avg', 'max'])))
    assert output_shape == (None, last_dim)


def test_resnet():
    app = random.choice(RESNET_LIST)
    module = keras_applications.resnet
    last_dim = 2048
    _test_application_basic(app, module=module)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_resnetv2():
    app = random.choice(RESNETV2_LIST)
    module = keras_applications.resnet_v2
    last_dim = 2048
    _test_application_basic(app, module=module)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_resnext():
    app = random.choice(RESNEXT_LIST)
    module = keras_applications.resnext
    _test_application_basic(app, module=module)


def test_vgg():
    app = random.choice([vgg16.VGG16, vgg19.VGG19])
    module = vgg16
    last_dim = 512
    _test_application_basic(app, module=module)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_xception():
    app = xception.Xception
    module = xception
    last_dim = 2048
    _test_application_basic(app, module=module)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_inceptionv3():
    app = inception_v3.InceptionV3
    module = inception_v3
    last_dim = 2048
    _test_application_basic(app, module=module)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_inceptionresnetv2():
    app = inception_resnet_v2.InceptionResNetV2
    module = inception_resnet_v2
    last_dim = 1536
    _test_application_basic(app, module=module)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_mobilenet():
    app, module, last_dim = random.choice(MOBILENET_LIST)
    _test_application_basic(app, module=module)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_densenet():
    app, last_dim = random.choice(DENSENET_LIST)
    module = densenet
    _test_application_basic(app, module=module)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_nasnet():
    app, last_dim = NASNET_LIST[0]  # NASNetLarge is too heavy to test on Travis
    module = nasnet
    _test_application_basic(app, module=module)
    # _test_application_notop(app, last_dim)
    # _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


if __name__ == '__main__':
    pytest.main([__file__])
