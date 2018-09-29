import pytest
import numpy as np
from numpy.testing import assert_allclose

# We don't use keras.applications.imagenet_utils here
# because we also test _obtain_input_shape which is not exposed.
from keras_applications import imagenet_utils as utils
from keras import backend
from keras import models
from keras import layers
from keras import utils as keras_utils


def decode_predictions(*args, **kwargs):
    kwargs['backend'] = backend
    kwargs['utils'] = keras_utils
    return utils.decode_predictions(*args, **kwargs)


def preprocess_input(*args, **kwargs):
    kwargs['backend'] = backend
    return utils.preprocess_input(*args, **kwargs)


def test_preprocess_input():
    # Test image batch with float and int image input
    x = np.random.uniform(0, 255, (2, 10, 10, 3))
    xint = x.astype('int32')
    assert preprocess_input(x).shape == x.shape
    assert preprocess_input(xint).shape == xint.shape

    out1 = preprocess_input(x, 'channels_last')
    out1int = preprocess_input(xint, 'channels_last')
    out2 = preprocess_input(np.transpose(x, (0, 3, 1, 2)), 'channels_first')
    out2int = preprocess_input(np.transpose(xint, (0, 3, 1, 2)), 'channels_first')
    assert_allclose(out1, out2.transpose(0, 2, 3, 1))
    assert_allclose(out1int, out2int.transpose(0, 2, 3, 1))

    # Test single image
    x = np.random.uniform(0, 255, (10, 10, 3))
    xint = x.astype('int32')
    assert preprocess_input(x).shape == x.shape
    assert preprocess_input(xint).shape == xint.shape

    out1 = preprocess_input(x, 'channels_last')
    out1int = preprocess_input(xint, 'channels_last')
    out2 = preprocess_input(np.transpose(x, (2, 0, 1)), 'channels_first')
    out2int = preprocess_input(np.transpose(xint, (2, 0, 1)), 'channels_first')
    assert_allclose(out1, out2.transpose(1, 2, 0))
    assert_allclose(out1int, out2int.transpose(1, 2, 0))

    # Test that writing over the input data works predictably
    for mode in ['torch', 'tf']:
        x = np.random.uniform(0, 255, (2, 10, 10, 3))
        xint = x.astype('int')
        x2 = preprocess_input(x, mode=mode)
        xint2 = preprocess_input(xint)
        assert_allclose(x, x2)
        assert xint.astype('float').max() != xint2.max()
    # Caffe mode works differently from the others
    x = np.random.uniform(0, 255, (2, 10, 10, 3))
    xint = x.astype('int')
    x2 = preprocess_input(x, data_format='channels_last', mode='caffe')
    xint2 = preprocess_input(xint)
    assert_allclose(x, x2[..., ::-1])
    assert xint.astype('float').max() != xint2.max()


def test_preprocess_input_symbolic():
    # Test image batch
    x = np.random.uniform(0, 255, (2, 10, 10, 3))
    inputs = layers.Input(shape=x.shape[1:])
    outputs = layers.Lambda(preprocess_input, output_shape=x.shape[1:])(inputs)
    model = models.Model(inputs, outputs)
    assert model.predict(x).shape == x.shape

    outputs1 = layers.Lambda(
        lambda x: preprocess_input(x, 'channels_last'),
        output_shape=x.shape[1:])(inputs)
    model1 = models.Model(inputs, outputs1)
    out1 = model1.predict(x)
    x2 = np.transpose(x, (0, 3, 1, 2))
    inputs2 = layers.Input(shape=x2.shape[1:])
    outputs2 = layers.Lambda(
        lambda x: preprocess_input(x, 'channels_first'),
        output_shape=x2.shape[1:])(inputs2)
    model2 = models.Model(inputs2, outputs2)
    out2 = model2.predict(x2)
    assert_allclose(out1, out2.transpose(0, 2, 3, 1))

    # Test single image
    x = np.random.uniform(0, 255, (10, 10, 3))
    inputs = layers.Input(shape=x.shape)
    outputs = layers.Lambda(preprocess_input, output_shape=x.shape)(inputs)
    model = models.Model(inputs, outputs)
    assert model.predict(x[np.newaxis])[0].shape == x.shape

    outputs1 = layers.Lambda(
        lambda x: preprocess_input(x, 'channels_last'),
        output_shape=x.shape)(inputs)
    model1 = models.Model(inputs, outputs1)
    out1 = model1.predict(x[np.newaxis])[0]
    x2 = np.transpose(x, (2, 0, 1))
    inputs2 = layers.Input(shape=x2.shape)
    outputs2 = layers.Lambda(
        lambda x: preprocess_input(x, 'channels_first'),
        output_shape=x2.shape)(inputs2)
    model2 = models.Model(inputs2, outputs2)
    out2 = model2.predict(x2[np.newaxis])[0]
    assert_allclose(out1, out2.transpose(1, 2, 0))


def test_decode_predictions():
    x = np.zeros((2, 1000))
    x[0, 372] = 1.0
    x[1, 549] = 1.0
    outs = decode_predictions(x, top=1)
    scores = [out[0][2] for out in outs]
    assert scores[0] == scores[1]

    # the numbers of columns and ImageNet classes are not identical.
    with pytest.raises(ValueError):
        decode_predictions(np.ones((2, 100)))


def test_obtain_input_shape():
    # input_shape and default_size are not identical.
    with pytest.raises(ValueError):
        utils._obtain_input_shape(
            input_shape=(224, 224, 3),
            default_size=299,
            min_size=139,
            data_format='channels_last',
            require_flatten=True,
            weights='imagenet')

    # Test invalid use cases
    for data_format in ['channels_last', 'channels_first']:
        # test warning
        shape = (139, 139)
        if data_format == 'channels_last':
            input_shape = shape + (99,)
        else:
            input_shape = (99,) + shape
        with pytest.warns(UserWarning):
            utils._obtain_input_shape(
                input_shape=input_shape,
                default_size=None,
                min_size=139,
                data_format=data_format,
                require_flatten=False,
                weights='fake_weights')

        # input_shape is smaller than min_size.
        shape = (100, 100)
        if data_format == 'channels_last':
            input_shape = shape + (3,)
        else:
            input_shape = (3,) + shape
        with pytest.raises(ValueError):
            utils._obtain_input_shape(
                input_shape=input_shape,
                default_size=None,
                min_size=139,
                data_format=data_format,
                require_flatten=False)

        # shape is 1D.
        shape = (100,)
        if data_format == 'channels_last':
            input_shape = shape + (3,)
        else:
            input_shape = (3,) + shape
        with pytest.raises(ValueError):
            utils._obtain_input_shape(
                input_shape=input_shape,
                default_size=None,
                min_size=139,
                data_format=data_format,
                require_flatten=False)

        # the number of channels is 5 not 3.
        shape = (100, 100)
        if data_format == 'channels_last':
            input_shape = shape + (5,)
        else:
            input_shape = (5,) + shape
        with pytest.raises(ValueError):
            utils._obtain_input_shape(
                input_shape=input_shape,
                default_size=None,
                min_size=139,
                data_format=data_format,
                require_flatten=False)

        # require_flatten=True with dynamic input shape.
        with pytest.raises(ValueError):
            utils._obtain_input_shape(
                input_shape=None,
                default_size=None,
                min_size=139,
                data_format='channels_first',
                require_flatten=True)

    # test include top
    assert utils._obtain_input_shape(
        input_shape=(3, 200, 200),
        default_size=None,
        min_size=139,
        data_format='channels_first',
        require_flatten=True) == (3, 200, 200)

    assert utils._obtain_input_shape(
        input_shape=None,
        default_size=None,
        min_size=139,
        data_format='channels_last',
        require_flatten=False) == (None, None, 3)

    assert utils._obtain_input_shape(
        input_shape=None,
        default_size=None,
        min_size=139,
        data_format='channels_first',
        require_flatten=False) == (3, None, None)

    assert utils._obtain_input_shape(
        input_shape=None,
        default_size=None,
        min_size=139,
        data_format='channels_last',
        require_flatten=False) == (None, None, 3)

    assert utils._obtain_input_shape(
        input_shape=(150, 150, 3),
        default_size=None,
        min_size=139,
        data_format='channels_last',
        require_flatten=False) == (150, 150, 3)

    assert utils._obtain_input_shape(
        input_shape=(3, None, None),
        default_size=None,
        min_size=139,
        data_format='channels_first',
        require_flatten=False) == (3, None, None)


if __name__ == '__main__':
    pytest.main([__file__])
