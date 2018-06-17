import pytest
import sys

if sys.version_info > (3, 4):
    from importlib import reload


def test_that_internal_imports_are_not_overriden():
    # Test that changing the keras module after importing
    # Keras does not override keras.preprocessing's keras module
    import keras_applications
    reload(keras_applications)
    assert keras_applications._KERAS_BACKEND is None

    import keras
    if not hasattr(keras.applications, 'keras_applications'):
        return  # Old Keras, don't run.

    # TODO: enable the following tests
    # if get_source_inputs is included in tf.keras.utils
    return

    import tensorflow as tf
    keras_applications.set_keras_submodules(
        backend=tf.keras.backend,
        engine=tf.keras.engine,
        layers=tf.keras.layers,
        models=tf.keras.models,
        utils=tf.keras.utils)
    assert keras.applications.vgg16.backend is keras.backend

    # Now test the reverse order
    del keras
    reload(keras_applications)
    assert keras_applications._KERAS_BACKEND is None

    keras_applications.set_keras_submodules(
        backend=tf.keras.backend,
        engine=tf.keras.engine,
        layers=tf.keras.layers,
        models=tf.keras.models,
        utils=tf.keras.utils)
    import keras
    assert keras.applications.vgg16.backend is keras.backend


if __name__ == '__main__':
    pytest.main([__file__])
