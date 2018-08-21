"""Enables dynamic setting of underlying Keras module.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None


def set_keras_submodules(backend=None,
                         layers=None,
                         models=None,
                         utils=None,
                         engine=None):
    # Deprecated, will be removed in the future.
    global _KERAS_BACKEND
    global _KERAS_LAYERS
    global _KERAS_MODELS
    global _KERAS_UTILS
    _KERAS_BACKEND = backend
    _KERAS_LAYERS = layers
    _KERAS_MODELS = models
    _KERAS_UTILS = utils


def get_keras_submodule(name):
    # Deprecated, will be removed in the future.
    if name not in {'backend', 'layers', 'models', 'utils'}:
        raise ImportError(
            'Can only retrieve one of "backend", '
            '"layers", "models", or "utils". '
            'Requested: %s' % name)
    if _KERAS_BACKEND is None:
        raise ImportError('You need to first `import keras` '
                          'in order to use `keras_applications`. '
                          'For instance, you can do:\n\n'
                          '```\n'
                          'import keras\n'
                          'from keras_applications import vgg16\n'
                          '```\n\n'
                          'Or, preferably, this equivalent formulation:\n\n'
                          '```\n'
                          'from keras import applications\n'
                          '```\n')
    if name == 'backend':
        return _KERAS_BACKEND
    elif name == 'layers':
        return _KERAS_LAYERS
    elif name == 'models':
        return _KERAS_MODELS
    elif name == 'utils':
        return _KERAS_UTILS


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils

__version__ = '1.0.5'
