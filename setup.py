from setuptools import setup
from setuptools import find_packages

long_description = '''
Keras Applications is the `applications` module of
the Keras deep learning library.
It provides model definitions and pre-trained weights for a number
of popular archictures, such as VGG16, ResNet50, Xception, MobileNet, and more.

Read the documentation at: https://keras.io/applications/

Keras Applications may be imported directly
from an up-to-date installation of Keras:

```
from keras import applications
```

Keras Applications is compatible with Python 2.7-3.6
and is distributed under the MIT license.
'''

setup(name='Keras_Applications',
      version='1.0.4',
      description='Reference implementations of popular deep learning models',
      long_description=long_description,
      author='Keras Team',
      url='https://github.com/keras-team/keras-applications',
      download_url='https://github.com/keras-team/'
                   'keras-applications/tarball/1.0.4',
      license='MIT',
      install_requires=['keras>=2.1.6',
                        'numpy>=1.9.1',
                        'h5py'],
      extras_require={
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
