from distutils.core import setup
import os

try:
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'),
            'rt') as f:
        long_desc = f.read()
except FileNotFoundError:
    # Doesn't need to be in PyPI package.
    long_desc = None

setup(
        name='keras-pickle-wrapper',
        packages=['keras_pickle_wrapper'],
        install_requires=['keras', 'h5py'],
        version='1.0.4',
        description='A small library that wraps Keras models to pickle them.',
        long_description=long_desc,
        author='Walt Woods',
        author_email='woodswalben@gmail.com',
        url='https://github.com/wwoods/keras_pickle_wrapper',
        keywords=['keras', 'pickle'],
)

