# `keras_pickle_wrapper`

This small library exposes a KerasPickleWrapper class that allows keras models to be pickled, and even used across multiprocessing (or with a library like [job\_stream](https://github.com/wwoods/job_stream)).

It won't be necessary if https://github.com/fchollet/keras/issues/789 is ever properly resolved.

## Installation

`pip install keras-pickle-wrapper`

## Usage

```python

import keras
import pickle

from keras_pickle_wrapper import KerasPickleWrapper

ins = keras.layers.Input((2,))
x = ins
x = keras.layers.Dense(3)(x)
x = keras.layers.Dense(1)(x)
m = keras.models.Model(inputs=ins, outputs=x)
m.compile(loss='mse', optimizer='sgd')

# Wrap a compiled model
mw = KerasPickleWrapper(m)

# Calling the object returns the wrapped Keras model
mw().fit([[0,0], [0,1], [1,0], [1,1]], [[0], [1], [1], [0]])

# Pickle / unpickle the wrapper
data = pickle.dumps(mw)
mw2 = pickle.loads(data)
output_1 = mw().predict(np.asarray([[0, 0]]))
output_2 = mw2().predict(np.asarray([[0, 0]]))

# You can unload the object from memory as well
mw.unload()

# The object will remain unloaded until requested again
output_3 = mw().predict(np.asarray([[0, 0]]))

print("All outputs:")
print(output_1)
print(output_2)
print(output_3)

# If using tensorflow and pickling / unpickling a lot, be sure to clear the 
# session:
keras.backend.clear_session()

# When using a custom layer class, be sure to register it so the pickler works
class MyLayer(keras.layers.Layer):
    pass  # ...
KerasPickleWrapper.register(MyLayer)
```

If your model takes 1GB of RAM, the default approach should require 2GB additional RAM to encode, as it dumps to shared memory by default.  To disable this, set `KerasPickleWrapper.NO_SHM = True`.  Temporary files will then be written to the standard temporary directory.  Using `KerasPickleWrapper.unload(clear_session=True)` prior to pickling combined with `NO_SHM` should eliminate excess memory consumption, but clears the session (session clearing only applies to Tensorflow at the moment).


## Changelog

* 2017-9-19 - Fix for Python 2, up to V1.0.3.
* 2017-8-17 - Renamed to keras-pickle-wrapper because PyPI.  V1.0.2.
* 2017-8-17 - Packaged up KerasWrapper for PyPI distribution.

