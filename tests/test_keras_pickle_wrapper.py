
import unittest

import keras_pickle_wrapper as kpw

class TestKerasPickleWrapper(unittest.TestCase):
    def test_example(self):
        ## Example from README.md, without import KerasPickleWrapper line
        KerasPickleWrapper = kpw.KerasPickleWrapper
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------
        import keras
        import numpy as np
        import pickle

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

        # When using a custom layer class, be sure to register it so the pickler works
        class MyLayer(keras.layers.Layer):
            pass  # ...
        KerasPickleWrapper.register(MyLayer)
        ## -------------------------------------------------------------------
        ## -------------------------------------------------------------------

        self.assertEqual(output_1, output_2)
        self.assertEqual(output_1, output_3)

