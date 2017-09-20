"""Provides KerasPickleWrapper, a class which can wrap a keras model, allowing
it to be pickled.
"""

import keras
import os
import tempfile
import threading

# Dump files to shared memory, if possible
def _get_shm():
    """Return memory-mapped folder path, or None to use default tempfile 
    mechanics.
    """
    d = '/run/shm'
    if not os.path.lexists(d):
        d = '/dev/shm'
    if not os.path.lexists(d):
        # Use normal temporary folder
        d = None
    return d

_SHM_DIR = _get_shm()

# Registration mechanism for custom classes
_custom_classes = {}
def _register_custom_class(cls):
    """Registers a class (layer or whatever) for when keras needs it to
    deserialized.
    """
    name = cls.__name__
    if name in _custom_classes:
        raise ValueError("Cannot add {} twice".format(name))
    _custom_classes[name] = cls


# Keras, especially with tensorflow, must be single-threaded.  Therefore, it is
# paramount that we always load and save from the same pid/tid.  For job_stream
# at least, this is a sufficient condition to test for bad user behavior.  This
# approach is not strict enough for general use, though.
_keras_pickle_pid = [None]
def _keras_pickle_pid_check():
    my_pid = (os.getpid(), threading.current_thread().ident)
    if _keras_pickle_pid[0] is None:
        _keras_pickle_pid[0] = my_pid
    elif _keras_pickle_pid[0] != my_pid:
        raise ValueError("Must do all keras activity in a single process and "
                "thread.  It is OK if forks happen before keras activity, "
                "but not after.")

def _load_model(data):
    """data - [str].  Passed as array because the member is popped to decrease
    the refcount of the string, allowing the gc to collect it if needed.
    """
    _keras_pickle_pid_check()

    ofile = tempfile.NamedTemporaryFile('wb', dir=_SHM_DIR, suffix='.h5',
            delete=False)
    try:
        ofile.write(data[0])
        ofile.close()

        # decref on the data string
        data.pop()

        model = keras.models.load_model(ofile.name,
                custom_objects=_custom_classes)
        return model
    finally:
        os.unlink(ofile.name)
_load_model.__safe_for_unpickling__ = True


_reduce_should_clear_session = [False]
def _reduce_ex(self, protocol=None):
    _keras_pickle_pid_check()

    ofile = tempfile.NamedTemporaryFile(dir=_SHM_DIR, suffix='.h5',
            delete=False)
    try:
        ofile.close()
        self.save(ofile.name)
        # Before reading it out, clear the session if we're supposed to
        if _reduce_should_clear_session[0]:
            del self
            if hasattr(keras.backend, 'clear_session'):
                keras.backend.clear_session()
        with open(ofile.name, 'rb') as f:
            return (_load_model, ([f.read()],))
    finally:
        os.unlink(ofile.name)
# Please use KerasWrapper instead (see below)
#keras.models.Model.__reduce_ex__ = _reduce_ex


class KerasPickleWrapper(object):
    """Class providing pickle semantics for keras models.
    """

    # Class attributes / methods
    NO_SHM = False
    """NO_SHM should be set to `True` to prevent :class:`KerasPickleWrapper` 
    from using memory mapped storage when possible.
    """

    @classmethod
    def register(cls, custom_cls):
        """Required for loading pickled models with custom classes deriving
        from keras classes.  Registers the custom class with the unpickler.
        """
        _register_custom_class(custom_cls)

    # Instance attributes / methods
    __slots__ = ['_obj']

    @property
    def is_loaded(self):
        return not isinstance(self._obj, tuple)


    def __init__(self, obj):
        self._obj = obj


    def unload(self, clear_session=False):
        """Pickles our object so that we may be pickled.  Optionally clears the
        keras session.
        """
        if not self.is_loaded:
            raise ValueError("Must be loaded first")
        _reduce_should_clear_session[0] = clear_session
        self._obj = _reduce_ex(self._obj)


    def __call__(self):
        if not self.is_loaded:
            self._obj = self._obj[0](*self._obj[1])
        return self._obj


    def __reduce_ex__(self, protocol):
        if self.is_loaded:
            # Live pickling
            _reduce_should_clear_session[0] = False
            return (KerasPickleWrapper, (_reduce_ex(self._obj),))
        else:
            # Unloaded pickling
            return (KerasPickleWrapper, (self._obj,))

