import numpy as np


class VIM(object):
    _mat = None
    _names = None

    def __init__(self, size, names):
        # TODO: add doc
        self._mat = np.zeros([size, size])
        self._names = names

