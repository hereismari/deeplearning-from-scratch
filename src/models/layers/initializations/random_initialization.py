import numpy as np
from models.layers.initializations.initialization import Initialization


class RandomInitialization(object):
    @staticmethod
    def init(shape, delta=0.001, **kwargs):
        data = np.random.randn(*shape) * delta
        return data