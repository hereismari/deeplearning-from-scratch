import numpy as np
from models.layers.initializations.initialization import Initialization


class RandomInitialization(object):
    @staticmethod
    def init(shape, **kwargs):
        '''He-et-all initialization'''
        return np.random.randn(*shape) * np.sqrt(2/shape[0]))