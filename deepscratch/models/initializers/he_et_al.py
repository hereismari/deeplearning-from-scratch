import numpy as np
from deepscratch.models.initializers.initialization import Initialization


class HeEtAl(object):
    @staticmethod
    def init(shape, **kwargs):
        '''He-et-al initialization'''
        return np.random.randn(*shape) * np.sqrt(2/shape[0])