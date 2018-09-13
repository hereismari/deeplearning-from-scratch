import numpy as np
from deepscratch.models.initializers.initialization import Initialization


class RandomInitialization(object):
    @staticmethod
    def init(shape, delta=0.001, **kwargs):
        data = np.random.randn(*shape) * delta
        return data