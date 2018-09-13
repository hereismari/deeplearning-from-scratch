import numpy as np


class Initialization(object):
    @staticmethod
    def init(shape, **kwargs):
        data = np.zeros(shape)
        return data