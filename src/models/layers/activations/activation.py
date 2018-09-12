import numpy as np


class Activation(object):
    def __call__(self, data):
        return data
    
    def grads(self, data):
        return np.ones(data.shape)