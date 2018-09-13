import numpy as np


class Layer(object):
    def forward(self, data):
        return data
    
    def backward(self, grads):
        return np.ones(grads.shape)
    
    def initialize(self, **kwargs):
        pass