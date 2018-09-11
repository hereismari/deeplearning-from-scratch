import numpy as np


class Activation(object):
    @staticmethod
    def forward(data):
        return data
    
    
    @staticmethod
    def backward(data):
        return np.ones(data.shape)
    
    def __call__(self, data):
        return self.forward(data)
    
    def grads(self, data):
        return self.backward(data)