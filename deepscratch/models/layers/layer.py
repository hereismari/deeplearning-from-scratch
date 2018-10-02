import numpy as np


class Layer(object):
    def __init__(self):
        self.input_shape = None
    
    def name(self):
        return self.__class__.__name__

    def forward(self, data):
        return NotImplementedError()
    
    def backward(self, grads, **kwargs):
        return NotImplementedError()

    def initialize(self, initializer, otimizer, input_shape, **kwargs):
        pass

    def output_shape(self):
        return NotImplementedError()
    
    def params(self):
        return NotImplementedError()
    
    def dparams(self):
        return NotImplementedError()