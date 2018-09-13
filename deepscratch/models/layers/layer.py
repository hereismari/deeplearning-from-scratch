import numpy as np


class Layer(object):
    def __init__(self):
        self.input_shape = None

    def forward(self, data):
        return NotImplementedError()
    
    def backward(self, grads):
        return NotImplementedError()

    def initialize(self, initializer, otimizer, input_shape, **kwargs):
        pass

    def output_shape(self):
        return NotImplementedError()