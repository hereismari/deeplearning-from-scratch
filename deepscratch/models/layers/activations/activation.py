import numpy as np


class Activation(object):
    def __call__(self, data):
        raise NotImplementedError()
    
    def grads(self, data):
        raise NotImplementedError()