import numpy as np


class Linear(object):
    def __call__(self, data):
        return data
    
    def grads(self, data):
        return data