import numpy as np
from deepscratch.models.layers.activations.activation import Activation


class Softmax(Activation):
    def __call__(self, data):
        exp = np.exp(data - np.max(data, axis=-1, keepdims=True))
        return exp / np.sum(exp, axis=-1, keepdims=True)
    
    def backward(self, data):
        softmax = self.__call__(data)
        return softmax * (1 - softmax)