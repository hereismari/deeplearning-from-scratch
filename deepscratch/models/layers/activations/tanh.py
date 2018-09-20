import numpy as np
from deepscratch.models.layers.activations.activation import Activation


class Tanh(Activation):
    def __call__(self, data):
        return np.tanh(data)
    
    
    def grads(self, data):
        return 1 - self.__call__(data) ** 2