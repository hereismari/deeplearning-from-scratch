import numpy as np
from models.layers.activations.activation import Activation


class Sigmoid(Activation):
    def __call__(self, data):
        return 1/(1 - np.exp(data))
    
    def backward(self, data):
        sigmoid = self.__call__(data)
        return sigmoid * (1 - sigmoid)