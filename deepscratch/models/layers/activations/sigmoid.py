import numpy as np
from deepscratch.models.layers.activations.activation import Activation


class Sigmoid(Activation):
    def __call__(self, data):
        exp = np.exp(data)
        return exp / (1 + exp)

    def backward(self, data):
        sigmoid = self.__call__(data)
        return sigmoid * (1 - sigmoid)