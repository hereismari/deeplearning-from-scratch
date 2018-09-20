import numpy as np
from deepscratch.models.layers.activations.activation import Activation


class Relu(Activation):
    def __call__(self, data):
        return np.maximum(0, data)
    
    def grads(self, data):
        relu = self.__call__(data)
        return (relu > 0) * 1.0