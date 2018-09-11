import numpy as np
from models.layers.activations.activation import Activation


class Sigmoid(Activation):
    @staticmethod
    def forward(data):
        return 1/(1 - np.exp(data))
    
    
    @staticmethod
    def backward(data):
        sigmoid = Sigmoid.forward(data)
        return sigmoid * (1 - sigmoid)