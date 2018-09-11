import numpy as np
from models.layers.activations.activation import Activation


class Tanh(Activation):
    @staticmethod
    def forward(data):
        return np.tanh(data)
    
    
    @staticmethod
    def backward(data):
        return 1 - Tanh.forward(data) ** 2