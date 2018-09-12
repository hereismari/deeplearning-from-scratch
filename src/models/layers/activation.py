import numpy as np

from models.layers.layer import Layer

from models.layers.activations.activation import Activation as Linear
from models.layers.activations.relu import Relu
from models.layers.activations.sigmoid import Sigmoid
from models.layers.activations.tanh import Tanh


class Activation(Layer):
    _activations = {
        'relu': Relu,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'linear': Linear
    }
    
    def __init__(self, name):
        if name not in self._activations:
            raise ValueError('Activation unknown %s' % name)
        self._activation = self._activations[name]
        
    def __call__(self, data):
        return self.forward(data)
    
    def forward(self, data):
        self._current_input_data = data
        return self._activation(data)

    def backward(self, grads):
        return grads * self._activation.grads(self._current_input_data)
    