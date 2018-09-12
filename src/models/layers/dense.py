import numpy as np
from models.layers.layer import Layer
from models.layers.activations.activation import Activation

class Dense(Layer):
    def __init__(self, n_units, input_shape=None, activation=Activation, trainable=True):
        self._n_units = n_units
        self._input_shape = input_shape
        self.activation = activation
        self.trainable = trainable

        self.W = None
        self.b = None

        self._dW = None
        self._db = None
        self._current_input_data = None
        self._current_ouput_data = None


    def forward(self, data):
        if self.W is None or self.b is None:
            raise Exception('Layer must be initialized')
        self._current_input_data = data
        self._current_ouput_data = np.dot(data, self.W) + self.b 
        return self.activation(self._current_ouput_data)
    

    def backward(self, grads):
        if self.trainable:
            self._dW = np.dot(self.current_input_data.T, grads)
            self._db = np.sum(grads, axis=0, keepdims=True)
        
            self.W = self.optimize(self.W, self._dW)
            self.W = self.optimize(self.W, self._dW)

        return np.dot(grads, self.W.T)
    
    def run_batch(self, batch):
        self._current_batch = [batch]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            output = np.dot(self._current_batch[-1], w) + b
            output = self.activations[i](output)
            self._current_batch.append(output)
        
        self._current_batch = self._current_batch[::-1]
        return output
    
    def run_batch(self, batch):
        self._current_batch = [batch]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            output = np.dot(self._current_batch[-1], w) + b
            output = self.activations[i](output)
            self._current_batch.append(output)
        
        self._current_batch = self._current_batch[::-1]
        return output