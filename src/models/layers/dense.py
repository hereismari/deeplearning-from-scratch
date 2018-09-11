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
        d = grads * self.activation.grads(self._current_output_data)
        if self.trainable:
            self._dW = np.dot(self.current_input_data, d)
            self._db = np.sum(d, axis=0, keepdims=True)
        
            self.W = self.optimize(self.W, self._dW)
            self.W = self.optimize(self.W, self._dW)

        return np.dot(d, self.W.T)


    def update_params(self, gradients, learning_rate=0.1):
        assert len(gradients) == len(self.weights), (len(gradients), len(self.weights))
        assert len(gradients) == len(self.biases), (len(gradients), len(self.biases))
        
        for i, grad in enumerate(gradients[::-1]):
            assert grad['weights'].shape == self.weights[i].shape
            self.weights[i] -= learning_rate * grad['weights']
            assert grad['biases'].shape == self.biases[i].shape
            self.biases[i] -= learning_rate * grad['biases']

    
    def run_batch(self, batch):
        self._current_batch = [batch]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            output = np.dot(self._current_batch[-1], w) + b
            output = self.activations[i](output)
            self._current_batch.append(output)
        
        self._current_batch = self._current_batch[::-1]
        return output

    def update_params(self, gradients, learning_rate=0.1):
        assert len(gradients) == len(self.weights), (len(gradients), len(self.weights))
        assert len(gradients) == len(self.biases), (len(gradients), len(self.biases))
        
        for i, grad in enumerate(gradients[::-1]):
            assert grad['weights'].shape == self.weights[i].shape
            self.weights[i] -= learning_rate * grad['weights']
            assert grad['biases'].shape == self.biases[i].shape
            self.biases[i] -= learning_rate * grad['biases']

    
    def run_batch(self, batch):
        self._current_batch = [batch]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            output = np.dot(self._current_batch[-1], w) + b
            output = self.activations[i](output)
            self._current_batch.append(output)
        
        self._current_batch = self._current_batch[::-1]
        return output