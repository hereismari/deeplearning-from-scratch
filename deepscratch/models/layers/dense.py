import numpy as np
import warnings
import copy

from deepscratch.models.layers.layer import Layer
from deepscratch.models.layers.activation import Activation
import deepscratch.models.initializers as initializers


class Dense(Layer):
    def __init__(self, n_units, input_shape=None, trainable=True, initializer=None, **kwargs):
        self.n_units = n_units
        self.input_shape = input_shape
        self.trainable = trainable

        self.W = None
        self.b = None

        self._dW = None
        self._db = None

        self.W_opt = None
        self.b_opt = None

        self._current_input_data = None
        self._current_ouput_data = None

        self.initializer = initializers.load(initializer, **kwargs) if type(initializer) is str else initializer
    

    def initialize(self, initializer, optimizer, input_shape, **kwargs):
        if self.initializer is None:
            self.initializer = initializers.load(initializer, **kwargs) if type(initializer) is str else initializer
        else:
            warnings.warn('Layer already has a initializer so the model initializer was ignored')

        if self.input_shape is None:
            self.input_shape = input_shape
        else:
            assert self.input_shape is not None, 'First layer must have the input shape explicit defined'
            assert input_shape is None or self.input_shape == input_shape, 'Input shape %s is different than the one previously defined %s' % (input_shape, self.input_shape)

        self.W = self.initializer.init((self.input_shape[0], self.n_units))
        self.b = self.initializer.init((1, self.n_units))

        self.W_opt = optimizer(shape=self.W.shape, **kwargs)
        self.b_opt = optimizer(shape=self.b.shape, **kwargs)


    def forward(self, data):
        #print ('weight, bias')
        #print(self.W, self.b)
        if self.W is None or self.b is None:
            raise Exception('Layer must be initialized')
        
        self._current_input_data = data
        return np.dot(data, self.W) + self.b


    def backward(self, grads, update=True, **kwargs):
        dx = np.dot(grads, self.W.T)
        if self.trainable:
            self._dW = np.dot(self._current_input_data.T, grads)
            self._db = np.sum(grads, axis=0, keepdims=True)

            if update:
                self.W = self.W_opt(self.W, self._dW)
                self.b = self.b_opt(self.b, self._db)

        return dx


    def output_shape(self):
        return (self.n_units,)
    

    def params(self):
        return [self.W, self.b]


    def dparams(self):
        return [self._dW, self._db]