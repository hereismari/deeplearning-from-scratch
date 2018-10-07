from deepscratch.models.model import Model
import deepscratch.models.optimizers as optimizers
import deepscratch.models.initializers as initializers


class NeuralNetwork(Model):
    def __init__(self, layers=[], initializer='random', optimizer='sgd', **kwargs):
        self.layers = layers
        self.optimizer = optimizers.load(optimizer, **kwargs) if type(optimizer) is str else optimizer
        self.initializer = initializers.load(initializer, **kwargs) if type(initializer) is str else initializer


    def __iter__(self):
        return iter(self.layers)


    def next(self): 
        for layer in self.layers:
            yield layer


    def __next__(self): # Python 3
        return next


    def add(self, layer):
        self.layers.add(layer)


    def pop(self):
        return self.layers.pop()


    def initialize(self, **kwargs):
        for i, layer in enumerate(self.layers):
            input_shape = self.layers[i-1].output_shape() if i > 0 else None
            layer.initialize(self.initializer, self.optimizer, input_shape, **kwargs)


    def forward(self, data):
        output = data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    
    def backward(self, grads, **kwargs):
        for layer in reversed(self.layers):
            grads = layer.backward(grads, **kwargs)
        return grads