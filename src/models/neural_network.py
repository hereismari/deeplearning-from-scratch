class NeuralNetwork(object):
    def __init__(self, layers=[]):
        self.layers=layers

    def add_layer(self, layer):
        self.layers.add(layer)

    def pop_layer(self):
        return self.layers.pop()
    
    def initialize(self):
        for layer in self.layers:
            layer.initialize()
    
    def forward(self, data):
        output = data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grads):
        for layer in self.layers[::-1]:
            grads = layer.backward(grads)