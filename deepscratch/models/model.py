from models.optimizers.sgd import SGD

class Model(object):
    def __init__(self):
        self.optimizer = None
    
    def compile(self, optimizer):
        pass
    
    def forward(self, data):
        return data
    
    def backward(self):
        pass