class Model(object):
    def __init__(self):
        self.optimizer = None
    
    def initialize(self, **kwargs):
        raise NotImplementedError()
    
    def forward(self, data):
        raise NotImplementedError()
    
    def backward(self, grads, **kwargs):
        raise NotImplementedError()