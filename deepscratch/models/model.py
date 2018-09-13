class Model(object):
    def __init__(self):
        self.optimizer = None
    
    def initialize(self, **kwargs):
        pass
    
    def forward(self, data):
        return data
    
    def backward(self):
        pass