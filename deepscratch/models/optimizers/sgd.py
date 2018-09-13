

class SGD(object):

    def __init__(self, lr=0.001):
        self.lr = lr
    
    def __call__(self, data, grads):
        return self.optimize(data, grads)
    
    def optimize(self, data, grads):
        assert grads.shape == data.shape, (grads.shape, data.shape)
        return data -1 * self.lr * grads