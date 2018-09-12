

class SGD(object):

    def __init__(self, lr=0.001):
        self.lr = lr
    

    def optimize(self, data, grads):
        assert grads.shape == data.shape, grads.shape, data.shape
        return data -1 * lr * grads
