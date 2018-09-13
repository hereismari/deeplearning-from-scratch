import numpy as np
from deepscratch.models.optimizers.sgd import SGD

class RMSProp(SGD):
    def __init__(self, shape, lr=0.001, decay=.9, eps=1e-8, mu=.9):
        super().__init__(lr=lr)
        self.decay = decay
        self.eps = eps
        self.mu = mu

        self.shape = shape
        self.mean_sqrt = np.zeros(shape, dtype=float)
        self.momentum = np.zeros(shape, dtype=float)


    def optimize(self, data, grads):
        assert data.shape == grads.shape, (data.shape, grads.shape)
        dx_mean_sqr = self.decay * self.mean_sqrt + (1 - self.decay) * grads ** 2
        momentum = self.mu * self.momentum + self.lr * grads / (np.sqrt(dx_mean_sqr) + self.eps)
        self._momentum = momentum
        self.mean_sqrt = dx_mean_sqr
        return data - momentum