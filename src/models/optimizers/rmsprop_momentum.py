from models.optimizers.sgd import SGD

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
        assert data.shape == grads.shape, data.shape, grads.shape
        dx = grad[w_type]
        dx_mean_sqr = decay * self.mean_sqrt + (1 - decay) * dx ** 2
        momentum = mu * self.momentum + lr * dx / (np.sqrt(dx_mean_sqr) + eps)
        self._momentum = momentum
        self.mean_sqrt = dx_mean_sqr
        return data - momentum