import numpy as np


class Loss(object):
    def __init__(self):
        pass

    def __call__(self):
        pass
    
    def grads(self):
        pass


class SquaredLoss(object):
    def __init__(self):
        pass

    def __call__(self, pred, real):
        return 0.5 * np.power(real - pred, 2)
    
    def grads(self, pred, real):
        return -1 * (real - pred) * (1.0/len(pred))


class CrossEntropy(object):
    def __init__(self):
        pass
    
    def _avoid_div_by_zero(self, p):
        return np.clip(p, 1e-15, 1 - 1e-15)

    def __call__(self, pred, real):
        pred = self._avoid_div_by_zero(pred)
        return -real * np.log(pred) - (1 - real) * np.log(1 - pred)

    def grads(self, pred, real):
        pred = self._avoid_div_by_zero(pred)
        return (-(real / pred) + (1 - real) / (1 - pred)) * (1.0/len(pred))


losses = {
    'squared-loss': SquaredLoss,
    'mean-square': SquaredLoss,
    'cross-entropy': CrossEntropy
}


def load(loss, **kwargs):
    if loss not in losses:
        raise ValueError('Unknown loss: %s' % loss)
    else:
        return losses[loss](**kwargs)