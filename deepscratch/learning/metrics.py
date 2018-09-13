import numpy as np


class Metric(object):
    def __init__(self):
        pass
    
    def __call__(self):
        pass


class Accuracy(Metric):
    def __call__(self, pred=None, real=None, **kwargs):
        return np.sum(np.argmax(pred, axis=1) == real) / len(pred)