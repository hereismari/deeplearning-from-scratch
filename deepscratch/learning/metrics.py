import numpy as np

class Metric(object):
    def __init__(self):
        pass
    
    def __call__(self):
        pass


class Accuracy(Metric):
    def __call__(self, pred=None, real=None, **kwargs):
        return np.sum(np.argmax(pred, axis=1) == np.argmax(real, axis=1)) / (len(pred) * 1.0)



metrics = {
    'accuracy': Accuracy
}


def load(metric, **kwargs):
    if metric not in metrics:
        raise ValueError('Unknown metric: %s' % metric)
    else:
        return metrics[metric](**kwargs)