from deepscratch.models.optimizers.sgd import SGD
from deepscratch.models.optimizers.rmsprop_momentum import RMSProp

optimizers = {
    'sgd': SGD,
    'rmsprop': RMSProp
}


def load(optimizer, **kwargs):
    if optimizer not in optimizers:
        raise ValueError('Optimizer unknown %s' % optimizer)
    else:
        return optimizers[optimizer](**kwargs)