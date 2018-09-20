from deepscratch.dataloader.mnist import MNIST
from deepscratch.dataloader.xor import XOR


datasets = {
    'mnist': MNIST(),
    'xor': XOR()
}


def load(dataset, split=True):
    if dataset not in datasets:
        raise ValueError('Dataset unknown %s' % dataset)
    else:
        return datasets[dataset].load(split=split)