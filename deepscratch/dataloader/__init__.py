from deepscratch.dataloader.mnist import MNIST

datasets = {
    'mnist': MNIST()
}

def load(dataset):
    if dataset not in datasets:
        raise ValueError('Dataset unknown %s' % dataset)
    else:
        return datasets[dataset].load()