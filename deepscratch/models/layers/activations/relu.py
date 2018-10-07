import numpy as np
from deepscratch.models.layers.activations.activation import Activation


class Relu(Activation):
    # https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/activation_functions.py
    def __call__(self, data):
        return np.where(data >= 0, data, 0)
    
    def grads(self, data):
        return np.where(data >= 0, 1, 0)


class LeakyRelu():
    # https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/activation_functions.py
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, data):
        return np.where(data >= 0, data, self.alpha * data)

    def grads(self, data):
        return np.where(data >= 0, 1, self.alpha)


class Elu():
    # https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/activation_functions.py
    def __init__(self, alpha=0.2):
        self.alpha = alpha 

    def __call__(self, data):
        return np.where(data >= 0.0, data, self.alpha * (np.exp(data) - 1))

    def grads(self, data):
        return np.where(data >= 0.0, 1, self.__call__(data) + self.alpha)


class Selu():
    # https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/activation_functions.py
    # https://arxiv.org/abs/1706.02515,
    # https://github.com/bioinf-jku/SNNs/blob/master/SelfNormalizingNetworks_MLP_MNIST.ipynb
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946 

    def __call__(self, data):
        return self.scale * np.where(data >= 0.0, data, self.alpha*(np.exp(data)-1))

    def grads(self, data):
        return self.scale * np.where(data >= 0.0, 1, self.alpha * np.exp(data))
