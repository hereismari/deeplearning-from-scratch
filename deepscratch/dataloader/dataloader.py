import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader(object):
    def __init__(self):
        self.data = np.random.rand(10)
        self.x = np.random.rand(10)
        self.y = np.random.rand(10)
    
    def load(self):
        return train_test_split(self.x, self.y)