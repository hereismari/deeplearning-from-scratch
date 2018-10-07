import numpy as np

from deepscratch.dataloader.dataloader import DataLoader
from sklearn.datasets.mldata import fetch_mldata
from sklearn.preprocessing import OneHotEncoder


class MNIST(DataLoader):
    def __init__(self):
        self.data = fetch_mldata('MNIST original')
        self._one_hot = OneHotEncoder(sparse=False)
        self._preprocess_data()
    
    def _preprocess_data(self):
        self.x = (self.data.data * 1.0 / np.max(self.data.data))
        self.orig_y = self.data.target
        self.y = self._one_hot.fit_transform(self.orig_y.reshape(len(self.orig_y), 1))