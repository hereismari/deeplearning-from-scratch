import numpy as np

from deepscratch.dataloader.dataloader import DataLoader
from sklearn.datasets.mldata import fetch_mldata


class MNIST(DataLoader):
    def __init__(self):
        self.data = fetch_mldata('MNIST original')
        self.x = self.data.data
        self.y = self.data.target