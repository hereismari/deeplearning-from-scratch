import numpy as np

from deepscratch.dataloader.dataloader import DataLoader
from sklearn.datasets.mldata import fetch_mldata
from sklearn.preprocessing import OneHotEncoder


class XOR(DataLoader):
    def __init__(self):
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])