from deepscratch.models.neural_network import NeuralNetwork
from deepscratch.learning.trainer import Trainer
import deepscratch.models.layers as layers
import deepscratch.dataloader as dataloader


def main():
    # data
    train_x, test_x, train_y, test_y = dataloader.load('mnist')
    # net
    nn = NeuralNetwork(layers=[layers.Dense(10, input_shape=784)])
    trainer = Trainer(nn)

if __name__ == "__main__":
    main()