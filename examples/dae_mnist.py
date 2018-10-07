from deepscratch.models.neural_network import NeuralNetwork
from deepscratch.learning.trainer import Trainer
import deepscratch.models.layers as layers
import deepscratch.dataloader as dataloader

import matplotlib.pyplot as plt
import numpy as np

def main():
    # data
    train_x, test_x, train_y, test_y = dataloader.load('mnist')

    # net
    nn = NeuralNetwork(
        layers=[
            layers.Dense(500, input_shape=(784, )),
            layers.Activation('relu'),
            layers.Dense(200),
            layers.Activation('relu'),
            layers.Dense(500),
            layers.Activation('relu'),
            layers.Dense(784)
        ], optimizer='rmsprop', initializer='he-et-al')
    
    trainer = Trainer(nn, loss='mean-square')

    epochs = 10
    for i in range(epochs):
        trainer.train(train_x, train_x, epochs=1)
        plt.imshow(test_x[0].reshape(28, 28))
        plt.show()
        plt.imshow(trainer.predict(np.array([test_x[0]])).reshape(28, 28))
        plt.show()

if __name__ == "__main__":
    main()