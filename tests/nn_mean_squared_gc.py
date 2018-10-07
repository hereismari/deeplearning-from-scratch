import numpy as np

from deepscratch.models.neural_network import NeuralNetwork
from deepscratch.learning.trainer import Trainer
import deepscratch.models.layers as layers
import deepscratch.dataloader as dataloader

from deepscratch.learning.checker import GradientChecker


def main():
    # data
    train_x, train_y = dataloader.load('xor', split=False)

    # use a random batch
    BATCH_SIZE = 4
    used_batch = np.random.permutation(len(train_x))[:BATCH_SIZE]
    batch_x = train_x[used_batch]
    batch_y = train_y[used_batch]

    # Network implementation
    nn = NeuralNetwork(
        layers=[
            layers.Dense(300, input_shape=(2,)),
            layers.Activation('tanh'),
            layers.Dense(200),
            layers.Activation('leaky-relu'),
            layers.Dense(1)
        ],
        optimizer='rmsprop'
    )

    # Training
    trainer = Trainer(nn, loss='mean-square', print_step_mod=1000)
    gc = GradientChecker(nn, trainer)

    # Check for N steps
    N = 10000
    for _ in range(N):
        gc.check(batch_x, batch_y)
        trainer.batch_train(batch_x, batch_y)


if __name__ == "__main__":
    main()