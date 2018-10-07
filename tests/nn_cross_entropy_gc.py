import numpy as np

from deepscratch.models.neural_network import NeuralNetwork
from deepscratch.learning.trainer import Trainer
import deepscratch.models.layers as layers
import deepscratch.dataloader as dataloader

from deepscratch.learning.checker import GradientChecker


def main():
    # data
    train_x, test_x, train_y, test_y = dataloader.load('mnist')
    # train_x, train_y = dataloader.load('xor', split=False)

    # use a random batch
    BATCH_SIZE = 32
    used_batch = np.random.permutation(len(train_x))[:BATCH_SIZE]
    batch_x = train_x[used_batch]
    batch_y = train_y[used_batch]

    nn = NeuralNetwork(
        layers=[
            layers.Dense(512, input_shape=(784, )),
            layers.Activation('tanh'),
            layers.Dense(256),
            layers.Activation('tanh'),
            layers.Dense(10),
            layers.Activation('softmax')
        ], optimizer='rmsprop', initializer='he-et-al')
    
    trainer = Trainer(nn, loss='cross-entropy', metrics=['accuracy'])

    # Training    
    gc = GradientChecker(nn, trainer)
    
    # Check for N steps
    N = 10000
    for _ in range(N):
        gc.check(batch_x, batch_y)
        trainer.batch_train(batch_x, batch_y)

    print (trainer.predict(batch_x))

if __name__ == "__main__":
    main()