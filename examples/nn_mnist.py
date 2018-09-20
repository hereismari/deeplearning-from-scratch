from deepscratch.models.neural_network import NeuralNetwork
from deepscratch.learning.trainer import Trainer
import deepscratch.models.layers as layers
import deepscratch.dataloader as dataloader


def main():
    # data
    train_x, test_x, train_y, test_y = dataloader.load('mnist')

    # net
    nn = NeuralNetwork(
        layers=[
            layers.Dense(512, input_shape=(784, )),
            layers.Activation('relu'),
            layers.Dense(10),
            layers.Activation('softmax')
        ])
    
    trainer = Trainer(nn, loss='cross-entropy', metrics=['accuracy'])

    trainer.train(train_x, train_y, epochs=10, test_size=0.25)
    print(trainer.eval(test_x, test_y))


if __name__ == "__main__":
    main()