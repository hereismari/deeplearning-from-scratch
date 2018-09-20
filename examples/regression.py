import numpy as np
import matplotlib.pyplot as plt


from deepscratch.models.neural_network import NeuralNetwork
from deepscratch.learning.trainer import Trainer
import deepscratch.models.layers as layers
import deepscratch.dataloader as dataloader


def main():
    # Generating fake data: 7 * X + 15
    SYNT_TRAIN_SIZE = 200
    train_x = np.random.rand(SYNT_TRAIN_SIZE)
    np.random.normal()
    train_y = np.reshape(7 * train_x + 15 + np.random.normal(0, 0.8, size=SYNT_TRAIN_SIZE), (SYNT_TRAIN_SIZE, 1))
    train_x = np.reshape(train_x, (SYNT_TRAIN_SIZE, 1))
    
    plt.plot(train_x, train_y, 'ro', alpha=0.5)
    plt.show()

    # Network implementation
    nn = NeuralNetwork(layers=[layers.Dense(1, input_shape=(1,))], optimizer='sgd')
    
    # Training
    trainer = Trainer(nn, loss='mean-square', print_step_mod=1)
    trainer.train(train_x, train_y, epochs=30000, batch_size=-1)
    
    print('Learned parameters:')
    print('weights:', nn.layers[0].W)
    print('biases:', nn.layers[0].b)
    print('Input function: 7 * X + 15')
    plt.plot(train_x, nn.forward(train_x), 'bo', train_x, train_y, 'ro', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()