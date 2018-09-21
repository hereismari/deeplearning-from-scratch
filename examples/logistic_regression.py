import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs


from deepscratch.models.neural_network import NeuralNetwork
from deepscratch.learning.trainer import Trainer
import deepscratch.models.layers as layers
import deepscratch.dataloader as dataloader


def main():
    # Generating fake data: Class1 around (0, 0) and Class2 around (10, 10)
    train_x, train_y = make_blobs(n_samples=200, n_features=2, cluster_std=1.0,
                                  centers=[(0, 0), (10, 10)], shuffle=False, random_state=42)
    one_hot = OneHotEncoder(sparse=False)
    print(train_x)
    train_y_one_hot = one_hot.fit_transform(train_y.reshape(len(train_y), 1))

    plt.scatter(train_x[:,0], train_x[:,1], alpha=0.5, c=train_y)
    plt.show()

    # Network implementation
    nn = NeuralNetwork(
        layers=[
            layers.Dense(2, input_shape=(2,)),
            layers.Activation('softmax')
        ],
        optimizer='sgd')
    
    # Training
    trainer = Trainer(nn, loss='cross-entropy', print_step_mod=100, metrics=['accuracy'])
    trainer.train(train_x, train_y_one_hot, epochs=45, batch_size=20)
    
    print('Learned parameters:')
    print('weights:', nn.layers[0].W)
    print('biases:', nn.layers[0].b)
    
    eval_x = np.random.uniform(low=-3, high=13, size=(200, 2))

    plt.scatter(train_x[:,0], train_x[:,1], alpha=0.5)
    plt.scatter(eval_x[:,0], eval_x[:,1], alpha=0.5, c=(np.argmax(nn.forward(eval_x), axis=1) + 2))
    plt.show()


if __name__ == "__main__":
    main()