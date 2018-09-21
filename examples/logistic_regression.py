import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


from deepscratch.models.neural_network import NeuralNetwork
from deepscratch.learning.trainer import Trainer
import deepscratch.models.layers as layers
import deepscratch.dataloader as dataloader


def main():
    # Generating fake data: Class1 = (x, 4 * x), Class2 = (x, 10 * x + 1) 
    SYNT_TRAIN_SIZE = 200

    # Generating classes
    x = np.random.rand(SYNT_TRAIN_SIZE)
    class_1_x = np.column_stack((x, x * 4 + np.random.normal(0, 4, size=SYNT_TRAIN_SIZE)))
    x = np.random.rand(SYNT_TRAIN_SIZE) + 2
    class_2_x = np.column_stack((x, x * 10 + 1 + np.random.normal(0, 4, size=SYNT_TRAIN_SIZE)))

    class_1_y = np.zeros(SYNT_TRAIN_SIZE)
    class_2_y = np.ones(SYNT_TRAIN_SIZE)

    # Generating train data
    train_x = np.row_stack((class_1_x, class_2_x))
    train_y = np.row_stack((class_1_y, class_2_y)).reshape(-1)
    one_hot = OneHotEncoder(sparse=False)
    train_y_one_hot = one_hot.fit_transform(train_y.reshape(len(train_y), 1))

    plt.scatter(train_x[:,0], train_x[:,1], alpha=0.5, c=train_y)
    plt.show()

    # Network implementation
    nn = NeuralNetwork(layers=[layers.Dense(2, input_shape=(2,)), layers.Activation('softmax')], optimizer='sgd')
    
    # Training
    trainer = Trainer(nn, loss='cross-entropy', print_step_mod=100, metrics=['accuracy'])
    trainer.train(train_x, train_y_one_hot, epochs=300, batch_size=20)
    
    print('Learned parameters:')
    print('weights:', nn.layers[0].W)
    print('biases:', nn.layers[0].b)
    
    plt.scatter(train_x[:,0], train_x[:,1], alpha=0.5, c=train_y)
    plt.scatter(train_x[:,0], train_x[:,1], alpha=0.5, c=nn.forward(train_x))
    plt.show()


if __name__ == "__main__":
    main()