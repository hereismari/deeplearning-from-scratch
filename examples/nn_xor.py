from deepscratch.models.neural_network import NeuralNetwork
from deepscratch.learning.trainer import Trainer
import deepscratch.models.layers as layers
import deepscratch.dataloader as dataloader


def main():
    # Getting data
    train_x, train_y = dataloader.load('xor', split=False)
    
    # Network implementation
    nn = NeuralNetwork(
        layers=[
            layers.Dense(10, input_shape=(2,)),
            layers.Activation('relu'),
            layers.Dense(1)
        ],
        optimizer='rmsprop'
    )
    
    # Training
    trainer = Trainer(nn, loss='mean-square', print_step_mod=1000)
    trainer.train(train_x, train_y, epochs=5000, batch_size=-1)
    
    print (trainer.predict(train_x))


if __name__ == "__main__":
    main()