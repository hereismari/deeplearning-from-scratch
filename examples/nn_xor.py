from deepscratch.models.neural_network import NeuralNetwork
from deepscratch.learning.trainer import Trainer
import deepscratch.models.layers as layers
import deepscratch.dataloader as dataloader


def main():
    # Getting data
    train_x, train_y = dataloader.load('xor', split=False)
    
    # Network implementation
    nn = NeuralNetwork(layers=[layers.Dense(1, input_shape=(2,))])
    
    # Training
    trainer = Trainer(nn, loss='mean-square', print_step_mod=1)
    trainer.train(train_x, train_y, epochs=300, batch_size=-1)
    
    print (trainer.predict(train_x))


if __name__ == "__main__":
    main()