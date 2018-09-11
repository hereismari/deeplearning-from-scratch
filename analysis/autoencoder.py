
# coding: utf-8

# #### Dependências

# In[7]:


# Gráficos
import matplotlib.pyplot as plt

# Matemática + manipualação de vetores
import math
import numpy as np

# "Fixar" números aleatórios a serem gerados
np.random.seed(0)

# Trabalhar com os dados
import pandas as pd
from sklearn.datasets.mldata import fetch_mldata
from sklearn.model_selection import train_test_split

# Utilidades
import utils

# Recarregar automaticamente dependências caso elas mudem
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Implementações já feitas
import nbimporter
import neural_net as nn

np.seterr(all='raise')


# # AutoEncoder

# In[33]:


class NeuralNetwork(object):
    def __init__(self, layers=[1], input_size=1, activations=[None]):
        assert len(layers) == len(activations)
        self.input_size = input_size
        self.layers = layers
        self.activations, self._act_devs = self.get_act(activations)
        
        self.weights, self.biases = self.define_params()
        self._current_batch = []

        
    def get_act(self, act_names):
        def _no_act(x):
            return x
        def _dev_no_act(x):
            return np.ones(x.shape)

        def _sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def _dev_sigmoid(x):
            return x * (1 - x)
        
        def _relu(x):
            return np.maximum(1e-15, x)
        
        def _dev_relu(x):
            try:
                return (x > 0) * 1.0
            except:
                print('relu exception')
                print(x)
        activations = []
        act_devs = []
        for act_name in act_names:
            if act_name is None:
                act, dev_act = _no_act, _dev_no_act
            elif act_name == 'sigmoid':
                act, dev_act = _sigmoid, _dev_sigmoid
            elif act_name == 'relu':
                act, dev_act = _relu, _dev_relu
            else:
                raise ValueError('Activation function is not valid: %s' % act_name)
            
            activations.append(act)
            act_devs.append(dev_act)
        return activations, act_devs
    

    def define_params(self):
        '''He-et-all initialization'''
        weights = []
        biases = []
        for i, (in_dim, out_dim) in enumerate(zip([self.input_size] + self.layers, self.layers)):
            weights.append(np.random.randn(in_dim, out_dim) * np.sqrt(2/in_dim))
            biases.append(np.random.randn(out_dim) * np.sqrt(2/in_dim))
            
            print('Weight %d shape =' % i, weights[i].shape)
            print('Bias %d shape =' % i, biases[i].shape)
            

        return weights, biases


    def update_params(self, gradients, learning_rate=0.1):
        assert len(gradients) == len(self.weights), (len(gradients), len(self.weights))
        assert len(gradients) == len(self.biases), (len(gradients), len(self.biases))
        
        for i, grad in enumerate(gradients[::-1]):
            assert grad['weights'].shape == self.weights[i].shape
            self.weights[i] -= learning_rate * grad['weights']
            # assert grad['biases'].shape == self.biases[i].shape
            self.biases[i] -= learning_rate * grad['biases']

    
    def run_batch(self, batch):
        self._current_batch = [batch]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            output = np.dot(self._current_batch[-1], w) + b
            output = self.activations[i](output)
            self._current_batch.append(output)
        
        self._current_batch = self._current_batch[::-1]
        return output


# In[38]:


class Trainer(object):
    def __init__(self, model, learning_rate = 0.01, loss_name='l2',
                 print_mod=1000, verbose=True):
        
        def _accuracy(pred_y, real_y):
            p = np.argmax(self.softmax(pred_y), axis=1)
            return np.sum(p == real_y) / len(pred_y)
            
        
        self.model = model
        self.loss_name = loss_name
        self.learning_rate = learning_rate
        self.loss, self.loss_dev = self._define_loss()
        
        self.train_step = 0
        self.eval_steps = []
        
        self.verbose = verbose
        self.print_mod = print_mod
        
        self.train_losses = []
        self.eval_losses = []
        
        self._metrics = {
            'accuracy': _accuracy
        }
    
    def softmax(self, x):
        exps = np.exp(x)
        return (exps / np.sum(exps, axis=1, keepdims=True))
        
    def _define_loss(self):
        def _l2(pred_y, real_y):
            n = len(pred_y)
            return (1.0/2) * (1.0/n) * np.sum(np.power(pred_y - real_y, 2))
        
        def _l2_dev(pred_y, real_y):
            n = len(pred_y)
            return (pred_y - real_y) * (1.0/n)
        
        def _cross_entropy(pred_y, real_y):
            m = real_y.shape[0]
            p = self.softmax(pred_y)
            # We use multidimensional array indexing to extract 
            # softmax probability of the correct label for each sample.
            # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
            log_likelihood = -np.log(p[range(m), real_y.astype(int)])
            loss = np.sum(log_likelihood) / m
            return loss
    
        
        def _cross_entropy_dev(pred_y, real_y):
            m = real_y.shape[0]
            grad = self.softmax(pred_y)
            grad[range(m), real_y.astype(int)] -= 1
            grad = grad / m
            return grad

        if self.loss_name == 'l2':
            return _l2, _l2_dev
        elif self.loss_name == 'cross-entropy':
            return _cross_entropy, _cross_entropy_dev
        else:
            raise ValueError('Invalid loss name: %s' % self.loss_name)


    def train(self, batch_x, batch_y):
        self.train_step += 1
        # run feed forward network
        pred_y = self.model.run_batch(batch_x)
        # save loss
        self.train_losses.append(self.loss(pred_y, batch_y))
        # get gradients
        grads = self.generate_gradients(pred_y, batch_y, batch_x)
        # update parameters
        self.model.update_params(grads, self.learning_rate)

        if self.verbose and (self.train_step - 1) % self.print_mod == 0:
            print('Loss: %.4f for step %d' % (self.train_losses[-1], self.train_step))


    def eval(self, batch_x, batch_y, metrics=[]):
        # run feed forward network
        pred_y = self.model.run_batch(batch_x)
        # loss
        loss = self.loss(pred_y, batch_y)
        self.eval_losses.append(loss)
        # metrics
        res_metrics = []
        for m in metrics:
            if m in self._metrics:
                res_metrics.append(self._metrics[m](pred_y, batch_y))
            else:
                raise ValueError('Invalid metric: %s' % m)
        
        self.eval_steps.append(self.train_step)
            
        return loss, res_metrics

    
    def plot_losses(self):
        if len(self.eval_losses) > 0:
            plt.title('Train Loss: %.4f | Test Loss: %.4f for step %d' % (self.train_losses[-1], self.eval_losses[-1], self.train_step))
        else:
            plt.title('Train Loss: %.4f for step %d' % (self.train_losses[-1], self.train_step))    
        plt.plot([i for i in range(self.train_step)], self.train_losses)
        plt.plot([i for i in self.eval_steps], self.eval_losses)
        
        
    def generate_gradients(self, pred_y, real_y, data_x):
        grad = []
        input_size = pred_y.shape[0]
        j = len(self.model.activations) - 1
        k = len(self.model.weights) - 1
        dly = self.loss_dev(pred_y, real_y) * self.model._act_devs[j](self.model._current_batch[0])
        dlx = np.dot(dly, self.model.weights[k].T)

        for i, (w, b) in enumerate(zip(self.model.weights[::-1], self.model.biases[::-1])):
            dlw = np.dot(self.model._current_batch[i+1].T, dly)
            dlb = np.sum(dly)
            # print('weight:', w.shape, 'bias:', b.shape)
            # print('dlw:', dlw.shape, 'dlb:', dlb.shape)
            # print('dly:', dly.shape, 'dlx:', dlx.shape)
            grad.append({
                'weights': dlw,
                'biases': dlb
            })
            
            j -= 1
            k -= 1
            if i != len(self.model.weights)-1:
                dly = dlx * self.model._act_devs[j](self.model._current_batch[i+1])
                dlx = np.dot(dly, self.model.weights[k].T)
        return grad


# ### mnist

# In[39]:


# load mnist
mnist = fetch_mldata('MNIST original')

x = mnist.data / np.max(mnist.data)
y = mnist.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[ ]:


autoencoder = NeuralNetwork(layers=[10, 2, 10, 784], input_size=784, activations=['relu', 'relu', 'relu', 'sigmoid'])
t = Trainer(autoencoder, verbose=False, learning_rate=0.001)


# In[41]:


for i in range(40):
    for batch_x, batch_y in nn.batches(x, y, 64):
        t.train(batch_x, batch_x)
    if i % 1 == 0:
        loss, metrics = t.eval(x_test, x_test, metrics=[])
        print('Test loss = %.5f' % loss)

t.plot_losses()


# ### Implementing SGD trainer

# In[207]:


class Trainer(object):
    def __init__(self, model, learning_rate = 0.01, loss_name='l2',
                 print_mod=1000, verbose=True):
        
        def _accuracy(pred_y, real_y):
            p = np.argmax(self.softmax(pred_y), axis=1)
            return np.sum(p == real_y) / len(pred_y)
            
        
        self.model = model
        self.loss_name = loss_name
        self.learning_rate = learning_rate
        self.loss, self.loss_dev = self._define_loss()
        
        self.train_step = 0
        self.eval_steps = []
        
        self.verbose = verbose
        self.print_mod = print_mod
        
        self.train_losses = []
        self.eval_losses = []
        
        self._metrics = {
            'accuracy': _accuracy
        }
    
    def softmax(self, x):
        exps = np.exp(x)
        return (exps / np.sum(exps, axis=1, keepdims=True))
        
    def _define_loss(self):
        def _l2(pred_y, real_y):
            n = len(pred_y)
            return (1.0/2) * (1.0/n) * np.sum(np.power(pred_y - real_y, 2))
        
        def _l2_dev(pred_y, real_y):
            n = len(pred_y)
            return (pred_y - real_y) * (1.0/n)
        
        def _cross_entropy(pred_y, real_y):
            m = real_y.shape[0]
            p = self.softmax(pred_y)
            # We use multidimensional array indexing to extract 
            # softmax probability of the correct label for each sample.
            # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
            log_likelihood = -np.log(p[range(m), real_y.astype(int)])
            loss = np.sum(log_likelihood) / m
            return loss
    
        
        def _cross_entropy_dev(pred_y, real_y):
            m = real_y.shape[0]
            grad = self.softmax(pred_y)
            grad[range(m), real_y.astype(int)] -= 1
            grad = grad / m
            return grad

        if self.loss_name == 'l2':
            return _l2, _l2_dev
        elif self.loss_name == 'cross-entropy':
            return _cross_entropy, _cross_entropy_dev
        else:
            raise ValueError('Invalid loss name: %s' % self.loss_name)


    def train(self, batch_x, batch_y):
        self.train_step += 1
        
        # run feed forward network
        pred_y = self.model.run_batch(batch_x)
        # save loss
        self.train_losses.append(self.loss(pred_y, batch_y))
        # get gradients
        grads = self.generate_gradients(pred_y, batch_y, batch_x)
        # update parameters
        self.model.update_params(grads, self.learning_rate)

        if self.verbose and (self.train_step - 1) % self.print_mod == 0:
            print('Loss: %.4f for step %d' % (self.train_losses[-1], self.train_step))


    def eval(self, batch_x, batch_y, metrics=[]):
        # run feed forward network
        pred_y = self.model.run_batch(batch_x)
        # loss
        loss = self.loss(pred_y, batch_y)
        self.eval_losses.append(loss)
        # metrics
        res_metrics = []
        for m in metrics:
            if m in self._metrics:
                res_metrics.append(self._metrics[m](pred_y, batch_y))
            else:
                raise ValueError('Invalid metric: %s' % m)
        
        self.eval_steps.append(self.train_step)
            
        return loss, res_metrics

    
    def plot_losses(self):
        if len(self.eval_losses) > 0:
            plt.title('Train Loss: %.4f | Test Loss: %.4f for step %d' % (self.train_losses[-1], self.eval_losses[-1], self.train_step))
        else:
            plt.title('Train Loss: %.4f for step %d' % (self.train_losses[-1], self.train_step))    
        plt.plot([i for i in range(self.train_step)], self.train_losses)
        plt.plot([i for i in self.eval_steps], self.eval_losses)
        
        
    def generate_gradients(self, pred_y, real_y, data_x):
        grad = []
        input_size = pred_y.shape[0]
        j = len(self.model.activations) - 1
        k = len(self.model.weights) - 1
        dly = self.loss_dev(pred_y, real_y) * self.model._act_devs[j](self.model._current_batch[0])
        dlx = np.dot(dly, self.model.weights[k].T)

        for i, (w, b) in enumerate(zip(self.model.weights[::-1], self.model.biases[::-1])):
            dlw = np.dot(self.model._current_batch[i+1].T, dly)
            dlb = np.sum(dly)
            # print('weight:', w.shape, 'bias:', b.shape)
            # print('dlw:', dlw.shape, 'dlb:', dlb.shape)
            # print('dly:', dly.shape, 'dlx:', dlx.shape)
            grad.append({
                'weights': dlw,
                'biases': dlb
            })
            
            j -= 1
            k -= 1
            if i != len(self.model.weights)-1:
                dly = dlx * self.model._act_devs[j](self.model._current_batch[i+1])
                dlx = np.dot(dly, self.model.weights[k].T)
        return grad


# ### Gradients
# 
# ##### L2 loss with 1 layer, no activation
# 
# **Loss**
# 
# $$L = 1/2 * 1/n * \sum{(y_i - ŷ_i)^{2}}$$
# $$L = 1/2 * 1/n * \sum{(y_i - w_i * x_i + b_i)^{2}}$$
# 
# **Gradients**
# 
# $$\frac{\partial L}{\partial w_i} = 1/2 * 1/n * 2 * \sum{(y_i - ŷ_i)} * \frac{\partial {ŷ_i}}{\partial w_i} $$
# $$\frac{\partial L}{\partial w_i} = 1/n * \sum{(y_i - ŷ_i)} * x_i$$
# 
# ---
# 
# $$\frac{\partial L}{\partial b_i} = 1/2 * 1/n * 2 * \sum{(y_i - ŷ_i)} * \frac{\partial {ŷ_i}}{\partial b_i} $$
# $$\frac{\partial L}{\partial b_i} = 1/n * \sum{(y_i - ŷ_i)} * 1$$
# 
# 
# ##### L2 loss with 2 layers, relu activation in the hidden layer
# 
# **Loss**
# 
# $$L = 1/2 * 1/n * \sum{(y_i - ŷ_i)^{2}}$$
# $$L = 1/2 * 1/n * \sum{(y_i - (w_j * x_j + b_j))^{2}}$$
# $$x_j = relu(w_i * x_i + b_i)$$
# 
# 
# **Gradients**
# 
# $$\frac{\partial L}{\partial w_i} = 1/n * \sum{(y_i - ŷ_i)} * x_j $$
# $$\frac{\partial L}{\partial b_i} = 1/n * \sum{(y_i - ŷ_i)} * 1$$
# 
# $$\frac{\partial L}{\partial w_j} = 1/n * \sum{(y_i - ŷ_i)} * x_j * x_i, se relu() > 0$$
# $$\frac{\partial L}{\partial b_j} = 1/n * \sum{(y_i - ŷ_i)} * x_j, se relu() > 0$$

# In[16]:


nn = NeuralNetwork()


# In[17]:


t = Trainer(nn, verbose=False)
for i in range(100000):
    t.train(synt_x, synt_y)

t.plot_losses()


# In[18]:


print('Learned parameters:')
print('weights:', nn.weights)
print('biases:', nn.biases)
print('Input function: 7 * X + 15')
plt.plot(synt_x, nn.run_batch(synt_x), 'bo', synt_x, synt_y, 'ro', alpha=0.5)


# ### What if the data is non linear?

# #### XOR example

# In[27]:


xor_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_y = np.array([[0], [1], [1], [0]])


# In[154]:


nn = NeuralNetwork(layers=[10, 2], input_size=2, activations=['relu', None])
t = Trainer(nn, verbose=False)
for i in range(100000):
    t.train(xor_x, xor_y)

t.plot_losses()


# In[155]:


plt.plot(xor_x, nn.run_batch(xor_x), 'bo', xor_x, xor_y, 'ro', alpha=0.5)


# In[156]:


nn = NeuralNetwork(layers=[10, 1], input_size=2, activations=[None, None])
t = Trainer(nn, verbose=False)
for i in range(100000):
    t.train(xor_x, xor_y)

t.plot_losses()


# In[157]:


plt.plot(xor_x, nn.run_batch(xor_x), 'bo', xor_x, xor_y, 'ro', alpha=0.5)


# #### Synthetic data

# In[158]:


def get_random_error(size, mu=0, std_dev=0.8):
    return np.random.normal(mu, std_dev, size)

# generate the function: 7 * X + 15, it can be a more complicated function, but let's keep it simple for now
synt_x = np.random.rand(SYNT_TRAIN_SIZE)
synt_y = np.reshape(7 * np.log(synt_x) + 1 + get_random_error(SYNT_TRAIN_SIZE), (SYNT_TRAIN_SIZE, 1))

synt_x = np.reshape(synt_x, (SYNT_TRAIN_SIZE, 1))


# In[159]:


plt.plot(synt_x, synt_y, 'ro', alpha=0.5)


# In[160]:


nn = NeuralNetwork(layers=[10, 1], activations=['sigmoid', None])
t = Trainer(nn)


# In[161]:


for i in range(10000):
    t.train(synt_x, synt_y)

t.plot_losses()


# In[162]:


print('Learned parameters:')
print('weights:', nn.weights)
print('biases:', nn.biases)
print('Input function: 7 * log(X) + 15')
plt.plot(synt_x, nn.run_batch(synt_x), 'bo', synt_x, synt_y, 'ro', alpha=0.5)


# ### Iris classification

# In[163]:


from sklearn.model_selection import train_test_split
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[209]:


def batches(x, y, batch_size=True):
    idx = np.random.permutation(len(x))
    x = x[idx]
    y = y[idx]
    
    for i in range(0, len(x)-batch_size-1, batch_size):
        batch_x = x[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        yield batch_x, batch_y


# In[165]:


nn = NeuralNetwork(layers=[10, 4], input_size=4, activations=[None, None])
t = Trainer(nn, verbose=False, loss_name='cross-entropy')
for i in range(100):
    for batch_x, batch_y in batches(x, y, 16):
        t.train(batch_x, batch_y)
    if i % 10 == 0:
        loss, metrics = t.eval(x_test, y_test, metrics=['accuracy'])
        print('Test loss = %.5f, accuracy %.5f' % (loss, metrics[0]))

t.plot_losses()


# ### MNIIST classification

# In[197]:


# load mnist
mnist = fetch_mldata('MNIST original')


# In[198]:


x = mnist.data / np.max(mnist.data)
y = mnist.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[199]:


y


# In[212]:


nn = NeuralNetwork(layers=[512, 256, 10], input_size=784, activations=['relu', 'relu', None])
t = Trainer(nn, verbose=False, loss_name='cross-entropy', learning_rate=0.001)
for i in range(20):
    for batch_x, batch_y in batches(x, y, 64):
        t.train(batch_x, batch_y)
    if i % 1 == 0:
        loss, metrics = t.eval(x_test, y_test, metrics=['accuracy'])
        print('Test loss = %.5f, accuracy %.5f' % (loss, metrics[0]))

t.plot_losses()

