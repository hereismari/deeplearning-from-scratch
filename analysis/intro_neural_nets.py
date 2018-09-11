
# coding: utf-8

# #### Dependências

# In[9]:


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

# Utilidades
import utils

# Recarregar automaticamente dependências caso elas mudem
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Introdução a redes neurais
# 
# Esse conteúdo é bastante inspirado no [Capítulo 3](https://github.com/iamtrask/Grokking-Deep-Learning) de Grokking Deep Learning.

# Redes neurais podem ser vistas de maneira bastante simplificadas como **funções matemáticas**. Desse modo, similar à funções as redes neurais recebem uma ou mais entradas e a mapeiam para uma ou mais saídas. Aqui, mapear = realizar computação.

# Falando em Deep Learning, essas entradas são normalmente **tensores**, isto é, vetores de diferentes dimensões.

# ![](https://www.kdnuggets.com/wp-content/uploads/scalar-vector-matrix-tensor.jpg)
# Imagem de: https://www.kdnuggets.com/2018/05/wtf-tensor.html

# Tensor, [**em Deep Learning**](https://www.youtube.com/watch?v=f5liqUk0ZTw), é um nome bonito para vetores multi-dimensionais, onde um tensor de uma dimensão é um escalar, com duas dimensões vetor, 3 dimensões é uma matriz, e N dimensões onde N > 3 não há nomencaltura específica.

# In[10]:


class MyFirstNeuralNetwork(object):

    def __init__(self, weights=0.5):
        self._weights = weights
    
    def function(self, _input):
        return self._activation_function(_input * self._weights)
    
    def _activation_function(self, data):
        return data


# In[11]:


nn = MyFirstNeuralNetwork()

