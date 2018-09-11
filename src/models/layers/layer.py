class Layer(object):
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
            return np.maximum(0, x)
        
        def _dev_relu(x):
            return (x > 0) * 1.0
        
        def _tanh(x):
            return np.tanh(x)
        
        def _dev_tanh(x):
            return 1 - x ** 2
        
        activations = []
        act_devs = []
        for act_name in act_names:
            if act_name is None:
                act, dev_act = _no_act, _dev_no_act
            elif act_name == 'sigmoid':
                act, dev_act = _sigmoid, _dev_sigmoid
            elif act_name == 'relu':
                act, dev_act = _relu, _dev_relu
            elif act_name == 'tanh':
                act, dev_act = _tanh, _dev_tanh
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
            assert grad['biases'].shape == self.biases[i].shape
            self.biases[i] -= learning_rate * grad['biases']

    
    def run_batch(self, batch):
        self._current_batch = [batch]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            output = np.dot(self._current_batch[-1], w) + b
            output = self.activations[i](output)
            self._current_batch.append(output)
        
        self._current_batch = self._current_batch[::-1]
        return output