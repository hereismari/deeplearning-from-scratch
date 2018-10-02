import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from random import uniform

import deepscratch.learning.losses as losses
import deepscratch.learning.metrics as ds_metrics
import deepscratch.models.optimizers as optimizers


class GradientChecker(object):
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
    
    def backpropagate(self, batch_x, batch_y):
        # run feed forward network
        pred_y = self.trainer.predict(batch_x)
        # run backpropagation
        self.model.backward(self.trainer.loss.grads(pred_y, batch_y), update_weights=False)


    def loss(self, batch_x, batch_y):
        # run feed forward network
        pred_y = self.trainer.predict(batch_x)
        return self.trainer.loss(pred_y, batch_y)


    def check(self, batch_x, batch_y, layers=['Dense'], delta=1e-5, error_limit=1e-6):
        # store gradients for each layer
        self.backpropagate(batch_x, batch_y)

        # check gradients
        for i, layer in enumerate(self.model):
            if layer.name() in layers:
                self.check_layer(batch_x, batch_y, i, delta=delta)
    

    def check_layer(self, batch_x, batch_y, layer_indx, delta=1e-5, error_limit=1e-6):
        layer = self.model.layers[layer_indx]

        for i, (param, dparam) in enumerate(zip(layer.params(), layer.dparams())):
            assert param.shape == dparam.shape, 'Error dims should match: %s and %s' % (param.shape, dparam.shape)
            ri = int(uniform(0, param.size))

            # evaluate cost at [x + delta] and [x - delta]
            old_val = param.flat[ri]
            self.model.layers[layer_indx].params()[i].flat[ri] = old_val + delta
            cg0 = self.loss(batch_x, batch_y)
            self.model.layers[layer_indx].params()[i].flat[ri] = old_val - delta
            cg1 = self.loss(batch_x, batch_y)

            # reset old value for this parameter
            self.model.layers[layer_indx].params()[i].flat[ri] = old_val
            
            # fetch both numerical and analytic gradient
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / (2 * delta)
            
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            print (rel_error <= error_limit, '%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
