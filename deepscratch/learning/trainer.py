import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import deepscratch.learning.losses as losses
import deepscratch.learning.metrics as ds_metrics
import deepscratch.models.optimizers as optimizers


class Trainer(object):
    def __init__(self, model, loss, metrics=[], print_step_mod=1000, verbose=True, batch_size=16):
        self.model = model
        self.model.initialize()

        self.loss = losses.load(loss) if type(loss) is str else loss
        self.metrics = self._define_metrics(metrics)
        self.batch_size = batch_size

        self.train_step = 0
        self.eval_steps = []
        
        self.verbose = verbose
        self.print_step_mod = print_step_mod
        
        self.train_losses = []
        self.eval_losses = []
    

    def _define_metrics(self, metrics):
        new_metrics = []
        for metric in metrics:
            new_metrics.append(ds_metrics.load(metric) if type(metric) is str else metric)
        return new_metrics


    def batch_train(self, batch_x, batch_y):
        self.train_step += 1
        # run feed forward network
        pred_y = self.predict(batch_x)
        # save loss
        self.train_losses.append(self.loss(pred_y, batch_y))
        # run backpropagation
        self.model.backward(self.loss.grads(pred_y, batch_y))
        if self.verbose and (self.train_step - 1) % self.print_step_mod == 0:
            print('Loss: %.4f for step %d' % (self.train_losses[-1], self.train_step))
    

    def batches(self, x, y, batch_size):
        idx = np.random.permutation(len(x))
        x = x[idx]
        y = y[idx]

        for i in range(0, len(x)-batch_size-1, batch_size):
            batch_x = x[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            yield batch_x, batch_y


    def train(self, data_x, data_y, epochs=1, batch_size=None, test_size=0.0, eval_x=None, eval_y=None):
        if batch_size is None:
            batch_size = self.batch_size
        
        if test_size > 0.0:
            data_x, eval_x, data_y, eval_y = train_test_split(data_x, data_y, test_size=test_size)

        for epoch in range(1, epochs + 1):
            self.train_losses = []
            
            # Train
            if batch_size == -1:
                self.batch_train(data_x, data_y)
            else:
                for batch_x, batch_y in self.batches(data_x, data_y, batch_size):
                    self.batch_train(batch_x, batch_y)
            
            # Eval
            train_loss, train_metrics = self.eval(data_x, data_y, batch_size=batch_size)
            print ('Epoch {} | Train loss: {:5f} | Train metrics: {}'.format(epoch, train_loss, train_metrics))

            if eval_x is not None and eval_y is not None:
                eval_loss, eval_metrics = self.eval(eval_x, eval_y, batch_size=batch_size)
                print ('Eval loss: {:5f} | Eval metrics: {}'.format(eval_loss, eval_metrics))


    def eval(self, data_x, data_y, metrics=None, batch_size=16, log=False):
        if log:
            self.eval_steps.append(self.train_step)
                
        # metrics
        if metrics is None:
            metrics = self.metrics
        
        self.eval_losses = []
        if batch_size == -1:
            return self.batch_eval(data_x, data_y, metrics)
        else:
            batches = 0
            res_metrics = []
            for batch_x, batch_y in self.batches(data_x, data_y, batch_size):
                batches += 1
                _, batch_metrics = self.batch_eval(batch_x, batch_y, metrics)
                res_metrics.append(batch_metrics)
        
            avg_loss = sum(self.eval_losses) / batches
            
            avg_metrics = [0] * len(metrics)
            for batch_metric in res_metrics:
                for i in range(len(avg_metrics)):
                    avg_metrics[i] += batch_metric[i]
            avg_metrics = [m/batches for m in avg_metrics]

            return avg_loss, avg_metrics

    
    def batch_eval(self, data_x, data_y, metrics):
        # run feed forward network
        pred_y = self.predict(data_x)
        
        # loss
        loss = self.loss(pred_y, data_y)
        self.eval_losses.append(loss)

        res_metrics = []
        for metric in metrics:
            res_metrics.append(metric(pred_y, data_y))

        return loss, res_metrics


    def predict(self, data_x):
        return self.model.forward(data_x)

    
    def plot_losses(self):
        if len(self.eval_losses) > 0:
            plt.title('Train Loss: %.4f | Test Loss: %.4f for step %d' % (self.train_losses[-1], self.eval_losses[-1], self.train_step))
        else:
            plt.title('Train Loss: %.4f for step %d' % (self.train_losses[-1], self.train_step))    
        plt.plot([i for i in range(self.train_step)], self.train_losses)
        plt.plot([i for i in self.eval_steps], self.eval_losses)