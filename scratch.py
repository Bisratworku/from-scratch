from typing import Any
import random
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
X,y = spiral_data(samples = 100, classes = 2)
y = y.reshape(-1, 1) 
#=nb
class layer:
    #  this class will help us make layers of neural nerworks automaticaly it will be initialyzed by how much input we have and how many neurons we want for each layer the weights are random numbers that have the shape the number of inputs by the number of neurons we want. 
    def __init__(self, inputs, neurons, weight_regularaization_l1 = 0, weight_regularaization_l2 = 0, bias_regularaization_l1 = 0, bias_regularization_l2 = 0):
        self.data = inputs
        try:
            self.weight =  np.random.randn(self.data, neurons)
        except:
            self.weight =  np.random.randn(len(self.data.shape), neurons)
        self.bias = np.zeros((1, neurons))
        self.weight_regularization_l1 = weight_regularaization_l1
        self.weight_regularization_l2 = weight_regularaization_l2
        self.bias_regularization_l1 = bias_regularaization_l1
        self.bias_regularization_l2 = bias_regularization_l2
    def forward(self, data):
        self.input = data
        self.output = np.dot(self.input,self.weight) + self.bias
    def backward(self, relu):
        self.dweight = np.dot((self.input).T, relu)
        self.dinputs = np.dot(relu, self.weight.T)
        self.dbias = np.sum(relu, axis=0, keepdims=True)

        if self.weight_regularization_l1 > 0:
            dl1 = np.ones_like(self.weight)
            dl1[dl1 < 0] = -1
            self.dweight += self.weight_regularization_l1 * dl1
        if self.weight_regularization_l2 > 0:
            self.dweight += self.weight_regularization_l2 * 2 * self.weight
        if self.bias_regularization_l1 > 0:
            dl1 = np.ones_like(self.bias)
            dl1[dl1 < 0] = -1
            self.dbias += self.bias_regularization_l1 * dl1
        if self.bias_regularization_l2 > 0:
            self.dbias += self.bias_regularization_l2 * self.bias
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backwarx(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis= 1, keepdims= True))
        probabilities = exp_values / np.sum(exp_values, axis= 1, keepdims = True)
        self.output = probabilities
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index , (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrics = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrics, single_dvalues)
class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
class loss:
    def regularization_loss(self, layer):
        regularization_loss = 0
        if layer.weight_regularaization_l1 > 0 :
            regularization_loss += layer.weight_regularaization_l1 * np.sum(np.abs(layer.weight))
        if layer.weight_regularaization_l2 > 0 :
            regularization_loss += layer.weight_regularaization_l2 * np.sum(layer.weight * layer.weight)
        if layer.bias_regularaization_l1 > 0:
            regularization_loss += layer.bias_regularaization_l1 * np.sum(np.abs(layer.bias))
        if layer.bias_regularizatiob_l2 > 0 :
            regularization_loss += layer.bias_regularizatiob_l2 * np.sum(layer.bias * layer.bias)
        return regularization_loss
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
class Loss_CatagoricalCrossEntropy(loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1 :
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
    def backward(self, dvalues, y_true): # dvalues stands for the output of the actvation function in this case that activation function is softmax 
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
class derivative_softmax_crossEntropy:
        def __init__(self, target) :
            self.target = target
            self.softmax = Activation_Softmax()
            self.loss = Loss_CatagoricalCrossEntropy()
        def forward(self, inputs, y_true):
            self.softmax.forward(inputs)
            self.output = self.softmax.output
            return self.loss.calculate(self.output, y_true)
        def backward(self, dvalues, y_true): # dvalues stands for the output of the activation function in this case that activation function is sigmoid 
            samples = len(dvalues)
            if len(y_true.shape) == 2:
                y_true = np.argmax(y_true, axis= 1)
            self.dinputs = dvalues.copy()
            self.dinputs[range(samples), y_true] -= 1
            self.dinputs = self.dinputs / samples
class loss_binaryCrossentropy(loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1- 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples
class mse:
    def forward(self, y_pred, y_true):
        
        self.output = np.mean((y_pred - y_true) **2, axis= -1)
        return self.output
    def backward(self, dvalues, y_true):
        samples =   len(dvalues)
        output = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues)/output
        self.dinputs = self.dinputs / samples
class mae:
    def forward(self, y_pred , y_true):
        self.output = np.mean(np.abs(y_pred - y_true), axis= -1)
        return self.output
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        output = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / output # np.sign will return 1 if the number is greater than 0 and -1 if the number is less than 0 
        self.dinputs = self.dinputs / samples
class optimizerSGD:
    def __init__(self, learning_rate = 1. , decay = 0., momentem = 0.):
        self.learning_rate = learning_rate
        self.current_leraning_rate = learning_rate
        self.decay = decay
        self.step = 0
        self.momentum = momentem
    def pre_param_update(self):
        self.current_leraning_rate = (1/(1 + self.decay * self.step)) * self.learning_rate
        #self.current_leraning_rate = (np.power(self.decay, self.step)) * self.learning_rate
    def update_params (self , layer):
        if self.momentum:
            if not hasattr(layer , "weight_momentem"):
                layer.weight_momentem = np.zeros_like(layer.weight)
                layer.bias_momentem = np.zeros_like(layer.bias) 
            weight_update = self.momentum * layer.weight_momentem - self.current_leraning_rate * layer.dweight
            layer.weight_momentem = weight_update
            bias_update = self.momentum * layer.bias_momentem - self.current_leraning_rate * layer.dbias
            layer.bias_momentem = bias_update
        else:
            weight_update = -self.current_leraning_rate * layer.dweight
            bias_update = -self.current_leraning_rate * layer.dbias
        layer.weight += weight_update
        layer.bias += bias_update 
    def after_update_params(self):
        self.step += 1
class Ada_grad:
    "'Ada_grad or adaptive grad works by assigning a custom learning rate to each hyper parameter'"
    def __init__(self, learning_rate = 1., norm = 1e-7, decay = 1e-3):
        self.current_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.norm = norm # norm will helps us to avoid the singularity problem which is when the learning become so small
        self.decay = decay
        self.step = 0
    def pre_update_params(self):
        self.current_learning_rate = (1/(1 + self.decay * self.step)) * self.learning_rate
    def update_params(self, layer):
        if not hasattr(layer , "weight_catch"):
            layer.weight_catch  = np.zeros_like(layer.weight)
            layer.bias_catch = np.zeros_like(layer.bias)
        layer.weight_catch += layer.dweight**2
        layer.bias_catch += layer.dbias**2
        catch_weight = -self.current_learning_rate *layer.dweight / (np.sqrt(layer.weight_catch) + self.norm) 
        layer.weight += catch_weight
        catch_bias = -self.current_learning_rate * layer.dbias / (np.sqrt(layer.bias_catch) + self.norm) 
        layer.bias += catch_bias 
    def after_update_params(self):
        self.step += 1
class Rms_prop:
    def __init__(self, learning_rate = 1.,decay = 0., norm = 1e-7, rho = 1e-4):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.norm = norm
        self.rho = rho
        self.step = 0
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1 + self.decay * self.step))
    def update_params(self, layer):
        if not hasattr(layer, "weight_catch"):
            layer.weight_catch = np.zeros_like(layer.weight)
            layer.bias_catch = np.zeros_like(layer.bias)
        layer.weight_catch = self.rho * layer.weight_catch + (1 - self.rho) * layer.dweight**2
        layer.bias_catch = self.rho * layer.bias_catch + (1 - self.rho) * layer.dbias**2
        layer.weight += -self.current_learning_rate * layer.dweight/(np.sqrt(layer.weight_catch + self.norm))
        layer.bias += -self.current_learning_rate * layer.dbias/(np.sqrt(layer.bias_catch + self.norm) )
    def after_update_params(self):
        self.step += 1 
class Adam:
    def __init__(self, learning_rate = 0.001,decay = 0., norm = 1e-8, rho =0.999 , beta = 0.9):
        self.current_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.norm = norm
        self.rho = rho
        self.beta = beta
        self.step = 0
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = (1./(1. + self.decay * self.step)) * self.learning_rate
    def update_params(self, layer):
        if not hasattr(layer, "weight_momentem"):
            layer.weight_momentem = np.zeros_like(layer.weight)
            layer.weight_catch = np.zeros_like(layer.dweight)
            layer.bias_momentem = np.zeros_like(layer.bias)
            layer.bias_catch = np.zeros_like(layer.bias)
        layer.weight_momentem = self.beta * layer.weight_momentem + (1 - self.beta) * layer.dweight 
        layer.weight_catch = self.rho * layer.weight_catch + (1 - self.rho) * layer.dweight **2 
        weight_momentem_corrected = layer.weight_momentem/(1 - self.beta ** (self.step + 1))
        weight_catch_corrected = layer.weight_catch/(1 - self.rho ** (self.step + 1))

        layer.bias_momentem = self.beta * layer.bias_momentem + (1 - self.beta) * layer.dbias
        layer.bias_catch = self.rho * layer.bias_catch + (1 - self.rho) * layer.dbias ** 2
        bias_momentem_corrected = layer.bias_momentem / (1- self.beta ** (self.step + 1))
        bias_catch_corrected = layer.bias_catch / (1 - self.rho ** (self.step + 1))

        layer.weight += -self.current_learning_rate * weight_momentem_corrected/(np.sqrt(weight_catch_corrected) + self.norm)
        layer.bias += -self.current_learning_rate * bias_momentem_corrected/(np.sqrt(bias_catch_corrected) + self.norm)
    def after_update_params(self):
        self.step += 1
class dropOut:
    def __init__(self , rate : float = 0.5):
        self.rate = 1 - rate
    def forward(self, inputs):
        self.inputs = inputs
        self.bernouli = np.random.binomial(1, self.rate, size = inputs.shape)/self.rate
        self.output = inputs * self.bernouli
    def backward(self, dinputs):
        self.doutput = self.bernouli * dinputs
class accuracy:
    def __init__(self , softmax_output, y):
        self.sotmax_output = softmax_output
        self.label = y
    def calculate(self):
        pred = np.argmax( self.sotmax_output, axis=1)
        if len(self.label.shape) == 2:
            self.label = np.argmax(self.label, axis=1)
        correct = np.mean(pred == self.label)
        return correct



























