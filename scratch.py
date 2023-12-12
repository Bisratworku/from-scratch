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
    def __init__(self, inputs, neurons, weight_regularaization_l1 = 0, weight_regularaization_l2 = 0, bias_regularaization_l1 = 0, bias_regularizatiob_l2 = 0):
        self.data = inputs
        try:
            self.weight =  np.random.randn(self.data, neurons)
        except:
            self.weight =  np.random.randn(len(self.data.shape), neurons)
        self.bias = np.zeros((1, neurons))
        self.weight_regularization_l1 = weight_regularaization_l1
        self.weight_regularization_l2 = weight_regularaization_l2
        self.bias_regularization_l1 = bias_regularaization_l1
        self.bias_regularization_l2 = bias_regularizatiob_l2
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
class activation:
    def __init__(self , output):
        self.output = output
    def step(self):
        step = np.array(self.output)
        step[step > 0] = 1
        step[step <= 0] = 0
        return step
    def sigmoid(self):
        self.sigmoid =  1/(1 + np.exp(-self.output))
        return self.sigmoid
    def sigmoid_backward(self, dvalues):
        self.dinputs = dvalues * (self.sigmoid * (1 - self.sigmoid))
    def RELU(self):
       self.value = np.array(self.output)
       self.value[self.value < 1] = 0
       return self.value
    def RELU_backward(self,dvalue):
        self.dinputs = dvalue.copy()
        self.dinputs[self.value < 1] = 0
    def softmax(self):
        exp_value = np.exp(self.output - np.max(self.output, axis = 1, keepdims = True))
        self.output =  exp_value/np.sum(exp_value, axis = 1 , keepdims = True)
        return self.output
    def softmax_backward(self, dvalues):
        dinputs = np.empty_like(dvalues)
        for i , (sample_output,sample_derivative) in enumerate(zip(self.output, dvalues)):
            kronicer_delta = np.diagflat(sample_output.reshape(-1,1))
            jacobian_matrics = kronicer_delta - np.dot(sample_output,sample_output.T)
            dinputs[i] = np.dot(jacobian_matrics,sample_derivative)
        return dinputs

class loss:
    def __init__(self,pred,target) :
        self.pred = pred # this is the output of the softmax activation 
        self.target = target #this the true lables

    def regularization_loss(self, layer):
        self.regularization = 0
        if layer.weight_regularization_l1 > 0:
            self.regularization += layer.weight_regularization_l1 * np.sum(abs(layer.weight))
        if layer.weight_regularization_l2 > 0:
            self.regularization += layer.weight_regularization_l2 * np.sum(layer.weight * layer.weight)
        if layer.bias_regularization_l1 > 0 :
            self.regularization += layer.bias_regularization_l1 * np.sum(abs(layer.bias))
        if layer.bias_regularization_l2 > 0:
            self.regularization += layer.bias_regularization_l2 * np.sum(layer.bias * layer.bias)
        return self.regularization
    def catagorical_crossentropy(self):
        if len(self.target.shape) == 1: #if the labels are not one hot encoded 
            elem = self.pred[[range(len(self.pred))],[self.target]] 
            clipped = np.clip(elem, 1e-7, 1- 1e-7)
            self.output = np.mean(-np.log(clipped))
            return self.output
        elif len(self.target.shape) == 2:
            sum = np.sum(self.pred * self.target , axis = 1)
            clip = np.clip(sum, 1e-7, 1 - 1e-7)
            self.output = np.mean(-np.log(clip))
            return self.output
    def catagory_backward(self):
        label = len(self.pred[0]) #we use this to get the number of elements in the first #array we use it to one hot encode vectors
        if len(self.target.shape) == 1: #if it is not one hot encoded
            self.target = np.eye(label) [self.target]  # we hot code encode it what this will do is first we create a diagonal vector with shape equvalent to [label] and the vector will be 
            #duplicated to match the number elements in [self.target] then the elements in [self.target] will be used as a index of 1.

        self.values = -self.target/self.pred #this is the derivative of catagorical cross entropy loss
        self.output = self.values/len(self.pred) # to normalize them this will sava us from changing the learning rate every stape. 
        return self.output
    def binary_cross_entropy_loss(self):
        clipped_pred = np.clip(self.pred, 1e-7, 1 - 1e-7) # we clipped thr zero values to avoid a run time error which is the zero logarithm error
        self.output = -(self.target * np.log(self.pred)) + (1 - self.target) * np.log(1 - self.pred)
        self.output = np.mean(self.output)
        return self.output
    def backward_binary_cross(self):
        outputs = len(self.pred[0])
        clipped_dvalues = np.clip(self.pred, 1e-7, 1 - 1e-7)
        self.dinputs = -(self.target / clipped_dvalues - (1 - self.target /1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / len(self.pred)

class derivative_softmax_crossEntropy:
        def __init__(self, Relu_output,target) :
            self.target = target
            self.softmax = activation(Relu_output).softmax()
            self.loss = loss(self.softmax, target).catagorical_crossentropy()
        def backward(self):
            sample = len(self.softmax)
            if len(self.target.shape) == 2: # if the label is one hot encoded we change them to descrite values
                self.target = np.argmax(self.target)
            self.output = self.softmax.copy()  
            self.output[range(sample) , self.target] -= 1
            self.output = self.output/sample 
            return self.output

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
    def forward(self, input):
        self.bernouli = np.random.binomial(1, self.rate, input.shape)/self.rate
        self.output = input * self.bernouli
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

l1 = layer(2, 64, weight_regularaization_l2=5e-4, bias_regularizatiob_l2 = 5e-4 )
l1.forward(X)
a1 = activation(l1.output)
l2 = layer(64,1)
l2.forward()



























