# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Feed Forward Neural Network
"""
import time
import numpy as np
from tqdm import tqdm

class NeuralNetwork:

    '''Neural Network used for feedforward and partially for the autoencoder'''

    def __init__(self, layers, lr, active_fn1, active_fn2, active_out, pretrained_weights=None):
        ##number of layers in network
        self.layers = layers
        ## training learn rate
        self.lr = lr
        ##define first activation function
        self.active_fn1 = active_fn1
        ##define second activation function
        self.active_fn2 = active_fn2
        ##output layer
        self.active_out = active_out

    def initializeWeights(self):

        '''Initializes Weights for Each Layer'''

        self.weights = []
        ##initialize random weights for input layers to first hidden layers
        for i in np.arange(0, len(self.layers) - 2):
            ##generate random values for weights
            w = np.random.randn(self.layers[i] + 1, self.layers[i + 1] + 1)
            ##normalize the weights and append to list
            self.weights.append(w / np.sqrt(self.layers[i]))
            ##initialize weights for last two layers
            w = np.random.randn(self.layers[-2] + 1, self.layers[-1])
            ##normalize weights and append to list
            self.weights.append(w / np.sqrt(self.layers[-2]))

    def summary(self):

        '''outputs a network summary that lists layers'''

        print('\n----- Network Summary -----')
        print('Input Layer Size: ', self.layers[0])
        print('Hidden Layer1: ', self.layers[1], 'Activation Function: ', self.active_fn1)
        print('Output Layer: ', self.layers[2], 'Activation Function: ', self.active_fn2)
        print('Output Activation: ', self.active_out)

    def oneHot(self, y):

        '''One hot encoding for cross entropy calculations'''

        ##create array of zeros of shape of one hot encoded labels
        oh = np.zeros((y.shape[0], np.unique(y).size))
        ##if value == 0, o_h_y[0], o_h_y[1] = [1, 0] elif value == 1 o_h_y[0], o_h_y[1] = [0, 1] ...
        oh[np.arange(y.shape[0]), y] = 1
        ##return o_h_y transposed for derivative calculation
        return oh

    def _addOnesColumn(self, X):

        '''adds a ones column at the end of X to play nice with the weights'''

        return np.c_[X, np.ones((X.shape[0]))]

    def _updateWeights(self, activations, d):

        '''updates weights'''
        ##The weights in the network are updated for each layer
        ##in this case, the bias is baked into the weights
        for layer in np.arange(0, len(self.weights)):
            self.weights[layer] += -self.lr * activations[layer].T.dot(d[layer])

    def _backprop(self, xi, yi):

        '''back propagation calculation'''

        ##create activation list for each of the
        activations = [np.atleast_2d(xi)]
        for layer in np.arange(0, len(self.weights)):
            
            ## feedforward linear step for each layer
            self.linear_output = activations[layer].dot(self.weights[layer])
            
            ## the nonlinear step is calculated by inputting the linear_output
            ## to the activation function 
            self.output = self.active_fn1.forward(self.linear_output)
            ##append outputs to activations
            activations.append(self.output)

        #the error is calculated by taking the difference between truth and the
        ##last activation output
        error = activations[-1] - yi
        
        ##y is not taken into account for the autoencoder case
        # error = activations[-1]
        ##debug
        # self.a = activations

        ##calculate derivative list of the last activation or output
        d = [error * self.active_fn1.derivative(activations[-1])]
        ##take the derivative of each layer in reverse order
        for layer in np.arange(len(activations) - 2, 0, -1):
            
            ##to find what was learned, the derivative of each layer is calculated
            ##starting with the last layer in the direction of the input layer
            dif = d[-1].dot(self.weights[layer].T)
            dif = dif * self.active_fn2.derivative(activations[layer])
            
            ##the derivative is appended to a list
            d.append(dif)
            
        ##And because the derivative was appended in reverse order, the list is reversed
        d = d[::-1]
        ##update weights
        self._updateWeights(activations, d)

    def train(self, X, y, iterations):
        ##add constant adds colum of ones
        X = self._addOnesColumn(X)
        ##start iteration loop
        for iteration in tqdm(range(iterations), desc = 'Training', colour = 'blue'):
            ##backpropagate through each sample associated label
            for (xi, yi) in zip(X, y):
                self._backprop(xi, yi)
                
                ##print output at interval
                # if (iteration % 50 == 0):
                #     print('\n-------------------- UPDATES --------------------')
                #     print('\n Feed Forward')
                #     print('\n linear output: ', self.linear_output)
                #     print('\n Updated weights: ', self.weights)
                #     print('\n Predictions: ', self.output)
                #     time.sleep(3)

    def predict(self, X):

        '''prediction with single output label'''

        ##make sure values are atleast 2d else make them 2d
        predictions = np.atleast_2d(X)
        ##add bias column
        predictions = np.c_[predictions, np.ones((predictions.shape[0]))]
        ##loop over network layers
        for layer in np.arange(0, len(self.weights)):
            predictions = self.active_out.forward(np.dot(predictions, self.weights[layer]))
        ##returns a single input value
        return predictions

    def predictCrossEntropy(self, X):

        '''prediction using one hot encoded labels'''

        ##make sure values are atleast 2d else make them 2d
        predictions = np.atleast_2d(X)
        ##add bias column
        predictions = np.c_[predictions, np.ones((predictions.shape[0]))]
        ##loop over network layers
        for layer in np.arange(0, len(self.weights)):
            predictions = self.active_out.forward(np.dot(predictions, self.weights[layer]))
        ##returns a single input value
        return predictions

    def mse(self, truth, predictions):

        '''calculate the predictions all at once using the predict function'''

        return np.square(np.sum(np.subtract(truth, predictions))) / truth.shape[0]

class Perceptron:

    '''Single Perceptron'''

    def __init__(self, lr, n_iterations, activation):
        ##define learning rate
        self.lr = lr
        ##number of iterations to train
        self.n_iterations = n_iterations
        ##activation funtion to use
        self.activation = activation

    def _initializeWeights(self, num_features):

        '''initializes weights'''

        return np.zeros(num_features)

    def _onesAndZeros(self, y):

        '''converts all of the y values in y to either ones or zeros'''

        return np.array([1 if i > 0 else 0 for i in y])

    def train(self, X, y):

        '''train the model'''

        ##num of samples, num features
        m, n = X.shape
        ##initialize weights
        self.weights = self._initializeWeights(n)
        ##initialize bias
        self.bias = 0
        
        ##start training
        for _ in tqdm(range(self.n_iterations), desc = 'Perceptron Train Loop', colour = 'green'):
            for idx, x in enumerate(X):
                
                ##calculate the linear output 
                linear_output = np.dot(x, self.weights) + self.bias
                
                ##prediction is made by inputting the linear linear_output to the activation function
                y_pred = self.activation.forward(linear_output)
                
                ##weights are then updated by multiplying the learning rate with the difference
                ##of the truth and the prediction multiplied by the sample
                self.weights += self.lr * (y[idx] - y_pred) * x
                
                ##the bias is then updated by multiplying the learning rate with the difference
                ##of the truth and the prediction
                self.bias +=  self.lr * (y[idx] - y_pred)
                
                # ##print output at interval
                # if (idx % 50 == 0):
                #     print('\n-------------------- UPDATES --------------------')
                #     print('\n linear output: ', linear_output)
                #     print('\n Updated weights: ', self.weights)
                #     print('\n Predictions: ', y_pred)
                #     print('\n Updated bias: ', self.bias)
                #     time.sleep(3)

    def predict(self, X):

        '''predict using the trained model'''

        ##calculate the linear output
        linear_output = np.dot(X, self.weights) + self.bias
        ##y prediction is the activation of the linear output
        self.y_pred = self.activation.forward(linear_output)

    def accuracy(self, y, y_pred):

        '''calculate the accuracy of the trained model'''

        self.accuracy = sum(y == y_pred) / len(y)