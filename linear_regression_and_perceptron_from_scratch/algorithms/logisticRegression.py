# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Logistic Regression
"""
import time
import numpy as np
from tqdm import tqdm

class LogisticRegression:
    
    '''Logistic Regression Model'''
    
    def __init__(self, iterations, lr):
        self.iterations = iterations ##num iterations to run on dataset
        self.lr = lr ##learn rate
        self.weights = None
        self.bias = None
    
    def _sigmoid(self, x):
        
        '''Calculates the Sigmoid for Each sample'''
        
        return 1 / (1 + np.exp(-x))
    
    def _initializeWeights(self, num_features):
        
        '''initializes weights'''
        
        return np.zeros(num_features)
    
    def train(self, X, y):
        
        '''initializes weights and biases and starts training '''
        
        ##number of observations, number of features, and bias
        m, n = X.shape
        ##initialize weights and bias
        self.weights = self._initializeWeights(n)
        self.bias = 0
        
        ##start training loop
        for iteration in tqdm(range(self.iterations), desc = 'Train Loop', colour = 'green'):
            
            ##linear function is calulated
            linear_output = np.dot(X, self.weights) + self.bias
            
            ##the predictions is calculated by inputing the linear output
            ## to the sigmoid
            y_pred = self._sigmoid(linear_output)
            
            ##updates to the weights are made by taking the 
            ##partial derivative with respect to weights (dJ/dw)
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            
            ##updates to the bias are made by taking the 
            ##partial derivative with respect to bias (dJ/dB)
            db = 1 / m * np.sum(y_pred - y)
            
            ##finally, the weights and biases are updated by multiplying the learning
            ##rate to the weights and biases
            self.weights -= self.lr * dw.T
            self.bias -= self.lr * db
            
            ##print output at interval
            if (iteration % 50 == 0):
                print('\n-------------------- UPDATES --------------------')
                print('\n Logistic Regression')
                print('\n linear output: ', linear_output)
                print('\n Updated weights: ', self.weights)
                print('\n Predictions: ', y_pred)
                print('\n Updated bias: ', self.bias)
                time.sleep(3)
        
    def predict(self, X):
        
        '''will predict test set given the test set and trained weights and biases'''
        
        ##linear function
        linear_mod = np.dot(X, self.weights) + self.bias
        ##probability to create logistic
        y_pred = self._sigmoid(linear_mod)
        ##get predictions
        self.predictions = [1 if i > 0.5 else 0 for i in y_pred]
        return self.predictions
    
    def calcAccuracy(self, y):
         
        '''calculates accuracy given truth and predicted value'''
        
        ##change target values to zeros and 1s 
        y[y <= 0], y[y > 0] = 0, 1
        
        return np.sum(y == self.predictions) / len(y)
        