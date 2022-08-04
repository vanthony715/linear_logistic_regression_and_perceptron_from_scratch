# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Activation functions such as sigmoid, relu, etc..
"""

import numpy as np

class ReLU:

    '''Rectified Linear Activation Object'''

    def forward(self, inputs):

        return np.maximum(inputs, 0)

    def derivative(self, inputs):

        return inputs > 0

class Sigmoid:

    '''Calculate the Sigmoid'''

    def forward(self, inputs):

        '''given an input value x, the output is either 1 or 0'''

        return 1 / (1 + np.exp(-inputs))

    def derivative(self, inputs):

        '''outputs the dericative of sigmoid'''

        return inputs * (1 - inputs)

class Signum:

    '''Signum Activation for the Perceptron'''

    def forward(self, inputs):
        
        '''the output is 1 if the input >= 0 else it is 0'''
        
        return np.where(inputs >= 0, 1, 0)

class Step:

    '''prediction step function'''

    def forward(self, prediction):

       return 1 if prediction > 0.5 else 0

class Softmax:

    '''Softmax Activation'''

    def forward(self, inputs):

        ##calculate exponential values
        self.exponents = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    
        ##calculate probabilities
        return np.exp(inputs) / np.sum(np.exp(inputs))

class Tanh:

    '''Hyperbolic Tangent Activation'''

    def forward(self, inputs):

        return (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))

    def derivative(self, inputs):

        return 1 - ((np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs)))**2
