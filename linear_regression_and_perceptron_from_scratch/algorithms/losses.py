# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Loss calculations
"""

import numpy as np

class CrossEntropyLoss:

    '''calculates the cross entropy loss'''

    def calculate(self, truth, predicted):

        return -np.sum(truth * np.log(predicted)) / float(truth.shape[0])

class MSE:

    '''calculate the loss using mse for regression datasets'''

    def calculate(self, truth, predicted):

        self.mse = np.square(np.sum(np.subtract(truth, predicted))) / truth.shape[0]

