#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anthony J. Vasquez
email: avasque1@jh.edu
phone: 315-920-8877

Class: Introduction to Machine Learning

Description: Project 4 - Main
"""
##standard python libraries
import os
import sys
import warnings
import argparse
import time
import gc

import pandas as pd
import numpy as np

##preprocess pipeline
from utils.dataLoaders import LoadCsvData
from utils.preprocess import PreprocessData
from utils.splitData import SplitData
from utils.splitDataClassless import SplitDataClassless

##neural networks
from algorithms.logisticRegression import LogisticRegression
from algorithms.networks import NeuralNetwork
from algorithms.networks import Perceptron

##activation functions
from algorithms.activations import ReLU
from algorithms.activations import Signum
from algorithms.activations import Softmax
from algorithms.activations import Tanh

##import losses
from algorithms.losses import MSE
from algorithms.losses import CrossEntropyLoss

##turn off all warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

gc.collect()

##command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder_name', type = str ,default = '/data',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--dataset_name', type = str ,default = '/abalone',
                    help='Name of the folder where the data and names files are located'),

parser.add_argument('--namespath', type = str , default = 'data/abalone',
                    help='Path to dataset names'),

parser.add_argument('--discretize_data', type = bool ,default = False,
                    help='Should dataset be discretized?'),

parser.add_argument('--quantization_number', type = int , default = 2,
                    help='If discretized, then quantization number'),

parser.add_argument('--standardize_data', type = bool , default = True,
                    help='Should data be standardized?'),

parser.add_argument('--k_folds', type = int , default = 5,
                    help='Number of folds for k-fold validation'),

parser.add_argument('--stratified', type = bool, default = False,
                    help='split the dataset evenly based on classes'),

parser.add_argument('--min_examples', type = int , default = 1,
                    help='Drop classes with less examples then this value'),

parser.add_argument('--remove_orig_cat_col', type = bool , default = True,
                    help='Remove the original categorical columns for data encoding'),

parser.add_argument('--model', type = str , default = 'autoencoder',
                    help='Can be perceptron, logistic_regression, feedforward, or autoencoder'),
args = parser.parse_args()
## =============================================================================
##                                  MAIN
## =============================================================================
if __name__ == "__main__":
    ##start timer
    tic = time.time()
## =============================================================================
##                              PATHS / ARGUMENTS
## =============================================================================
    ##define paths
    cwd = os.getcwd().replace('\\', '/') ##get current working directory
    data_folder_name = cwd + args.data_folder_name
    datapath = data_folder_name + args.dataset_name + '.data'
    namespath = data_folder_name + args.dataset_name + '.names'
    dataset_name = args.dataset_name
## =============================================================================
##                                  PREPROCESS
## ============================================================================
    ##list of categorical data columns per dataset
    car = ['buying', 'maint','lug_boot', 'safety']
    abalone = ['sex']
    forestfires= ['month', 'day']
    practice = ['gender', 'travel', 'income_level']

    if dataset_name == '/car':
        category_list = car
    elif dataset_name == '/abalone':
        category_list = abalone
    elif dataset_name == '/forestfires':
        category_list = forestfires
    elif dataset_name == '/practice':
        category_list = practice
    else:
        category_list = []

    classification_list = ['car','breast-cancer-wisconsin','house-votes-84', 'practice']
    regression_list = ['abalone', 'forestfires', 'machine', 'iris']

    if args.dataset_name.split('/', 1)[1] in classification_list:
        mode = 'classification'
        test_dict = {'testset': [], 'dataset': [], 'score': [],
                 'score_pruned': [], 'decision_node_cnt': [],
                 'decision_node_cnt_pruned': [], 'stop_criteria': []}

    elif args.dataset_name.split('/', 1)[1] in regression_list:
        mode = 'regression'
        test_dict = {'testset': [], 'dataset': [], 'score': [],
                 'decision_node_cnt': [], 'stop_criteria': []}
    else:
        'None'

    print('Dataset: ', args.dataset_name.split('/', 1)[1])
    print('Train mode: ', mode)

    print('\n******************** ML Pipeline Started ********************')
    ##define tuple of values to drop from dataframe
    values_to_replace = ('na', 'NA', 'nan', 'NaN', 'NAN', '?', ' ')
    # values_to_change = {'place_holder':0}
    values_to_change = {'5more':6, 'more': 5}

    # ##load data
    load_data_obj = LoadCsvData(datapath, namespath, dataset_name)
    names = load_data_obj.loadNamesFromText() ##load names from text
    data = load_data_obj.loadData() ##data to process

    ##preprocess pipeline
    proc_obj = PreprocessData(data, values_to_replace, values_to_change,
                              args.dataset_name, args.discretize_data,
                              args.quantization_number, args.standardize_data,
                              args.remove_orig_cat_col, category_list, mode)
    proc_obj.dropRowsBasedOnListValues() ##replaces values from list
    proc_obj.changeValues() ##changes values from values_to_change list
    proc_obj.encodeData() ##encodes data
    proc_obj.createUniqueAttribNames()
    proc_obj.convertDatatypes()
    proc_obj.standardizeData() ##standardizes data
    df_encoded = proc_obj.discretizeData() ##discretizes dat

    if mode == 'regression':
        encoded_targs = {'target': []}
        for i in range(len(data['target'])):
            mean = np.mean(df_encoded.target)
            std = np.std(df_encoded.target)
            z = (df_encoded.target[i] - mean) / std
            encoded_targs['target'].append(z)
        df_encoded = df_encoded.drop('target', axis = 1)
        encoded_targs = pd.DataFrame(encoded_targs)
        df_encoded['target'] = encoded_targs

        ##round the data before split
        df_encoded = round(df_encoded, 3)

    if args.stratified and mode == 'classification':
        split_obj = SplitData(df_encoded, args.k_folds, args.min_examples)
        split_obj.removeSparseClasses() ##removes classes that do not meet the min_examples criteria
        split_obj.countDataClasses() ##counts data classes
        split_obj.splitPipeline() ##start of the stratefied k-fold validation split
        train_test_sets = split_obj.createTrainSets() ##k train and test sets returned as a dictionary
        train_columns = split_obj.getTrainColumns() ##gets all columns but target
    else:
        split_obj = SplitDataClassless(df_encoded, args.k_folds, args.min_examples)
        split_obj.splitPipeline() ##start of the stratefied k-fold validation split
        train_test_sets = split_obj.createTrainSets() ##k train and test sets returned as a dictionary
        train_columns = split_obj.getTrainColumns() ##gets all columns but target

    def writeToFile(writepath, variable, description, file_ext):

        '''input is numpy array which is converted to a dataframe and written to file'''

        dataframe = pd.DataFrame(variable)
        dataframe.to_csv(writepath + description + file_ext)
        

# ## =============================================================================
# ##                                  TRAIN
# ## =============================================================================
    for k in range(len(train_test_sets['train_set'])):
        ##get train data and labels
        X_train = train_test_sets['train_set'][k].loc[:, train_columns]
        y_train = train_test_sets['train_set'][k]['target']
        ##get test data and labels
        X_test = train_test_sets['test_set'][k].loc[:, train_columns]
        y_test = train_test_sets['test_set'][k]['target']
        ##indicate the train test iteration
        print('\nTrain/Test Set: ', k)

        ##convert to numpy
        X_train = X_train.values
        y_train = y_train.values
        X_test = X_test.values
        y_test = y_test.values

# =============================================================================
#         Perceptron
# =============================================================================
        if args.model == 'perceptron':
            ##hyperparameters
            train_iterations = 10
            learn_rate = 0.01
            ##instantiate activation
            signum = Signum()
            ##instantiate perceptron
            perceptron = Perceptron(learn_rate, train_iterations, signum)
            ##train perceptron
            perceptron.train(X_train, y_train)
            ##make predictions
            perceptron.predict(X_test)

            print('----------Perceptron Results----------')
            if mode == 'classification':
                ##accuracy
                perceptron.accuracy(y_test, perceptron.y_pred)
                print('The Perceptron Accuracy is: ', perceptron.accuracy)
            else:
                mse = MSE()
                mse.calculate(y_test, perceptron.y_pred)
                print('The Perceptron MSE is: ', mse.mse)

## =============================================================================
##         Logistic Regression
## =============================================================================
        elif args.model == 'logistic_regression':
            ##hyperparameters
            train_iterations = 100
            learn_rate = 0.001
            ##instantiate logistic regression model
            logistic = LogisticRegression(train_iterations, learn_rate)
            ##train the logistic regression model
            logistic.train(X_train, y_train)
            ##make predictions
            predictions = logistic.predict(X_test)
            ##calculate accuracy
            accuracy = logistic.calcAccuracy(y_test)
            print('\n----------Logistic Results----------')
            print('Logistic Regression Model Accuracy: ', accuracy)

## =============================================================================
##         Feed Forward
## =============================================================================
        elif args.model == 'feedforward':

            ##hyperparameters
            train_iterations = 100
            learn_rate = 0.001

            ##instantiate activation functions
            active_fn1 = ReLU()
            active_fn2 = Tanh()
            # active_out = Sigmoid()
            active_out = Softmax()

            ##the number of classes for classification, one for regression
            if mode == 'classification':
                n_outputs = np.unique(y_test).size
            else:
                n_outputs = 1

            ##define network
            nn = NeuralNetwork([X_train.shape[1], X_train.shape[0], n_outputs], lr = learn_rate,
                               active_fn1=active_fn1, active_fn2=active_fn2,
                               active_out=active_out)

            ##print the network summary
            nn.summary()
            ##initialize weights
            nn.initializeWeights()
            ##kickoff training
            nn.train(X=X_train, y=y_train, iterations=train_iterations)

            ##use mse for the loss function here
            if mode == 'regression':
                ##print summary
                mse_dict = {'preds': [], 'truth': []}

                for (xi, yi) in zip(X_test, y_test):
                    ##truth is the label
                    truth = yi

                    ##pass prediction through step function to calculate prediction and normalize result
                    prediction = nn.predict(xi)

                    mse_dict['preds'].append(prediction[0][0])
                    mse_dict['truth'].append(yi)

                mse = nn.mse(np.array(mse_dict['truth']), np.array(mse_dict['preds']))
                print('\n----- Feed Forward Performance Summary -----')
                print('MSE: ', mse)

            ##use cross entropy as the loss function here
            if mode == 'classification':

                ##one hot encode for cross entropy
                ohy = nn.oneHot(y_test)

                ##instantiate cross entropy loss
                cross_entropy_loss = CrossEntropyLoss()

                ##book keeping to calculate the mean cross entropy loss
                ce_losses = []
                for (xi, yi) in zip(X_test, ohy):
                    ##truth is the label
                    truth = yi

                    ##debug
                    # print('xi: ', xi)
                    # print('yi: ', yi)

                    ##pass prediction through step function to calculate prediction and normalize result
                    prediction = np.absolute(nn.predict(xi))

                    ##demonstration
                    # print('prediction: ', prediction)

                    ##calculate the cross entropy loss
                    loss = cross_entropy_loss.calculate(truth, prediction)

                    ##calculate cross entropy loss
                    ce_losses.append(loss)

                ce_loss = 1 / y_test.size * sum(ce_losses)

                ##print summary
                print('\n----- Feed Forward Performance Summary -----')
                print('Cross Entropy Loss: ', ce_loss)

## =============================================================================
##                     Autoencoder
## =============================================================================
##                      Encoder
## =============================================================================

        elif args.model == 'autoencoder':
            ##hyperparameters
            train_iterations = 150
            learn_rate = 0.001
            active_fn1 = Tanh()
            active_fn2 = active_fn1
            
            #input layer is the number of features
            in_layer1 = X_train.shape[1]
            
            ##the encoder hidden layer is half of the size of the input layer
            hid_layer1 = int(in_layer1 / 2)
            
            ##the bottleneck layer downsamples to size two
            bottleneck = 2

            ##define network
            encoder = NeuralNetwork([in_layer1, hid_layer1, bottleneck], lr = learn_rate,
                               active_fn1=active_fn1, active_fn2=active_fn2,
                               active_out=None)

            ##print the network summary
            print('\n------ Encoder ------')
            encoder.summary()
            ##initialize weights
            encoder.initializeWeights()
            ##kickoff training
            encoder.train(X=X_train, y=y_test, iterations=train_iterations)

## =============================================================================
##                      Decode
## =============================================================================
            ##The input layer values are retrieved from the encoder as a transfer learning step
            X_train_transfer = encoder.weights[-1]
            
            ##the input layer for the decoder is the size of the bottleneck
            in_layer2 = bottleneck
            
            ##the hidden layer upsamples to the size of the encoder hidden layer
            hid_layer2 = hid_layer1
            
            ##finally, the output layer
            out_layer = in_layer1

            ##define network
            decoder = NeuralNetwork([in_layer2, hid_layer2, out_layer], lr = learn_rate,
                                active_fn1=active_fn1, active_fn2=active_fn2,
                                active_out=None)

            ##print the network summary
            print('\n------ Decoder ------')
            decoder.summary()
            ##initialize weights
            decoder.initializeWeights()
            ##kickoff training
            decoder.train(X=X_train_transfer, y=y_train, iterations=train_iterations)

            ##get this layer from the encoder
            layer = 0
            encoder_layer1 = encoder.weights[layer]

            ##transpose encoders first layer values to match decoder layer shape
            encoder_layer1 = encoder_layer1.T

            ##remove encoder bias column from the layer
            encoder_layer1 = np.delete(encoder_layer1, -1, 1)

            ##get last layer of the decoder
            decoder_out_layer = decoder.weights[-1]

            ##calculate mse on the features
            mse_dict = {'original': [], 'learned': []}

            for (original, learned) in zip(encoder_layer1, decoder_out_layer):

                mse_dict['original'].append(original)
                mse_dict['learned'].append(learned)

            mse = decoder.mse(np.array(mse_dict['original']), np.array(mse_dict['learned']))
            print('\n----- Feed Forward Performance Summary -----')
            print('MSE: ', mse)

        else:
            print('\nModel: ', args.model, ' does not Exist')

    toc = time.time()
    tf = round((toc - tic), 2)
    print('Total Time: ', tf)