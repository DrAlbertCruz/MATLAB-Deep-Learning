%% makeDefaults.m - Make the defaults values and flavor text
%   This function will set some default values and provides a central place
%   to control flavor text
clear all
close all
clc

%% Flavor text
default.msgNumArgs = 'This function requires one or more arguments; the training data, followed by a variable number of argument-value pairs';

%% Default values
default.EPOCH = 5;                          % Epoch limit for training
default.MINIBATCH_SIZE = 10;     .001           % Mini-batch on each it.
default.INITIAL_LEARNING_RATE = 0.001;      % Learning rate
default.L2_REGULARIZATION = 0.0001;         % Regularization
default.FLAG_SHUFFLE = true;                % Shuffle the samples on epoch
default.VERBOSE_FREQUENCY = 10;             % Number of iterations to display by default
default.FLAG_GPU = true;                    % Whether or not to use the GPU
default.FREEZE_MODE = 2;                    % Amount of freezing
default.NETWORK_NAME = 'alexnet';           % Default name for the network

default.optimizer = 'sgdm';                 % Training method used in training options

% Parameters for final fully connected layer used in classification
default.WeightLearnRateFactor = 20;
default.WeightL2Factor = 1;
default.BiasLearnRateFactor = 20;
default.BiasL2Factor = 0;

default.FOLDS = 3;

save default