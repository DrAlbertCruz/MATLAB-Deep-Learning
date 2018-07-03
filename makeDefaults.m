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
default.MINIBATCH_SIZE = 64;                % Mini-batch on each it.
default.INITIAL_LEARNING_RATE = 0.01;       % Learning rate
default.L2_REGULARIZATION = 0.0001;         % Regularization
default.FLAG_SHUFFLE = true;                % Shuffle the samples on epoch
default.VERBOSE_FREQUENCY = 1;              % Number of iterations to display by default
default.FLAG_GPU = true;                    % Whether or not to use the GPU

default.optimizer = 'adam';                 % Training method used in training options

% Parameters for final fully connected layer used in classification
default.WeightLearnRateFactor = 5;
default.WeightL2Factor = 1;
default.BiasLearnRateFactor = 5;
default.BiasL2Factor = 0;

save default