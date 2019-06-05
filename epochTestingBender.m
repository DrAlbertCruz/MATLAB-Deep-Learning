clear all
close all
clc

DATA_LOCATION = '~/data/Salento-Grapevine-Yellows-Dataset/localized';
SAVE_LOCATION = '~/MATLAB-Deep-Learning/localized3/';
FOLDS = 5;

% Resnet101 jobs have to be done sequentially because of how much memory
% they take up
for net = { 'squeezenet', 'inceptionv3', 'resnet50', 'resnet101', 'googlenet', 'alexnet' }
    for epoch = 6:40
        SAVE_NAME = [ cell2mat(net), '_frozen_e', num2str(epoch) ];
        NET_NAME = cell2mat(net);
        crossValidate( DATA_LOCATION, SAVE_LOCATION, SAVE_NAME, NET_NAME, FOLDS, 1, true );
    end
end