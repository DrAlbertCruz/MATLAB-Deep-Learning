clear all
close all
clc

DATA_LOCATION = '~/data/Salento-Grapevine-Yellows-Dataset/localized';
SAVE_LOCATION = '~/MATLAB-Deep-Learning/localized3/';
FOLDS = 5;

% Resnet101 jobs have to be done sequentially because of how much memory
% they take up
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet101_frozen_e1', 'resnet101', FOLDS, 1, true );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet101_frozen_e1', 'resnet101', FOLDS, 2, true );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet101_frozen_e1', 'resnet101', FOLDS, 3, true );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet101_frozen_e1', 'resnet101', FOLDS, 4, true );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet101_frozen_e1', 'resnet101', FOLDS, 5, true );

try
    delete(gcp)
catch e
    display( "No parpool open" );
end
pool = parpool(2);

alexnet_1 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e1', 'alexnet', FOLDS, 1, true );
alexnet_2 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e2', 'alexnet', FOLDS, 2, true );
alexnet_3 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e3', 'alexnet', FOLDS, 3, true );
alexnet_4 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e4', 'alexnet', FOLDS, 4, true );
alexnet_5 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e5', 'alexnet', FOLDS, 5, true );
resnet50_1 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'resnet50_frozen_e1', 'resnet50', FOLDS, 1, true );
resnet50_2 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'resnet50_frozen_e2', 'resnet50', FOLDS, 2, true );
resnet50_3 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'resnet50_frozen_e3', 'resnet50', FOLDS, 3, true );
resnet50_4 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'resnet50_frozen_e4', 'resnet50', FOLDS, 4, true );
resnet50_5 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'resnet50_frozen_e5', 'resnet50', FOLDS, 5, true );
googlenet_1 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'googlenet_frozen_e1', 'googlenet', FOLDS, 1, true );
googlenet_2 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'googlenet_frozen_e2', 'googlenet', FOLDS, 2, true );
googlenet_3 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'googlenet_frozen_e3', 'googlenet', FOLDS, 3, true );
googlenet_4 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'googlenet_frozen_e4', 'googlenet', FOLDS, 4, true );
googlenet_5 = parfeval( pool, @crossValidate, 0, DATA_LOCATION, SAVE_LOCATION, 'googlenet_frozen_e5', 'googlenet', FOLDS, 5, true );
