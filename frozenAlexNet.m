%% frozenAlexNet
%   Retrain AlexNet, but freeze some of the initial layers
function net = frozenAlexNet( trainingData, varargin )
load default                                            % Load defaults
numClasses = length(unique( trainingData.Labels ));     % Num classes
%% Input validation
if ~mod( nargin, 2 )
    error( default.msgNumArgs );
end
%% Default arguments
EPOCH = default.EPOCH;
MINIBATCH_SIZE = default.MINIBATCH_SIZE;
INITIAL_LEARNING_RATE = default.INITIAL_LEARNING_RATE;
L2_REGULARIZATION = default.L2_REGULARIZATION;
FLAG_SHUFFLE = default.FLAG_SHUFFLE;
VERBOSE_FREQUENCY = default.VERBOSE_FREQUENCY;
FLAG_GPU = false;   % 7/4 Fix this when using on Bender
for i=1:2:length(varargin)
    arg_ = cell2mat(varargin(i));
    val_ = cell2mat(varargin(i+1));
    switch arg_
        case 'epoch'
            EPOCH = val_;
        case 'miniBatchSize'
            MINIBATCH_SIZE = val_;
        case 'initialLearningRate'
            INITIAL_LEARNING_RATE = val_;
        case 'L2Regularization'
            L2_REGULARIZATION = val_;
        case 'shuffle'
            FLAG_SHUFFLE = val_;
        case 'verbosity'
            VERBOSE_FREQUENCY = val_;
        case 'gpu'
            FLAG_GPU = val_;
        otherwise
            error( 'Invalid vararg pair!' );
    end
end
%% Switches for converting flags to strings if needed
% Set the flag to shuffle the data on each epoch. Unused: 'once'.
if FLAG_SHUFFLE == true
    SHUFFLE = 'every-epoch';
else
    SHUFFLE = 'never';
end
% Set the flag to use the gpu or the cpu. Unused: multi-worker
if FLAG_GPU == true
    GPU = 'gpu';
else
    GPU = 'cpu';
end
%% Set training options
opts = trainingOptions( default.optimizer, ...
    'InitialLearnRate', INITIAL_LEARNING_RATE, ...
    'L2Regularization', L2_REGULARIZATION, ...
    'MaxEpochs', EPOCH, ...
    'MiniBatchSize', MINIBATCH_SIZE, ...
    'Verbose', true, ...
    'VerboseFrequency', VERBOSE_FREQUENCY, ...
    'Shuffle', SHUFFLE, ...
    'ExecutionEnvironment', GPU ...
    );
%% Replace final classification layer
alex = alexnet;
layers = alex.Layers;
myFCName = 'acc_fcout';
% The following code will 'freeze' the network by setting the learning rate
% factor to 0.
% Conv layers: 2, 6, 10, 12, 14
% FC layers: 17, 20
% Classification: 23
for layer = [2, 6, 10, 12, 14, 17, 20]
        layers(layer).WeightLearnRateFactor = 0;
        layers(layer).WeightL2Factor = 0;
        layers(layer).BiasLearnRateFactor = 0;
        layers(layer).BiasL2Factor = 0;
end
% Output of FC at 20 is 4096. Use function getFinalFCLayer to get the final
% set of classification layers
newlayers = getFinalFCLayer( 4096, myFCName, FLAG_GPU, numClasses );
% The following code replaces the layers in the old network
layers(23) = newlayers(1);
layers(24) = newlayers(2);
layers(25) = newlayers(3);
%% Training step
net = trainNetwork(trainingData,layers,opts);