function net = retrainAlexNet( trainingData, varargin )
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
FLAG_GPU = default.FLAG_GPU;

% Freezing the network. Freeze values:
%   0 - Do not freeze. Let all layers train.
%   1 - Partial freeze. Freeze the front layers according to literature and
%   let the back layers train.
%   2 - Full freeze. Freeze everything except for the FC layer we're adding
%   to the end for classification.
FREEZE_MODE = default.FREEZE_MODE;

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
        case 'freeze' % Freeze mode
            FREEZE_MODE = val_;
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

%% Begin loading AlexNet, replace final classification layer
alex = alexnet;
layers = alex.Layers;
myFCName = 'acc_fcout';
newLayers = getFinalFCLayer( 4096, myFCName, GPU, numClasses );
% Remove the last few layers
layers(23:25) = [];
% Add the new layers
layers = [layers; newLayers];

%% Code to freeze parts of the network
endOfNet = 22;
netMidpoint = 16;
frontLayers = 1:netMidpoint;
endLayers = (netMidpoint+1):endOfNet;
if FREEZE_MODE > 0
    layers(frontLayers) = freezeWeights(layers(frontLayers));
end
if FREEZE_MODE > 1
    layers(endLayers) = freezeWeights(endLayers);
end

%% Training step
net = trainNetwork(trainingData,layers,opts);