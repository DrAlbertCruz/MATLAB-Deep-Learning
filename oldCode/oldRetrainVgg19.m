%% retrainVgg16.m
%   A function to load a pre-trained VGG-16 CNN and retrain it on a given
%   dataset.
function net = retrainVgg19( trainingData, varargin )
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
%% vararg switch
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
alex = vgg19;
layers = alex.Layers;
myFCName = 'acc_fcout';
layers_ =  getFinalFCLayer( 4096, myFCName, GPU, numClasses );
layers(39) = layers_(1);
layers(40) = layers_(2);
layers(41) = layers_(3);
%% Training step
net = trainNetwork(trainingData,layers,opts);