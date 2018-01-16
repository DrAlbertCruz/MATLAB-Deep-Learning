%% Description
%   Generate one MHI image for each video.
function net = retrainAlexNet( trainingData, varargin )
if ~mod( nargin, 2 )
    error( 'Variable arguments must be name and value pairs!' );
end
%% Default arguments
EPOCH = 5;
MINIBATCH_SIZE = 100;
INITIAL_LEARNING_RATE = 0.01;
L2_REGULARIZATION = 0.0001;
FLAG_SHUFFLE = true;
VERBOSE_FREQUENCY = 10;
FLAG_GPU = true;
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
%% Fixed/other arguments
numClasses = length(unique( trainingData.Labels ));
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
opts = trainingOptions( 'sgdm', ...
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
finalFCLayer = fullyConnectedLayer(numClasses,'Name',myFCName);
finalFCLayerInputSize = 4096;
finalFCLayer.Weights = gpuArray(single(randn([numClasses finalFCLayerInputSize])*0.0001));
finalFCLayer.Bias = gpuArray(single(randn([numClasses 1])*0.0001));
finalFCLayer.WeightLearnRateFactor = 1;
finalFCLayer.WeightL2Factor = 1;
finalFCLayer.BiasLearnRateFactor = 1;
finalFCLayer.BiasL2Factor = 0;
layers(23) = finalFCLayer;
layers(24) = softmaxLayer('Name','acc_fcout_softmax');
layers(25) = classificationLayer('Name','acc_fcout_output');
%% Training step
net = trainNetwork(trainingData,layers,opts);