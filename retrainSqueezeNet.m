function net = retrainSqueezeNet( trainingData, varargin )
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
%% Begin loading AlexNet, replace final classification layer
net = squeezenet;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'prob','ClassificationLayer_predictions'});
% Initialize the final FC layer as best we can
myFCName = 'acc_fcout';
finalFCLayer = fullyConnectedLayer(numClasses,'Name',myFCName);
finalFCLayerInputSize = 1000; % Output of pool10 in SqueezeNet is 1x1x1000
finalFCLayer.Weights = gpuArray(single(randn([numClasses finalFCLayerInputSize])*0.0001));
finalFCLayer.Bias = gpuArray(single(randn([numClasses 1])*0.0001));
finalFCLayer.WeightLearnRateFactor = default.WeightLearnRateFactor;
finalFCLayer.WeightL2Factor = default.WeightL2Factor;
finalFCLayer.BiasLearnRateFactor = default.BiasLearnRateFactor;
finalFCLayer.BiasL2Factor = default.BiasL2Factor;
% Create the new layers
newLayers = [
    finalFCLayer;
    softmaxLayer('Name','acc_fcout_softmax');
    classificationLayer('Name','acc_fcout_output')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool10',myFCName);
%% Training step
net = trainNetwork(trainingData,lgraph,opts);