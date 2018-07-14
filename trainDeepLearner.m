%% trainDeepLearner Train a deep learning neural network
%   trainDeepLearner( dataset ) trains a deep learning algorithm on the
%   image datastore object 'dataset'.
%
%   trainDeepLearner(___,PARAM1,VAL1,PARAM2,VAL2,___) trains a deep
%   learning algorithm with varying parameters. Parameters names cannot be
%   abbreviated, and case matters.
%
%   Parameters include:
%
%   'gpu'
%   Flag for enabling training on GPU
%
%   'network'
%   Specify the neural network architecture. Valid architectures are:
%   'alexnet' (default), 'googlenet', 'resnet50', 'resnet101'

function net = trainDeepLearner( trainingData, varargin )
load default                                            % Load defaults
numClasses = length(unique( trainingData.Labels ));     % Num classes
%% Input validation
if ~mod( nargin, 2 )
    error( default.msgNumArgs );
end
%% DEFAULT VALUES
% As of 7/14 we do not use the 'makeDefaults' file
EPOCH = 5;
MINIBATCH_SIZE = 10;
INITIAL_LEARNING_RATE = 0.001;
L2_REGULARIZATION = 0.0001;
FLAG_SHUFFLE = true;
VERBOSE_FREQUENCY = 1;
FLAG_GPU = true;
% Make sure these two are correct. Alexnet is not a DAG.
NETWORK = 'alexnet';
IS_DAG = false;
OPTIMIZER = 'sgdm';

% Freezing the network. Freeze values:
%   0 - Do not freeze. Let all layers train.
%   1 - Partial freeze. Freeze the front layers according to literature and
%   let the back layers train.
%   2 - Full freeze. Freeze everything except for the FC layer we're adding
%   to the end for classification.
FREEZE_MODE = default.FREEZE_MODE;

%% VARARGIN SWITCH
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
        case 'network'
            NETWORK = val_;
        case 'optimizer'
            OPTIMIZER = val_;
        otherwise
            error( 'Invalid vararg pair!' );
    end
end

%% SET SWITCHES, FLAGS, AND DO INPUT VALIDATION
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
% Recurrent nets need to set 'is_dag'
if strcmp( NETWORK, 'googlenet' ) || ...
        strcmp( NETWORK, 'resnet50' ) || ...
        strcmp( NETWORK, 'resnet101' )
    IS_DAG = true;
end

%% SET TRAINING OPTIONS
opts = trainingOptions( OPTIMIZER, ...
    'InitialLearnRate', INITIAL_LEARNING_RATE, ...
    'L2Regularization', L2_REGULARIZATION, ...
    'MaxEpochs', EPOCH, ...
    'MiniBatchSize', MINIBATCH_SIZE, ...
    'Verbose', true, ...
    'VerboseFrequency', VERBOSE_FREQUENCY, ...
    'Shuffle', SHUFFLE, ...
    'ExecutionEnvironment', GPU ...
    );

%% LOAD NETWORK AND ADD FC LAYERS TO TOP
myFCName = 'acc_fcout';
if strcmp( NETWORK, 'alexnet' )
    % Load network
    alex = alexnet;
    layers = alex.Layers;
    clear alex
    % Structure network
    newLayers = getFinalFCLayer( 4096, myFCName, GPU, numClasses );
    layers(23:25) = []; % Remove the last few layers
    layers = [layers; newLayers]; % Add the new layers
    clear newLayers
    % Midpoint is the start of the FC layers
    netMidpoint = 16;
    endOfNet = 22;
elseif strcmp( NETWORK, 'googlenet' )
    % Load network
    net = googlenet;
    lgraph = layerGraph(net);
    clear googlenet
    % Remove the classification layers
    lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
    % Structure network
    newLayers =  getFinalFCLayer( 1024, myFCName, GPU, numClasses );
    lgraph = addLayers(lgraph,newLayers); % Add the layers to the graph
    clear newLayers
    connection_point = 'pool5-drop_7x7_s1'; % Name of connection point
    % MATLAB tutorial states that for half-way freezing you freeze
    % everything up to and including inception_5
    netMidpoint = 111;
    endOfNet = 141;
elseif strcmp( NETWORK, 'resnet50' )
    net = resnet50;
    lgraph = layerGraph(net);
    clear resnet50
    % Remove the classification layers
    lgraph = removeLayers(lgraph, {'fc1000','fc1000_softmax','ClassificationLayer_fc1000'});
    newLayers =  getFinalFCLayer( 2048, myFCName, GPU, numClasses );
    lgraph = addLayers(lgraph,newLayers); % Add the layers to the graph
    connection_point = 'avg_pool';
    % For a partial freeze, we freeze up to 80% of the network. This is
    % approximately up to the last three modules
    netMidpoint = 140;
    % There are 174 non-output layers in the network
    endOfNet = 174;
elseif strcmp( NETWORK, 'resnet101' )
    net = resnet101;
    lgraph = layerGraph(net);
    clear resnet101
    % Remove the classification layers
    lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});
    newLayers =  getFinalFCLayer( 2048, myFCName, GPU, numClasses );
    lgraph = addLayers(lgraph,newLayers); % Add the layers to the graph
    connection_point = 'pool5';
    % For a partial freeze, we freeze up to 80% of the network. For
    % resnet101 this is approximately up to the 7th module (not inclusive
    % of res4b_18)
    netMidpoint = 269;
    % There are 174 non-output layers in the network
    endOfNet = 344;
end

%% CODE TO FREEZE A PART OF THE NETWORK
% If it is a DAG, need to separate layers and connections
if IS_DAG
    % This is where the DAG networks are joined
    lgraph = connectLayers(lgraph,connection_point,myFCName);
    % Need to separate layers and connections to run freeze code
    layers = lgraph.Layers;
    connections = lgraph.Connections;
end
% Indexing for front and end part of net
frontLayers = 1:netMidpoint;
endLayers = (netMidpoint+1):endOfNet;
% Now freeze, based on freeze level
if FREEZE_MODE > 0
    layers(frontLayers) = freezeWeights(layers(frontLayers));
end
if FREEZE_MODE > 1
    layers(endLayers) = freezeWeights(layers(endLayers));
end
clear endOfNet netMidpoint

%% TRAINING STEP
if IS_DAG
    % If this is a DAG network, we need to connect it first before sending
    % to 'trainNetwork'
    lgraph = createLgraphUsingConnections(layers,connections);
    net = trainNetwork(trainingData,lgraph,opts);
else
    net = trainNetwork(trainingData,layers,opts);
end