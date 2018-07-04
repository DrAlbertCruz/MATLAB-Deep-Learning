%% getFinalFCLayer
%   Get the output layer of the network. Creates a fully-connected layer
%   followed by a softmax activation and a classification layer.
function layers =  getFinalFCLayer( inputSize, myFCName, GPU, numClasses )
load default
finalFCLayer = fullyConnectedLayer(numClasses,'Name',myFCName);
finalFCLayerInputSize = inputSize;
if ~GPU
    finalFCLayer.Weights = single(randn([numClasses finalFCLayerInputSize])*0.0001);
    finalFCLayer.Bias = single(randn([numClasses 1])*0.0001);
else
    finalFCLayer.Weights = gpuArray(single(randn([numClasses finalFCLayerInputSize])*0.0001));
    finalFCLayer.Bias = gpuArray(single(randn([numClasses 1])*0.0001));
end
finalFCLayer.WeightLearnRateFactor = default.WeightLearnRateFactor;
finalFCLayer.WeightL2Factor = default.WeightL2Factor;
finalFCLayer.BiasLearnRateFactor = default.BiasLearnRateFactor;
finalFCLayer.BiasL2Factor = default.BiasL2Factor;
layers = [ finalFCLayer; ...
           softmaxLayer('Name','acc_fcout_softmax'); ...
           classificationLayer('Name','acc_fcout_output') ];