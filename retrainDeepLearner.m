%% retrainDeepLearner
% Inputs:
%   1) Epoch range to iterate over
%   2) Input data folder
%   3) Output mat folder
%   4) Save file prefix
function retrainDeepLearner( PARAM_LIMIT_IN, ... Epoch range to it. over
                             DATA_LOCATION, ... Input data folder
                             SAVE_LOCATION, ... Save location
                             SAVE_PREFIX, ... Prefix for mat file
                             F, ... Function pointer to the deep learner
                             N ) % The size required by the deep learner
load default
clc
%DATA_LOCATION = '~/data/Salento-Grapevine-Yellows-Dataset/raw';
%SAVE_LOCATION = '~/MATLAB-Deep-Learning/Salento1/';

disp( 'Getting data ready' ), tic;
images = imageDatastore(DATA_LOCATION,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
disp( 'Full dataset' );
images.ReadFcn = @(filename)readAndPreprocessImage(filename, N);
tbl = countEachLabel( images );
minSetCount = min(tbl{:,2});
% Images are the original set

for PARAM_LIMIT = PARAM_LIMIT_IN
    [imagesInner,~] = splitEachLabel(images,minSetCount,'randomized');
    results(1).fold_results = ([]);
    for fold = 1:default.FOLDS
        tic;
        [trainingImages,validationImages] = splitEachLabel(imagesInner,0.7,'randomized');
        net = trainAFold( F, trainingImages, PARAM_LIMIT );
        predictedLabels = classify(net,validationImages);
        results(fold).fold_results = sum(predictedLabels == validationImages.Labels) ...
            / length(predictedLabels);
        results(fold).prediction = predictedLabels;
        results(fold).groundTruth = validationImages.Labels;
        results(fold).time = toc;
    end
    
    save( fullfile( SAVE_LOCATION, ...
                    [ SAVE_PREFIX num2str(PARAM_LIMIT) '.mat'] ), ...
                    'results' );
end
end

function net = trainAFold( F, images, PARAM_LIMIT )
% Training the net
disp( 'Training the network' );
net = feval( F, images, 'epoch', PARAM_LIMIT );
%net = retrainAlexNet( images, ...
%    'epoch', PARAM_LIMIT );
end