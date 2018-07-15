function crossValidate( DATA_LOCATION, ... Location to load images from
    SAVE_LOCATION, ... Folder where to save results file
    SAVE_PREFIX, ... Folder where to save results file
    NET_ARCH, ... Network architecture
    FOLDS, ... Number of random folds
    EPOCH, ... Number of epochs. Default is 5.
    USE_GPU ) % Whether or not to use the GPU 

%% INITIALIZATION PARAMS
if strcmp( NET_ARCH, 'alexnet' )
    N = 227;
else
    N = 224; % Size of image
end
% Image datastore
images = imageDatastore(DATA_LOCATION,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
% Set image read function
images.ReadFcn = @(filename)readAndPreprocessImage(filename, N);
% The following used when randomly creating folds with equal priors
tbl = countEachLabel( images );
minSetCount = min(tbl{:,2});

%% FOLD ITERATION LOOP
for foldNo = 1:FOLDS
    % Randomly create a dataset with equal priors
    [imagesInner,~] = splitEachLabel(images,minSetCount,'randomized');
    
    tic;
    [trainingImages,validationImages] = splitEachLabel(imagesInner,0.7,'randomized');
    
    % Train the network
    net = trainDeepLearner( trainingImages, ...
        'epoch', EPOCH, ...
        'network', NET_ARCH, ...
        'miniBatchSize', 25, ...
        'freeze', 2, ...
        'gpu', USE_GPU );
    
    predictedLabels = classify(net,validationImages);
    
    results(foldNo).fold_results = sum(predictedLabels == validationImages.Labels) ...
        / length(predictedLabels);
    results(foldNo).prediction = predictedLabels;
    results(foldNo).groundTruth = validationImages.Labels;
    results(foldNo).time = toc;
end

save( fullfile( SAVE_LOCATION, [ SAVE_PREFIX, '.mat' ] ), ...
    'results' );