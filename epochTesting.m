clear all

DATA_LOCATION = 'C:\data\Salento-Grapevine-Yellows-Dataset\localized';
SAVE_LOCATION = 'localized3\';

for netArch = { 'alexnet', 'resnet50', 'resnet101', 'googlenet' }
    for epochLimit = 1:5
        disp( [ cell2mat( netArch ), ': Starting simulation ', num2str(epochLimit) ] );
        
        SAVE_PREFIX = [ cell2mat(netArch), '_fullfrozen_e_', num2str(epochLimit) ];
        
        if strcmp( cell2mat(netArch), 'alexnet' )
            N = 227;
        else
            N = 224; % Size of image
        end
        
        % Load default values
        load default
        
        images = imageDatastore(DATA_LOCATION,...
            'IncludeSubfolders',true,...
            'LabelSource','foldernames');
        
        % Set image read function
        images.ReadFcn = @(filename)readAndPreprocessImage(filename, N);
        
        % Randomly create a dataset with equal priors
        tbl = countEachLabel( images );
        minSetCount = min(tbl{:,2});
        [imagesInner,~] = splitEachLabel(images,minSetCount,'randomized');
        
        tic;
        [trainingImages,validationImages] = splitEachLabel(imagesInner,0.7,'randomized');
        
        % Train the network
        net = trainDeepLearner( trainingImages, ...
            'epoch', epochLimit, ...
            'network', cell2mat(netArch), ...
            'miniBatchSize', 20, ...
            'freeze', 2 );
        
        predictedLabels = classify(net,validationImages);
        
        results(epochLimit).fold_results = sum(predictedLabels == validationImages.Labels) ...
            / length(predictedLabels);
        results(epochLimit).prediction = predictedLabels;
        results(epochLimit).groundTruth = validationImages.Labels;
        results(epochLimit).time = toc;
    end
    
    save( fullfile( SAVE_LOCATION, [ SAVE_PREFIX, '.mat' ] ), ...
        'results' );
end