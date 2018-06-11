function t_ashrafGrapesProjectAlexNet_getNet
PARAM_LIMIT = 30;
DATA_LOCATION = 'G:\data\Pierces-Disease-Grapes-2017\localized';
SAVE_FOLDER = 'G:\MATLAB-Deep-Learning\UCCoopExt\';

disp( 'Getting data ready' ), tic;
images = imageDatastore(DATA_LOCATION,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
disp( 'Full dataset' );
images.ReadFcn = @(filename)readAndPreprocessImage(filename);
tbl = countEachLabel( images );
minSetCount = min(tbl{:,2});
% Images are the original set
[images,~] = splitEachLabel(images,minSetCount,'randomized');

% for PARAM_LIMIT = PARAM_LIMIT_IN
% results(1).fold_results = ([]);
% for fold = 1:5
    tic;
%     [trainingImages,validationImages] = splitEachLabel(images,0.7,'randomized');
    results.net = trainAFold( images, PARAM_LIMIT );
    predictedLabels = classify(results.net,images);
    results.fold_results = sum(predictedLabels == images.Labels) ...
        / length(predictedLabels);
    results.prediction = predictedLabels;
    results.groundTruth = images.Labels;
    results.time = toc;

save( ...
    fullfile( SAVE_FOLDER, [ 'alexNet-net-final-' num2str(PARAM_LIMIT) '.mat'] ), ...
    'results' );
t_mailtest( [ 'VICTORY JOB SCHEDULER: AlexNet done with epoch ' num2str( PARAM_LIMIT ) ], 'VICTORY JOB SCHEDULER: Progress notification' );
% end
end

function net = trainAFold( images, PARAM_LIMIT )
% Training the net
disp( 'Training the network' );
net = retrainAlexNet( images, ...
    'epoch', PARAM_LIMIT, ...
    'miniBatchSize', fix(150*2.5) );
end

function Iout = readAndPreprocessImage(filename)

I = imread(filename);

% Some images may be grayscale. Replicate the image 3 times to
% create an RGB image.
if ismatrix(I)
    I = cat(3,I,I,I);
end

% % From the top
% % Maintain aspect ratio by clipping from the bottom
% if size(I,1) > size(I,2)
%     I = I( 1:size(I,2), :, : );
% end

% Resize the image as required for the CNN.
Iout = imresize(I, [227 227]);

% Typecast into single [0,1]
Iout = single(mat2gray(Iout));

end