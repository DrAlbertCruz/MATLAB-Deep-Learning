function SalentoGoogLeNet( PARAM_LIMIT_IN )
load default
if nargin==0
    PARAM_LIMIT_IN = 1:40;
end
clc
DATA_LOCATION = '~/data/Salento-Grapevine-Yellows-Dataset/raw';
SAVE_LOCATION = '~/MATLAB-Deep-Learning/Salento1/';

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

for PARAM_LIMIT = PARAM_LIMIT_IN
    [imagesInner,~] = splitEachLabel(images,minSetCount,'randomized');
    results(1).fold_results = ([]);
    for fold = 1:default.FOLDS
        tic;
        [trainingImages,validationImages] = splitEachLabel(imagesInner,0.7,'randomized');
        net = trainAFold( trainingImages, PARAM_LIMIT );
        predictedLabels = classify(net,validationImages);
        results(fold).fold_results = sum(predictedLabels == validationImages.Labels) ...
            / length(predictedLabels);
        results(fold).prediction = predictedLabels;
        results(fold).groundTruth = validationImages.Labels;
        results(fold).time = toc;
    end
    
    save( fullfile( SAVE_LOCATION, ...
                    [ 'GoogLeNet_e' num2str(PARAM_LIMIT) '.mat'] ), ...
                    'results' );
end

end

function net = trainAFold( images, PARAM_LIMIT )
% Training the net
disp( 'Training the network' ), tic;
gpuDevice(1);
net = retrainGoogLenet( images, ...
    'epoch', PARAM_LIMIT );
% Doesnt work: 150
% Does work: 50
toc;
end

function Iout = readAndPreprocessImage(filename)
I = imread(filename);
% Some images may be grayscale. Replicate the image 3 times to
% create an RGB image.
if ismatrix(I)
    I = cat(3,I,I,I);
end
% Resize the image as required for the CNN.
Iout = imresize(I, [224 224]);
% Typecast into single [0,1]
%Iout = single(Iout) ./ 255;
end
