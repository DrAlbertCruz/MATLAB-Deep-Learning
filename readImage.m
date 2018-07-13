function Iout = readImage(filename, n)
I = imread(filename);
% Some images may be grayscale. Replicate the image 3 times to
% create an RGB image.
if ismatrix(I)
    I = cat(3,I,I,I);
end
% Resize the image as required for the CNN.
Iout = imresize(I, [n n]);
% Typecast into single [0,1]
Iout = single(Iout)./255;
end