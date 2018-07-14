try
    delete(gcp)
catch e
    display( "No parpool open" );
end
pool = parpool(3);

alexnet = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/localized', '/home/acruz/MATLAB-Deep-Learning/localized2', 'AlexNet_e', @retrainAlexNet, 227 );
resnet50 = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/localized', '/home/acruz/MATLAB-Deep-Learning/localized2', 'ResNet50_e', @retrainResNet50, 224 );
squeezenet = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/localized', '/home/acruz/MATLAB-Deep-Learning/localized2', 'SqueezeNet_e', @retrainSqueezeNet, 227 );
resnet101 = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/localized', '/home/acruz/MATLAB-Deep-Learning/localized2', 'ResNet101_e', @retrainResNet101, 224 );
googlenet = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/localized', '/home/acruz/MATLAB-Deep-Learning/localized2', 'GoogLeNet_e', @retrainGoogLenet, 224 );