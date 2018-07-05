try
    delete(gcp)
catch e
    display( "No parpool open" );
end
pool = parpool(3);
parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/raw', '/home/acruz/MATLAB-Deep-Learning/Salento2', 'InceptionResnetV2_e', @retrainInceptionResnetV2, 299 );
parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/raw', '/home/acruz/MATLAB-Deep-Learning/Salento2', 'AlexNet_e', @retrainAlexNet, 227 );
parfeval( pool, @retrainDeepLearner, 0, 13:40, '~/data/Salento-Grapevine-Yellows-Dataset/raw', '/home/acruz/MATLAB-Deep-Learning/Salento2', 'ResNet50_e', @retrainResNet50, 224 );
parfeval( pool, @retrainDeepLearner, 0, 22:40, '~/data/Salento-Grapevine-Yellows-Dataset/raw', '/home/acruz/MATLAB-Deep-Learning/Salento2', 'GoogLeNet_e', @retrainGoogLenet, 224 );
parfeval( pool, @retrainDeepLearner, 0, 24:40, '~/data/Salento-Grapevine-Yellows-Dataset/raw', '/home/acruz/MATLAB-Deep-Learning/Salento2', 'AlexNet_e', @retrainSqueezeNet, 227 );
parfeval( pool, @retrainDeepLearner, 0, 8:40, '~/data/Salento-Grapevine-Yellows-Dataset/raw', '/home/acruz/MATLAB-Deep-Learning/Salento2', 'ResNet101_e', @retrainResNet101, 224 );