try
    delete(gcp)
catch e
    display( "No parpool open" );
end
pool = parpool(3);
%F1 = parfeval( pool, @retrainDeepLearner, 0, 1:40,
%'~/data/Salento-Grapevine-Yellows-Dataset/raw',
%'/home/acruz/MATLAB-Deep-Learning/Salento3', 'InceptionResnetV2_e',
%@retrainInceptionResnetV2, 299 ); Cancelled
%F2 = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/raw', '/home/acruz/MATLAB-Deep-Learning/Salento3', 'AlexNet_e', @retrainAlexNet, 227 );
%F3 = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/raw', '/home/acruz/MATLAB-Deep-Learning/Salento3', 'ResNet50_e', @retrainResNet50, 224 );
%F4 = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/raw', '/home/acruz/MATLAB-Deep-Learning/Salento3', 'GoogLeNet_e', @retrainGoogLenet, 224 );
%F5 = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/raw', '/home/acruz/MATLAB-Deep-Learning/Salento3', 'SqueezeNet_e', @retrainSqueezeNet, 227 );
%F6 = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/raw', '/home/acruz/MATLAB-Deep-Learning/Salento3', 'ResNet101_e', @retrainResNet101, 224 );
% F7 = parfeval( pool, @retrainDeepLearner, 0, 1:40,
% '~/data/Salento-Grapevine-Yellows-Dataset/localized',
% '/home/acruz/MATLAB-Deep-Learning/localized', 'InceptionResnetV2_e',
% @retrainInceptionResnetV2, 299 ); %Exception, cancelled
%F8 = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/localized', '/home/acruz/MATLAB-Deep-Learning/localized', 'AlexNet_e', @retrainAlexNet, 227 );
F9 = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/localized', '/home/acruz/MATLAB-Deep-Learning/localized', 'ResNet50_e', @retrainResNet50, 224 );
%F10 = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/localized', '/home/acruz/MATLAB-Deep-Learning/localized', 'GoogLeNet_e', @retrainGoogLenet, 224 );
%F11 = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/localized', '/home/acruz/MATLAB-Deep-Learning/localized', 'SqueezeNet_e', @retrainSqueezeNet, 227 );
F12 = parfeval( pool, @retrainDeepLearner, 0, 1:40, '~/data/Salento-Grapevine-Yellows-Dataset/localized', '/home/acruz/MATLAB-Deep-Learning/localized', 'ResNet101_e', @retrainResNet101, 224 );