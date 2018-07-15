DATA_LOCATION = '~/data/Salento-Grapevine-Yellows-Dataset/localized';
SAVE_LOCATION = '~/MATLAB-Deep-Learning/localized3/';
FOLDS = 5;



crossValidate( DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e1', 'alexnet', FOLDS, 1, true );
