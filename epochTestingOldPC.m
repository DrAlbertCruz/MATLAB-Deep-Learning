DATA_LOCATION = 'C:\data\Salento-Grapevine-Yellows-Dataset\localized';
SAVE_LOCATION = 'localized3\';
FOLDS = 5;

crossValidate( DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e1', 'alexnet', FOLDS, 1, false );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet50_frozen_e1', 'resnet50', FOLDS, 1, false );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet101_frozen_e1', 'resnet101', FOLDS, 1, false );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'googlenet_frozen_e1', 'googlenet', FOLDS, 1, false );