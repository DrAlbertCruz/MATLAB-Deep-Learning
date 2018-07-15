DATA_LOCATION = 'C:\data\Salento-Grapevine-Yellows-Dataset\localized';
SAVE_LOCATION = 'localized3\';
FOLDS = 5;

crossValidate( DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e1', 'alexnet', FOLDS, 1 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e2', 'alexnet', FOLDS, 2 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e3', 'alexnet', FOLDS, 3 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e4', 'alexnet', FOLDS, 4 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e5', 'alexnet', FOLDS, 5 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet50_frozen_e1', 'resnet50', FOLDS, 1 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet50_frozen_e2', 'resnet50', FOLDS, 2 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet50_frozen_e3', 'resnet50', FOLDS, 3 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet50_frozen_e4', 'resnet50', FOLDS, 4 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet50_frozen_e5', 'resnet50', FOLDS, 5 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet101_frozen_e1', 'resnet101', FOLDS, 1 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet101_frozen_e2', 'resnet101', FOLDS, 2 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet101_frozen_e3', 'resnet101', FOLDS, 3 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet101_frozen_e4', 'resnet101', FOLDS, 4 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'resnet101_frozen_e5', 'resnet101', FOLDS, 5 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'googlenet_frozen_e1', 'googlenet', FOLDS, 1 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'googlenet_frozen_e2', 'googlenet', FOLDS, 2 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'googlenet_frozen_e3', 'googlenet', FOLDS, 3 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'googlenet_frozen_e4', 'googlenet', FOLDS, 4 );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'googlenet_frozen_e5', 'googlenet', FOLDS, 5 );