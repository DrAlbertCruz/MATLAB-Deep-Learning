DATA_LOCATION = 'C:\data\Salento-Grapevine-Yellows-Dataset\localized';
SAVE_LOCATION = 'localized3\';
FOLDS = 5;



crossValidate( DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e1', 'alexnet', FOLDS, 1, false );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e2', 'alexnet', FOLDS, 2, true );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e3', 'alexnet', FOLDS, 3, true );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e4', 'alexnet', FOLDS, 4, true );
crossValidate( DATA_LOCATION, SAVE_LOCATION, 'alexnet_frozen_e5', 'alexnet', FOLDS, 5, true );

