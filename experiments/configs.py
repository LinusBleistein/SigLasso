import numpy as np


test = {
        'niter': 1,
        'name': 'test',
        'data_config': {
            'model_X': 'cubic',
            'model_Y': 'cde',
            'n_train': 10,
            'n_test': 10,
            'n_val': 10,
            'n_points_true': 500,
            'dim_X': 3,
            'dim_Y': 1,
            'non_linearity_Y': 'Tanh',
        },
        'exp_config': {
            'n_points_X': [100], #TODO: check somewhere that n_points_X > n_points_true +2
            'n_points_Y': [3],
            'model_names': [['ncde', 'lasso', 'gru']],
            'model_hyperparams': [{
                'lasso': {
                    'alpha_grid': 10 ** np.linspace(-5, 5, 50),
                    'sig_order': [3],
                    'normalize': True,
                    'weighted': True,
                },
                'ncde': {
                    'vector_field': 'original',
                    'num_epochs': 1,
                    'lr': 1e-3},
                'gru': {
                    'gru_width': 128,
                    'num_epochs': 5,
                    'lr': 1e-3},
                }]
            }
        }


grid_n_points_Y = {
        'niter': 1,
        'name': 'test',
        'data_config': {
            'model_X': 'cubic',
            'model_Y': 'cde',
            'n_train': 50,
            'n_test': 50,
            'n_val': 50,
            'n_points_true': 1000,
            'dim_X': 3,
            'dim_Y': 2,
            'non_linearity_Y': 'Tanh',
        },
        'exp_config': {
            'n_points_X': [100], #TODO: check somewhere that n_points_X > n_points_true +2
            'n_points_Y': [0, 1, 2, 3, 4, 5],
            'model_names': [['ncde', 'lasso', 'gru']],
            'model_hyperparams': [{
                'lasso': {
                    'alpha_grid': 10 ** np.linspace(-5, 5, 50),
                    'sig_order': [1, 2, 3, 4, 5],
                    'normalize': True,
                    'weighted': True,
                },
                'ncde': {
                    'vector_field': 'original',
                    'num_epochs': 25,
                    'lr': 1e-3},
                'gru': {
                    'gru_width': 128,
                    'num_epochs': 50,
                    'lr': 1e-3},
                }]
            }
        }
