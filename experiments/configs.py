import numpy as np


config_test = {
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
            'model_names': [['lasso']],
            'model_hyperparams': [{
                'lasso': {
                    'alpha_grid': 10 ** np.linspace(-7, 1, 5),
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