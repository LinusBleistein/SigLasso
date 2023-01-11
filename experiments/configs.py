import numpy as np


test = {
        'niter': 4,
        'n_cpu': 3, # number of cpus for parallelization
        'name': 'test_multiprocessing',
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



y_sampling = {
        'niter': 10,
        'name': 'y_sampling_y_cde',
        'data_config': {
            'model_X': 'cubic',
            'model_Y': 'cde',
            'n_train': 50,
            'n_test': 50,
            'n_val': 50,
            'n_points_true': 1000,
            'dim_X': 3,
            'dim_Y': 1,
            'non_linearity_Y': 'Tanh',
        },
        'exp_config': {
            'n_points_X': [100],
            'n_points_Y': [0, 1, 2, 3, 4, 9, 14, 19],
            'model_names': [['gru', 'lasso', 'ncde']],
            'model_hyperparams': [{
                'lasso': {
                    'sig_order': [1, 2, 3, 4, 5],
                    'normalize': True,
                    'weighted': True,
                },
                'ncde': {
                    'vector_field': 'original',
                    'num_epochs': 30,
                    'lr': 1e-3},
                'gru': {
                    'gru_width': 128,
                    'num_epochs': 100,
                    'lr': 1e-3},
                }]
            }
        }

x_sampling = {
        'niter': 1,
        'name': 'x_sampling',
        'data_config': {
            'model_X': 'cubic',
            'model_Y': 'cde',
            'n_train': 200,
            'n_test': 200,
            'n_val': 200,
            'n_points_true': 1000,
            'dim_X': 3,
            'dim_Y': 1,
            'non_linearity_Y': 'Tanh',
        },
        'exp_config': {
            'n_points_X': [5,10,20,30,50],
            'n_points_Y': [0],
            'model_names': [['lasso', 'ncde', 'gru']],
            'model_hyperparams': [{
                'lasso': {
                    'sig_order': [1, 2, 3, 4],
                    'normalize': True,
                    'weighted': True,
                },
                'ncde': {
                    'vector_field': 'original',
                    'num_epochs': 30,
                    'lr': 1e-3},
                'gru': {
                    'gru_width': 128,
                    'num_epochs': 100,
                    'lr': 1e-3},
                }]
            }
        }

dimension_test = {
        'niter': 1,
        'name': 'dimension_test',
        'data_config': {
            'model_X': 'cubic',
            'model_Y': 'cde',
            'n_train': 200,
            'n_test': 200,
            'n_val': 200,
            'n_points_true': 1000,
            'dim_X': [3,4,5,6],
            'dim_Y': 1,
            'non_linearity_Y': 'Tanh',
        },
        'exp_config': {
            'n_points_X': [20],
            'n_points_Y': [0],
            'model_names': [['lasso', 'ncde', 'gru']],
            'model_hyperparams': [{
                'lasso': {
                    'sig_order': [1, 2, 3, 4],
                    'normalize': True,
                    'weighted': True,
                },
                'ncde': {
                    'vector_field': 'original',
                    'num_epochs': 30,
                    'lr': 1e-3},
                'gru': {
                    'gru_width': 128,
                    'num_epochs': 100,
                    'lr': 1e-3},
                }]
            }
        }

ornstein_uhlenbeck = {
        'niter': 1,
        'name': 'ou_test',
        'data_config': {
            'model_X': 'brownian',
            'model_Y': 'ornstein_uhlenbeck',
            'n_train': 200,
            'n_test': 100,
            'n_val': 100,
            'n_points_true': 1000,
            'dim_X': 3,
            'dim_Y': 1,
            'non_linearity_Y': 'Tanh',
        },
        'exp_config': {
            'n_points_X': [100],
            'n_points_Y': [0, 1, 2, 3, 4, 20,50,100],
            'model_names': [['lasso', 'ncde', 'gru']],
            'model_hyperparams': [{
                'lasso': {
                    'sig_order': [1, 2, 3, 4],
                    'normalize': True,
                    'weighted': True,
                },
                'ncde': {
                    'vector_field': 'original',
                    'num_epochs': 30,
                    'lr': 1e-3},
                'gru': {
                    'gru_width': 128,
                    'num_epochs': 100,
                    'lr': 1e-3},
                }]
            }
        }
