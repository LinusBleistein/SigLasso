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
            'n_points_X': [100],
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
                    'num_epochs': 30,
                    'lr': 1e-3},
                'gru': {
                    'gru_width': 128,
                    'num_epochs': 30,
                    'lr': 1e-3},
                }]
            }
        }

y_sampling = {
        'niter': 10,
        'name': 'y_sampling',
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
            'model_names': [['ncde', 'gru', 'lasso']],
            'model_hyperparams': [{
                'lasso': {
                    'sig_order': [1, 2, 3, 4, 5],
                    'normalize': True,
                    'weighted': True,
                },
                'ncde': {
                    'vector_field': 'original',
                    'num_epochs': 100,
                    'lr': 1e-4},
                'gru': {
                    'gru_width': 128,
                    'num_epochs': 100,
                    'lr': 1e-3},
                }]
            }
        }

tumor = {
        'niter': 10,
        'name': 'tumor',
        'data_config': {
            'model_X': 'cubic_positive',
            'model_Y': 'tumor_growth',
            'n_train': 50,
            'n_test': 50,
            'n_val': 50,
            'n_points_true': 1000,
            'dim_X': 2,
            'dim_Y': None,
            'non_linearity_Y': None,
        },
        'exp_config': {
            'n_points_X': [100],
            'n_points_Y': [4],
            'model_names': [['lasso', 'ncde', 'gru']],
            'model_hyperparams': [{
                'lasso': {
                    'sig_order': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'normalize': True,
                    'weighted': True,
                },
                'ncde': {
                    'vector_field': 'original',
                    'num_epochs': 100,
                    'lr': 1e-3},
                'gru': {
                    'gru_width': 128,
                    'num_epochs': 100,
                    'lr': 1e-3},
                }]
            }
        }


ou = {
        'niter': 10,
        'name': 'ou',
        'data_config': {
            'model_X': 'brownian',
            'model_Y': 'ornstein_uhlenbeck',
            'n_train': 50,
            'n_test': 50,
            'n_val': 50,
            'n_points_true': 1000,
            'dim_X': 2,
            'dim_Y': 1,
            'non_linearity_Y': 'Tanh',
        },
        'exp_config': {
            'n_points_X': [100],
            'n_points_Y': [4],
            'model_names': [['ncde', 'lasso', 'gru']],
            'model_hyperparams': [{
                'lasso': {
                    'sig_order': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'normalize': True,
                    'weighted': True,
                },
                'ncde': {
                    'vector_field': 'original',
                    'num_epochs': 100,
                    'lr': 1e-3},
                'gru': {
                    'gru_width': 128,
                    'num_epochs': 100,
                    'lr': 1e-3},
                }]
            }
        }

ill_specified = {
        'niter': 10,
        'name': 'ill_specified',
        'data_config': {
            'model_X': 'cubic',
            'model_Y': 'lognorm',
            'n_train': 50,
            'n_test': 50,
            'n_val': 50,
            'n_points_true': 1000,
            'dim_X': 2,
            'dim_Y': 1,
            'non_linearity_Y': 'Tanh',
        },
        'exp_config': {
            'n_points_X': [100],
            'n_points_Y': [4],
            'model_names': [['ncde', 'lasso', 'gru']],
            'model_hyperparams': [{
                'lasso': {
                    'sig_order': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'normalize': True,
                    'weighted': True,
                },
                'ncde': {
                    'vector_field': 'original',
                    'num_epochs': 100,
                    'lr': 1e-4},
                'gru': {
                    'gru_width': 128,
                    'num_epochs': 100,
                    'lr': 1e-3},
                }]
            }
        }


