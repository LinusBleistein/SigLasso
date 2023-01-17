# TODO: run expe OU n_train = 50, 100, 200, n_points_Y = 4
# TODO: n_train = 50 avec lognorm et TumorGrowth
# TODO: modèle bien specifié, on fait varier dim_X et n_points_X


test = {
        'niter': 1,
        'name': 'test',
        'data_config': {
            'model_X': 'brownian',
            'model_Y': 'ornstein_uhlenbeck',
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


tumor = {
        'niter': 1,
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
                    'num_epochs': 30,
                    'lr': 1e-3},
                'gru': {
                    'gru_width': 128,
                    'num_epochs': 100,
                    'lr': 1e-3},
                }]
            }
        }


y_sampling = {
        'niter': 1,
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

y_sampling_ou_100 = {
        'niter': 1,
        'name': 'y_sampling_ou_2',
        'data_config': {
            'model_X': 'brownian',
            'model_Y': 'ornstein_uhlenbeck',
            'n_train': 100,
            'n_test': 100,
            'n_val': 100,
            'n_points_true': 1000,
            'dim_X': 2,
            'dim_Y': 1,
            'non_linearity_Y': 'Tanh',
        },
        'exp_config': {
            'n_points_X': [100],
            'n_points_Y': [0, 4, 9, 14, 19],
            'model_names': [['lasso', 'gru', 'ncde']],
            'model_hyperparams': [{
                'lasso': {
                    'sig_order': [1, 2, 3, 4, 5, 6, 7, 8, 9],
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


y_sampling_ou_50 = {
        'niter': 1,
        'name': 'y_sampling_ou_2',
        'data_config': {
            'model_X': 'brownian',
            'model_Y': 'ornstein_uhlenbeck',
            'n_train': 50,
            'n_test': 100,
            'n_val': 50,
            'n_points_true': 1000,
            'dim_X': 2,
            'dim_Y': 1,
            'non_linearity_Y': 'Tanh',
        },
        'exp_config': {
            'n_points_X': [100],
            'n_points_Y': [0, 4, 9, 14, 19],
            'model_names': [['lasso', 'gru', 'ncde']],
            'model_hyperparams': [{
                'lasso': {
                    'sig_order': [1, 2, 3, 4, 5, 6, 7, 8, 9],
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


y_sampling_ou_1000 = {
        'niter': 1,
        'name': 'y_sampling_ou_2',
        'data_config': {
            'model_X': 'brownian',
            'model_Y': 'ornstein_uhlenbeck',
            'n_train': 1000,
            'n_test': 100,
            'n_val': 1000,
            'n_points_true': 1000,
            'dim_X': 2,
            'dim_Y': 1,
            'non_linearity_Y': 'Tanh',
        },
        'exp_config': {
            'n_points_X': [100],
            'n_points_Y': [0, 4, 9, 14, 19],
            'model_names': [['lasso', 'gru', 'ncde']],
            'model_hyperparams': [{
                'lasso': {
                    'sig_order': [1, 2, 3, 4, 5, 6, 7, 8, 9],
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
#
# x_sampling = {
#         'niter': 1,
#         'name': 'x_sampling',
#         'data_config': {
#             'model_X': 'cubic',
#             'model_Y': 'cde',
#             'n_train': 200,
#             'n_test': 200,
#             'n_val': 200,
#             'n_points_true': 1000,
#             'dim_X': 3,
#             'dim_Y': 1,
#             'non_linearity_Y': 'Tanh',
#         },
#         'exp_config': {
#             'n_points_X': [5,10,20,30,50],
#             'n_points_Y': [0],
#             'model_names': [['lasso', 'ncde', 'gru']],
#             'model_hyperparams': [{
#                 'lasso': {
#                     'sig_order': [1, 2, 3, 4],
#                     'normalize': True,
#                     'weighted': True,
#                 },
#                 'ncde': {
#                     'vector_field': 'original',
#                     'num_epochs': 30,
#                     'lr': 1e-3},
#                 'gru': {
#                     'gru_width': 128,
#                     'num_epochs': 100,
#                     'lr': 1e-3},
#                 }]
#             }
#         }
#
# dimension_test = {
#         'niter': 1,
#         'name': 'dimension_test',
#         'data_config': {
#             'model_X': 'cubic',
#             'model_Y': 'cde',
#             'n_train': 200,
#             'n_test': 200,
#             'n_val': 200,
#             'n_points_true': 1000,
#             'dim_X': [3,4,5,6],
#             'dim_Y': 1,
#             'non_linearity_Y': 'Tanh',
#         },
#         'exp_config': {
#             'n_points_X': [20],
#             'n_points_Y': [0],
#             'model_names': [['lasso', 'ncde', 'gru']],
#             'model_hyperparams': [{
#                 'lasso': {
#                     'sig_order': [1, 2, 3, 4],
#                     'normalize': True,
#                     'weighted': True,
#                 },
#                 'ncde': {
#                     'vector_field': 'original',
#                     'num_epochs': 30,
#                     'lr': 1e-3},
#                 'gru': {
#                     'gru_width': 128,
#                     'num_epochs': 100,
#                     'lr': 1e-3},
#                 }]
#             }
#         }
#
# ornstein_uhlenbeck = {
#         'niter': 1,
#         'name': 'ou_test',
#         'data_config': {
#             'model_X': 'brownian',
#             'model_Y': 'ornstein_uhlenbeck',
#             'n_train': 200,
#             'n_test': 100,
#             'n_val': 100,
#             'n_points_true': 1000,
#             'dim_X': 3,
#             'dim_Y': 1,
#             'non_linearity_Y': 'Tanh',
#         },
#         'exp_config': {
#             'n_points_X': [100],
#             'n_points_Y': [10],
#             'model_names': [['lasso', 'ncde', 'gru']],
#             'model_hyperparams': [{
#                 'lasso': {
#                     'sig_order': [1, 2, 3, 4],
#                     'normalize': True,
#                     'weighted': True,
#                 },
#                 'ncde': {
#                     'vector_field': 'original',
#                     'num_epochs': 30,
#                     'lr': 1e-3},
#                 'gru': {
#                     'gru_width': 128,
#                     'num_epochs': 100,
#                     'lr': 1e-3},
#                 }]
#             }
#         }
