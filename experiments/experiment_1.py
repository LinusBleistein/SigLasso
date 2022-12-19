import numpy as np
import os
from sacred import Experiment
from configs import *
from src import datagen, models, utils, utils_exp
import sys
import time
import torch

from src.sampling import downsample


ex = Experiment()


@ex.main
def run_exp(_run, dim_X, dim_Y, n_points, n_points_true, non_linearity_Y, 
            model_X, model_Y, n_train, n_test, n_val, model_names,
            model_hyperparams):
    # Add the try/except conditions this if you launch many runs so that
    # everything does not stop if there is an error
    # try:

    X_train_true, Y_train, X_val_true, Y_val, X_test_true, Y_test = datagen.get_train_val_test(
        model_X, model_Y, n_train, n_test, n_val, dim_X, dim_Y, n_points_true,
        non_linearity_Y=non_linearity_Y)

    X_train, sampling_X_train = downsample(X_train_true, n_points)
    X_val, sampling_X_val = downsample(X_val_true, n_points)
    X_test, sampling_X_test = downsample(X_test_true, n_points)

    print(X_train.shape, Y_train.shape)

    for model in model_names:
        print(model)
        if model == 'ncde':
            lr = model_hyperparams['ncde']['lr']
            ncde = cdemodel.NeuralCDE(
                dim_X, dim_Y,
                vector_field=model_hyperparams['ncde']['vector_field'])

            print('Train ncde')
            time_1 = time.time()
            ncde = cdemodel.train_neural_cde(
                ncde, X_train, Y_train[:, -1, :],
                model_hyperparams['ncde']['num_epochs'], lr=lr)
            time_2 = time.time()

            time_true = torch.linspace(0, 1, n_points_true)
            Y_test_pred = ncde.get_trajectory(X_test, time_true)
            Y_train_pred = ncde.get_trajectory(X_train, time_true)

            print(f'NCDE output: {Y_train_pred.shape}')
            _run.log_scalar(f'l2_train_{model}',
                            utils.l2_distance(Y_train_pred, Y_train))
            _run.log_scalar(f'l2_test_{model}',
                            utils.l2_distance(Y_test_pred, Y_test))
            _run.log_scalar(f'mse_train_{model}',
                            utils.mse_last_point(Y_train, Y_train_pred))
            _run.log_scalar(f'mse_test_{model}',
                            utils.mse_last_point(Y_test, Y_test_pred))
            _run.log_scalar(f'time_{model}', time_2 - time_1)
            _run.log_scalar(f'best_lr_{model}', lr)

        elif model == 'gru':
            lr = model_hyperparams['gru']['lr']
            gru_model = models.GRUModel(
                dim_X, model_hyperparams['gru']['gru_width'], dim_Y)

            print('Train GRU')
            time_1 = time.time()
            gru_model = models.train_gru(
                gru_model, X_train, Y_train[:, -1, :],
                model_hyperparams['gru']['num_epochs'], lr=lr)
            time_2 = time.time()

            Y_test_pred = gru_model.get_trajectory(X_test_true)
            Y_train_pred = gru_model.get_trajectory(X_train_true)

            Y_train = Y_train.numpy()
            Y_test = Y_test.numpy()

            print(f'GRU output: {Y_train_pred.shape}')
            _run.log_scalar(f'l2_train_{model}',
                            utils.l2_distance(Y_train_pred, Y_train))
            _run.log_scalar(f'l2_test_{model}',
                            utils.l2_distance(Y_test_pred, Y_test))
            _run.log_scalar(f'mse_train_{model}',
                            utils.mse_last_point(Y_train, Y_train_pred))
            _run.log_scalar(f'mse_test_{model}',
                            utils.mse_last_point(Y_test, Y_test_pred))
            _run.log_scalar(f'time_{model}', time_2 - time_1)
            _run.log_scalar(f'best_lr_{model}', lr)

        elif model == 'lasso':
            val_mse = []
            for sig_order in model_hyperparams['lasso']['sig_order']:
                print(f'Train lasso for signature order {sig_order}')
                lasso_sig = models.SigLasso(
                    sig_order, model_hyperparams['lasso']['alpha_grid'],
                    max_iter=1e04, standardize=False)
                lasso_sig.train(X_train, Y_train[:, -1, :].flatten())
                Y_val_pred = lasso_sig.predict_trajectory(X_val)

                print(f'Lasso val output: {Y_val_pred.shape}')
                val_mse.append(
                    utils.mse_last_point(Y_val, Y_val_pred))

            best_sig_order = model_hyperparams['lasso']['sig_order'][np.argmin(val_mse)]
            print('Train lasso')
            lasso_sig = models.SigLasso(
                best_sig_order, model_hyperparams['lasso']['alpha_grid'],
                max_iter=1e04, standardize=False)

            time_1 = time.time()
            lasso_sig.train(X_train, Y_train[:, -1, :].flatten())
            time_2 = time.time()

            Y_train_pred = lasso_sig.predict_trajectory(X_train_true)
            Y_test_pred = lasso_sig.predict_trajectory(X_test_true)

            print(f'Lasso output: {Y_train_pred.shape}')
            _run.log_scalar(f'lasso_best_alpha', lasso_sig.reg.alpha_)
            _run.log_scalar(f'l2_train_{model}',
                            utils.l2_distance(Y_train_pred, Y_train))
            _run.log_scalar(f'l2_test_{model}',
                            utils.l2_distance(Y_test_pred, Y_test))
            _run.log_scalar(f'mse_train_{model}',
                            utils.mse_last_point(Y_train, Y_train_pred))
            _run.log_scalar(f'mse_test_{model}',
                            utils.mse_last_point(Y_test, Y_test_pred))
            _run.log_scalar(f'time_{model}', time_2 - time_1)
            _run.log_scalar(f'best_sig_{model}', best_sig_order)

        else:
            raise ValueError('model does not exist')

    # except Exception as e:
    #     _run.log_scalar('error', str(e))

    # rand_indiv = np.random.choice(Y_test.shape[0])
    # plt.plot(Y_pred_ncde[rand_indiv, :, :].detach().numpy(), label='NCDE', c='red')
    # plt.plot(Y_test[rand_indiv, :, :], label='True' ,c='blue')
    # plt.plot(Y_pred_lasso[rand_indiv, :, :],label='Lasso',c='orange')
    # plt.legend()
    # plt.show()


config = globals()[str(sys.argv[1])]
niter = config['niter']
filepath = f"results/{config['name']}"
os.makedirs(filepath, exist_ok=True)
utils_exp.gridsearch(ex, config['exp_config'], dirname=filepath, niter=niter)

