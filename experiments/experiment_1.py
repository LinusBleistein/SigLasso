import os
from sacred import Experiment
import sys
import time

# Set working directory to source
abspath = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(abspath))
os.chdir(BASE_DIR)

from configs import *

from src.models import GRUModel, NeuralCDE, SigLasso
from src.sampling import downsample
from src.train import train_gru, train_neural_cde
from src.utils import l2_distance, mse_on_grid
from src.utils_exp import load_data
from src import utils_exp

# TODO: problem with NCDE + Siglasso
ex = Experiment()


@ex.main
def run_exp(_run, data_path, n_points_X, n_points_Y, model_names,
            model_hyperparams):
    # Add the try/except conditions this if you launch many runs so that
    # everything does not stop if there is an error
    # try:
    X_raw_train, Y_raw_train, X_raw_val, Y_raw_val, X_raw_test, Y_raw_test = \
        load_data(data_path)

    X_train, grid_X_train = downsample(
        X_raw_train, n_points_X, keep_first=True, keep_last=True)
    Y_train, grid_Y_train = downsample(
        Y_raw_train, n_points_Y, keep_first=False, keep_last=True,
        on_grid=grid_X_train)

    X_val, grid_X_val = downsample(
        X_raw_val, n_points_X, keep_first=True, keep_last=True)
    Y_val, grid_Y_val = downsample(
        Y_raw_val, n_points_Y, keep_first=False, keep_last=True,
        on_grid=grid_X_val)

    # X_test, grid_X_test = downsample(
    #     X_raw_test, n_points_X, keep_first=True, keep_last=True)
    # Y_test, grid_Y_test = downsample(
    #    Y_raw_test, n_points_Y, keep_first=False, keep_last=True,
    #    on_grid=grid_X_test)

    print(f'X_train.shape={X_train.shape}')
    print(f'Y_train.shape={Y_train.shape}')

    for model in model_names:
        print(model)
        if model == 'ncde':
            lr = model_hyperparams['ncde']['lr']
            ncde_model = NeuralCDE(
                X_train.shape[2], Y_train.shape[2],
                vector_field=model_hyperparams['ncde']['vector_field'])

            print('Train ncde')
            time_1 = time.time()

            ncde_model = train_neural_cde(
                ncde_model,
                X_train,
                Y_train,
                model_hyperparams['ncde']['num_epochs'],
                grid_Y=grid_Y_train,
                lr=lr)
            time_2 = time.time()

            Y_test_pred = ncde_model.predict_trajectory(X_raw_test)
            Y_train_pred = ncde_model.predict_trajectory(X_raw_train)

            print(f'NCDE: Y_train_pred.shape={Y_train_pred.shape}')

            _run.log_scalar(f'l2_train_{model}',
                            l2_distance(Y_train_pred, Y_raw_train))
            _run.log_scalar(f'l2_test_{model}',
                            l2_distance(Y_test_pred, Y_raw_test))

            _run.log_scalar(f'mse_last_point_train_{model}',
                            mse_on_grid(Y_train_pred, Y_raw_train))
            _run.log_scalar(f'mse_last_point_test_{model}',
                            mse_on_grid(Y_test_pred, Y_raw_test))

            _run.log_scalar(f'time_{model}', time_2 - time_1)
            _run.log_scalar(f'best_lr_{model}', lr)

        elif model == 'gru':
            lr = model_hyperparams['gru']['lr']
            gru_model = GRUModel(
                X_train.shape[2],
                model_hyperparams['gru']['gru_width'],
                Y_train.shape[2])

            print('Train GRU')
            time_1 = time.time()
            gru_model = train_gru(
                gru_model,
                X_train,
                Y_train,
                model_hyperparams['gru']['num_epochs'],
                grid_Y=grid_Y_train,
                lr=lr)
            time_2 = time.time()

            Y_test_pred = gru_model.predict_trajectory(X_raw_test)
            Y_train_pred = gru_model.predict_trajectory(X_raw_train)

            print(f'GRU: Y_train_pred.shape={Y_train_pred.shape}')

            _run.log_scalar(f'l2_train_{model}',
                            l2_distance(Y_train_pred, Y_raw_train))
            _run.log_scalar(f'l2_test_{model}',
                            l2_distance(Y_test_pred, Y_raw_test))

            _run.log_scalar(f'mse_last_point_train_{model}',
                            mse_on_grid(Y_train_pred, Y_raw_train))
            _run.log_scalar(f'mse_last_point_test_{model}',
                            mse_on_grid(Y_test_pred, Y_raw_test))

            _run.log_scalar(f'time_{model}', time_2 - time_1)
            _run.log_scalar(f'best_lr_{model}', lr)

        elif model == 'lasso':
            val_mse = []
            for sig_order in model_hyperparams['lasso']['sig_order']:
                print(f'Train lasso for signature order {sig_order}')
                lasso_sig = SigLasso(
                    sig_order,
                    Y_train.shape[2],
                    normalize=model_hyperparams['lasso']['normalize'],
                    weighted=model_hyperparams['lasso']['weighted']
                )
                lasso_sig.train(X_train, Y_train, grid_Y_train)

                Y_val_pred = lasso_sig.predict(X_val)

                print(f'Lasso val output: {Y_val_pred.shape}')
                # The best signature order is selected as the one minimizing
                # the mse at the last time step of Y_val
                val_mse.append(
                    mse_on_grid(Y_val_pred, Y_val,
                                grid_1=grid_X_val, grid_2=grid_Y_val))

            best_sig_order = model_hyperparams['lasso']['sig_order'][
                np.argmin(val_mse)]
            print('Train lasso')
            lasso_sig = SigLasso(
                best_sig_order,
                Y_train.shape[2],
                normalize=model_hyperparams['lasso']['normalize'],
                weighted=model_hyperparams['lasso']['weighted']
            )

            time_1 = time.time()
            lasso_sig.train(X_train, Y_train, grid_Y_train)
            time_2 = time.time()

            Y_train_pred = lasso_sig.predict(X_raw_train)
            Y_test_pred = lasso_sig.predict(X_raw_test)

            # print(f'SigLasso: Y_train_pred.shape={Y_train_pred.shape}')

            _run.log_scalar(f'val_mse_array_{model}', val_mse)
            _run.log_scalar(
                f'l2_train_{model}',
                l2_distance(Y_train_pred, Y_raw_train))
            _run.log_scalar(f'l2_test_{model}',
                            l2_distance(Y_test_pred, Y_raw_test))

            _run.log_scalar(f'mse_last_point_train_{model}',
                            mse_on_grid(Y_train_pred, Y_raw_train))
            _run.log_scalar(f'mse_last_point_test_{model}',
                            mse_on_grid(Y_test_pred, Y_raw_test))

            _run.log_scalar(f'time_{model}', time_2 - time_1)
            _run.log_scalar(f'lasso_best_alpha', lasso_sig.reg.alpha_)
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

utils_exp.gridsearch(ex, config, BASE_DIR)
