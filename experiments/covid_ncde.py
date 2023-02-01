import numpy as np
import pandas as pd
import torch
from src.models import NeuralCDE
from src.train import train_neural_cde

#Specify the paths

result_path = "/Users/linusbleistein/Documents/Thèse/SigLasso/results/"
data_path = "/Users/linusbleistein/Documents/Thèse/SigLasso/data/"

#Start

[horizons,horizon_back,label_tensor_train] = torch.load(data_path + "/data_label_train_preprocessed.pkl")

[horizons,dates,regions,horizon_back,data_tensor_train,features_names,index_train_day,index_train_region] =\
                    torch.load(data_path + "/data_features_train_preprocessed.pkl")

[horizons,horizon_back,label_tensor_test] = torch.load(data_path + "/data_label_test_preprocessed.pkl")

[horizons,dates,regions,horizon_back,data_tensor_test,\
     features_names,index_test_day,index_test_region] = torch.load(data_path + "/data_features_test_preprocessed.pkl")

h , n_obs , D_time , d_feature = data_tensor_train.shape

results_NCDE = pd.DataFrame({'days' : index_train_day , "region" : index_train_region})
results_test_NCDE = pd.DataFrame({'days' : index_test_day , "region" : index_test_region})

model_hyperparams = {
                'ncde': {
                    'vector_field': 'almost_original',
                    'num_epochs': 20,
                    'lr': 1e-3 }
                }

X_train = data_tensor_train[2][:,:,:]
Y_train = label_tensor_train[2][:]

def mse_simple(X, Y):
    """
    X and Y must be of shape (n_samples, dim)

    """
    return np.mean(np.linalg.norm(np.array(X) - np.array(Y), axis=1) ** 2)


for horizon in list(horizons):
    X_train = data_tensor_train[horizon - 1][:, :, :]
    keep_ind_train = []
    for i in np.arange(X_train.shape[0]):
        if X_train[i].sum() != 0:
            keep_ind_train.append(i)
    X_train = X_train[keep_ind_train, :, :]

    #TODO: virer ce truc dégueulasse et le faire directement dans le data-preprocessing
    time_column = X_train[:, :, -1].clone()
    first_column = X_train[:, :, 0].clone()

    X_train[:, :, 0] = time_column
    X_train[:, :, -1] = first_column

    Y_train = label_tensor_train[horizon - 1][keep_ind_train].unsqueeze(2)

    X_test = data_tensor_test[horizon - 1][:, :, :]
    Y_test = label_tensor_test[horizon - 1][:].unsqueeze(2)

    dim_X = X_train.shape[2]
    dim_Y = Y_train.shape[2]

    ncde_model = NeuralCDE(
        dim_X, dim_Y,
        vector_field=model_hyperparams['ncde']['vector_field'])

    ncde = train_neural_cde(
        ncde_model, X_train, Y_train,
        model_hyperparams['ncde']['num_epochs'], lr=model_hyperparams['ncde']['lr'])

    Y_test_pred = ncde_model.predict_trajectory(X_test)
    Y_train_pred = ncde_model.predict_trajectory(X_train)

    if horizon == 1:
        results_NCDE['truth'] = Y_train.reshape(-1,1)
        results_test_NCDE['truth'] = Y_test.reshape(-1,1)

    results_NCDE['pred' + str(horizon)] = Y_train_pred[:, -1]
    results_test_NCDE['pred' + str(horizon)] = Y_test_pred[:, -1]

results_NCDE.to_csv(result_path + "results_covid/results_NCDE.cvs")
results_test_NCDE.to_csv(result_path + "results_covid/results_test_NCDE.cvs")