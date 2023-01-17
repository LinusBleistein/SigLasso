import numpy as np
import pandas as pd


import torch

from src.models import GRUModel
from src.train import train_gru

#Specify the paths

result_path = "/Users/linusbleistein/Documents/Thèse/SigLasso/results/"
data_path = "/Users/linusbleistein/Documents/Thèse/SigLasso/data/"

#Start

[horizons,horizon_back,label_tensor_train] = torch.load(data_path + "/data_label_train_preprocessed.pkl")
print(label_tensor_train.shape)

[horizons,dates,regions,horizon_back,data_tensor_train,features_names,index_train_day,index_train_region] =\
                    torch.load(data_path + "/data_features_train_preprocessed.pkl")

[horizons,horizon_back,label_tensor_test] = torch.load(data_path + "/data_label_test_preprocessed.pkl")
print(label_tensor_test.shape)

[horizons,dates,regions,horizon_back,data_tensor_test,\
     features_names,index_test_day,index_test_region] = torch.load(data_path + "/data_features_test_preprocessed.pkl")

h , n_obs , D_time , d_feature = data_tensor_train.shape

results_GRU = pd.DataFrame({'days' : index_train_day , "region" : index_train_region})
results_test_GRU = pd.DataFrame({'days' : index_test_day , "region" : index_test_region})

model_hyperparams = {
                'gru': {
                    'gru_width': 128,
                    'num_epochs': 1,
                    'lr': 1e-4},
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

    Y_train = label_tensor_train[horizon - 1][keep_ind_train].unsqueeze(2)

    X_test = data_tensor_test[horizon - 1][:, :, :]
    Y_test = label_tensor_test[horizon - 1][:].unsqueeze(2)

    dim_X = X_train.shape[2]
    dim_Y = Y_train.shape[2]

    gru_model = GRUModel(dim_X, model_hyperparams['gru']['gru_width'], dim_Y)

    gru = train_gru(gru_model, X_train, Y_train,
                    model_hyperparams['gru']['num_epochs'], lr=model_hyperparams['gru']['lr'])

    Y_test_pred = gru_model.predict_trajectory(X_test)
    Y_train_pred = gru_model.predict_trajectory(X_train)

    if horizon == 1:
        results_GRU['truth'] = Y_train.reshape(-1,1)
        results_test_GRU['truth'] = Y_test.reshape(-1,1)

    results_GRU['pred' + str(horizon)] = Y_train_pred[:, -1]
    results_test_GRU['pred' + str(horizon)] = Y_test_pred[:, -1]

results_GRU.to_csv(result_path + "results_covid/results_GRU.cvs")
results_test_GRU.to_csv(result_path + "results_covid/results_test_GRU.cvs")