import numpy as np
import pandas as pd
import torch
from src.models import SigLasso

#Specify paths

result_path = "SET_RESULT_PATH"
data_path = "SET_DATA_PATH/"

#Import data

[horizons,horizon_back,label_tensor_train] = torch.load(data_path + "data_label_train_preprocessed.pkl")

[horizons,dates,regions,horizon_back,data_tensor_train,
     features_names,index_train_day,index_train_region] = torch.load(data_path + "data_features_train_preprocessed.pkl")


[horizons,horizon_back,label_tensor_test] = torch.load(data_path + "data_label_test_preprocessed.pkl")

[horizons,dates,regions,horizon_back,data_tensor_test,features_names,
     index_test_day,index_test_region] = torch.load(data_path + "data_features_test_preprocessed.pkl")

h , n_obs , D_time , d_feature = data_tensor_train.shape


#TODO est-ce qu'on peut remplacer ce truc par autre chose ? Une fonction util déjà codée ?
def mse_simple(X, Y):
    """
    X and Y must be of shape (n_samples, dim)

    """
    return np.mean(np.linalg.norm(np.array(X) - np.array(Y), axis=1) ** 2)

#Create the result tensors

results_SigSparse_n = pd.DataFrame({'days' : index_train_day , "region" : index_train_region})
results_test_SigSparse_n = pd.DataFrame({'days' : index_test_day , "region" : index_test_region})

results_SigSparse_nw = pd.DataFrame({'days' : index_train_day , "region" : index_train_region})
results_test_SigSparse_nw = pd.DataFrame({'days' : index_test_day , "region" : index_test_region})

model_hyperparams = {'lasso': {
                #    'alpha_grid': 10 ** np.linspace(-7, 1, 30),
                    'sig_order': [1 , 2 , 3 ]}}

X_train = data_tensor_train[2][:,:,:]
Y_train = label_tensor_train[2][:].unsqueeze(2)#reshape((1,Y_train.shape[0],1)) #

#Train SigLasso without weighting

for horizon in list(horizons):

    val_mse_test_order_n = []

    X_train = data_tensor_train[horizon - 1][:, :, :]
    keep_ind_train = []
    for i in np.arange(X_train.shape[0]):
        if X_train[i].sum() != 0:
            keep_ind_train.append(i)
    X_train = X_train[keep_ind_train, :, :]
    Y_train = label_tensor_train[horizon - 1][keep_ind_train].unsqueeze(2)
    X_test = data_tensor_test[horizon - 1][:, :, :]
    Y_test = label_tensor_test[horizon - 1][:].unsqueeze(2)

    for sig_o in model_hyperparams['lasso']['sig_order']:
        lasso_sig_n = SigLasso(sig_o, 1, max_iter=1e09, normalize=True, weighted=False)
        lasso_sig_n.train(X_train, Y_train)

        Y_test_pred = lasso_sig_n.predict(X_test)
        val_mse_test_order_n.append(
            mse_simple(Y_test, Y_test_pred[:, -1]))

    best_sig_order_n = model_hyperparams['lasso']['sig_order'][np.argmin(val_mse_test_order_n)]
    lasso_sig_n = SigLasso(best_sig_order_n, 1, normalize=True, weighted=False,
                           max_iter=1e09)
    lasso_sig_n.train(X_train, Y_train)

    Y_train_pred = lasso_sig_n.predict(X_train)
    Y_test_pred = lasso_sig_n.predict(X_test)
    if horizon == 1:
        results_SigSparse_n['truth'] = Y_train.numpy().reshape(-1)
        results_test_SigSparse_n['truth'] = Y_test.numpy().reshape(-1)

    results_SigSparse_n['pred' + str(horizon)] = Y_train_pred[:, -1].numpy().reshape(-1)
    results_test_SigSparse_n['pred' + str(horizon)] = Y_test_pred[:, -1].numpy().reshape(-1)

results_SigSparse_n.to_csv(result_path + "results_SigLasso_normalized.cvs")
results_test_SigSparse_n.to_csv(result_path + "results_test_SigLasso_normalized.cvs")

#Train the weighted SigLasso

for horizon in list(horizons):

    val_mse_test_order_nw = []

    X_train = data_tensor_train[horizon - 1][:, :, :]
    keep_ind_train = []
    for i in np.arange(X_train.shape[0]):
        if X_train[i].sum() != 0:
            keep_ind_train.append(i)
    X_train = X_train[keep_ind_train, :, :]
    Y_train = label_tensor_train[horizon - 1][keep_ind_train].unsqueeze(2)
    X_test = data_tensor_test[horizon - 1][:, :, :]
    Y_test = label_tensor_test[horizon - 1][:].unsqueeze(2)

    for sig_o in model_hyperparams['lasso']['sig_order']:
        lasso_sig_n = SigLasso(sig_o, 1, max_iter=1e09, normalize=True, weighted=True)
        lasso_sig_n.train(X_train, Y_train)

        Y_test_pred = lasso_sig_n.predict(X_test)
        val_mse_test_order_nw.append(
            mse_simple(Y_test, Y_test_pred[:, -1]))

    best_sig_order_n = model_hyperparams['lasso']['sig_order'][np.argmin(val_mse_test_order_nw)]
    lasso_sig_n = SigLasso(best_sig_order_n, 1, normalize=True, weighted=True,
                           max_iter=1e09)
    lasso_sig_n.train(X_train, Y_train)

    Y_train_pred = lasso_sig_n.predict(X_train)
    Y_test_pred = lasso_sig_n.predict(X_test)

    if horizon == 1:
        results_SigSparse_nw['truth'] = Y_train.numpy().reshape(-1)
        results_test_SigSparse_nw['truth'] = Y_test.numpy().reshape(-1)

    results_SigSparse_nw['pred' + str(horizon)] = Y_train_pred[:, -1].numpy().reshape(-1)
    results_test_SigSparse_nw['pred' + str(horizon)] = Y_test_pred[:, -1].numpy().reshape(-1)

#Save the results for

results_SigSparse_nw.to_csv(result_path + "results_covid/results_SigLasso_normalized_weighted.cvs")
results_test_SigSparse_nw.to_csv(result_path + "results_covid/results_test_SigLasso_normalized_weighted.cvs")