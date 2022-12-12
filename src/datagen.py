import numpy as np
from stochastic.processes.diffusion import OrnsteinUhlenbeckProcess
import torch
import torchcde

from siglearning.cdemodel import CDEModel


def create_X(model_X, n_samples, n_points, dim_X, n_points_hermite=15):
    if model_X == 'cubic':
        t = torch.linspace(0, 1, n_points_hermite)
        t_ = t.unsqueeze(0).unsqueeze(-1).expand(
            n_samples, n_points_hermite, 1)
        x_ = torch.randn(n_samples, n_points_hermite, dim_X - 1)
        x = torch.cat([t_, x_], dim=2)

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            x, t=t)
        Xfunc = torchcde.CubicSpline(coeffs, t=t)
        X = Xfunc.evaluate(torch.linspace(0, 1, n_points))
        return X
    elif model_X == 'cubic_diffusion':
        t = torch.linspace(0, 1, n_points)
        t_ = t.unsqueeze(0).unsqueeze(-1).expand(
            n_samples, n_points, 1)
        diffusion = OrnsteinUhlenbeckProcess()
        x_ = torch.zeros((n_samples, n_points, dim_X - 1))
        for i in np.arange(n_samples):
            for dimension in np.arange(dim_X - 1):
                x_[i, :, dimension] = torch.tensor(diffusion.sample(n=n_points - 1))
        X = torch.cat([t_, x_], dim=2)

        #coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            #x, t=t)
        #Xfunc = torchcde.CubicSpline(coeffs, t=t)
        #X = Xfunc.evaluate(torch.linspace(0, 1, n_points))
        return X
    else:
        raise ValueError("model_X does not exist")



def get_train_val_test(
        model_X, model_Y, n_train, n_test, n_val, dim_X, dim_Y, n_points_true,
        non_linearity_Y=None):

    time_true = torch.linspace(0, 1, n_points_true)
    X_train = create_X(model_X, n_train, n_points_true, dim_X)
    X_test = create_X(model_X, n_test, n_points_true, dim_X)
    X_val = create_X(model_X, n_val, n_points_true, dim_X)

    if model_Y == 'cde':
        gen_cde = CDEModel(dim_X, dim_Y, non_linearity=non_linearity_Y)
        Y_train = gen_cde.get_Y(X_train, time_true)
        Y_test = gen_cde.get_Y(X_test, time_true)
        Y_val = gen_cde.get_Y(X_val, time_true)

    elif model_Y == 'lognorm':
        Y_train = torch.log(torch.linalg.norm(X_train, axis=2)).unsqueeze(-1)
        Y_test = torch.log(torch.linalg.norm(X_test, axis=2)).unsqueeze(-1)
        Y_val = torch.log(torch.linalg.norm(X_val, axis=2)).unsqueeze(-1)

    else:
        raise ValueError("model_Y does not exist")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test



