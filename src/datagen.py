import numpy as np
from stochastic.processes.diffusion import OrnsteinUhlenbeckProcess
import torch
import torchcde

from src.utils import matrix_to_function
from src.vector_fields import SimpleVectorField


def create_X(model_X: str, n_samples: int, n_points: int, dim_X: int,
             n_knots: int = 15):
    """ Simulation of trajectories for predictor.

    """
    if model_X == 'cubic':
        # Fit polynomial to knots
        t = torch.linspace(0, 1, n_knots)
        t_ = t.unsqueeze(0).unsqueeze(-1).expand(
            n_samples, n_knots, 1)
        knots_ = torch.randn(n_samples, n_knots, dim_X - 1)
        knots = torch.cat([t_, knots_], dim=2)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            knots, t=t)

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
                x_[i, :, dimension] = torch.tensor(
                    diffusion.sample(n=n_points - 1))
        X = torch.cat([t_, x_], dim=2)

        #coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            #x, t=t)
        #Xfunc = torchcde.CubicSpline(coeffs, t=t)
        #X = Xfunc.evaluate(torch.linspace(0, 1, n_points))
        return X
    else:
        raise NotImplementedError(f"{model_X} not implemented.")


class CDEModel():
    """
    CDE model with vector field one layer neural network initialized randomly.
    """
    def __init__(self, dim_X, dim_Y, non_linearity=None):
        self.dim_X = dim_X
        self.dim_Y = dim_Y
        self.vector_field = SimpleVectorField(
            dim_Y, dim_X, non_linearity=non_linearity)
        self.Y0 = torch.randn(dim_Y)

    def get_Y(self, X, time, interpolation_method='cubic', with_noise=True,
              noise_Y_var=0.01):
        """
        Samples Y from X

        """
        #Path interpolation to obtain smooth paths
        Xfunc = matrix_to_function(X, time, interpolation_method)

        #Set uniform initial random condition (every individual has the same)
        z0 = self.Y0 * torch.ones(X.shape[0], self.dim_Y)
        Y = torchcde.cdeint(
            X=Xfunc, func=self.vector_field, z0=z0, t=time)

        if with_noise:
            noise_Y = noise_Y_var * torch.randn(Y.shape)
            return Y.detach() + noise_Y
        else:
            return Y.detach()


def get_train_val_test(
        model_X: str, model_Y: str, n_train: int, n_val: int, n_test: int,
        dim_X: int, dim_Y: int, n_points_true: int,
        non_linearity_Y: str = None):

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
        NotImplementedError(f"{model_Y} not implemented.")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test



