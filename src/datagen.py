import numpy as np
from stochastic.processes.diffusion import OrnsteinUhlenbeckProcess
from stochastic.processes.continuous import BrownianMotion
import torch
import torchcde

from src.utils import matrix_to_function, get_cumulative_moving_sum
from src.vector_fields import SimpleVectorField


def create_X(model_X: str, n_samples: int, n_points: int, dim_X: int = None,
             n_knots: int = 15):
    """ Simulation of trajectories for predictor. The 1st coordinate of X is
    always time

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

    if model_X == 'brownian':
        t = torch.linspace(0,1,n_points)
        bm = BrownianMotion(t=10,scale=0.1)
        sample = torch.empty((n_samples,n_points,dim_X))
        for i in np.arange(n_samples):
            brownian_sample = torch.tensor([bm.sample_at(t) for i in np.arange(dim_X-1)]).T
            sample[i,:,1:] = brownian_sample
            sample[i,:,0] = t

        return sample


    if model_X == 'squared_brownian':
        t = torch.linspace(0, 10, n_points)
        bm = BrownianMotion(t=10)
        x_ = torch.zeros((n_samples, n_points, 2))

        for i in np.arange(n_samples):
            x_[i, :, 0] = t
            x_[i, :, 1] = 0.25 * torch.tensor(bm.sample_at(t) ** 2)

        return x_

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
        return X

    else:
        raise NotImplementedError(
            f"{model_X} not implemented. Accepted values for model_X are "
            f"'cubic_diffusion', 'squared_brownian', 'brownian' and 'cubic' ")


class OrnsteinUhlenbeck:
    def __init__(self,theta,mu,omega):
        self.theta = theta
        self.mu = mu
        self.omega = omega
        self.Y0 = torch.randn(1)

    def get_Y(self,X):

        sample = torch.empty((X.shape[0],X.shape[1],1))
        time_grid = X[0,:,0]
        dt = time_grid[1]-time_grid[0]

        for i in np.arange(X.shape[0]):
            y = self.Y0.clone()
            sample[i,0] = y
            for j,t in enumerate(time_grid[:-1]):
                y += self.theta*(self.mu-y)*dt + self.omega@(X[i,j+1,1:]-X[i,j,1:])
                sample[i,j+1] = y

        return sample


class TumorGrowth:
    def __init__(self, lambda_0=0.9, lambda_1=0.7, k_1=10, k_2=0.5, psi=20):
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.k_1 = k_1
        self.k_2 = k_2
        self.psi = psi

    def tumor_growth(self, u, y, x):
        assert u.ndim == 1, "u must be a one dimensional array of length 4"
        assert u.shape[0] == 4, "u must be a one dimensional array of length 4"

        assert isinstance(x, float), "x must be a float"
        assert isinstance(y, float), "y must be a float"

        du_1 = self.lambda_0 * u[0] * (
                1 + (self.lambda_0 / self.lambda_1 * y) ** self.psi) ** (
                -1 / self.psi) - self.k_2 * x * u[0]
        du_2 = self.k_2 * x * u[0] - self.k_1 * u[1]
        du_3 = self.k_1 * (u[1] - u[2])
        du_4 = self.k_1 * (u[2] - u[3])
        return np.array([du_1, du_2, du_3, du_4])

    def tumor_trajectory(self, x):
        assert x.ndim == 1, "x must be a one-dimensional array"
        y_track = []
        u = np.array([2, 0, 0, 0])
        y = 2.
        dt = 10/len(x)
        y_track.append(y)

        for i, t in enumerate(np.arange(0, 10-dt, step=dt)):
            u = u + dt * self.tumor_growth(u, float(y_track[i]), float(x[i]))
            y = np.sum(u)
            y_track.append(y)

        return torch.tensor(y_track)

    def get_Y(self, X):
        assert X.ndim == 3, " X must have 3 dimensions: n_samples, time, channels"
        assert X.shape[2] == 2, "To generate tumor trajectories, X must be " \
                                "1-dimensional, hence we must have " \
                                "X.shape[2] == 2 (time being included in X)"

        Y = torch.zeros(X.shape[0], X.shape[1], 1)
        for i in np.arange(X.shape[0]):
            Y[i, :, 0] = self.tumor_trajectory(X[i, :, 1].flatten())
        return Y


class CDEModel():
    """
    CDE model with vector field one layer neural network initialized randomly.
    """
    def __init__(self, dim_X: int, dim_Y: int, non_linearity: str = None):
        self.dim_X = dim_X
        self.dim_Y = dim_Y
        self.vector_field = SimpleVectorField(
            dim_X, dim_Y, non_linearity=non_linearity)
        self.Y0 = torch.randn(dim_Y)

    def get_Y(self, X: torch.Tensor, time: torch.Tensor,
              interpolation_method: str = 'cubic', with_noise: bool = False,
              noise_Y_var: float = 0.01):
        """
        Samples Y from X

        """
        # Path interpolation to obtain smooth paths
        Xfunc = matrix_to_function(X, time, interpolation_method)

        # Set uniform initial random condition (every individual has the same)
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
        n_points_true: int, dim_X: int = None, dim_Y: int = None,
        non_linearity_Y: str = None, window_Y: int = 3):

    time_true = torch.linspace(0, 1, n_points_true)

    X_train = create_X(model_X, n_train, n_points_true, dim_X=dim_X)
    X_test = create_X(model_X, n_test, n_points_true, dim_X=dim_X)
    X_val = create_X(model_X, n_val, n_points_true, dim_X=dim_X)

    if dim_X is not None:
        assert X_train.shape[2] == dim_X, "Inconsistency between model_X " \
                                          "and dim_X"
    else:
        dim_X = X_train.shape[2]

    if model_Y == 'cde':
        assert dim_Y is not None, "if model_Y is 'cde' then dim_Y is required"

        gen_cde = CDEModel(dim_X, dim_Y, non_linearity=non_linearity_Y)
        Y_train = gen_cde.get_Y(X_train, time_true)
        Y_test = gen_cde.get_Y(X_test, time_true)
        Y_val = gen_cde.get_Y(X_val, time_true)

    if model_Y == 'tumor_growth':
        gen_tumor = TumorGrowth()
        Y_train = gen_tumor.get_Y(X_train)
        Y_test = gen_tumor.get_Y(X_test)
        Y_val = gen_tumor.get_Y(X_val)

    elif model_Y == 'lognorm':
        Y_train = torch.log(
            torch.linalg.norm(
                get_cumulative_moving_sum(X_train, window=window_Y),
                axis=2)
        ).unsqueeze(-1)
        Y_test = torch.log(
            torch.linalg.norm(
                get_cumulative_moving_sum(X_test, window=window_Y),
                axis=2)
        ).unsqueeze(-1)
        Y_val = torch.log(
            torch.linalg.norm(
                get_cumulative_moving_sum(X_val, window=window_Y),
                axis=2)
        ).unsqueeze(-1)

    else:
        NotImplementedError(f"{model_Y} not implemented.")

    if dim_Y is not None:
        assert Y_train.shape[2] == dim_Y, "Inconsistency between model_Y " \
                                          "and dim_Y"
    return X_train, Y_train, X_val, Y_val, X_test, Y_test




