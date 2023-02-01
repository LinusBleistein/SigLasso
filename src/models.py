import iisignature
import iisignature as isig
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.preprocessing import StandardScaler
import torch
import torchcde
from typing import Tuple
import warnings

from src.utils import l2_distance, normalize_path, get_weight_matrix
from src.vector_fields import OriginalVectorField, AlmostOriginalVectorField

# Remove some warnings
warnings.simplefilter('once', UserWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


class SigLasso:
    def __init__(self, sig_order: int, dim_Y, max_iter=1e3,
                 normalize: bool = True, weighted: bool = False):
        """Implements our SigLasso algorithm: a linear regression on signature
        features with a specific L1 regularization.
        """
        self.sig_order = sig_order
        self.weighted = weighted
        self.scaler = StandardScaler()
        self.dim_Y = dim_Y
        if self.dim_Y == 1:
            self.reg = LassoCV(max_iter=int(max_iter))
        else:
            self.reg = MultiTaskLassoCV(max_iter=int(max_iter))
        self.normalize = normalize

    def train(self, X: torch.Tensor, Y: torch.Tensor,
              grid_Y: torch.Tensor = None, grid_X: torch.Tensor = None):
        assert X.ndim == 3, \
            "X must have 3 dimensions: (n_samples, n_points, dim_X)"
        assert Y.ndim == 3, \
            "Y must have 3 dimensions: (n_samples, n_points, dim_X)"

        sigX, Yfinal = self.get_final_matrices(
            X, Y, grid_Y=grid_Y, grid_X=grid_X)

        # In the 1d case, LassoCV raises an error when Y is 2d
        if self.dim_Y == 1:
            Yfinal = Yfinal.squeeze(-1)

        if self.weighted:
            self.reg.fit(
                sigX @ get_weight_matrix(X.shape[2], self.sig_order),
                Yfinal)
        else:
            self.reg.fit(sigX, Yfinal)

    def get_final_matrices(
            self, X: torch.Tensor, Y: torch.Tensor,
            grid_Y: torch.Tensor = None,
            grid_X: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        # Check that the order of signature is not too large
        if iisignature.siglength(X.shape[2], self.sig_order) > 10 ** 6:
            raise ValueError("Length of signatures are larger than 10**6, "
                             "pick a smaller truncation order.")

        if self.normalize:
            X = normalize_path(X)

        if Y.shape[1] == 1:
            return isig.sig(X, self.sig_order), Y[:, 0, :]

        if grid_Y is None:
            raise ValueError('If Y has more than one observation, the '
                             'indices of the observations must be passed.')
        list_sigXs = []
        list_Yfinal = []

        grid_Y = grid_Y.numpy().astype(int)
        for i in range(Y.shape[0]):
            for j in range(grid_Y.shape[1]):
                index_Y = grid_Y[i, j]
                index_X = int(np.where(grid_X[i, :] == grid_Y[i, j])[0])

                if index_Y == 0:
                    warnings.warn(
                        'An observation of Y at time 0 has been skipped '
                        'since we need at least two observations of X up '
                        'to observation of Y')
                else:
                    # Signature of the path up to observation time of Y
                    sub_X_i = X[i, :index_X, :]

                    list_sigXs.append(
                        isig.sig(sub_X_i, self.sig_order))
                    # Y has already been downsampled so you should use j
                    # and not index_Y here
                    list_Yfinal.append(Y[i, j, :])

        return (torch.from_numpy(np.stack(list_sigXs)),
                torch.stack(list_Yfinal))

    def predict(self, X: torch.Tensor, on_grid: torch.Tensor = None,
                pass_sigs: bool = False) -> torch.Tensor:
        if on_grid is not None:
            assert on_grid.ndim == 2, " grid must have 2 dimensions: " \
                                      "(n_samples, n_points)"
            on_grid = on_grid.numpy().astype(int)
        else:
            # If on_grid has not been passed as argument, we predict Y at
            # all points of X
            on_grid = np.tile(np.arange(X.shape[1]), (X.shape[0], 1))

        if pass_sigs:
            return self.reg.predict(X)

        # new_Y = np.zeros((X.shape[0], on_grid.shape[1], self.dim_Y))

        assert X.ndim == 3, " X must have 3 dimensions: " \
                            "(n_samples, n_points, dim_X)"

        if self.normalize:
            X = normalize_path(X)
        list_pred = []
        for i in range(X.shape[0]):
            pred_Y = []
            for j in range(on_grid.shape[1]):
                index_grid = on_grid[i, j]

                if index_grid == 0:
                    pred_Y.append([self.reg.intercept_])
                else:
                    sub_X_i = X[i, :index_grid, :]
                    sigX_i = isig.sig(sub_X_i, self.sig_order).reshape(1,
                                                                       -1)
                    if self.weighted:
                        pred_Y.append(self.reg.predict(
                            sigX_i @ get_weight_matrix(X.shape[2],
                                                       self.sig_order)))
                    else:
                        pred_Y.append(self.reg.predict(sigX_i))
            list_pred.append(np.stack(pred_Y, axis=1).squeeze(0))
        if self.dim_Y == 1:
            return torch.from_numpy(np.stack(list_pred)).unsqueeze(-1)
        else:
            return torch.from_numpy(np.stack(list_pred))

    def get_l2_error(self, X: torch.Tensor, Y_full: torch.Tensor,
                     grid_Y: torch.Tensor, pass_sigs=False) -> float:
        assert X.ndim == 3, " X must have 3 dimensions: " \
                            "(n_samples, n_points, dim_X)"
        assert Y_full.ndim == 3, "Y must have 3 dimensions: " \
                                 "(n_samples, n_points, dim_Y)"
        assert grid_Y.ndim == 2, "grid_Y must have 2 dimensions: " \
                                 "(n_samples, n_points)"

        new_Y = self.predict(X, on_grid=grid_Y, pass_sigs=pass_sigs)
        return l2_distance(Y_full, new_Y)


class GRUModel(torch.nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int,
                 output_dim: int,
                 layer_dim: int = 1):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_channels = hidden_channels

        # GRU layers
        self.gru = torch.nn.GRU(
            input_channels, hidden_channels, layer_dim, batch_first=True)

        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_channels, output_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        assert X.ndim == 3, " X must have 3 dimensions: " \
                            "(n_samples, n_points, dim_X)"
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(
            self.layer_dim, X.size(0), self.hidden_channels).requires_grad_()

        # Forward propagation by passing in the input and hidden state
        # into the model
        out, _ = self.gru(X, h0.detach())

        # Reshaping the outputs in the shape of
        # (batch_size, seq_length, hidden_size) so that it can fit into the
        # fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape
        # (batch_size, output_dim)
        out = self.fc(out)

        return out

    def predict_trajectory(self, X: torch.Tensor) -> torch.Tensor:
        assert X.ndim == 3, " X must have 3 dimensions: " \
                            "(n_samples, n_points, dim_X)"

        h0 = torch.zeros(self.layer_dim, X.size(0),
                         self.hidden_channels).requires_grad_()
        hidden_states, _ = self.gru(X, h0.detach())
        out = []
        for i in range(X.shape[1]):
            out.append(self.fc(hidden_states[:, i, :]))
        Y = torch.stack(out, dim=1)
        return Y.detach()  # Need to detach to later compute l2 distance

    def get_l2_error(self, X: torch.Tensor, Y:torch.Tensor) -> float:
        Y_pred = self.predict_trajectory(X)
        return l2_distance(Y, Y_pred)


class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int,
                 vector_field: str = 'original'):
        super(NeuralCDE, self).__init__()
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        if vector_field == 'original':
            self.func = OriginalVectorField(hidden_channels, input_channels)

        elif vector_field == 'almost_original':
            self.func = AlmostOriginalVectorField(hidden_channels,
                                                  input_channels)

        else:
            raise ValueError(
                "vector_field must be one of ['original, 'almost_original']")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            X)
        Xfunc = torchcde.CubicSpline(coeffs)
        X0 = Xfunc.evaluate(Xfunc.interval[0])
        z0 = self.initial(X0)

        step_size = (Xfunc.grid_points[1:] - Xfunc.grid_points[:-1]).min()
        # Here, changing the NeuralCDE solver to a non-implicit scheme makes
        # training drastically faster
        z_T = torchcde.cdeint(X=Xfunc,
                              z0=z0,
                              func=self.func,
                              t=Xfunc.interval,
                              method='rk4',
                              options=dict(step_size=step_size))
        z_T = z_T[:, 1]
        return z_T

    def predict_trajectory(self, X: torch.Tensor) -> torch.Tensor:
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            X)
        Xfunc = torchcde.CubicSpline(coeffs)

        time_Y = torch.linspace(0, Xfunc.interval[1], X.shape[1])

        X0 = Xfunc.evaluate(Xfunc.interval[0])
        z0 = self.initial(X0)

        step_size = (Xfunc.grid_points[1:] - Xfunc.grid_points[:-1]).min()
        Y = torchcde.cdeint(X=Xfunc,
                              z0=z0,
                              func=self.func,
                              t=time_Y,
                              method='rk4',
                              options=dict(step_size=step_size))

        return Y.detach()

    def get_l2_error(self, X: torch.tensor, Y: torch.Tensor) -> float:
        Y_pred = self.predict_trajectory(X)
        return l2_distance(Y, Y_pred)
