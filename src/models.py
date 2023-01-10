import iisignature as isig
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
import torch
import torchcde
from typing import Tuple
import warnings

from src.utils import l2_distance, normalize_path, get_weight_matrix
from src.vector_fields import SimpleVectorField, MultiLayerVectorField, \
    OriginalVectorField

# Remove some warnings
warnings.simplefilter('once', UserWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


# TODO: vérifier si la signature sort un 1 au début et si on n'a pas 2 intercepts

# TODO: un propos quelque part sur quoi faire si le nombre de sampling
#  points des X sont différents d'un individu à l'autre: faire du
#  remplissage feed-forward car ça ne perturbe pas la signature ?

# TODO: check everywhere if we need numpy or tensor arrays!

# TODO: résoudre la question de Y au temps 0

#TODO: check shape of signature not too large, otherwise pass


class SigLasso:
    def __init__(self, sig_order: int, dim_Y, max_iter=1e3,
                 normalize: bool = True, weighted: bool = False):
        """
        Parameters
        ----------
        sig_order: depth of the signature to compute.
        alpha_grid: hyperparameter validation grid for the Lasso.
        max_iter: maximum iterations for solving the penalized regression.
        pass_sigs: if True, SigLasso can be trained directly with signatures instead of paths.
        n_points: if pass_sigs is True, this is the number of sampling points of X.
        normalize: if True, paths are normalized by their total variation before signature computation.
        weighted: if True, layers of the signature vectors are normalized to enforce layer-wise specific
        penalization.
        """
        self.sig_order = sig_order
        self.weighted = weighted

        self.dim_Y = dim_Y
        if self.dim_Y == 1:
            self.reg = LassoCV(max_iter=int(max_iter))
        else:
            self.reg = MultiTaskLassoCV(max_iter=int(max_iter))
        self.normalize = normalize

    def train(self, X: torch.Tensor, Y: torch.Tensor,
              grid_Y: torch.Tensor = None, pass_sigs: bool = False):
        if pass_sigs:
            self.reg.fit(X, Y)
        else:
            assert X.ndim == 3, \
                "X must have 3 dimensions: n_samples, time, channels"
            assert Y.ndim == 3, \
                "Y must have 3 dimensions: n_samples, time, channels"

            sigX, Yfinal = self.get_final_matrices(X, Y, grid_Y=grid_Y)

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
            self, X: torch.Tensor, Y: torch.Tensor, grid_Y: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # TODO : problem in normalization: do we do it here or for each subpath ?
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

                if index_Y == 0:
                    warnings.warn(
                        'An observation of Y at time 0 has been skipped '
                        'since we need at least two observations of X up '
                        'to observation of Y')
                else:
                    # Signature of the path up to observation time of Y
                    list_sigXs.append(
                        isig.sig(X[i, :index_Y, :], self.sig_order))
                    # Y has already been downsampled so you should use j
                    # and not index_Y here
                    list_Yfinal.append(Y[i, j, :])

        return (torch.from_numpy(np.stack(list_sigXs)),
                torch.stack(list_Yfinal))

    def predict(self, X: torch.Tensor, on_grid: torch.Tensor = None,
                pass_sigs: bool = False) -> torch.Tensor:
        if on_grid is not None:
            assert on_grid.ndim == 2, " grid must have 2 dimensions: " \
                                      "(n_samples, time)"
            on_grid = on_grid.numpy().astype(int)
        else:
            # If on_grid has not been passed as argument, we predict Y at
            # all points of X
            on_grid = np.tile(np.arange(X.shape[1]), (X.shape[0], 1))

        if pass_sigs:
            return self.reg.predict(X)

        # new_Y = np.zeros((X.shape[0], on_grid.shape[1], self.dim_Y))

        assert X.ndim == 3, " X must have 3 dimensions: " \
                            "n_samples, time, channels"
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
                            "n_samples, time, channels"
        assert Y_full.ndim == 3, "Y must have 3 dimensions: " \
                                 "n_samples, time, channels"
        assert grid_Y.ndim == 2, "grid_Y must have 2 dimensions: " \
                                 "n_samples, time"

        if self.normalize:
            X = normalize_path(X)
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

    def forward(self, X: torch.Tensor):
        assert X.ndim == 3, " X must have 3 dimensions: " \
                            "n_samples, time, channels"
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
                            "n_samples, time, channels"

        h0 = torch.zeros(self.layer_dim, X.size(0),
                         self.hidden_channels).requires_grad_()
        hidden_states, _ = self.gru(X, h0.detach())
        out = []
        for i in range(X.shape[1]):
            out.append(self.fc(hidden_states[:, i, :]))
        Y = torch.stack(out, dim=1)
        return Y.detach()  # Need to detach to later compute l2 distance

    def get_l2_error(self, X, Y):
        Y_pred = self.predict_trajectory(X)
        return l2_distance(Y, Y_pred)


class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, interpolation="cubic",
                 activation='Tanh', vector_field='original', width=None,
                 depth=None):
        super(NeuralCDE, self).__init__()
        if vector_field == 'simple':
            self.func = SimpleVectorField(hidden_channels, input_channels,
                                          non_linearity=activation)
        elif vector_field == 'multilayer':
            self.func = MultiLayerVectorField(
                hidden_channels, input_channels, width, depth,
                activation=activation)
        elif vector_field == 'original':
            self.func = OriginalVectorField(hidden_channels, input_channels)
        else:
            raise ValueError(
                "vector_field must be one of ['simple, 'multilayer']")
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        # self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, X):
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            X)
        Xfunc = torchcde.CubicSpline(coeffs)
        X0 = Xfunc.evaluate(Xfunc.interval[0])
        z0 = self.initial(X0)
        z_T = torchcde.cdeint(X=Xfunc,
                              z0=z0,
                              func=self.func,
                              t=Xfunc.interval)
        z_T = z_T[:, 1]
        return z_T

    def predict_trajectory(self, X):
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            X)
        Xfunc = torchcde.CubicSpline(coeffs)

        time_Y = torch.linspace(0, Xfunc.interval[1], X.shape[1])

        X0 = Xfunc.evaluate(Xfunc.interval[0])
        z0 = self.initial(X0)

        Y = torchcde.cdeint(X=Xfunc, z0=z0, func=self.func, t=time_Y)

        return Y.detach().numpy()

    def get_l2_error(self, X, Y):
        Y_pred = self.predict_trajectory(X)
        return l2_distance(Y, Y_pred)
