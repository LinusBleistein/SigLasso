import iisignature as isig
import numpy as np
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
import torch
import torchcde

from src.utils import l2_distance, normalize_path, weight_matrix
from src.vector_fields import SimpleVectorField


class SigLasso:
    def __init__(self, sig_order: int, dim_Y, max_iter=int(1e3),
                 pass_sigs=False, n_points=False, normalize=True,
                 weighted=False,
                 alpha_grid: np.ndarray = 10 ** np.linspace(-7, 1, 50)):
        """
        TODO: move pass_sigs argument to the functions: there is no reason that it should be an attribute
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
        self.alpha_grid = alpha_grid

        self.weighted = weighted
        if dim_Y == 1:
            self.reg = LassoCV(alphas=self.alpha_grid, max_iter=int(max_iter))
        else:
            self.reg = MultiTaskLassoCV(alphas=self.alpha_grid,
                                        max_iter=int(max_iter))
        self.normalize = normalize
        self.pass_sigs = pass_sigs
        if pass_sigs and n_points == False:
            raise ValueError('When passing signatures directly,'
                             ' you must also pass the number of sampling points of X.')
        else:
            self.n_points = n_points

    def train(self, X, Y, indices_Y=None):
        if self.pass_sigs:
            self.reg.fit(X, Y)
        else:
            assert X.ndim == 3, " X must have 3 dimensions: n_samples, time, dim"
            assert Y.ndim == 3, " Y must have 3 dimensions: n_samples, time, dim"

            if self.normalize:
                X = normalize_path(X)

            sigX, Yfinal = self.get_fit_matrices(X, Y, indices_Y=indices_Y)

            if self.weighted == True:
                dim_X = X.shape[2]
                self.reg.fit(sigX@weight_matrix(dim_X,self.sig_order), Yfinal)
            else:
                self.reg.fit(sigX, Yfinal)

    def get_final_matrices(self, X, Y, indices_Y=None):
        # TODO: un propos quelque part sur quoi faire si le nombre de sampling
        #  points des X sont différents d'un individu à l'autre: faire du
        #  remplissage feed-forward car ça ne perturbe pas la signature !

        # TODO: deal with initial value of Y

        # TODO: check everywhere if we need numpy or tensor arrays!
        if Y.shape[1] == 1:
            return isig.sig(X, self.sig_order), Y[:, 0, :]
        else:
            if indices_Y is None:
                raise ValueError('If Y has more than one observation, the '
                                 'indices of the observations must be passed.')
            list_sigXs = []
            list_Yfinal = []
            indices_Y = indices_Y.numpy().astype(int)
            for i in range(Y.shape[0]):
                for j in range(indices_Y.shape[1]):
                    index_Y = indices_Y[i, j]
                    # print(index_Y)
                    # print(f'Size of X: {X[i, :index_Y, :].shape}')
                    list_sigXs.append(
                        isig.sig(X[i, :index_Y, :], self.sig_order)) # Signature of the path up to observation time of Y
                    list_Yfinal.append(Y[i, j, :])
            return np.stack(list_sigXs), np.stack(list_Yfinal)

    def predict(self, X):
        if self.pass_sigs:
            return self.reg.predict(X)
        else:
            if self.normalize:
                X = normalize_path(X)
            sigX = isig.sig(X, self.sig_order)
            if self.weighted == True:
                dim_X = X.shape[2]
                sigX = sigX@weight_matrix(dim_X,self.sig_order)
            return self.reg.predict(sigX)

    def predict_trajectory(self, X):
        new_Y = np.zeros((X.shape[0], X.shape[1], 1))
        for i in range(0, X.shape[1]):
            if self.normalize:
                X_i = normalize_path(X[:, :i + 1, :])
            else:
                X_i = X[:, :i + 1, :]
            Xsig_i = isig.sig(X_i, self.sig_order)
            if self.weighted:
                dim_X = X.shape[2]
                Xsig_i = Xsig_i @ weight_matrix(dim_X, self.sig_order)
            new_Y[:, i, 0] = ((Xsig_i @ self.reg.coef_.T).flatten()
                              + self.reg.intercept_)
        return new_Y

    def get_l2_error(self, X, Y_full):
        # if self.normalize:
        #     X = normalize_path(X)
        new_Y = self.predict_trajectory(X)
        return l2_distance(Y_full, new_Y)


class GRUModel(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_dim,
                 layer_dim=1):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_channels = hidden_channels

        # GRU layers
        self.gru = torch.nn.GRU(
            input_channels, hidden_channels, layer_dim, batch_first=True)

        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_channels, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_channels).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

    def get_trajectory(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0),
                         self.hidden_channels).requires_grad_()
        hidden_states, _ = self.gru(x, h0.detach())
        out = []
        for i in range(x.shape[1]):
            out.append(self.fc(hidden_states[:, i, :]))
        Y = torch.stack(out, dim=1)
        return Y.detach().numpy()

    def get_l2_error(self, X, Y):
        Y_pred = self.get_trajectory(X)
        return l2_distance(Y, Y_pred)



class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, interpolation="cubic",
                 activation='Tanh', vector_field='simple', width=None,
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

    def get_trajectory(self, X, n_points_Y):
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            X)
        Xfunc = torchcde.CubicSpline(coeffs)

        time_Y = torch.linspace(0, Xfunc.interval[1], 100)

        X0 = Xfunc.evaluate(Xfunc.interval[0])
        z0 = self.initial(X0)

        Y = torchcde.cdeint(X=Xfunc, z0=z0, func=self.func, t=time_Y)

        return Y.detach().numpy()

    def get_l2_error(self, X, Y, time_X, n_points_Y):
        Y_pred = self.get_trajectory(X, time_X, n_points_Y)
        return l2_distance(Y, Y_pred)
