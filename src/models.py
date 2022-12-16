import iisignature as isig
import numpy as np
from sklearn.linear_model import LassoCV
from siglearning.utils import l2_distance, normalize_path, weight_matrix
from vector_fields import SimpleVectorField
import torch


class SigLasso:
    def __init__(self, sig_order, alpha_grid, max_iter=int(1e3),
                 pass_sigs=False, n_times=False, standardize=True, weighted=False):
        """

        Parameters
        ----------
        sig_order: depth of the signature to compute.
        alpha_grid: hyperparameter validation grid for the Lasso.
        max_iter: maximum iterations for solving the penalized regression.
        pass_sigs: if True, SigLasso can be trained directly with signatures instead of paths.
        n_times: if pass_sigs is True, this is the number of sampling points of X.
        standardize: if True, paths are normalized by their total variation before signature computation.
        weighted: if True, layers of the signature vectors are normalized to enforce layer-wise specific
        penalization.
        """
        self.sig_order = sig_order
        self.alpha_grid = alpha_grid
        self.weighted = weighted
        self.reg = LassoCV(alphas=self.alpha_grid, max_iter=int(max_iter))
        self.standardize = standardize
        self.pass_sigs = pass_sigs
        if pass_sigs and n_times == False:
            raise ValueError('When passing signatures directly,'
                             ' you must also pass the number of sampling points of X.')
        else:
            self.n_times = n_times

    def train(self, X, Y):
        if self.pass_sigs:
            self.reg.fit(X, Y)
        else:
            if self.standardize:
                X = normalize_path(X)

            sigX = isig.sig(X, self.sig_order)

            if self.weighted == True:
                dim_X = X.shape[2]
                self.reg.fit(sigX@weight_matrix(dim_X,self.sig_order),Y)
            else:
                self.reg.fit(sigX, Y)

    def predict(self, X):
        if self.pass_sigs:
            return self.reg.predict(X)
        else:
            if self.standardize:
                X = normalize_path(X)
            sigX = isig.sig(X, self.sig_order)
            if self.weighted == True:
                dim_X = X.shape[2]
                sigX = sigX@weight_matrix(dim_X,self.sig_order)
            return self.reg.predict(sigX)

    def predict_trajectory(self, X):
        new_Y = np.zeros((X.shape[0], X.shape[1], 1))
        for i in range(0, X.shape[1]):
            if self.standardize:
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
        # if self.standardize:
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
