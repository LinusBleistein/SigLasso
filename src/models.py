import iisignature as isig
import numpy as np
from sklearn.linear_model import LassoCV
from siglearning.utils import l2_distance, normalize_path, weight_matrix
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


def train_gru(model, X, Y, num_epochs, lr=0.001, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dataset = torch.utils.data.TensorDataset(X, Y)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch_X, batch_y = batch
            pred_y = model(batch_X)
            loss = torch.nn.MSELoss()(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

    return model
