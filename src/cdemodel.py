import torch
import torchcde
from siglearning.utils import matrix_to_function, l2_distance


class MultiLayerVectorField(torch.nn.Module):
    """
    Multilayer vector field.
    """
    def __init__(
            self, hidden_channels, input_channels, width, depth,
            activation='Tanh'):
        super(MultiLayerVectorField, self).__init__()
        self.width = width
        self.depth = depth
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input = torch.nn.Linear(self.hidden_channels, self.width)
        self.layers = torch.nn.Sequential(
                *[torch.nn.Linear(self.width, self.width) for _ in
                  range(self.depth)])
        self.activation = getattr(torch.nn, activation)()
        self.output = torch.nn.Linear(width, hidden_channels * input_channels)

    def forward(self, t, z):
        h = self.input(z)
        for k in range(self.depth):
            h = self.activation(self.layers[k](h))
        return self.output(h).view(
            z.shape[0], self.hidden_channels, self.input_channels)


class SimpleVectorField(torch.nn.Module):
    """
    Simple vector field.
    """
    def __init__(self, hidden_channels, input_channels, non_linearity=None):
        super(SimpleVectorField, self).__init__()
        super().__init__()
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.linear = torch.nn.Linear(hidden_channels,
                                      hidden_channels * input_channels)
        if non_linearity is not None:
            self.non_linearity = getattr(torch.nn, non_linearity)()
        else:
            self.non_linearity = non_linearity

    def forward(self, t, z):
        batch = z.shape[0]
        if self.non_linearity is not None:
            return self.non_linearity(
                self.linear(z).view(
                    batch, self.hidden_channels, self.input_channels))
        else:
            new_z = self.linear(z)
            return new_z.view(
                batch, self.hidden_channels, self.input_channels)


class OriginalVectorField(torch.nn.Module):
    """
    Original vector field from "Neural Controlled Differential Equations for Irregular Time Series" (Kidger, 2020).
    """
    def __init__(self, hidden_channels, input_channels):
        super(OriginalVectorField, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self, t, z):
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class CDEModel():
    """
    Generative linear CDE model with random dynamics.
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
        Samples interpolated paths and target time series on n_sampling_points points.

        Parameters
        ----------
        batch: number of individuals to sample
        n_sampling_points: number of points on which to sample them.

        Returns
        -------
        Y: target time series sampled on a regular grid of n_sampling_points.
        X: driving signal sampled on a regular grid of n_sampling points.
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


def train_neural_cde(model, X, Y, num_epochs, lr=0.001, batch_size=32):
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





