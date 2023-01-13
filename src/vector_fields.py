import torch


class SimpleVectorField(torch.nn.Module):
    """
    Simple vector field.
    """
    def __init__(self, input_channels, hidden_channels, non_linearity=None):
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
            return self.linear(z).view(
                batch, self.hidden_channels, self.input_channels)


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


class OriginalVectorField(torch.nn.Module):
    """
    Original vector field from "Neural Controlled Differential Equations for
    Irregular Time Series" (Kidger, 2020).
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

class AlmostOriginalVectorField(torch.nn.Module):
    """
    Original vector field from "Neural Controlled Differential Equations for
    Irregular Time Series" (Kidger, 2020) but with a different activation function (Tanh instead of Relu).
    """
    def __init__(self, hidden_channels, input_channels):
        super(AlmostOriginalVectorField, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self, t, z):
        z = self.linear1(z)
        z = z.tanh()
        z = self.linear2(z)
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z