import torch


class SimpleVectorField(torch.nn.Module):
    """
    Simple vector field.
    """
    def __init__(self, input_channels: int, hidden_channels: int,
                 non_linearity: str = None):
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

    def forward(self, t: torch.Tensor, z: torch.Tensor):
        batch = z.shape[0]
        if self.non_linearity is not None:
            return self.non_linearity(
                self.linear(z).view(
                    batch, self.hidden_channels, self.input_channels))
        else:
            return self.linear(z).view(
                batch, self.hidden_channels, self.input_channels)


class OriginalVectorField(torch.nn.Module):
    """
    Original vector field from "Neural Controlled Differential Equations for
    Irregular Time Series" (Kidger, 2020).
    """
    def __init__(self, hidden_channels: int, input_channels: int):
        super(OriginalVectorField, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class AlmostOriginalVectorField(torch.nn.Module):
    """
    Original vector field from "Neural Controlled Differential Equations for
    Irregular Time Series" (Kidger, 2020) but with a different activation
    function (Tanh instead of Relu).
    """
    def __init__(self, hidden_channels: int, input_channels: int):
        super(AlmostOriginalVectorField, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self, t: torch.Tensor, z: torch.Tensor):
        z = self.linear1(z)
        z = z.tanh()
        z = self.linear2(z)
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z
