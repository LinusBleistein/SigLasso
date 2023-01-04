import torch

from src.models import GRUModel
from src.utils import split_XY_on_grid


def train_gru(
        model: GRUModel,
        X_raw: torch.Tensor,
        Y_raw: torch.Tensor,
        num_epochs: int,
        lr: float = 0.001,
        batch_size: int = 1, # Use a batch_size of 1 to allow for variable length sequences
        grid_Y: torch.Tensor = None):

    # Split the paths in case Y is observed at several time points
    X, Y = split_XY_on_grid(X_raw, Y_raw, grid_Y=grid_Y)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train_dataset = torch.utils.data.IterableDataset(X, Y)
    # train_dataloader = torch.utils.data.DataLoader(
    #   train_dataset, batch_size=batch_size)

    for epoch in range(num_epochs):
        # for batch in train_dataloader:
        for i in range(len(X)): # Custom training loop to allow variable length sequences
            batch_X = X[i].unsqueeze(0)
            batch_y = Y[i].unsqueeze(0)

            pred_y = model(batch_X)
            loss = torch.nn.MSELoss()(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

    return model


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