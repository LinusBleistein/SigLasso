import torch


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