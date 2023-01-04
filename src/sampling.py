import numpy as np
import torch


def downsample(X: torch.Tensor, n_points_kept: int, with_noise: bool = False,
               noise_X_var: float = 0.5, keep_first: bool = False,
               keep_last: bool = False, on_grid=None):
    """
    TODO: update doc here
    Downsamples a dataframe, keeping n_points_kept whose index is between limit_down and limit_up.
    Example: if n_points_kept is set to 5 and limit_down == 3, limit_up == 10, this function will choose 5 points,
    between the third and the 10th sampling point, for every individual.
    Little catch: we need the first sampling point to be included for every individual. Therefore, only n_point_kept - 1
    will be sampled at random.

    Parameters
    ----------
    X: matrix to downsample.
    n_points_kept: number of points to keep for every individual, excluding the first and last points
    limit_down: lower index limit.
    limit_up: upper index limit.

    Returns
    -------
    A dataframe whose shape is the same than df, expect for the second dimension which is now of shape n_points_kept.
    A second dataframe which logs the sampling points for every individual.
    """
    assert X.ndim == 3, "X must have 3 dimensions: n_samples, n_points, dim"
    if on_grid is not None:
        assert on_grid.ndim == 2, 'If not None, on_grid must have 2 ' \
                              'dimensions: n_samples, n_points'
        assert on_grid.shape[0] == X.shape[0], 'First dimension of on_grid ' \
                                               'and X must be the same'
    n_samples = X.shape[0]
    n_dim = X.shape[2]
    n_choices = n_points_kept

    if keep_first:
        n_choices += 1
    if keep_last:
        n_choices += 1

    assert n_choices <= X.shape[1], f'n_choices + keep_first + keep_last is ' \
                                    f'equal to {n_choices} which is larger ' \
                                    f'than X.shape[1]'

    downsampled_X = np.zeros((n_samples, n_choices, n_dim))
    times_X = np.zeros((n_samples, n_choices))

    if on_grid is None:
        on_grid = np.tile(np.arange(X.shape[1]), (X.shape[0], 1))

    for i in range(n_samples):
        # Do not select randomly the first or the last point
        sampling_points = np.random.choice(on_grid[i, 1:-1],
                                               size=n_points_kept,
                                               replace=False)
        #Keep the first sampling point
        if keep_first:
            sampling_points = np.insert(sampling_points, 0, on_grid[i, 0])
        if keep_last:
            sampling_points = np.insert(sampling_points, 0, on_grid[i, -1])

        #Sort the sampling points
        sampling_points = np.sort(sampling_points)
        # print(sampling_points)
        downsampled_X[i, :, :] = X[i, sampling_points, :]
        times_X[i, :] = sampling_points
        # time_X[i, :] = sampling_points / X.shape[1]

    if with_noise:
        noise_X = noise_X_var * torch.randn(downsampled_X.shape)
        return torch.Tensor(downsampled_X) + noise_X, torch.Tensor(times_X)
    else:
        return torch.Tensor(downsampled_X), torch.Tensor(times_X)


def multi_downsample(X, Y, n_points_X, n_points_Y, with_noise=True, noise_X_var=0.5):
    # TODO: delete this function ?
    n_samples = X.shape[0]
    n_dim = X.shape[2]
    X_out = np.zeros((n_samples * n_points_Y, n_points_X, n_dim))
    Y_out = np.zeros((n_samples * n_points_Y, n_dim))
    running_i = 0
    for i in np.arange(n_samples):
        sampling_points = np.random.choice(np.arange(1, X.shape[1]),
                                           size=n_points_X - 1,
                                           replace=False)
        sampling_points_Y = np.random.choice(sampling_points,
                                             size=n_points_Y,
                                             replace=False)
        # Keep the first sampling point
        sampling_points = np.insert(sampling_points, 0, 0)
        # Sort the sampling points
        sampling_points = np.sort(sampling_points)
        X_downsampled = X[i, sampling_points, :]

        for j in range(n_points_Y):
            Y_out[running_i, :] = Y[i, sampling_points_Y[j], :]
            X_out[running_i, :, :] = X_downsampled[:sampling_points_Y[j], :]
            running_i += 1

    if with_noise:
        noise_X = noise_X_var * torch.randn(X_out.shape)
        return torch.Tensor(X_out) + noise_X, Y_out
    else:
        return torch.Tensor(X_out), Y_out
