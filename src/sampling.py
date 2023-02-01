import numpy as np
import torch
from typing import Tuple


def downsample(X: torch.Tensor, n_points_kept: int, with_noise: bool = False,
               noise_X_var: float = 0.5, keep_first: bool = False,
               keep_last: bool = False, on_grid: torch.Tensor = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """Downsample a dataframe, sampling n_points_kept randomly. If on_grid is
    passed, the points are subsampled on this grid.
    """
    assert X.ndim == 3, "X must have 3 dimensions: " \
                        "(n_samples, n_points, dim_X)"
    if on_grid is not None:
        assert on_grid.ndim == 2, 'If not None, on_grid must have 2 ' \
                              'dimensions: (n_samples, n_points)'
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


