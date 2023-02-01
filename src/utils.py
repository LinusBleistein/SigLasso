import iisignature as isig
import itertools
import math
import numpy as np
import torch
import torchcde
from typing import Tuple
import warnings


def get_cumulative_moving_sum(X: torch.Tensor, window: int = 3) \
        -> torch.Tensor:
    """
    Returns the cumulative moving sum of a path X along its time dimension
    """
    assert X.ndim == 3, " X must have 3 dimensions: " \
                        "(n_samples, n_points, dim_X)"

    X_cumsum = torch.cumsum(X, dim=1)
    output = X_cumsum[:, window:, :] - X_cumsum[:, :-window, :]
    return torch.cat([X_cumsum[:, :window, :], output], dim=1)


def split_XY_on_grid(X: torch.Tensor, Y: torch.Tensor,
                     grid_Y: torch.Tensor = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    if Y.shape[1] == 1:
            return X, Y[:, 0, :]
    else:
        if grid_Y is None:
            raise ValueError('If Y has more than one observation, the '
                                 'indices of the observations must be passed.')
        assert grid_Y.shape[1] == Y.shape[1], "There should be as many " \
                                              "measurement time indexes in " \
                                              "grid_Y as measurements in Y."
        list_Xs = []
        list_Yfinal = []

        grid_Y = grid_Y.numpy().astype(int)
        for i in range(Y.shape[0]):
            for j in range(grid_Y.shape[1]):
                index_Y = grid_Y[i, j]
                if index_Y == 0:
                    warnings.warn(
                            'An observation of Y at time 0 has been skipped '
                            'since we need at least two observations of X up '
                            'to observation of Y')
                else:
                    # Y has already been downsampled so you should use j
                    # and not index_Y here
                    list_Xs.append(X[i, :index_Y, :])
                    list_Yfinal.append(Y[i, j, :])

        return list_Xs, torch.stack(list_Yfinal)


def fill_forward(X: torch.Tensor, max_length: int) -> torch.Tensor:
    """Fill X to have second dimension max_length."""
    return torch.cat(
        [X, X[-1].unsqueeze(0).expand(max_length - X.size(0), X.size(1))])


def split_and_fill_XY_on_grid(X: torch.Tensor, Y: torch.tensor,
                              grid_Y: torch.Tensor) -> torch.Tensor:
    """
    Output a feature tensor in which every row is X subsampled up to a
    measurement time of the target Y, specified in grid_Y. The submatrix is
    completed using fillfoward as specified here:
    https://github.com/patrick-kidger/torchcde/blob/master/example/irregular_data.py
    """
    assert X.ndim == 3, \
        " X must have 3 dimensions: (n_samples, n_points, dim_X)"
    assert Y.ndim == 3, \
        "Y must have 3 dimensions: (n_samples, n_points, dim_Y)"
    assert grid_Y.ndim == 2, \
        "grid_Y must have 2 dimensions: (n_samples, n_points)"

    # Split sampled to be observed up to observations of Y
    list_Xs, Y_final = split_XY_on_grid(X, Y, grid_Y)

    # Compute the maximum length of the Xs: it is the time horizon until which
    # we will have to fill forward for all individuals.
    max_length = 0
    for i in range(len(list_Xs)):
        if list_Xs[i].shape[0] > max_length:
            max_length = list_Xs[i].shape[0]

    list_filled_Xs = []
    for i in range(len(list_Xs)):
        list_filled_Xs.append(fill_forward(list_Xs[i], max_length))

    return torch.stack(list_filled_Xs), Y_final


def matrix_to_function(X: torch.Tensor, time: torch.Tensor,
                       interpolation_method: str):
    """Turns X into an interpolated function, to use in torchcde.cdeint."""
    assert X.ndim == 3, \
        " X must have 3 dimensions: (n_samples, n_points, dim_X)"

    if interpolation_method == 'cubic':
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            X, t=time)
        return torchcde.CubicSpline(coeffs, t=time)

    elif interpolation_method == 'linear':
        coeffs = torchcde.linear_interpolation_coeffs(X, t=time)
        return torchcde.LinearInterpolation(coeffs, t=time)
    else:
        raise ValueError("interpolation_method must be one of 'linear' or "
                         "'cubic'.")


#TODO: delete this function?
def get_cfi(theta,feature,dim,order):
    """
    Computes the normalized CFI of feature i (numbering of features starts at 0).

    Parameters
    ----------
    theta : coefficient vector
    feature : feature number (numbering starts at 0).
    dim : number of dimensions of the feature path.
    order : signature truncation order.

    Returns
    -------
    The normalized CFI.

    """

    dimension_set = np.arange(0,dim)
    word_list = []
    for m in np.arange(1,order+1):
        word_list.append([p for p in itertools.product(dimension_set,repeat=m)])
    word_list = list(itertools.chain(*word_list))
    cfi = 0
    for i,word in enumerate(word_list):
        if feature in word:
            cfi += theta[i]
    normalizing_constant = (dim**(order)-1)/(dim-1)
    return 1/normalizing_constant*cfi


#TODO: delete this function?
def get_pfi(theta,feature,dim,order):
    """
    Computes the normalized PFI of feature i (numbering starts at 0).

    Parameters
    ----------
    theta: coefficient vector.
    feature: feature number.
    dim: number of dimensions of the feature path.
    order: signature truncation order.

    Returns
    -------
    The normalized PFI.
    """
    pfi = 0
    rolling_dim = -1
    for k in np.arange(order):
        rolling_dim += dim**k
        pfi += np.linalg.norm(theta[rolling_dim+feature])
    return (1/order)*pfi


def get_weights(dim_X: int, order: int) -> np.ndarray:
    """Gets the weighting factor for layer-wise Lasso penalty."""
    weight_vector = np.ones(isig.siglength(dim_X, order))
    position = 1
    for j in range(1, order+1):
        factor = math.factorial(j) / math.sqrt(j)
        weight_vector[position:position + dim_X ** j] *= factor
        position += dim_X ** j
    return weight_vector


def get_weight_matrix(dim_X: int, sig_order: int) -> np.ndarray:
    """Create the diagonal weighting matrix. """
    return np.diag(get_weights(dim_X, sig_order))


def normalize_path(X: torch.Tensor) -> torch.Tensor:
    """Normalize the paths contained in X by their total variation."""
    n_sample = X.shape[0]
    X_copy = X.detach().clone()
    tv_norm = torch.linalg.norm(
        X_copy[:, 1:, :] - X_copy[:, :-1, :], axis=2).sum(axis=1)
    for i in np.arange(n_sample):
        X_copy[i, :, :] *= 1 / tv_norm[i]
    return X_copy


def l2_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Computes L2 distance between X and Y
    """
    assert X.ndim == 3, \
        " X must have 3 dimensions: (n_samples, n_points, dim_X)"
    assert Y.ndim == 3, \
        " X must have 3 dimensions: (n_samples, n_points, dim_Y)"
    return np.mean(
        np.mean(
            np.linalg.norm((np.array(X) - np.array(Y)), axis=2) ** 2,
            axis=1) * (1 / 2)
    )


def mse_on_grid(X_1: torch.Tensor, X_2: torch.tensor,
                grid_1: torch.Tensor = None, grid_2: torch.Tensor = None) \
        -> float:
    """X_1 and X_2 must be of shape (n_samples, n_points_X_1/X_2, channels).
    X_1 can be differentely sampled than X_2, in which case grids must be
    passed. If not None, grid_2 must be of shape (n_samples, n_points_grid)
    we must have grid_2 subset of grid_1."""
    if grid_2 is None:
        # If on_grid is None, compute the mse at the last point
        return np.mean(
            np.linalg.norm(
                np.array(X_1[:, -1, :]) - np.array(X_2[:, -1, :]), axis=1) ** 2)
    else:
        grid_2 = grid_2.numpy().astype(int)
        mse_values = []
        for i in range(X_1.shape[0]):
            mse_sample_i = []
            for j in range(len(grid_2[i, :])):
                pos_X_1 = np.where(grid_1[i, :] == grid_2[i, j])
                mse_sample_i.append(
                    np.linalg.norm(
                        np.array(X_1[i, pos_X_1, :]) - np.array(X_2[i, j, :]),
                        axis=1) ** 2)
            mse_values.append(np.mean(mse_sample_i))
        return np.mean(mse_values)
