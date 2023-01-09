import numpy as np
import iisignature as isig
import itertools
import math
import torch
import torchcde
from typing import Tuple
import warnings


def get_cumulative_moving_sum(X: torch.Tensor, window: int = 3):
    """
    Returns the cumulative moving sum of a path X along its time dimension
    """
    assert X.ndim == 3, " X must have 3 dimensions: n_samples, time, dim"
    X_cumsum = torch.cumsum(X, dim=1)
    output = X_cumsum[:, window:, :] - X_cumsum[:, :-window, :]
    return torch.cat([X_cumsum[:, :window, :], output], dim=1)


def split_XY_on_grid(X: torch.Tensor, Y: torch.Tensor,
                     grid_Y: torch.Tensor = None) -> Tuple[
    torch.Tensor, torch.Tensor]:
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


def fill_forward(X, max_length):
    return torch.cat(
        [X, X[-1].unsqueeze(0).expand(max_length - X.size(0), X.size(1))])


def split_and_fill_XY_on_grid(X, Y, grid_Y):
    """
    This function takes as input a tensor of feature time series measurements, a tensor of target time series
    measurements, and a matrix of indexes of measurement times of the target.

    It ouputs a feature tensor in which every submatrix is a feature time series sampled up to the corresponding
    measurement time of the target. The submatrix is completed using fillfoward as specified here:
    https://github.com/patrick-kidger/torchcde/blob/master/example/irregular_data.py

    X: feature time series
    Y: target time series
    grid_Y: index of the timepoints at which Y is measured. Must be common with points in time where X is measured.
    Be careful to pass indexes and not measurement times themselves.
    """

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


def get_weights(dim_X,order):
    """
    Gets the weighting factor for layer-wise Lasso penalty.

    Parameters
    ----------
    dim_X : dimension of the feature path.
    order : truncation order of the signature.

    Returns
    -------
    A vector of size s_(dim_X)(order) of weights for the layer-wise Lasso penalty.
    """
    weight_vector = np.ones(isig.siglength(dim_X, order))
    position = 1
    for j in range(1, order+1):
        factor = math.factorial(j) / math.sqrt(j)
        weight_vector[position:position + dim_X ** j] *= factor
        position += dim_X ** j
    return weight_vector


def get_weight_matrix(dim_X, sig_order):
    """
    Create the diagonal weighting matrix.
    Parameters
    ----------
    dim_X : dimension of the feature path.
    sig_order : truncation order of the signature.

    Returns
    -------
    Weighting matrix for the signature features.
    """
    return np.diag(get_weights(dim_X, sig_order))


def matrix_to_function(X, time, interpolation_method):
    """
    Turns data matrix of shape (n_samples, n_points, dim_X) into an interpolated
    function, to use in torchcde.cdeint

    Parameters
    ----------
    X

    Returns
    -------

    """
    if interpolation_method == 'cubic':
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            X, t=time)
        return torchcde.CubicSpline(coeffs, t=time)
    elif interpolation_method == 'linear':
        coeffs = torchcde.linear_interpolation_coeffs(X, t=time)
        return torchcde.LinearInterpolation(coeffs, t=time)
    # else:
    #     if time.shape[0] != X.shape[0]:
    #         raise ValueError('time and X must have same first dimension')
    #     coeffs_list = []
    #     for i in range(X.shape[0]):
    #         coeffs_i = torchcde.hermite_cubic_coefficients_with_backward_differences(
    #             X[i, :, :], t=time[i, :])


def normalize_path(X):
    """
    Normalizes the paths contained in X by their total variation norm.
    Parameters
    ----------
    X: dataframe of paths.

    Returns
    -------
    Normalized dataframe of paths.
    """
    n_sample = X.shape[0]
    X_copy = X.detach().clone()
    tv_norm = torch.linalg.norm(
        X_copy[:, 1:, :] - X_copy[:, :-1, :], axis=2).sum(axis=1)
    for i in np.arange(n_sample):
        X_copy[i, :, :] *= 1/ (0.9 * tv_norm[i])
    return X_copy


def add_noise(df,variance):
    """
    Util function that adds noise to a set of paths.

    Parameters
    ----------
    df : dataframe of paths.
    variance : scale of the noise.

    Returns
    -------
    Noisy version of the dataframe.
    """
    return df + np.sqrt(variance)*np.random.randn(df.shape[0],df.shape[1],df.shape[2])


def MSE(y, X, theta):
    """
    Computes the MSE between an array of output values y and an array of predicted values X@theta.

    Parameters
    ----------
    y : array_like
        Output values, of size n.
    X : array_like
        Feature matrix, of shape (n,d).
    theta : array_like
        parameters, of size d.

    Returns
    -------
    MSE : float
        MSE between y and X@theta.

    """
    return (1/X.shape[0])*np.linalg.norm(y - X @ theta) ** 2


def recontruct_Y(reg, X, length_Y, order):
    """
    Util function that reconstructs a target time series through Taylor expansion of a CDE.

    Parameters
    ----------
    reg: a fitted regression model. Needs to have coef_ and an intercept_ attributes.
    X: driving signals.
    length_Y: lenght of the time series to reconstruct.
    order: order of the signature used for Taylor expansion.

    Returns
    -------

    """
    new_Y = np.zeros((X.shape[0], length_Y, 1))
    for i in range(0, X.shape[1]):
        Xsig_i = isig.sig(X[:, :i+1, :], order)
        new_Y[:, i, 0] = (Xsig_i @ reg.coef_.T).flatten() + reg.intercept_
    return new_Y


def l2_distance(X, Y):
    """
    X and Y must be of shape (n_samples, n_points, channels)

   #TODO: ça ne va pas du tout cette formule

    """
    return np.mean(np.linalg.norm((np.array(X) - np.array(Y)) ** 2, axis=2))


def mse_last_point(X, Y):
    """
    X and Y must be of shape (n_samples, n_points, channels)

    """
    return np.mean(
        np.linalg.norm(
            np.array(X[:, -1, :]) - np.array(Y[:, -1, :]), axis=1) ** 2)
