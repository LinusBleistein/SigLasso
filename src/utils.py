import numpy as np
import iisignature as isig
import torch
import torchcde
import math



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
    weight_vector = np.ones(isig.siglength(dim_X,order))
    position=1
    for j in range(1,order+1):
        factor = math.sqrt(j)/math.factorial(j)
        weight_vector[position:position+dim_X**j]*= factor
        position += dim_X**j
    return weight_vector

def weight_matrix(dim_X,sig_order):
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
    return np.diag(get_weights(dim_X,sig_order))

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


def normalize_path(df):
    """
    Normalizes the paths contained in df by their total variation norm.
    Parameters
    ----------
    df: dataframe of paths.

    Returns
    -------
    Normalized dataframe of paths.
    """
    n_sample = df.shape[0]
    df_copy = df.detach().clone()
    tv_norm = torch.linalg.norm(df_copy[:, 1:, :] - df_copy[:, :-1, :], axis=1).sum(axis=1)
    for i in np.arange(n_sample):
        df_copy[i, :, :] *= 1/tv_norm[i]
    return df_copy


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


def multi_downsample(X, Y, n_points_X, n_points_Y, with_noise=True, noise_X_var=0.5):
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


def downsample(X, n_points_kept, with_noise=False, noise_X_var=0.5):
    """
    Downsamples a dataframe, keeping n_points_kept whose index is between limit_down and limit_up.
    Example: if n_points_kept is set to 5 and limit_down == 3, limit_up == 10, this function will choose 5 points,
    between the third and the 10th sampling point, for every individual.
    Little catch: we need the first sampling point to be included for every individual. Therefore, only n_point_kept - 1
    will be sampled at random.

    Parameters
    ----------
    X: matrix to downsample.
    n_points_kept: number of points to keep for every individual.
    limit_down: lower index limit.
    limit_up: upper index limit.

    Returns
    -------
    A dataframe whose shape is the same than df, expect for the second dimension which is now of shape n_points_kept.
    A second dataframe which logs the sampling points for every individual.
    """
    n_samples = X.shape[0]
    n_dim = X.shape[2]
    downsampled_X = np.zeros((n_samples, n_points_kept, n_dim))
    times_X = np.zeros((n_samples, n_points_kept))
    # time_X = np.zeros((n_samples, n_points_kept))
    for i in np.arange(n_samples):
        sampling_points = np.random.choice(np.arange(1, X.shape[1]),
                                           size=n_points_kept - 2,
                                           replace=False)
        #Keep the first sampling point
        sampling_points = np.insert(sampling_points, 0, 0)
        sampling_points = np.insert(sampling_points, 0, X.shape[1] - 1)
        #Sort the sampling points
        sampling_points = np.sort(sampling_points)
        downsampled_X[i, :, :] = X[i, sampling_points, :]
        times_X[i, :] = sampling_points
        # time_X[i, :] = sampling_points / X.shape[1]

    if with_noise:
        noise_X = noise_X_var * torch.randn(downsampled_X.shape)
        return torch.Tensor(downsampled_X) + noise_X, torch.Tensor(times_X)
    else:
        return torch.Tensor(downsampled_X), torch.Tensor(times_X)


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
    X and Y must be of shape (n_samples, n_points, dim)

    constant, not piecewise linear as we usually do in the theorems -> check this.

    """
    return np.mean(np.linalg.norm((np.array(X) - np.array(Y)) ** 2, axis=2))

def mse_last_point(X, Y):
    """
    X and Y must be of shape (n_samples, n_points, dim)

    """
    return np.mean(
        np.linalg.norm(
            np.array(X[:, -1, :]) - np.array(Y[:, -1, :]), axis=1) ** 2)
