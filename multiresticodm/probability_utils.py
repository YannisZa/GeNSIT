# import os
import torch
import numpy as np

from typing import Union, Tuple
from numba import njit, prange

from multiresticodm.global_variables import PROBABILITY_UTILS_CACHED
from multiresticodm.math_utils import log_factorial_vectorised,log_factorial

def uniform_binary_choice(n:int=1,choices:list=[-1,1]):
    x = np.random.rand(n)
    var = np.zeros(n)
    var[x<.5] = choices[0]
    var[x>=.5] = choices[1]
    if len(var) == 1:
        return var[0]
    else:
        return var

@njit(cache=PROBABILITY_UTILS_CACHED)
def product_multinomial_sample(log_intensity:np.ndarray,rsums:np.ndarray):
    # NUMBA IMPLEMENTATION OF NUMPY RANDOM MULTINOMIAL HAS NUMERICAL INSTABILITIES
    # AVOID USING

    # Compute probabilities for each row
    # log_rsums = np.array([logsumexp(log_intensity[i,:]) for i in range(rsums.shape[0])])
    p = np.exp(log_intensity-np.log(rsums).reshape((rsums.shape[0],1)))
    p = p / p.sum(axis=1).reshape((log_intensity.shape[1],1))
    # Initialise table
    tab = np.empty(p.shape,dtype='int32')
    # Loop through each row and sample it
    for i in range(rsums.shape[0]):
        tab[i,:] = np.random.multinomial(n=rsums[i],pvals=p[i,:])
    return tab


@njit(cache=PROBABILITY_UTILS_CACHED)
def log_odds_ratio_wrt_intensity(log_intensity: np.ndarray):
    # Extract dimensions of intensity
    nrows,ncols = np.shape(log_intensity)
    # Computes log of odds ratio of intensity
    log_intensity_rowsums = np.array([logsumexp(log_intensity[i,:]) for i in range(nrows)]).reshape((nrows,1))
    log_intensity_colsums = np.array([logsumexp(log_intensity[:,j]) for j in range(ncols)]).reshape((1,ncols))
    log_intensity_total = logsumexp(log_intensity_rowsums)
    # Computes log of odds ratio of intensity
    log_or = log_intensity + \
        log_intensity_total - \
        log_intensity_rowsums - \
        log_intensity_colsums
    return log_or


def log_odds_cross_ratio(log_intensity: np.ndarray, cell1: Union[Tuple, list, np.array], cell2: Union[Tuple, list, np.array]):
    # Computes log of odds ratio of intensity
    return log_intensity[cell1] + log_intensity[cell2] - log_intensity[(cell1[0], cell2[1])] - log_intensity[(cell2[0], cell1[1])]


# @njit(cache=PROBABILITY_UTILS_CACHED)
def log_table_likelihood_total_derivative_wrt_x(likelihood_grad,intensity_grad_x):
    # Dimensions
    nrows,ncols = likelihood_grad.shape
    # Reshape necessary objects
    likelihood_grad = likelihood_grad.reshape((nrows*ncols)).astype('float32')
    intensity_grad_x = intensity_grad_x.reshape((nrows*ncols,ncols)).astype('float32')
    # By default chain rule the total derivative is equal to the sum of
    # the derivative of the intensity wrt to x times the derivative of the table likelihood wrt x 
    return (likelihood_grad @ intensity_grad_x)

@njit(cache=PROBABILITY_UTILS_CACHED)
def log_poisson_pmf_unnormalised(log_intensity:np.ndarray,table:np.ndarray) -> float:
    # Compute log intensity total
    log_total = logsumexp(log_intensity.ravel())
    # Compute log pmf
    return -np.exp(log_total) + np.sum(table.astype('float32')*log_intensity) - log_factorial_vectorised(1,table.ravel()).sum()


@njit(cache=PROBABILITY_UTILS_CACHED)
def log_poisson_pmf_normalised(log_intensity:np.ndarray,table:np.ndarray) -> float:
    # Return log pmf
    return log_poisson_pmf_unnormalised(log_intensity,table)


@njit(cache=PROBABILITY_UTILS_CACHED)
def poisson_pmf_ground_truth(log_intensity:np.ndarray,table:np.ndarray,axis:int=None) -> float:
    return np.exp(log_intensity)


@njit(cache=PROBABILITY_UTILS_CACHED)
def log_poisson_pmf_jacobian_wrt_intensity(log_intensity:np.ndarray,table:np.ndarray) -> float:
    # Compute intensity
    intensity = np.exp(log_intensity)
    # Compute likelihood derivative wrt to intensity
    return -1 + table/intensity

@njit(cache=PROBABILITY_UTILS_CACHED)
def log_multinomial_pmf_unnormalised(log_intensity:np.ndarray,table:np.ndarray) -> float:
    # Normalise log intensites by log rowsums to create multinomial probabilities
    log_probabilities = (log_intensity - logsumexp(log_intensity.ravel())).astype('float32')
    # Compute log pmf
    # return np.einsum('ij,ij',table.astype('float32'),log_probabilities)
    return table.astype('float32').ravel().dot(log_probabilities.ravel()) - log_factorial_vectorised(1,table.ravel()).sum()

@njit(cache=PROBABILITY_UTILS_CACHED)
def log_multinomial_pmf_normalised(log_intensity:np.ndarray,table:np.ndarray,) -> float:
    # Return log pmf
    return np.float32(log_multinomial_pmf_unnormalised(log_intensity,table)) + log_factorial(1,table.sum())


@njit(cache=PROBABILITY_UTILS_CACHED)
def multinomial_pmf_ground_truth(log_intensity:np.ndarray,table:np.ndarray,axis:int=None) -> float:
    return np.sum(table)/np.sum(np.exp(log_intensity)) * np.exp(log_intensity)


@njit(cache=PROBABILITY_UTILS_CACHED)
def log_multinomial_pmf_jacobian_wrt_intensity(log_intensity:np.ndarray,table:np.ndarray) -> float:
    table = table.astype('float32')
    # Compute intensity
    intensity = np.exp(log_intensity).astype('float32')
    # Compute likelihood derivative wrt to intensity
    return table/intensity - np.sum(table)/np.sum(intensity)


@njit(cache=PROBABILITY_UTILS_CACHED)
def log_product_multinomial_pmf_unnormalised(log_intensity:np.ndarray,table:np.ndarray) -> float:
    # Get dimensions
    nrows,ncols = np.shape(table)
    # Compute log margins of intensity matrix
    log_rowsums = np.asarray([[logsumexp(log_intensity[i,:]) for i in range(nrows)]],dtype='float32')
    # Normalise log intensites by log rowsums to create multinomial probabilities
    log_probabilities = (log_intensity - log_rowsums.reshape((nrows,1))).astype('float32')
    # Compute log pmf
    return table.astype('float32').ravel().dot(log_probabilities.ravel()) - log_factorial_vectorised(1,table.ravel()).sum()


@njit(cache=PROBABILITY_UTILS_CACHED)
def log_product_multinomial_pmf_normalised(log_intensity:np.ndarray,table:np.ndarray) -> float:
    # Return log pmf
    log_target = log_product_multinomial_pmf_unnormalised(log_intensity,table) + \
        log_factorial_vectorised(1,table.sum(axis=1).astype('int32')).sum()    
    return log_target


@njit(cache=PROBABILITY_UTILS_CACHED)
def product_multinomial_pmf_ground_truth(log_intensity:np.ndarray,table:np.ndarray,axis:int=None) -> float:
    if axis is None:
        axis = 1
    return np.sum(table,axis=axis)/np.sum(np.exp(log_intensity),axis=axis) * np.exp(log_intensity)


@njit(cache=PROBABILITY_UTILS_CACHED)
def log_product_multinomial_pmf_jacobian_wrt_intensity(log_intensity:np.ndarray,table:np.ndarray):
    # Get dimensions
    nrows,ncols = np.shape(table)
    intensity = np.exp(log_intensity)
    # Compute log margins of intensity and table
    intensity_rowsums = intensity.sum(axis=1)
    table_rowsums = table.sum(axis=1)
    # Normalise log intensites by log colsums to create multinomial probabilities
    return table/intensity - (table_rowsums/intensity_rowsums).reshape((nrows,1))

@njit(cache=PROBABILITY_UTILS_CACHED)
def log_fishers_hypergeometric_pmf_unnormalised(log_intensity:np.ndarray,table:np.ndarray) -> float:
    # Get dimensions
    dims = np.shape(table)
    # Compute log odds ratio
    log_or = log_odds_ratio_wrt_intensity(log_intensity)
    # Compute log_probabilities
    log_or_colsums = np.asarray([[logsumexp(log_or[:,j]) for j in range(dims[1])]],dtype='float32')
    log_or_probabilities = log_or - log_or_colsums.reshape((1,dims[1]))
    # Compute log pmf
    return table.astype('float32').ravel().dot(log_or_probabilities.ravel()) - log_factorial_vectorised(1,table.ravel()).sum()

@njit(cache=PROBABILITY_UTILS_CACHED)
def log_fishers_hypergeometric_pmf_normalised(log_intensity:np.ndarray,table:np.ndarray) -> float:
    # Return log pmf
    return  log_fishers_hypergeometric_pmf_unnormalised(log_intensity,table) + \
        log_factorial_vectorised(1,table.sum(axis=0).astype('int32')).sum()


@njit(cache=PROBABILITY_UTILS_CACHED)
def fishers_hypergeometric_pmf_ground_truth(log_intensity:np.ndarray,table:np.ndarray,axis:int=None) -> float:
    if axis is None:
        axis = 0
    # Compute odds ratio
    odds_ratio = np.exp(log_odds_ratio_wrt_intensity(log_intensity=log_intensity))
    return np.sum(table,axis=axis) * (odds_ratio / np.sum(odds_ratio,axis=axis))


@njit(cache=PROBABILITY_UTILS_CACHED)
def log_fishers_hypergeometric_pmf_jacobian_wrt_intensity(log_intensity:np.ndarray,table:np.ndarray) -> float:
    # Dimensions
    nrows,ncols = np.shape(table)
    # Intensity
    intensity = np.exp(log_intensity)
    # Compute log margins of intensity
    intensity_rowsums = intensity.sum(axis=1)
    # Compute odd ratio
    odds_ratio = np.exp(log_odds_ratio_wrt_intensity(log_intensity))
    # Compute odd ratio margins
    odds_ratio_colsums = odds_ratio.sum(axis=0)
    # Compute log margins of table 
    table_rowsums = table.sum(axis=1)
    table_colsums = table.sum(axis=0)
    # Store temp quantity
    temp = (table_colsums/odds_ratio_colsums).T*odds_ratio
    # All entry contributions
    log_pmf_grad = (table-temp)/intensity
    # Rowsum contributions
    log_pmf_grad -= (table_rowsums-temp.sum(axis=1)).reshape((nrows,1))/(intensity_rowsums.reshape((nrows,1)))
    return log_pmf_grad

@njit(cache=PROBABILITY_UTILS_CACHED)
def table_similarity_measure(tab:np.ndarray,tab0:Union[float,np.ndarray],log_intensity:np.ndarray) -> float:
    if isinstance(tab0,float) or isinstance(tab0,int):
        # Take difference of log target
        return tab0 - log_product_multinomial_pmf_normalised(np.asarray(tab),np.asarray(log_intensity))
    else:
        # Take difference of log target
        return log_product_multinomial_pmf_normalised(np.asarray(tab0),np.asarray(log_intensity)) - log_product_multinomial_pmf_normalised(np.asarray(tab),np.asarray(log_intensity))

@njit(cache=PROBABILITY_UTILS_CACHED)
def multinomial_mode(n,p):
    # Implementing algorithm from "Determination of the modes of a Multinomial distribution"
    # Get length of probability vector
    r = np.shape(p)[0]
    # Make make first towards finding mode
    kis = np.floor((n+r/2)*p).astype('int32')
    # Sum all elements to check if they sum up to n
    n0 = int(np.sum(kis))
    # Generate random vector in [0,1]^r
    fis = np.random.uniform(0,1,r)

    if n0 < n:
        qis = np.divide(1-fis,kis+1)
        while n0 < n:
            min_index = np.argmin(qis)
            kis[min_index] += 1
            fis[min_index] -= 1
            qis[min_index] = (1-fis[min_index])/(kis[min_index]+1)
            n0 += 1

    if n0 > n:
        # Compute qis
        qis = np.divide(fis,kis)
        while n0 > n:
            min_index = np.argmin(qis)
            kis[min_index] -= 1
            fis[min_index] += 1
            qis[min_index] = fis[min_index]/kis[min_index]
            n0 -= 1

    return kis.astype('int32')


@njit
def generate_stopping_times(N:int,k_power:float,seed:int):
    if seed is not None:
        np.random.seed(seed)
    # Get number of stopping times
    stopping_times = np.empty(N)

    for i in prange(N):
        n = 1
        u = np.random.uniform(0, 1)
        while(u < np.power(n+1, -k_power)):
            n += 1
        stopping_times[i] = n
    return stopping_times

@njit
def compute_truncated_infinite_series(N:int,log_weights:np.array,k_power:float) -> np.array:

    # Compute S = Y[0] + sum_i (Y[i] - Y[i-1])/P(N > i) using logarithms
    ln_Y = np.empty(N+1)
    ln_Y_pos = np.empty(N+1)
    ln_Y_neg = np.empty(N)
    
    # Compute increasing averages estimator (Appendix C5)
    for i in range(0, N+1):
        ln_Y[i] = np.log(i+1) - logsumexp(log_weights[:i+1])
    # Compute first term in series
    ln_Y_pos[0] = ln_Y[0]
    # Compute log of Y[i]/P(N > i) and Y[i-1]/P(N > i)
    for i in range(1, N+1):
        ln_Y_pos[i] = ln_Y[i] + k_power*np.log(i)
        ln_Y_neg[i-1] = ln_Y[i-1] + k_power*np.log(i)
    # Sum of all positive and negative terms and convert back to log
    positive_sum = logsumexp(ln_Y_pos)
    negative_sum = logsumexp(ln_Y_neg)

    ret = np.empty(2)
    # If positive terms are larger in magnitude than negative terms
    if(positive_sum >= negative_sum):
        # This is just computing log(exp(positive_sum) - exp(negative_sum))
        ret[0] = positive_sum + np.log(1. - np.exp(negative_sum - positive_sum))
        ret[1] = 1.
    # Otherwise return a negative sign
    else:
        # This is just computing log(exp(negative_sum) - exp(positive_sum))
        ret[0] = negative_sum + np.log(1. - np.exp(positive_sum - negative_sum))
        ret[1] = -1.

    return ret


def random_tensor(
    *, distribution: str, parameters: dict, size: tuple, device: str = "cpu", **__
) -> torch.Tensor:

    """Generates a random tensor according to a distribution.

    :param distribution: the type of distribution. Can be 'uniform' or 'normal'.
    :param parameters: the parameters relevant to the respective distribution
    :param size: the size of the random tensor
    :param device: the device onto which to load the data
    :param __: additional kwargs (ignored)
    :return: the tensor of random variables
    """

    # Uniform distribution in an interval
    if distribution == "uniform":

        l, u = parameters.get("lower"), parameters.get("upper")
        if l > u:
            raise ValueError(
                f"Upper bound must be greater or equal to lower bound; got {l} and {u}!"
            )

        return torch.tensor((u - l), dtype=torch.float) * torch.rand(
            size, dtype=torch.float, device=device
        ) + torch.tensor(l, dtype=torch.float)

    # Normal distribution
    elif distribution == "normal":
        return torch.normal(
            parameters.get("mean"),
            parameters.get("std"),
            size=size,
            device=device,
            dtype=torch.float,
        )

    else:
        raise ValueError(f"Unrecognised distribution type {distribution}!")