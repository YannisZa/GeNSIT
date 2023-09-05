# import os
import sys
import torch
import numpy as np
import torch.distributions as distr

from typing import Union, Tuple
from torch import int32, float32

from multiresticodm.math_utils import log_factorial

def uniform_binary_choice(n:int=1):
    choices = [-1,1]
    x = torch.rand(n)
    var = torch.zeros(n)
    var[x<.5] = choices[0]
    var[x>=.5] = choices[1]
    if len(var) == 1:
        return var[0]
    else:
        return var

def product_multinomial_sample(log_intensity:torch.tensor,rsums:torch.tensor):
    # NUMBA IMPLEMENTATION OF NUMPY RANDOM MULTINOMIAL HAS NUMERICAL INSTABILITIES
    # AVOID USING

    # Compute probabilities for each row
    # log_rsums = torch.tensor([torch.logsumexp(log_intensity[i,:]) for i in range(rsums.shape[0])])
    p = torch.exp(log_intensity-torch.log(rsums).reshape((rsums.shape[0],1)))
    p = p / p.sum(dim=1).reshape((log_intensity.shape[1],1))
    # Initialise table
    tab = torch.empty(p.shape,dtype=int32)
    # Loop through each row and sample it
    for i in range(rsums.shape[0]):
        tab[i,:] = distr.multinomial.Multinomial(total_count=rsums[i],probs=p[i,:])
    return tab


def log_odds_ratio_wrt_intensity(log_intensity: torch.tensor):
    # Extract dimensions of intensity
    nrows,ncols = np.shape(log_intensity)
    # Computes log of odds ratio of intensity
    log_intensity_rowsums = torch.logsumexp(log_intensity,dim=1).unsqueeze(1)
    log_intensity_colsums = torch.logsumexp(log_intensity,dim=0).unsqueeze(0)
    log_intensity_total = torch.logsumexp(log_intensity_rowsums.ravel(),dim=0)
    # Computes log of odds ratio of intensity
    log_or = log_intensity + \
        log_intensity_total - \
        log_intensity_rowsums - \
        log_intensity_colsums
    return log_or


def log_odds_cross_ratio(log_intensity: torch.tensor, cell1: Union[Tuple, list, torch.tensor], cell2: Union[Tuple, list, torch.tensor]):
    # Computes log of odds ratio of intensity
    return log_intensity[cell1] + log_intensity[cell2] - log_intensity[(cell1[0], cell2[1])] - log_intensity[(cell2[0], cell1[1])]


# @njit(cache=PROBABILITY_UTILS_CACHED)
def log_table_likelihood_total_derivative_wrt_x(likelihood_grad,intensity_grad_x):
    # Dimensions
    nrows,ncols = likelihood_grad.shape
    # Reshape necessary objects
    likelihood_grad = likelihood_grad.reshape((nrows*ncols)).to(dtype=float32)
    intensity_grad_x = intensity_grad_x.reshape((nrows*ncols,ncols)).to(dtype=float32)
    # By default chain rule the total derivative is equal to the sum of
    # the derivative of the intensity wrt to x times the derivative of the table likelihood wrt x 
    return (likelihood_grad @ intensity_grad_x)

def log_poisson_pmf_unnormalised(log_intensity:torch.tensor,table:torch.tensor) -> float:
    # Compute log intensity total
    log_total = torch.logsumexp(log_intensity.ravel(),dim=0)
    # Compute log pmf
    return -torch.exp(log_total) + torch.sum(table.to(dtype=float32)*log_intensity) - log_factorial(1,table.ravel()).sum()


def log_poisson_pmf_normalised(log_intensity:torch.tensor,table:torch.tensor) -> float:
    # Return log pmf
    return log_poisson_pmf_unnormalised(log_intensity,table)


def poisson_pmf_ground_truth(log_intensity:torch.tensor,table:torch.tensor,axis:int=None) -> float:
    return torch.exp(log_intensity)


def log_poisson_pmf_jacobian_wrt_intensity(log_intensity:torch.tensor,table:torch.tensor) -> float:
    # Compute intensity
    intensity = torch.exp(log_intensity)
    # Compute likelihood derivative wrt to intensity
    return -1 + table/intensity

def log_multinomial_pmf_unnormalised(log_intensity:torch.tensor,table:torch.tensor) -> float:
    # Normalise log intensites by log rowsums to create multinomial probabilities
    log_probabilities = (log_intensity - torch.logsumexp(log_intensity.ravel(),dim=0)).to(dtype=float32)
    # Compute log pmf
    return table.to(dtype=float32).ravel().dot(log_probabilities.ravel()) - log_factorial(1,table.ravel()).sum()

def log_multinomial_pmf_normalised(log_intensity:torch.tensor,table:torch.tensor,) -> float:
    # Return log pmf
    return ((log_multinomial_pmf_unnormalised(log_intensity,table)) + log_factorial(1,table.sum())).to(dtype=float32)


def multinomial_pmf_ground_truth(log_intensity:torch.tensor,table:torch.tensor,axis:int=None) -> float:
    return torch.sum(table)/torch.sum(torch.exp(log_intensity)) * torch.exp(log_intensity)


def log_multinomial_pmf_jacobian_wrt_intensity(log_intensity:torch.tensor,table:torch.tensor) -> float:
    table = table.to(dtype=float32)
    # Compute intensity
    intensity = torch.exp(log_intensity).to(dtype=float32)
    # Compute likelihood derivative wrt to intensity
    return table/intensity - torch.sum(table)/torch.sum(intensity)


def log_product_multinomial_pmf_unnormalised(log_intensity:torch.tensor,table:torch.tensor) -> float:
    # Get dimensions
    nrows,ncols = np.shape(table)
    # Compute log margins of intensity matrix
    log_rowsums = torch.logsumexp(log_intensity,dim=1).unsqueeze(1)
    # Normalise log intensites by log rowsums to create multinomial probabilities
    log_probabilities = (log_intensity - log_rowsums).to(dtype=float32)
    # Compute log pmf
    return table.to(dtype=float32).ravel().dot(log_probabilities.ravel()) - log_factorial(1,table.ravel()).sum()


def log_product_multinomial_pmf_normalised(log_intensity:torch.tensor,table:torch.tensor) -> float:
    # Return log pmf
    log_target = log_product_multinomial_pmf_unnormalised(log_intensity,table) + \
        log_factorial(1,table.sum(dim=1).to(dtype=int32)).sum()    
    return log_target


def product_multinomial_pmf_ground_truth(log_intensity:torch.tensor,table:torch.tensor,axis:int=None) -> float:
    if axis is None:
        axis = 1
    return torch.sum(table,dim=axis)/torch.sum(torch.exp(log_intensity),dim=axis) * torch.exp(log_intensity)


def log_product_multinomial_pmf_jacobian_wrt_intensity(log_intensity:torch.tensor,table:torch.tensor):
    # Get dimensions
    nrows,ncols = np.shape(table)
    intensity = torch.exp(log_intensity)
    # Compute log margins of intensity and table
    intensity_rowsums = intensity.sum(dim=1)
    table_rowsums = table.sum(dim=1)
    # Normalise log intensites by log colsums to create multinomial probabilities
    return table/intensity - (table_rowsums/intensity_rowsums).reshape((nrows,1))

def log_fishers_hypergeometric_pmf_unnormalised(log_intensity:torch.tensor,table:torch.tensor) -> float:
    # Get dimensions
    dims = np.shape(table)
    # Compute log odds ratio
    log_or = log_odds_ratio_wrt_intensity(log_intensity)
    # Compute log_probabilities
    log_or_colsums = torch.logsumexp(log_or,dim=0).unsqueeze(0).to(dtype=float32)
    log_or_probabilities = log_or - log_or_colsums
    # Compute log pmf
    return table.to(dtype=float32).ravel().dot(log_or_probabilities.ravel()) - log_factorial(1,table.ravel()).sum()

def log_fishers_hypergeometric_pmf_normalised(log_intensity:torch.tensor,table:torch.tensor) -> float:
    # Return log pmf
    return  log_fishers_hypergeometric_pmf_unnormalised(log_intensity,table) + \
        log_factorial(1,table.sum(dim=0).to(dtype=int32)).sum()


def fishers_hypergeometric_pmf_ground_truth(log_intensity:torch.tensor,table:torch.tensor,axis:int=None) -> float:
    if axis is None:
        axis = 0
    # Compute odds ratio
    odds_ratio = torch.exp(log_odds_ratio_wrt_intensity(log_intensity=log_intensity))
    return torch.sum(table,dim=axis) * (odds_ratio / torch.sum(odds_ratio,dim=axis))


def log_fishers_hypergeometric_pmf_jacobian_wrt_intensity(log_intensity:torch.tensor,table:torch.tensor) -> float:
    # Dimensions
    nrows,ncols = np.shape(table)
    # Intensity
    intensity = torch.exp(log_intensity)
    # Compute log margins of intensity
    intensity_rowsums = intensity.sum(dim=1)
    # Compute odd ratio
    odds_ratio = torch.exp(log_odds_ratio_wrt_intensity(log_intensity))
    # Compute odd ratio margins
    odds_ratio_colsums = odds_ratio.sum(dim=0)
    # Compute log margins of table 
    table_rowsums = table.sum(dim=1)
    table_colsums = table.sum(dim=0)
    # Store temp quantity
    temp = (table_colsums/odds_ratio_colsums).T*odds_ratio
    # All entry contributions
    log_pmf_grad = (table-temp)/intensity
    # Rowsum contributions
    log_pmf_grad -= (table_rowsums-temp.sum(dim=1)).reshape((nrows,1))/(intensity_rowsums.reshape((nrows,1)))
    return log_pmf_grad

def table_similarity_measure(tab:torch.tensor,tab0:Union[float,torch.tensor],log_intensity:torch.tensor) -> float:
    if isinstance(tab0,float) or isinstance(tab0,int):
        # Take difference of log target
        return tab0 - log_product_multinomial_pmf_normalised(tab,log_intensity)
    else:
        # Take difference of log target
        return log_product_multinomial_pmf_normalised(tab0,log_intensity) - log_product_multinomial_pmf_normalised(tab,log_intensity)

def multinomial_mode(n,p):
    # Implementing algorithm from "Determination of the modes of a Multinomial distribution"
    # Get length of probability vector
    r = np.shape(p)[0]
    # Make make first towards finding mode
    kis = torch.floor((n+r/2)*p).to(dtype=int32)
    # Sum all elements to check if they sum up to n
    n0 = int(torch.sum(kis))
    # Generate random vector in [0,1]^r
    fis = torch.rand(r,dtype=float32)

    if n0 < n:
        qis = torch.divide(1-fis,kis+1)
        while n0 < n:
            min_index = torch.argmin(qis)
            kis[min_index] += 1
            fis[min_index] -= 1
            qis[min_index] = (1-fis[min_index])/(kis[min_index]+1)
            n0 += 1

    if n0 > n:
        # Compute qis
        qis = torch.divide(fis,kis)
        while n0 > n:
            min_index = torch.argmin(qis)
            kis[min_index] -= 1
            fis[min_index] += 1
            qis[min_index] = fis[min_index]/kis[min_index]
            n0 -= 1

    return kis.to(dtype=int32)


def generate_stopping_times(N:int,k_power:float,seed:int):
    if seed is not None:
        np.random.seed(seed)
    # Get number of stopping times
    stopping_times = np.empty(N)

    for i in range(N):
        n = 1
        u = np.random.uniform(0, 1)
        while(u < torch.pow(n+1, -k_power)):
            n += 1
        stopping_times[i] = n
    return stopping_times

def compute_truncated_infinite_series(N:int,log_weights:torch.tensor,k_power:float) -> torch.tensor:

    # Compute S = Y[0] + sum_i (Y[i] - Y[i-1])/P(N > i) using logarithms
    ln_Y = torch.empty(N+1)
    ln_Y_pos = torch.empty(N+1)
    ln_Y_neg = torch.empty(N)
    
    # Compute increasing averages estimator (Appendix C5)
    for i in range(0, N+1):
        ln_Y[i] = torch.log(i+1) - torch.logsumexp(log_weights[:i+1])
    # Compute first term in series
    ln_Y_pos[0] = ln_Y[0]
    # Compute log of Y[i]/P(N > i) and Y[i-1]/P(N > i)
    for i in range(1, N+1):
        ln_Y_pos[i] = ln_Y[i] + k_power*torch.log(i)
        ln_Y_neg[i-1] = ln_Y[i-1] + k_power*torch.log(i)
    # Sum of all positive and negative terms and convert back to log
    positive_sum = torch.logsumexp(ln_Y_pos)
    negative_sum = torch.logsumexp(ln_Y_neg)

    ret = torch.empty(2)
    # If positive terms are larger in magnitude than negative terms
    if(positive_sum >= negative_sum):
        # This is just computing log(exp(positive_sum) - exp(negative_sum))
        ret[0] = positive_sum + torch.log(1. - torch.exp(negative_sum - positive_sum))
        ret[1] = 1.
    # Otherwise return a negative sign
    else:
        # This is just computing log(exp(negative_sum) - exp(positive_sum))
        ret[0] = negative_sum + torch.log(1. - torch.exp(positive_sum - negative_sum))
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