# import os
import sys
import torch
import numpy as np
import xarray as xr
import torch.distributions as distr

from typing import Union, Tuple
from torch import int32, float32

from multiresticodm.utils.misc_utils import set_seed
from multiresticodm.utils.math_utils import log_factorial_sum, logfactorialsum, logsumexp

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

def log_odds_ratio_wrt_intensity_xarray(log_intensity: torch.tensor):
    # Extract dimensions of intensity
    origins = log_intensity.coords['origin'].values
    destinations = log_intensity.sizes['destination'].values
    # Computes log of odds ratio of intensity
    log_intensity_rowsums = logsumexp(log_intensity,dim=['destination']).expand_dims(dim={'origins':origins})
    log_intensity_colsums = logsumexp(log_intensity,dim=['origin']).expand_dims(dim={'destination':destinations})
    log_intensity, 
    log_intensity_rowsums,
    log_intensity_colsums = xr.align(
        log_intensity,
        log_intensity_rowsums,
        log_intensity_colsums,
        join='override'
    )
    log_intensity_total = logsumexp(log_intensity_rowsums,dim=['origin','destination'])
    # Computes log of odds ratio of intensity
    log_or = log_intensity + \
        log_intensity_total - \
        log_intensity_rowsums - \
        log_intensity_colsums
    return log_or


def log_odds_cross_ratio(log_intensity: torch.tensor, cell1: Union[Tuple, list, torch.tensor], cell2: Union[Tuple, list, torch.tensor]):
    # Computes log of odds ratio of intensity
    return log_intensity[cell1] + log_intensity[cell2] - log_intensity[(cell1[0], cell2[1])] - log_intensity[(cell2[0], cell1[1])]



def log_poisson_pmf_unnormalised(table:torch.tensor,log_intensity:torch.tensor,**kwargs) -> float:
    # Compute log pmf
    return - torch.sum(log_intensity) + (table.to(dtype=float32)*log_intensity).sum() - log_factorial_sum(table.ravel())

def log_poisson_loss(table:xr.DataArray,log_intensity:xr.DataArray,**kwargs) -> float:
    # Compute negative log pmf
    term1 = log_intensity.sum(['origin','destination'])
    term2 = -(table.astype('float32') * log_intensity).sum(['origin','destination']) 
    term3 = logfactorialsum(table,['origin','destination'])
    term1,term2,term3 = xr.align(term1,term2,term3, join = 'override')
    return term1 + term2 + term3

def log_poisson_pmf_normalised(table:torch.tensor,log_intensity:torch.tensor,**kwargs) -> float:
    # Return log pmf
    return log_poisson_pmf_unnormalised(log_intensity,table)

def poisson_pmf_ground_truth(table:torch.tensor,log_intensity:torch.tensor,axis:int=None) -> float:
    return torch.exp(log_intensity)



def log_multinomial_pmf_unnormalised(table:torch.tensor,log_intensity:torch.tensor,**kwargs) -> float:
    # Normalise log intensites by log rowsums to create multinomial probabilities
    log_probabilities = (log_intensity - torch.logsumexp(log_intensity.ravel()))
    # Compute log pmf
    return (table.to(dtype=float32).ravel()*log_probabilities.ravel()).sum() - log_factorial_sum(table.ravel())

def log_multinomial_loss(table:xr.DataArray,log_intensity:xr.DataArray,**kwargs) -> float:
    # Compute negative log pmf
    log_probabilities = (log_intensity - logsumexp(log_intensity,dim=['origin','destination']))
    # Compute log pmf
    table = table.astype('float32')
    
    term1 = -(table*log_probabilities).sum(['origin','destination']) 
    term2 = logfactorialsum(table,['origin','destination'])
    term1,term2 = xr.align(term1,term2, join = 'override')
    return term1 + term2


def log_multinomial_pmf_normalised(table:torch.tensor,log_intensity:torch.tensor,) -> float:
    # Return log pmf
    return ((log_multinomial_pmf_unnormalised(log_intensity,table)) + log_factorial_sum(table.sum())).to(dtype=float32)

def multinomial_pmf_ground_truth(table:torch.tensor,log_intensity:torch.tensor,axis:int=None) -> float:
    return torch.sum(table)/torch.sum(torch.exp(log_intensity)) * torch.exp(log_intensity)



def log_product_multinomial_pmf_unnormalised(table:torch.tensor,log_intensity:torch.tensor,**kwargs) -> float:
    # Compute log margins of intensity matrix
    log_rowsums = torch.logsumexp(log_intensity,dim=1).unsqueeze(1)
    # Normalise log intensites by log rowsums to create multinomial probabilities
    log_probabilities = (log_intensity - log_rowsums)
    # Compute log pmf
    return (table.to(dtype=float32).ravel()*log_probabilities.ravel()).sum() - log_factorial_sum(table.ravel())

def log_product_multinomial_loss(table:xr.DataArray,log_intensity:xr.DataArray,**kwargs) -> float:
    # Compute log margins of intensity matrix
    log_rowsums = logsumexp(log_intensity,dim=['origin']).expand_dims(dim={"origin": log_intensity.coords['origin'].values})
    log_intensity,log_rowsums = xr.align(log_intensity,log_rowsums,join='override')
    # Normalise log intensites by log rowsums to create multinomial probabilities
    log_probabilities = (log_intensity - log_rowsums)
    # Compute log pmf
    term1 = -(table.astype('float32')*log_probabilities).sum(['origin','destination']) 
    term2 = logfactorialsum(table,['origin','destination'])
    term1,term2 = xr.align(term1,term2, join = 'override')
    return term1 + term2


def log_product_multinomial_pmf_normalised(table:torch.tensor,log_intensity:torch.tensor,**kwargs) -> float:
    # Return log pmf
    log_target = log_product_multinomial_pmf_unnormalised(log_intensity,table) + \
        log_factorial_sum(table.sum(dim=1).to(dtype=int32))
    return log_target

def product_multinomial_pmf_ground_truth(table:torch.tensor,log_intensity:torch.tensor,axis:int=None) -> float:
    if axis is None:
        axis = 1
    return torch.sum(table,dim=axis)/torch.sum(torch.exp(log_intensity),dim=axis) * torch.exp(log_intensity)


def log_fishers_hypergeometric_pmf_unnormalised(table:torch.tensor,log_intensity:torch.tensor,**kwargs) -> float:
    # Compute log odds ratio
    log_or = log_odds_ratio_wrt_intensity(log_intensity)
    # Compute log_probabilities
    log_or_colsums = torch.logsumexp(log_or,dim=0).unsqueeze(0).to(dtype=float32)
    log_or_probabilities = log_or - log_or_colsums
    # Compute log pmf
    return (table.to(dtype=float32).ravel() * log_or_probabilities.ravel()).sum() - log_factorial_sum(table.ravel()).sum()

def log_fishers_hypergeometric_loss(table:xr.DataArray,log_intensity:xr.DataArray,**kwargs) -> float:
    origins = log_intensity.coords['origin'].values
    # Compute log odds ratio
    log_or = log_odds_ratio_wrt_intensity_xarray(log_intensity)
    # Compute log_probabilities
    log_or_colsums = logsumexp(log_or,dim=['origin']).expand_dims(dim={'origin':origins})
    log_or_probabilities = log_or - log_or_colsums
    # Align
    log_or,log_or_colsums = xr.align(log_or,log_or_colsums,join='override')
    table,log_or_probabilities = xr.align(table,log_or_probabilities,join='override')
    # Compute log pmf
    term1 = -(table.astype('float32') * log_or_probabilities).sum(['origin','destination'])
    term2 = logfactorialsum(table,['origin','destination'])
    term1,term2 = xr.align(term1,term2, join = 'override')
    return term1 + term2

def log_fishers_hypergeometric_pmf_normalised(table:torch.tensor,log_intensity:torch.tensor,**kwargs) -> float:
    # Return log pmf
    return  log_fishers_hypergeometric_pmf_unnormalised(log_intensity,table) + \
        log_factorial_sum(table.sum(dim=0).to(dtype=int32))

def fishers_hypergeometric_pmf_ground_truth(table:torch.tensor,log_intensity:torch.tensor,axis:int=None) -> float:
    if axis is None:
        axis = 0
    # Compute odds ratio
    odds_ratio = torch.exp(log_odds_ratio_wrt_intensity(log_intensity=log_intensity))
    return torch.sum(table,dim=axis) * (odds_ratio / torch.sum(odds_ratio,dim=axis))


def table_similarity_measure(tab:torch.tensor,tab0:Union[float,torch.tensor],log_intensity:torch.tensor,**kwargs) -> float:
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
        set_seed(seed)
    # Get number of stopping times
    stopping_times = torch.empty(N)

    for i in range(N):
        n = 1
        u = torch.rand(1)
        while(u < torch.pow(torch.tensor(n+1).float(), -k_power)):
            n += 1
        stopping_times[i] = n
    return stopping_times

def compute_truncated_infinite_series(N:int,log_weights:torch.tensor,k_power:float,device:str) -> torch.tensor:

    # Compute S = Y[0] + sum_{i>1} (Y[i] - Y[i-1])/P(N > i) using logarithms
    
    # Compute increasing averages estimator (Appendix C5)
    ln_Y = torch.log(torch.range(1, N+2,device=device))
    for i in range(0, N+1):
        ln_Y[i] -=  torch.logsumexp(log_weights[:i+1].ravel(),dim=0)
    # Compute log of Y[i]/P(N > i) and Y[i-1]/P(N > i)
    ln_Y_pos = ln_Y + torch.cat((torch.tensor([0],device=device),k_power*torch.log(torch.range(1, N+1,device=device))))
    ln_Y_neg = ln_Y[:(N+1)] + k_power*torch.log(torch.range(1, N+1,device=device))
    
    # Sum of all positive and negative terms and convert back to log
    positive_sum = torch.logsumexp(ln_Y_pos,dim=0)
    negative_sum = torch.logsumexp(ln_Y_neg,dim=0)

    ret = torch.empty(2,device=device)
    # If positive terms are larger in magnitude than negative terms
    if(positive_sum >= negative_sum):
        # This is just computing log(exp(positive_sum) - exp(negative_sum))
        ret[0] = positive_sum + torch.log(torch.tensor(1.) - torch.exp(negative_sum - positive_sum))
        ret[1] = 1.

        if not torch.all(torch.isfinite(ret[0])):
            print(log_weights)
            print(positive_sum,negative_sum)
            print(torch.exp(negative_sum - positive_sum))
            print(torch.log(torch.tensor(1.) - torch.exp(negative_sum - positive_sum)))
            raise Exception('Positive >= Negative')

    # Otherwise return a negative sign
    else:
        # This is just computing log(exp(negative_sum) - exp(positive_sum))
        ret[0] = negative_sum + torch.log(torch.tensor(1.) - torch.exp(positive_sum - negative_sum))
        ret[1] = -1.

        if not torch.all(torch.isfinite(ret[0])):
            print(log_weights)
            print(positive_sum,negative_sum)
            print(torch.exp(positive_sum - negative_sum))
            print( torch.log(torch.tensor(1.) - torch.exp(positive_sum - negative_sum)))
            raise Exception('Negative >= Positive')

    return ret


def random_vector(
    *, distribution: str, parameters: dict, size: tuple, **__
) -> np.ndarray:

    """Generates a random vector according to a distribution.

    :param distribution: the type of distribution. Can be 'uniform' or 'normal'.
    :param parameters: the parameters relevant to the respective distribution
    :param size: the size of the random tensor
    :param device: the device onto which to load the data
    :param __: additional kwargs (ignored)
    :return: the tensor of random variables
    """

    # Uniform distribution in an interval
    if distribution == "uniform":

        l, u = parameters.get("lower",0.0), parameters.get("upper",1.0)
        if l > u:
            raise ValueError(
                f"Upper bound must be greater or equal to lower bound; got {l} and {u}!"
            )

        return (u - l) * np.random.uniform(size=size,dtype='float32') + l

    # Normal distribution
    elif distribution == "normal":
        return np.random.normal(
            loc=parameters.get("mean",1.0),
            scale=parameters.get("std",0.1),
            size=size
        ).astype('float32')
    
    elif distribution == "poisson":
        return np.random.poisson(
            lam=parameters.get("lam",1.0),
            size=size
        ).astype('float32')

    else:
        raise ValueError(f"Unrecognised distribution type {distribution}!")
    
def sample_multinomial_row(i,msum,margin_probabilities,free_cells,axis_uncostrained_flat,device:str='cpu',ndims:int=2):
    # Get cells for specific row
    current_cells = free_cells[free_cells[:,axis_uncostrained_flat].ravel() == i,:]
    free_indices = [ current_cells[:,i] for i in range(ndims) ]
    updated_cells = distr.multinomial.Multinomial(
        total_count = msum.item(),
        probs = margin_probabilities[free_indices].ravel()
    ).sample()
    # Update free cells
    return updated_cells.to(device=device,dtype=int32)