from copy import deepcopy
import sys
import torch
import numpy as np
import xarray as xr

from numpy import shape 
from scipy import optimize
from torch import int32, float32
from itertools import chain, combinations
from multiresticodm.global_variables import DATA_SCHEMA

from multiresticodm.utils import flatten


def log_factorial_sum(arr):
    return torch.lgamma(arr+1).sum()

def positive_sigmoid(x,scale:float=1.0):
    return 2/(1+torch.exp(-x/scale)) - 1

def powerset(iterable):
    # Flatten list
    s = list(flatten(iterable))
    # Remove duplicates
    s = list(set(s))
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))


def normalised_manhattan_distance(tab:xr.DataArray,tab0:xr.DataArray):

    # Take difference
    difference = tab - tab0

    # Take absolute value of difference and divide by L1 norms
    return torch.mean(torch.divide(torch.absolute(difference), (torch.absolute(tab)+torch.absolute(tab0)), out=torch.zeros_like(tab,dtype=float32), where=tab!=0.0))


def map_distance_name_to_function(distance_name):
    if distance_name in globals():
        return globals()[distance_name]
    else:
        raise Exception(f"Distance function {distance_name} does not exist.")

def apply_norm(tab:xr.DataArray,tab0:xr.DataArray,name:str,**kwargs:dict):
    try:
        norm_function = globals()[name]
    except:
        raise Exception(f'Norm function name {name} not recognized')
    norm = norm_function(
        tab=tab,
        tab0=tab0,
        normalisation_constant=kwargs.get('normalisation_constant',None),
        progress_proxy=None
    )
    return norm

def l_0(tab:xr.DataArray,tab0:xr.DataArray,normalisation_constant:float=None,progress_proxy=None):
    N,I,J = shape(tab)
    if shape(tab0) != (N,I,J):
        tab0 = torch.unsqueeze(tab0,dim=0)
    res = (tab - tab0).to(dtype=float32)
    return res


def relative_l_0(tab:xr.DataArray,tab0:xr.DataArray,normalisation_constant:float=None,progress_proxy=None):
    N,I,J = shape(tab)
    if shape(tab0) != (N,I,J):
        tab0 = torch.unsqueeze(tab0,dim=0)
    res = (tab - tab0).to(device=float32)
    if normalisation_constant is None:
        res = ((tab - tab0)/torch.sum(tab0)).to(dtype=float32)
    else:
        res = ((tab - tab0)/normalisation_constant).to(dtype=float32)
    return res

def l_1(tab:xr.DataArray,tab0:xr.DataArray,normalisation_constant:float=None,progress_proxy=None):
    N,I,J = shape(tab)
    if shape(tab0) != (N,I,J):
        tab0 = torch.unsqueeze(tab0,dim=0)
    res = torch.absolute(tab - tab0).to(device=float32)
    return res

def relative_l_1(tab:xr.DataArray,tab0:xr.DataArray,normalisation_constant:float=None,progress_proxy=None):
    N,I,J = shape(tab)
    if shape(tab0) != (N,I,J):
        tab0 = torch.unsqueeze(tab0,dim=0)
    if normalisation_constant is None:
        res = (torch.absolute(tab - tab0)/torch.sum(torch.absolute(tab0))).astype('float32')
    else:
        res = (torch.absolute(tab - tab0)/normalisation_constant).astype('float32')
    return res

def l_2(tab:xr.DataArray,tab0:xr.DataArray,progress_proxy):
    N,I,J = shape(tab)
    if shape(tab0) != (N,I,J):
        tab0 = torch.unsqueeze(tab0,dim=0)
    res = torch.pow((tab - tab0),2).to(dtype=float32)
    return res


def p_distance(tab:xr.DataArray,tab0:xr.DataArray,**kwargs:dict):
    # Type of norm
    p = float(kwargs.get('p_norm',2))
    # Get dimensions
    dims = shape(tab)
    # Return difference in case of 0-norm
    if p == 0:
        return tab-tab0
    
    return torch.pow(torch.abs(tab-tab0),p).float().reshape(dims)

def torch_optimize(init,**kwargs):
    function = kwargs['function']

    def fit(xx):
        xx = torch.tensor(xx,dtype=float32,device=kwargs.get('device','cpu'),requires_grad=True)
        # Convert torch to numpy
        y,y_grad = function(xx,**kwargs)
        return y.detach().cpu().numpy(), y_grad.detach().cpu().numpy()

    try:
        res = optimize.minimize(
            fit,
            init,
            jac=True,
            options={'disp': False}
        )
        return res.x
    except:
        return None


def relative_l_2(tab:xr.DataArray,tab0:xr.DataArray,normalisation_constant:float=None,progress_proxy=None):
    N,I,J = shape(tab)
    if shape(tab0) != (N,I,J):
        tab0 = torch.unsqueeze(tab0,dim=0)
    if normalisation_constant is None:
        res = torch.pow(tab - tab0,2)/torch.sum(torch.pow(tab0,2)).to(dtype=float32)
    else:
        res = (torch.pow(tab - tab0,2)/normalisation_constant).to(dtype=float32)
    return res

def euclidean_distance(tab1:xr.DataArray,tab2:xr.DataArray,**kwargs):
    return torch.sqrt( torch.sum( torch.pow(((tab1 - tab2)/torch.sum(tab1.ravel())),2) ) )

def l_p_distance(tab1:xr.DataArray,tab2:xr.DataArray,**kwargs):
    return torch.linalg.norm( 
        ((tab1 - tab2).ravel()/torch.sum(tab2.ravel())), 
        ord=int(kwargs['ord']) if kwargs['ord'].isnumeric() else kwargs['ord']
    )

def edit_distance_degree_one(tab:xr.DataArray,tab0:xr.DataArray,**kwargs):
    dims = kwargs.get('dims',None)
    if dims is not None:
        tab = tab.reshape(dims)
        tab0 = tab0.reshape(dims)
    return torch.sum(torch.absolute(tab - tab0))/2

def edit_degree_one_error(tab:xr.DataArray,tab0:xr.DataArray,**kwargs):
    return torch.sum(torch.absolute(tab - tab0,dim=0))/2

def edit_distance_degree_higher(tab:xr.DataArray,tab0:xr.DataArray,**kwargs):
    return torch.sum((tab - tab0) > 0,axis=slice(1,None))

def chi_squared_row_distance(tab1:xr.DataArray,tab2:xr.DataArray,**kwargs):
    dims = kwargs.get('dims',None)
    tab1 = tab1.reshape(dims)
    tab2 = tab2.reshape(dims)
    rowsums1 = np.where(tab1.sum(axis=1)<=0,1,tab1.sum(axis=1)).reshape((dims[0],1))
    rowsums2 = np.where(tab2.sum(axis=1)<=0,1,tab2.sum(axis=1)).reshape((dims[0],1))
    colsums = np.where(tab1.sum(axis=0)<=0,1,tab1.sum(axis=0)).reshape((1,dims[1]))
    return np.sum((tab1/rowsums1 - tab2/rowsums2)**2 / colsums)

def chi_squared_column_distance(tab1:xr.DataArray,tab2:xr.DataArray,**kwargs):
    dims = kwargs.get('dims',None)
    tab1 = tab1.reshape(dims)
    tab2 = tab2.reshape(dims)
    colsums1 = np.where(tab1.sum(axis=0)<=0,1,tab1.sum(axis=0)).reshape((1,dims[1]))
    colsums2 = np.where(tab2.sum(axis=0)<=0,1,tab2.sum(axis=0)).reshape((1,dims[1]))
    rowsums = np.where(tab1.sum(axis=1)<=0,1,tab1.sum(axis=1)).reshape((dims[0],1))
    return np.sum((tab1/colsums1 - tab2/colsums2)**2 / rowsums)

def chi_squared_distance(tab1:xr.DataArray,tab2:xr.DataArray,**kwargs):
    dims = kwargs.get('dims',None)
    return chi_squared_column_distance(tab1,tab2,dims) + chi_squared_row_distance(tab1,tab2,dims)


def srmse(tab:xr.DataArray,tab0:xr.DataArray,**kwargs:dict):
    """ Computes standardised root mean square error. See equation (22) of
    "A primer for working with the Spatial Interaction modeling (SpInt) module
    in the python spatial analysis library (PySAL)" for more details.

    Parameters
    ----------
    table : xr.DataArray [NxM]
        Estimated flows.
    true_table : xr.DataArray [NxM]
        Actual flows.

    Returns
    -------
    float
        Standardised root mean square error of t_hat.

    """
    tab = tab.astype('float32')
    tab0 = tab0.astype('float32')
    tab,tab0 = xr.broadcast(tab, tab0)
    tab,tab0 = xr.align(tab, tab0, join='exact')
    numerator = ( ((tab0-tab)**2).sum(dim=['origin','destination']) / tab.size) ** 0.5
    denominator = tab0.sum(dim=['origin','destination']) / tab0.size
    srmse = numerator / denominator
    
    return srmse

def ssi(tab:xr.DataArray,tab0:xr.DataArray,**kwargs:dict):
    """ Computes Sorensen similarity index. See equation (23) of
    "A primer for working with the Spatial Interaction modeling (SpInt) module
    in the python spatial analysis library (PySAL)" for more details.

    Parameters
    ----------
    table : xr.DataArray [NxM]
        Estimated flows.
    true_table : xr.DataArray [NxM]
        Actual flows.

    Returns
    -------
    float
        Standardised root mean square error of t_hat.

    """
    # Compute denominator
    denominator = (tab0 + tab)
    denominator = xr.where(denominator <= 0, 1., denominator)
    # Compute numerator
    numerator = 2*xr.minimum(tab0,tab)
    # Compute SSI
    ssi = xr.divide(numerator,denominator).mean(dim=['origin','destination'])
    return ssi

def shannon_entropy(tab:xr.DataArray,tab0:xr.DataArray,**kwargs:dict):
    """Computes entropy for a table X
    E = sum_{i}^I sum_{j=1}^{J} X_{ij}log(X_{ij})
    
    tab0 : log intensity
    """
    try:
        log_distribution = globals()[kwargs['distribution_name']]
    except:
        raise Exception(f"No distribution function found for distribution name {kwargs['distribution_name']}")
    _tab = np.copy(tab)
    _tab0 = np.copy(tab0)
    # Apply distribution
    res = _shannon_entropy(
        _tab,
        _tab0,
        log_distribution,
        None
    )
    return res


def von_neumann_entropy(tab:xr.DataArray,tab0:xr.DataArray,**kwargs):
    # Convert matrix to square
    matrix = (tab@tab.T).astype('float32')
    # Add jitter
    matrix += kwargs['epsilon_threshold'] * torch.eye(matrix.shape[0],dtype='float32')
    # Find eigenvalues
    eigenval = torch.real(torch.linalg.eigvals(matrix))
    # Get all non-zero eigenvalues
    eigenval = eigenval[~torch.isclose(eigenval,0,atol=1e-08)]
    # Compute entropy
    res = torch.sum(-eigenval*torch.log(eigenval)).to(dtype=float32)

    return res

def sparsity(tab:xr.DataArray,tab0:xr.DataArray,**kwargs:dict):
    """Computes percentage of zero cells in table

    Parameters
    ----------
    table : xr.DataArray
        Description of parameter `table`.

    Returns
    -------
    float
        Description of returned object.

    """
    N,_,_ = shape(tab)
    res = np.count_nonzero(tab==0)/np.prod(tab.size)
    return res


def coverage_probability(tab:xr.DataArray,tab0:xr.DataArray,**kwargs:dict):
    # High posterior density mass
    alpha = 1-kwargs.get('region_mass',0.95)
    # Copy unstacked dimension sizes
    sizes = deepcopy(tab.sizes)
    unstacked_coords = deepcopy(tab.coords)
    # Stack iteration-related dimensions and space-related dimensions
    stacked_id_dims = set(['iter','seed']).intersection(tab.dims)
    tab = tab.stack(id=list(stacked_id_dims),space=['origin','destination'])
    # Get all dimensions comprising sweep
    sweep_dims = deepcopy(tab.indexes['sweep'].names)
    # Copy stacked dimensions and coordinates
    stacked_dims = deepcopy(tab.dims)
    # Get cell of table and sort all samples
    tab = np.sort(tab,axis=stacked_dims.index('space'))
    
    # Get lower and upper bound high posterior density regions
    lower_bound_hpdr,upper_bound_hpdr = calculate_min_interval(
        tab,
        alpha,
        dims=list(stacked_dims)
    )
    # # Reshape boundaries
    new_shape = [sizes['sweep'],
                 sizes['origin'],
                 sizes['destination']]
    lower_bound_hpdr = lower_bound_hpdr.reshape(tuple(new_shape))
    upper_bound_hpdr = upper_bound_hpdr.reshape(tuple(new_shape))
    # Reshape ground truth table to match 
    tab0 = np.repeat(tab0.values[np.newaxis,...], sizes['sweep'],axis=0)

    # Compute flag for whether ground truth table is covered
    cell_coverage = np.logical_and(np.greater_equal(tab0,lower_bound_hpdr), np.less_equal(tab0,upper_bound_hpdr))
    # Return xr data array
    return xr.DataArray(
        cell_coverage,
        name = 'coverage_probability',
        dims = ['sweep','origin','destination'],
        coords = dict(
            sweep = unstacked_coords['sweep'],
            origin = unstacked_coords['origin'],
            destination = unstacked_coords['destination']
        ),
        attrs = dict(
            region_mass = kwargs.get('region_mass',0.95)
        )
    )


def calculate_min_interval(x, alpha, dims):
    """
    Taken from https://github.com/aloctavodia/Doing_bayesian_data_analysis/blob/a34212340de7e2eb1723046dead980a3a13447ff/hpd.py#L7
    Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """
    N = x.shape[dims.index('id')]
    credible_interval_mass = 1.0-alpha
    # Get number of intervals within that bass
    interval_index0 = int(np.floor(credible_interval_mass*N))
    n_intervals = N - interval_index0
    # Get all possible credible_interval_mass% probability intervals
    right_boundary_index = tuple([slice(None) if d != 'id' else slice(interval_index0,None) for d in dims])
    right_boundary = x[right_boundary_index]
    left_boundary_index = tuple([slice(None) if d != 'id' else slice(0,n_intervals) for d in dims])
    left_boundary = x[left_boundary_index]
    interval_width = right_boundary - left_boundary

    if interval_width.shape[dims.index('id')] == 0:
        raise ValueError('Too few elements for interval calculation')
    # Find index of smallest probability interval
    min_idx = np.argmin(interval_width,axis=dims.index('id'))
    min_idx = np.expand_dims(min_idx,axis=dims.index('id'))
    # Get hpd boundaries
    hdi_min = np.take(x,min_idx)
    hdi_max = np.take(x,min_idx+interval_index0)
    return np.squeeze(hdi_min), np.squeeze(hdi_max)


def logsumexp(input, dim=None, keepdim=False):
    max_val, _ = input.max(dim=dim, keepdim=True)
    output = max_val + (input - max_val).exp().sum(dim=dim, keepdim=True).log()
    
    if not keepdim:
        output = output.squeeze(dim)
    
    return output