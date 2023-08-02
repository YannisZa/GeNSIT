import torch
import numpy as np

from tqdm import tqdm
from itertools import product
from itertools import chain, combinations

from numba_progress import ProgressBar
from numba import njit, vectorize, float32, int32, int64, int8, prange, guvectorize

from multiresticodm.utils import flatten
from multiresticodm.global_variables import MATH_UTILS_CACHED


@njit(cache=MATH_UTILS_CACHED)
def log_factorial(end:int,start:int=1):
    return sum([np.log(x) for x in prange(int(start)+1,int(end)+1)])

@vectorize([float32(int32, int32),float32(int64, int32),float32(int8, int32)])
def log_factorial_vectorised(start:np.ndarray,end:np.ndarray):
    return sum([np.log(x) for x in prange(start+1,end+1)])

@njit(cache=MATH_UTILS_CACHED)
def positive_sigmoid(x,scale:float=1.0):
    return 2/(1+np.exp(-x/scale)) - 1

def powerset(iterable):
    # Flatten list
    s = list(flatten(iterable))
    # Remove duplicates
    s = list(set(s))
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))


def normalised_manhattan_distance(tab:np.ndarray,tab0:np.ndarray):

    # Take difference
    difference = tab - tab0

    # Take absolute value of difference and divide by L1 norms
    return np.mean(np.divide(np.absolute(difference), (np.absolute(tab)+np.absolute(tab0)), out=np.zeros_like(tab,dtype=float), where=tab!=0.0))


def map_distance_name_to_function(distance_name):
    if distance_name in globals():
        return globals()[distance_name]
    else:
        raise Exception(f"Distance function {distance_name} does not exist.")

def apply_norm(tab:np.ndarray,tab0:np.ndarray,name:str,**kwargs:dict):
    try:
        norm_function = globals()[name]
    except:
        raise Exception(f'Norm function name {name} not recognized')
    with ProgressBar(total=int(np.shape(tab)[0]),leave=False) as progress:
        norm = norm_function(
            tab=tab,
            tab0=tab0,
            normalisation_constant=kwargs.get('normalisation_constant',None),
            progress_proxy=progress
        )
    return norm

@njit()
def l_0(tab:np.ndarray,tab0:np.ndarray,normalisation_constant:float=None,progress_proxy=None):
    N,I,J = np.shape(tab)
    res = np.zeros((N,I,J),dtype='float32')
    if np.shape(tab0) == (N,I,J):
        for n in prange(N):
            res[n] = (tab[n] - tab0[n]).astype('float32')
            if progress_proxy is not None:
                progress_proxy.update(1)
    else:
       for n in prange(N):
            res[n] = (tab[n] - tab0).astype('float32')
            if progress_proxy is not None:
                progress_proxy.update(1)
    return res

@njit()
def relative_l_0(tab:np.ndarray,tab0:np.ndarray,normalisation_constant:float=None,progress_proxy=None):
    N,I,J = np.shape(tab)
    res = np.zeros((N,I,J),dtype='float32')
    if np.shape(tab0) == (N,I,J):
        for n in prange(N):
            res[n] = (tab[n] - tab0[n]).astype('float32')
            if normalisation_constant is None:
                res[n] = ((tab[n] - tab0[n])/np.sum(tab0[n])).astype('float32')
            else:
                res[n] = ((tab[n] - tab0[n])/normalisation_constant).astype('float32')
            if progress_proxy is not None:
                progress_proxy.update(1)
    else:
       for n in prange(N):
            if normalisation_constant is None:
                res[n] = ((tab[n] - tab0)/np.sum(tab0)).astype('float32')
            else:
                res[n] = ((tab[n] - tab0)/normalisation_constant).astype('float32')
            if progress_proxy is not None:
                progress_proxy.update(1)
    return res

@njit()
def l_1(tab:np.ndarray,tab0:np.ndarray,normalisation_constant:float=None,progress_proxy=None):
    N,I,J = np.shape(tab)
    res = np.zeros((N,I,J),dtype='float32')
    if np.shape(tab0) == (N,I,J):
        for n in prange(N):
            res[n] = (np.absolute(tab[n] - tab0[n])).astype('float32')
            if progress_proxy is not None:
                progress_proxy.update(1)
    else:
       for n in prange(N):
            res[n] = (np.absolute(tab[n] - tab0)).astype('float32')
            if progress_proxy is not None:
                progress_proxy.update(1)
    return res

@njit()
def relative_l_1(tab:np.ndarray,tab0:np.ndarray,normalisation_constant:float=None,progress_proxy=None):
    N,I,J = np.shape(tab)
    res = np.zeros((N,I,J),dtype='float32')
    if int(np.shape(tab0)[0]) == int(N):
        for n in prange(N):
            if normalisation_constant is None:
                res[n] = (np.absolute(tab[n] - tab0[n])/np.sum(np.absolute(tab0[n]))).astype('float32')
            else:
                res[n] = (np.absolute(tab[n] - tab0[n])/normalisation_constant).astype('float32')
            if progress_proxy is not None:
                progress_proxy.update(1)
    else:
       for n in prange(N):
            if normalisation_constant is None:
                res[n] = (np.absolute(tab[n] - tab0)/np.sum(np.absolute(tab0))).astype('float32')
            else:
                res[n] = (np.absolute(tab[n] - tab0)/normalisation_constant).astype('float32')
            if progress_proxy is not None:
                progress_proxy.update(1)
    return res

@njit()
def l_2(tab:np.ndarray,tab0:np.ndarray,progress_proxy):
    N,I,J = np.shape(tab)
    res = np.zeros((N,I,J),dtype='float32')
    if np.shape(tab0) == (N,I,J):
        for n in prange(N):
            res[n] = ((tab[n] - tab0[n])**2).astype('float32')
            if progress_proxy is not None:
                progress_proxy.update(1)
    else:
       for n in prange(N):
            res[n] = ((tab[n] - tab0)**2).astype('float32')
            if progress_proxy is not None:
                progress_proxy.update(1)
    return res

def _p_distance(tab,p):
    if p == 0:
        return tab
    else:
        return np.abs(tab)**p


def p_distance(tab:np.array,tab0:np.array,**kwargs:dict):
    # Type of norm
    p = float(kwargs['kwargs'].get('p_norm',2))
    dims = np.shape(tab)
    if len(dims) > 2:
        return np.array([_p_distance(tab[n]-tab0,p=p) for n in range(dims[0])],dtype='float32').reshape(dims)
    else:
        return _p_distance(tab[0]-tab0,p=p).reshape(dims)

@njit()
def relative_l_2(tab:np.ndarray,tab0:np.ndarray,normalisation_constant:float=None,progress_proxy=None):
    N,I,J = np.shape(tab)
    res = np.zeros((N,I,J),dtype='float32')
    if np.shape(tab0) == (N,I,J):
        for n in prange(N):
            if normalisation_constant is None:
                res[n] = ((tab[n] - tab0[n])**2/np.sum(tab0[n]**2)).astype('float32')
            else:
                res[n] = ((tab[n] - tab0[n])**2/normalisation_constant).astype('float32')
            if progress_proxy is not None:
                progress_proxy.update(1)
    else:
       for n in prange(N):
            if normalisation_constant is None:
                res[n] = ((tab[n] - tab0)**2/np.sum(tab0**2)).astype('float32')
            else:
                res[n] = ((tab[n] - tab0)**2/normalisation_constant).astype('float32')
            if progress_proxy is not None:
                progress_proxy.update(1)
    return res

# @njit
def euclidean_distance(tab1:np.ndarray,tab2:np.ndarray,**kwargs):
    return np.sqrt( np.sum( ((tab1 - tab2)/np.sum(tab1.ravel()))**2 ) )

# @njit
def l_p_distance(tab1:np.ndarray,tab2:np.ndarray,**kwargs):
    return np.linalg.norm( 
        ((tab1 - tab2).ravel()/np.sum(tab2.ravel())), 
        ord=int(kwargs['ord']) if kwargs['ord'].isnumeric() else kwargs['ord']
    )

# @njit
def edit_distance_degree_one(tab:np.ndarray,tab0:np.ndarray,**kwargs):
    dims = kwargs.get('dims',None)
    if dims is not None:
        tab = tab.reshape(dims)
        tab0 = tab0.reshape(dims)
    return np.sum(np.absolute(tab - tab0))/2

def edit_degree_one_error(tab:np.ndarray,tab0:np.ndarray,**kwargs):
    @njit
    def _edit_degree_one_error(tab,tab0,progress_proxy):
        N = np.shape(tab)[0]
        K = np.shape(tab0)[0]
        res = np.zeros((N,1),dtype='float32')
        if K == N:
            for n in prange(N):
                res[n,0] += np.sum(np.absolute(tab[n] - tab0[n]))/2
                progress_proxy.update(1)
        elif K == 1:
            for n in prange(N):
                res[n,0] += np.sum(np.absolute(tab[n] - tab0[0]))/2
                progress_proxy.update(1)
        return res
    
    with ProgressBar(total=int(np.shape(tab)[0]),leave=False) as progress:
        error = _edit_degree_one_error(tab,tab0,progress)
    
    return error

def edit_distance_degree_higher(tab:np.ndarray,tab0:np.ndarray,**kwargs):
    dims = kwargs.get('dims',None)
    if dims is not None:
        tab = tab.reshape(dims)
        tab0 = tab0.reshape(dims)
    return np.sum((tab - tab0) > 0)


def edit_degree_higher_error(tab:np.ndarray,tab0:np.ndarray,**kwargs):
    @njit
    def _edit_degree_higher_error(tab,tab0,progress_proxy):
        N = np.shape(tab)[0]
        K = np.shape(tab0)[0]
        res = np.zeros((N,1),dtype='float32')
        if K == N:
            for n in prange(N):
                res[n,0] = np.sum((tab[n] - tab0[n]) > 0)
                progress_proxy.update(1)
        elif K == 1:
            for n in prange(N):
                res[n,0] = np.sum((tab[n] - tab0[0]) > 0)
                progress_proxy.update(1)
        return res
       
    with ProgressBar(total=int(np.shape(tab)[0]),leave=False) as progress:
        error = _edit_degree_higher_error(tab,tab0,progress)
    
    return error

# @njit
def chi_squared_row_distance(tab1:np.ndarray,tab2:np.ndarray,**kwargs):
    dims = kwargs.get('dims',None)
    tab1 = tab1.reshape(dims)
    tab2 = tab2.reshape(dims)
    rowsums1 = np.where(tab1.sum(axis=1)<=0,1,tab1.sum(axis=1)).reshape((dims[0],1))
    rowsums2 = np.where(tab2.sum(axis=1)<=0,1,tab2.sum(axis=1)).reshape((dims[0],1))
    colsums = np.where(tab1.sum(axis=0)<=0,1,tab1.sum(axis=0)).reshape((1,dims[1]))
    return np.sum((tab1/rowsums1 - tab2/rowsums2)**2 / colsums)

# @njit
def chi_squared_column_distance(tab1:np.ndarray,tab2:np.ndarray,**kwargs):
    dims = kwargs.get('dims',None)
    tab1 = tab1.reshape(dims)
    tab2 = tab2.reshape(dims)
    colsums1 = np.where(tab1.sum(axis=0)<=0,1,tab1.sum(axis=0)).reshape((1,dims[1]))
    colsums2 = np.where(tab2.sum(axis=0)<=0,1,tab2.sum(axis=0)).reshape((1,dims[1]))
    rowsums = np.where(tab1.sum(axis=1)<=0,1,tab1.sum(axis=1)).reshape((dims[0],1))
    return np.sum((tab1/colsums1 - tab2/colsums2)**2 / rowsums)

# @njit
def chi_squared_distance(tab1:np.ndarray,tab2:np.ndarray,**kwargs):
    dims = kwargs.get('dims',None)
    return chi_squared_column_distance(tab1,tab2,dims) + chi_squared_row_distance(tab1,tab2,dims)


def SRMSE(tab:np.array,tab0:np.array,**kwargs:dict):
    """ Computes standardised root mean square error. See equation (22) of
    "A primer for working with the Spatial Interaction modeling (SpInt) module
    in the python spatial analysis library (PySAL)" for more details.

    Parameters
    ----------
    table : np.array [NxM]
        Estimated flows.
    true_table : np.array [NxM]
        Actual flows.

    Returns
    -------
    float
        Standardised root mean square error of t_hat.

    """
    res = np.zeros((np.shape(tab)[0]),dtype='float32')
    @guvectorize([(int32[:,:,:], int32[:,:,:], float32[:]),
            (float32[:,:,:], int32[:,:,:], float32[:]),
            (int32[:,:,:], float32[:,:,:], float32[:]),
            (float32[:,:,:], float32[:,:,:], float32[:])], '(n,d1,d2),(k,d1,d2)->(n)')
    def _SRMSE(tab:np.array,tab0:np.array,res:np.array):
        N = np.shape(tab)[0]
        dims = np.shape(tab)[1:]
        for n in range(N):
            res[n] = (((np.sum((tab0[0] - tab[n])**2) / (np.prod(dims)))**.5) / (np.sum(tab0[0].ravel()) / (np.prod(dims)))).astype('float32')
    # print('\n')
    # print(tab[0])
    # print(np.shape(tab))
    # print(tab0[0].sum())
    # sys.exit()
    # Compute SRMSE
    _SRMSE(tab,tab0,res)

    return res

def SSI(tab:np.array,tab0:np.array,**kwargs:dict):
    """ Computes Sorensen similarity index. See equation (23) of
    "A primer for working with the Spatial Interaction modeling (SpInt) module
    in the python spatial analysis library (PySAL)" for more details.

    Parameters
    ----------
    table : np.array [NxM]
        Estimated flows.
    true_table : np.array [NxM]
        Actual flows.

    Returns
    -------
    float
        Standardised root mean square error of t_hat.

    """
    res = np.zeros((np.shape(tab)[0]),dtype='float32')
    @guvectorize([(int32[:,:,:], int32[:,:,:], float32[:]),
            (float32[:,:,:], int32[:,:,:], float32[:]),
            (int32[:,:,:], float32[:,:,:], float32[:]),
            (float32[:,:,:], float32[:,:,:], float32[:])], '(n,d1,d2),(k,d1,d2)->(n)')
    def _SSI(tab:np.array,tab0:np.array,res:np.array):
        N = np.shape(tab)[0]
        dims = np.shape(tab)[1:]
        for n in range(N):
            denominator = (tab0[0]+tab[n])
            denominator[denominator<=0] = 1
            ratio = ((2*np.minimum(tab0[0],tab[n]))/denominator)
            res[n] = ( np.sum(ratio) / (np.prod(dims))).astype('float32')
    # print('\n')
    # print(tab[0])
    # print(np.shape(tab))
    # print(tab0[0].sum())
    # sys.exit()
    # Compute SRMSE
    _SSI(tab,tab0,res)

    return res

def shannon_entropy(tab:np.array,tab0:np.array,**kwargs:dict):
    """Computes entropy for a table X
    E = sum_{i}^I sum_{j=1}^{J} X_{ij}log(X_{ij})
    
    tab0 : log intensity
    """
    try:
        log_distribution = globals()[kwargs['kwargs']['distribution_name']]
    except:
        raise Exception(f"No distribution function found for distribution name {kwargs['kwargs']['distribution_name']}")
    _tab = np.copy(tab)
    _tab0 = np.copy(tab0)
    # Apply distribution
    with ProgressBar(total=int(np.shape(_tab)[0]),leave=False) as progress:
        res = _shannon_entropy(
            _tab,
            _tab0,
            log_distribution,
            progress
        )
    return res

# @guvectorize([(int32[:,:,:], int32[:,:,:], float32[:], float32[:]),
#             (float32[:,:,:], int32[:,:,:], float32[:], float32[:]),
#             (int32[:,:,:], float32[:,:,:], float32[:], float32[:]),
#             (float32[:,:,:], float32[:,:,:], float32[:], float32[:])], '(n,i,j),(k,i,j),(),()', target='parallel')
@njit(parallel=True)
def _shannon_entropy(tab,tab0,distr,progress_proxy):
    N = np.shape(tab)[0]
    K = np.shape(tab0)[0]
    res = np.zeros((1,1),dtype='float32')
    if K == N:
        for n in prange(N):
            res[0,0] += distr(tab[n],tab0[n])
            progress_proxy.update(1)
    elif K == 1:
        for n in prange(N):
            res[0,0] += distr(tab[n],tab0[0])
            progress_proxy.update(1)
    # Take mean
    return -res/N

def von_neumann_entropy(tab:np.array,tab0:np.array,**kwargs):
    N = int(np.shape(tab)[0])
    res = np.zeros((N,1),dtype='float32')
    
    # with ProgressBar(total=N) as progress:
    for n in tqdm(range(N),leave=False):
        # Convert matrix to square
        matrix = (tab[n]@tab[n].T).astype('float32')
        # Add jitter
        matrix += kwargs['kwargs']['epsilon_threshold'] * np.eye(matrix.shape[0],dtype='float32')
        # Find eigenvalues
        eigenval = np.real(np.linalg.eigvals(matrix))
        # Get all non-zero eigenvalues
        eigenval = eigenval[~np.isclose(eigenval,0,atol=1e-08)]
        # Compute entropy
        res[n,0] = np.sum(-eigenval*np.log(eigenval),dtype='float32')
        # # Update progress bar
        # progress.update(1)
    return res

# @guvectorize([(int32[:,:], int32[:,:,:], float32[:]),
#             (float32[:,:], int32[:,:,:], float32[:]),
#             (int32[:,:], float32[:,:,:], float32[:]),
#             (float32[:,:], float32[:,:,:], float32[:])], '(i,j),(n,i,j)->(n)')
def _sparsity(tab:np.array,tab0:np.array,**kwargs:dict):
    """Computes percentage of zero cells in table

    Parameters
    ----------
    table : np.ndarray
        Description of parameter `table`.

    Returns
    -------
    float
        Description of returned object.

    """
    N,_,_ = np.shape(tab)
    res = np.zeros((N,1))
    for n in range(N):
        res[n] = np.count_nonzero(tab[n]==0)/np.prod(np.shape(tab[n]))
    return res


# @guvectorize([(int32[:,:], int32[:,:,:], float32[:,:,:]),
#             (float32[:,:], int32[:,:,:], float32[:,:,:]),
#             (int32[:,:], float32[:,:,:], float32[:,:,:]),
#             (float32[:,:], float32[:,:,:], float32[:,:,:])], '(i,j),(n,i,j)->(n,i,j)')
def coverage_probability(tab:np.array,tab0:np.array,**kwargs:dict):
    # High posterior density mass
    alpha = 1-kwargs['kwargs'].get('region_mass',0.95)
    # Get dimensions of table
    dims = np.shape(tab)[1:]
    # Ground truth table coverage flags
    cell_coverage = np.zeros((*tuple(dims),1),dtype='float32')
    # Get all cells of table
    cells = sorted([tuple(cell) for cell in product(*[range(dim) for dim in dims])])
    # Compute coverage prob
    for table_index in tqdm(cells,leave=False):
        # Get cell of table and sort all samples
        table_cell_value = np.sort(tab[(...,*table_index)])
        # Get lower and upper bound high posterior density regions
        lower_bound_hpdr,upper_bound_hpdr = calculate_min_interval(table_cell_value,alpha)
        # Compute flag for whether ground truth table is covered
        cell_covered = (tab0[(0,*table_index)] >= lower_bound_hpdr) and (tab0[(0,*table_index)] <= upper_bound_hpdr)
        # Add that flag to a table of flags
        cell_coverage[table_index] = cell_covered
    return cell_coverage


def calculate_min_interval(x, alpha):
    """
    Taken from https://github.com/aloctavodia/Doing_bayesian_data_analysis/blob/a34212340de7e2eb1723046dead980a3a13447ff/hpd.py#L7
    Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """

    N = len(x)
    credible_interval_mass = 1.0-alpha

    interval_index0 = int(np.floor(credible_interval_mass*N))
    n_intervals = N - interval_index0
    interval_width = x[interval_index0:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_index0]
    return hdi_min, hdi_max
