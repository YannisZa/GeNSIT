import sys
import torch
import numpy as np
import xarray as xr

from tqdm import tqdm
from numpy import shape 
from copy import deepcopy
from scipy import optimize,stats
from torch import int32, float32
from scipy.special import gammaln
from itertools import chain, combinations

from gensit.utils.misc_utils import flatten,is_sorted


def log_factorial_sum(arr):
    return torch.lgamma(arr+1).sum()

def positive_sigmoid(x,scale:float = 1.0):
    return 2/(1+torch.exp(-x/scale)) - 1

def powerset(iterable):
    # Flatten list
    s = list(flatten(iterable))
    # Remove duplicates
    s = list(set(s))
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))


def normalised_manhattan_distance(prediction:xr.DataArray,ground_truth:xr.DataArray=None):

    # Take difference
    difference = prediction - ground_truth

    # Take absolute value of difference and divide by L1 norms
    return torch.mean(torch.divide(torch.absolute(difference), (torch.absolute(prediction)+torch.absolute(ground_truth)), out = torch.zeros_like(prediction,dtype = float32), where = prediction!=0.0))


def map_distance_name_to_function(distance_name):
    if distance_name in globals():
        return globals()[distance_name]
    else:
        raise Exception(f"Distance function {distance_name} does not exist.")

def apply_norm(prediction:xr.DataArray,ground_truth:xr.DataArray=None,name:str='',**kwargs):
    try:
        norm_function = globals()[name]
    except:
        raise Exception(f'Norm function name {name} not recognized')
    norm = norm_function(
        prediction = prediction,
        ground_truth = ground_truth,
        normalisation_constant = kwargs.get('normalisation_constant',None),
        progress_proxy = None
    )
    return norm

def l_0(prediction:xr.DataArray,ground_truth:xr.DataArray=None,normalisation_constant:float = None,progress_proxy = None):
    N,I,J = shape(prediction)
    if shape(ground_truth) != (N,I,J):
        ground_truth = torch.unsqueeze(ground_truth,dim = 0)
    res = (prediction - ground_truth).to(dtype = float32)
    return res


def relative_l_0(prediction:xr.DataArray,ground_truth:xr.DataArray=None,normalisation_constant:float = None,progress_proxy = None):
    N,I,J = shape(prediction)
    if shape(ground_truth) != (N,I,J):
        ground_truth = torch.unsqueeze(ground_truth,dim = 0)
    res = (prediction - ground_truth).to(device = float32)
    if normalisation_constant is None:
        res = ((prediction - ground_truth)/torch.sum(ground_truth)).to(dtype = float32)
    else:
        res = ((prediction - ground_truth)/normalisation_constant).to(dtype = float32)
    return res

def l_1(prediction:xr.DataArray,ground_truth:xr.DataArray=None,**kwargs):
    prediction = prediction.astype('float32')
    ground_truth = ground_truth.astype('float32')
    prediction,ground_truth = xr.broadcast(prediction,ground_truth)
    prediction,ground_truth = xr.align(prediction,ground_truth, join='exact')
    mask = kwargs.get('mask',None)
    if mask is not None:
        # Apply mask
        prediction = prediction.where(mask)
        ground_truth = ground_truth.where(mask)
    
    l1_error = abs(
        prediction
        - ground_truth
    )
    return l1_error


def relative_l_1(prediction:xr.DataArray,ground_truth:xr.DataArray=None,**kwargs):
    prediction = prediction.astype('float32')
    ground_truth = ground_truth.astype('float32')
    prediction,ground_truth = xr.broadcast(prediction,ground_truth)
    prediction,ground_truth = xr.align(prediction,ground_truth, join='exact')
    mask = kwargs.get('mask',None)
    if mask is not None:
        # Apply mask
        prediction = prediction.where(mask)
        ground_truth = ground_truth.where(mask)
    
    relative_l1_error = abs(
        prediction
        - ground_truth
    )
    relative_l1_error /= np.min(ground_truth,1)
    return relative_l1_error

def l_2(prediction:xr.DataArray,ground_truth:xr.DataArray=None,progress_proxy=None):
    N,I,J = shape(prediction)
    if shape(ground_truth) != (N,I,J):
        ground_truth = torch.unsqueeze(ground_truth,dim = 0)
    res = torch.pow((prediction - ground_truth),2).to(dtype = float32)
    return res


def p_distance(prediction:xr.DataArray,ground_truth:xr.DataArray=None,**kwargs):
    # Type of norm
    p = float(kwargs.get('p_norm',2))
    # Get dimensions
    dims = shape(prediction)
    # Return difference in case of 0-norm
    if p == 0:
        return prediction-ground_truth
    
    return torch.pow(torch.abs(prediction-ground_truth),p).float().reshape(dims)

def torch_optimize(init,**kwargs):
    function = kwargs['function']

    def fit(xx):
        xx = torch.tensor(xx,dtype = float32,device = kwargs.get('device','cpu'),requires_grad = True)
        # Convert torch to numpy
        y,y_grad = function(xx,**kwargs)
        return y.detach().cpu().numpy(), y_grad.detach().cpu().numpy()

    try:
        res = optimize.minimize(
            fit,
            init,
            jac = True,
            options={'disp': False}
        )
        return res.x
    except:
        return None


def relative_l_2(prediction:xr.DataArray,ground_truth:xr.DataArray=None,normalisation_constant:float = None,progress_proxy = None):
    N,I,J = shape(prediction)
    if shape(ground_truth) != (N,I,J):
        ground_truth = torch.unsqueeze(ground_truth,dim = 0)
    if normalisation_constant is None:
        res = torch.pow(prediction - ground_truth,2)/torch.sum(torch.pow(ground_truth,2)).to(dtype = float32)
    else:
        res = (torch.pow(prediction - ground_truth,2)/normalisation_constant).to(dtype = float32)
    return res

def euclidean_distance(tab1:xr.DataArray,tab2:xr.DataArray,**kwargs):
    return torch.sqrt( torch.sum( torch.pow(((tab1 - tab2)/torch.sum(tab1.ravel())),2) ) )

def l_p_distance(tab1:xr.DataArray,tab2:xr.DataArray,**kwargs):
    return torch.linalg.norm( 
        ((tab1 - tab2).ravel()/torch.sum(tab2.ravel())), 
        ord = int(kwargs['ord']) if kwargs['ord'].isnumeric() else kwargs['ord']
    )

def edit_distance_degree_one(prediction:xr.DataArray,ground_truth:xr.DataArray=None,**kwargs):
    dims = kwargs.get("dims",None)
    if dims is not None:
        prediction = prediction.reshape(dims)
        ground_truth = ground_truth.reshape(dims)
    return torch.sum(torch.absolute(prediction - ground_truth))/2

def edit_degree_one_error(prediction:xr.DataArray,ground_truth:xr.DataArray=None,**kwargs):
    return torch.sum(torch.absolute(prediction - ground_truth,dim = 0))/2

def edit_distance_degree_higher(prediction:xr.DataArray,ground_truth:xr.DataArray=None,**kwargs):
    return torch.sum((prediction - ground_truth) > 0,axis = slice(1,None))

def chi_squared_row_distance(tab1:xr.DataArray,tab2:xr.DataArray,**kwargs):
    dims = kwargs.get("dims",None)
    tab1 = tab1.reshape(dims)
    tab2 = tab2.reshape(dims)
    rowsums1 = np.where(tab1.sum(axis = 1)<=0,1,tab1.sum(axis = 1)).reshape((dims[0],1))
    rowsums2 = np.where(tab2.sum(axis = 1)<=0,1,tab2.sum(axis = 1)).reshape((dims[0],1))
    colsums = np.where(tab1.sum(axis = 0)<=0,1,tab1.sum(axis = 0)).reshape((1,dims[1]))
    return np.sum((tab1/rowsums1 - tab2/rowsums2)**2 / colsums)

def chi_squared_column_distance(tab1:xr.DataArray,tab2:xr.DataArray,**kwargs):
    dims = kwargs.get("dims",None)
    tab1 = tab1.reshape(dims)
    tab2 = tab2.reshape(dims)
    colsums1 = np.where(tab1.sum(axis = 0)<=0,1,tab1.sum(axis = 0)).reshape((1,dims[1]))
    colsums2 = np.where(tab2.sum(axis = 0)<=0,1,tab2.sum(axis = 0)).reshape((1,dims[1]))
    rowsums = np.where(tab1.sum(axis = 1)<=0,1,tab1.sum(axis = 1)).reshape((dims[0],1))
    return np.sum((tab1/colsums1 - tab2/colsums2)**2 / rowsums)

def chi_squared_distance(tab1:xr.DataArray,tab2:xr.DataArray,**kwargs):
    dims = kwargs.get("dims",None)
    return chi_squared_column_distance(tab1,tab2,dims) + chi_squared_row_distance(tab1,tab2,dims)


def srmse(prediction:xr.DataArray,ground_truth:xr.DataArray=None,**kwargs):
    """ Computes standardised root mean square error. See equation (22) of
    "A primer for working with the Spatial Interaction modeling (SpInt) module
    in the python spatial analysis library (PySAL)" for more details.

    Parameters
    ----------
    prediction : xr.DataArray [NxM]
        Estimated flows.
    ground_truth : xr.DataArray [NxM]
        Actual flows.

    Returns
    -------
    float
        Standardised root mean square error of t_hat.

    """
    # print('prediction total',prediction.sum(['origin','destination']).values.tolist())
    # print('ground truth total',ground_truth.sum(['origin','destination']).values.tolist())

    prediction = prediction.astype('float32')
    ground_truth = ground_truth.astype('float32')
    prediction,ground_truth = xr.broadcast(prediction,ground_truth)
    prediction,ground_truth = xr.align(prediction,ground_truth, join='exact')
    mask = kwargs.get('mask',None)
    if mask is not None:
        # Apply mask
        prediction = prediction.where(mask)
        ground_truth = ground_truth.where(mask)
    

    numerator = ( ((prediction - ground_truth)**2).sum(dim=['origin','destination'],skipna=True) / (~np.isnan(prediction)).sum()) ** 0.5
    denominator = ground_truth.sum(dim=['origin','destination'],skipna=True) / (~np.isnan(ground_truth)).sum()
    srmse = numerator / denominator
    
    return srmse

def ssi(prediction:xr.DataArray,ground_truth:xr.DataArray=None,**kwargs):
    """ Computes Sorensen similarity index. See equation (23) of
    "A primer for working with the Spatial Interaction modeling (SpInt) module
    in the python spatial analysis library (PySAL)" for more details.

    Parameters
    ----------
    prediction : xr.DataArray [NxM]
        Estimated flows.
    ground_truth : xr.DataArray [NxM]
        Actual flows.

    Returns
    -------
    float
        Standardised root mean square error of t_hat.

    """
    prediction = prediction.astype('float32')
    ground_truth = ground_truth.astype('float32')
    prediction,ground_truth = xr.broadcast(prediction,ground_truth)
    prediction,ground_truth = xr.align(prediction,ground_truth, join='exact')
    mask = kwargs.get('mask',None)
    if mask is not None:
        # Apply mask
        ground_truth = ground_truth.where(mask)
        prediction = prediction.where(mask)

    # Compute denominator
    denominator = (ground_truth + prediction)
    # Compute numerator
    numerator = 2*np.minimum(ground_truth,prediction)
    ratio = numerator / denominator

    # Compute SSI
    ssi = ratio.mean(dim=['origin','destination'],skipna=True)
    return ssi

def markov_basis_distance(prediction:xr.DataArray,ground_truth:xr.DataArray=None,**kwargs):
    mask = kwargs.get('mask',None)
        
    if mask is not None:
        # Apply mask
        prediction = prediction.where(mask)
        ground_truth = ground_truth.where(mask)
    
    return np.abs(prediction - ground_truth).sum(
        dims = ['origin','destination'],
        dtype = 'float64',
        skipna = True
    ) / 2

def coverage_probability(prediction:xr.DataArray,ground_truth:xr.DataArray=None,**kwargs):
    # Deepcopy
    prediction = deepcopy(prediction)

    # Get region mass
    region_mass = kwargs.get('region_mass',0.95)
    
    # Get iteration dimension name
    dim = kwargs.get('dim','id')
    
    # Get test mask
    mask = kwargs.get('mask',None)
    
    # High posterior density mass
    alpha = 1-region_mass
    
    # Stack iteration-related dimensions and space-related dimensions
    prediction = prediction.stack(space=['origin','destination'])
    
    # Copy stacked dimensions and coordinates
    stacked_dims = deepcopy(prediction.dims)

    # Sort all samples by iteration-seed
    prediction[:] = np.sort(prediction.values, axis = stacked_dims.index(dim))
    
    # Get lower and upper bound high posterior density regions
    lower_bound_hpdr,upper_bound_hpdr = calculate_min_interval(
        prediction,
        alpha,
        dim = dim
    )
    # Compute flag for whether ground truth table is covered
    cell_coverage = (ground_truth >= lower_bound_hpdr) & (ground_truth <= upper_bound_hpdr)

    if mask is not None:
        # Apply mask
        cell_coverage = cell_coverage.where(mask)
    
    # Update coordinates to include region mass
    return cell_coverage


def calculate_min_interval(x, alpha, **kwargs):
    """
    Taken from https://github.com/aloctavodia/Doing_bayesian_data_analysis/blob/a34212340de7e2eb1723046dead980a3a13447ff/hpd.py#L7
    Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    """
    dim = kwargs.get('dim','id')
    N = x.sizes[dim]
    credible_interval_mass = 1.0-alpha
    
    # Get number of intervals within that mass
    interval_index0 = int(np.floor(credible_interval_mass*N))
    n_intervals = N - interval_index0

    # Get all possible credible_interval_mass% probability intervals
    left_boundary = x.isel(**{dim : slice(0,n_intervals)})
    right_boundary = x.isel(**{dim : slice(interval_index0,None)})
    left_boundary,right_boundary = xr.align(left_boundary,right_boundary,join='override')
    interval_width = right_boundary - left_boundary
    
    # Make sure that all samples are sorted
    right_boundary_sorted = is_sorted(right_boundary)
    left_boundary_sorted = is_sorted(left_boundary)
    try:
        assert left_boundary_sorted and right_boundary_sorted
    except:
        raise ValueError(f"Samples were not correctly sorted (left: {left_boundary_sorted}, right: {right_boundary_sorted})")
    
    # Make sure that the high posterior density interval is not zero
    if interval_width.sizes[dim] == 0:
        raise ValueError('Too few elements for interval calculation')
    
    # Find indices of tails of high density region
    min_idx = interval_width.argmin(dim).unstack('space')
    max_idx = min_idx.copy(deep = True)
    max_idx[:] = min_idx[:] + interval_index0
    
    # Remove space stack
    x = x.unstack('space')
    
    # Get hpd boundaries
    hdi_min = x.isel(**{dim:min_idx})
    hdi_max = x.isel(**{dim:max_idx})

    return hdi_min, hdi_max

def mean_absolute_residual_percentage_error(prediction:xr.DataArray,ground_truth:xr.DataArray,**kwargs):
    mask = kwargs.get('mask',None)
    dim = kwargs.get('dim','origin')
        
    if mask is not None:
        # Apply mask
        prediction = prediction.where(mask,drop=True)
        ground_truth = ground_truth.where(mask,drop=True)

    relative_marginal_l1_error = abs(
        prediction.sum(dim,dtype='float64')
        - ground_truth.sum(dim,dtype='float64')
    )
    relative_marginal_l1_error /= ground_truth.sum(dim,dtype='float64')
    if 'seed' in relative_marginal_l1_error.dims:
        relative_marginal_l1_error = relative_marginal_l1_error.mean(['seed'],dtype='float64')
    return relative_marginal_l1_error



def von_neumann_entropy(prediction:xr.DataArray,ground_truth:xr.DataArray=None,**kwargs):
    # Convert matrix to square
    matrix = (prediction@prediction.T).astype('float32')
    # Add jitter
    matrix += kwargs['epsilon_threshold'] * torch.eye(matrix.shape[0],dtype='float32')
    # Find eigenvalues
    eigenval = torch.real(torch.linalg.eigvals(matrix))
    # Get all non-zero eigenvalues
    eigenval = eigenval[~torch.isclose(eigenval,0,atol = 1e-08)]
    # Compute entropy
    res = torch.sum(-eigenval*torch.log(eigenval)).to(dtype = float32)

    return res

def sparsity(prediction:xr.DataArray,ground_truth:xr.DataArray=None,**kwargs):
    """Computes percentage of zero cells in table

    Parameters
    ----------
    prediction : xr.DataArray
        Description of parameter `table`.

    Returns
    -------
    float
        Description of returned object.

    """
    N,_,_ = shape(prediction)
    res = np.count_nonzero(prediction==0)/np.prod(prediction.size)
    return res

def logsumexp(input, dim = None):
    max_val = input.max(dim = dim)
    return max_val + np.log((np.exp((input - max_val)).sum(dim = dim)))

def logfactorialsum(arr, dim = None):
    if dim is None or len(dim) <= 0:
        return gammaln(arr+1).sum()
    else:
        return gammaln(arr+1).sum(dim = dim)
    
def roundint(data):
    return np.rint(data)

def sample_mean(data,**kwargs):
    if kwargs.get('dim',None) is None:
        return data.mean()
    else:
        return data.mean(kwargs.get('dim',None))

def signed_mean(data,signs,**kwargs):
    # Compute moments
    data,signs = xr.align(data,signs, join='exact')
    numerator = data.dot(signs,dims = kwargs['dim'])
    denominator = signs.sum(kwargs['dim'])
    numerator,denominator = xr.align(numerator,denominator, join='exact')
    return (numerator/denominator)

def signed_var(data,signs,**kwargs):
    # Compute mean
    samples_mean = signed_mean(data,signs,**kwargs)
    # Compute moments
    numerator = (data**2).dot(signs,dims = kwargs['dim'])
    denominator = signs.sum(kwargs['dim'])
    numerator,denominator = xr.align(numerator,denominator, join='exact')
    return (numerator/denominator - samples_mean**2)

def skew_sigmoid_torch(v):
    v = 1 / ( 1 + torch.exp(-50*v) )
    return v

def skew_sigmoid_numpy(v):
    v = 1 / ( 1 + np.exp(-50*v) )
    return v

# Calculates the gradient penalty loss for WGAN GP
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.FloatTensor(real_samples.shape[0]).fill_(1.0).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def reshape_null_data(x):
    # Create a mask for rows that are not all NaN
    non_nan_rows = ~np.all(np.isnan(x), axis=1)

    # Create a mask for columns that are not all NaN
    non_nan_cols = ~np.all(np.isnan(x), axis=0)

    # Apply the masks to filter out the rows and columns
    return x[non_nan_rows][:, non_nan_cols]


def subtract(x,y,**kwargs):
    x = x.astype('float32')
    y = y.astype('float32')
    x,y = xr.broadcast(x,y)
    x,y = xr.align(x,y, join='exact')
    mask = kwargs.get('mask',None)
    if mask is not None:
        # Apply mask
        y = y.where(mask)
        x = x.where(mask)

    return x - y

def l2(x,y,**kwargs):
    x = x.astype('float32')
    y = y.astype('float32')
    x,y = xr.broadcast(x,y)
    x,y = xr.align(x,y, join='exact')
    mask = kwargs.get('mask',None)
    if mask is not None:
        # Apply mask
        y = y.where(mask)
        x = x.where(mask)

    return (x - y)**2

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def perform_one_tailed_ttest(sample_a: np.ndarray, sample_b: np.ndarray, sample_a_name:str="sample_a", sample_b_name:str="sample_b", alpha: float = 0.05) -> dict:
    """
    Performs a one-tailed Welch's t-test:
        H0: mean(sample_a) >= mean(sample_b)
        H1: mean(sample_a) < mean(sample_b)

    Parameters:
        sample_a (np.ndarray): First sample (e.g., treatment group)
        sample_b (np.ndarray): Second sample (e.g., control group)
        alpha (float): Significance level (default: 0.05)

    Returns:
        dict: {
            "t_statistic": float,
            "p_value": float,
            "reject_null": bool,
            "conclusion": str
        }
    """
    
    # Input validation
    if not isinstance(sample_a, np.ndarray) or not isinstance(sample_b, np.ndarray):
        raise TypeError("Both sample_a and sample_b must be numpy arrays.")

    if sample_a.size < 2 or sample_b.size < 2:
        raise ValueError("Each sample must contain at least two observations.")

    if not (0 < alpha < 1):
        raise ValueError("Alpha must be a float between 0 and 1.")

    # Perform Welch's t-test
    t_statistic, p_value_two_tailed = stats.ttest_ind(sample_a, sample_b, equal_var=False)

    # Convert to one-tailed p-value (less-than test)
    p_value = p_value_two_tailed / 2 if t_statistic < 0 else 1 - (p_value_two_tailed / 2)

    # Determine result
    reject_null = p_value < alpha
    conclusion = (
        f"Reject H₀: Evidence suggests mean({sample_a_name}) < mean({sample_b_name})."
        if reject_null else
        f"Fail to reject H₀: No evidence that mean({sample_a_name}) < mean({sample_b_name})."
    )

    return {
        "t_statistic": t_statistic,
        "p_value": p_value,
        "reject_null": reject_null,
        "conclusion": conclusion
    }

def perform_one_tailed_ttest_using_summaries(
    mean1: float, std1: float, n1: int,
    mean2: float, std2: float, n2: int,
    name1:str = "mean1", name2:str = "mean2",
    alpha: float = 0.05
) -> dict:
    """
    Perform a one-tailed Welch's t-test from summary statistics.

    H₀: mean1 ≥ mean2
    H₁: mean1 < mean2

    Parameters:
        mean1, std1, n1: Mean, standard deviation, and size of sample 1
        mean2, std2, n2: Mean, standard deviation, and size of sample 2
        alpha: Significance level (default: 0.05)

    Returns:
        dict: {
            "t_statistic": float,
            "degrees_of_freedom": float,
            "p_value": float,
            "reject_null": bool,
            "conclusion": str
        }
    """
    # Input checks
    if n1 < 2 or n2 < 2:
        raise ValueError("Each sample must have at least 2 observations.")
    if std1 < 0 or std2 < 0:
        raise ValueError("Standard deviations must be non-negative.")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")

    # Compute standard errors
    se1_sq = (std1 ** 2) / n1
    se2_sq = (std2 ** 2) / n2

    # Compute t-statistic (Welch’s formula)
    t_stat = (mean1 - mean2) / np.sqrt(se1_sq + se2_sq)

    # Compute degrees of freedom (Welch–Satterthwaite equation)
    numerator = (se1_sq + se2_sq) ** 2
    denominator = ((se1_sq ** 2) / (n1 - 1)) + ((se2_sq ** 2) / (n2 - 1))
    df = numerator / denominator

    # One-tailed p-value (for H1: mean1 < mean2)
    p_value = stats.t.cdf(t_stat, df)

    reject_null = p_value < alpha
    conclusion = (
        f"Reject H₀: Evidence suggests {name1} < {name2}."
        if reject_null else
        f"Fail to reject H₀: No evidence that {name1} < {name2}."
    )

    return {
        "t_statistic": t_stat,
        "degrees_of_freedom": df,
        "p_value": p_value,
        "reject_null": reject_null,
        "conclusion": conclusion
    }