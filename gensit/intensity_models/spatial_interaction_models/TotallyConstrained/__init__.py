import torch
import xarray as xr

from numpy import arange
from torch import float32

SDE_POT_ARGS = ['log_destination_attraction','cost_matrix','grand_total','alpha','beta','delta','sigma','kappa','epsilon']
FLOW_MATRIX_ARGS = ['log_destination_attraction','cost_matrix','grand_total','alpha','beta']

def sde_pot_and_jacobian(**kwargs):
    # Calculate potential
    potential = sde_pot(**kwargs)
    # Calculate gradient of potential
    jacobian = sde_pot_jacobian(**kwargs)
    return potential,jacobian

def sde_pot_expanded(log_destination_attraction,cost_matrix,grand_total,alpha,beta,delta,sigma,kappa,epsilon):
    return sde_pot(
            **dict(
                log_destination_attraction = log_destination_attraction,
                cost_matrix = cost_matrix,
                grand_total = grand_total,
                alpha = alpha,
                beta = beta,
                delta = delta,
                sigma = sigma,
                kappa = kappa,
                epsilon = epsilon
            )
    )

def sde_pot(**kwargs):
    
    # Get parameters
    cost_matrix = kwargs['cost_matrix']
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    delta = kwargs['delta']
    kappa = kwargs['kappa']
    sigma = kwargs['sigma']
    epsilon = kwargs['epsilon']
    log_destination_attraction = kwargs['log_destination_attraction']

    # Compute log unnormalised expected flow
    log_utility = alpha*log_destination_attraction - beta*cost_matrix
    # Compute log normalisation factor
    log_normalisation = torch.logsumexp(log_utility.ravel(),dim = 0)
    
    # Compute potential
    if alpha == 0:
        potential = torch.tensor(-float('inf'))
    else:
        potential = -(1./alpha)*log_normalisation
        potential += kappa*(torch.exp(log_destination_attraction)).sum() - delta*torch.sum(log_destination_attraction)
        potential *= (2* epsilon) / (sigma**2)
    return potential

def sde_pot_jacobian(**kwargs):
    # Calculate gradient of potential
    return torch.autograd.functional.jacobian(
        sde_pot_expanded, 
        inputs = tuple([kwargs[k] for k in SDE_POT_ARGS]), 
        create_graph = True
    )

def sde_pot_hessian(**kwargs):

    # Calculate hessian of potential
    return torch.autograd.functional.hessian(
        sde_pot_expanded, 
        inputs = tuple([kwargs[k] for k in SDE_POT_ARGS]), 
        create_graph = True
    )


def log_flow_matrix(**kwargs):
    # Get data structure params
    device = kwargs.get('device','cpu')
    tensor = kwargs.get('torch',True)
    # Required inputs
    grand_total = kwargs.get('grand_total',1.0)
    cost_matrix = kwargs['cost_matrix']
    # Required outputs
    log_destination_attraction = kwargs['log_destination_attraction']
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    
    # Extract dimensions
    origin,destination = cost_matrix.size(dim = 0), cost_matrix.size(dim = 1)
    
    # If input is torch use the following code
    if tensor:
        if log_destination_attraction.ndim > 2:
            iter_sizes = [log_destination_attraction.size(dim = 0)]
            sweep = log_destination_attraction.size(dim = 2)
        else:
            iter_sizes = [1]
            sweep = 1
    # If input is xarray use the following code
    else:
        # Get iteration dims
        iter_dims = [x for x in alpha.dims if x in ['iter','seed','N']]
        iter_sizes = [dict(alpha.sizes)[x] for x in iter_dims]

        # Create dummy sweep coordinate
        if 'sweep' not in log_destination_attraction.dims:
            log_destination_attraction = log_destination_attraction.expand_dims(
                sweep = xr.DataArray(['dummy_sweep'], dims=['sweep'])
            )
        if 'sweep' not in alpha.dims:
            alpha = alpha.expand_dims(
                sweep = xr.DataArray(['dummy_sweep'], dims=['sweep'])
            )
        if 'sweep' not in beta.dims:
            beta = beta.expand_dims(
                sweep = xr.DataArray(['dummy_sweep'], dims=['sweep'])
            )

        sweep = len(log_destination_attraction.coords['sweep'].values.tolist())
        dims = iter_dims + ['sweep','origin','destination']
        # Merge all coordinates
        coords = log_destination_attraction.coords
        coords = coords.assign(origin = arange(1,origin+1,dtype='int32'))

        # Use the .sel() method to select the dimensions you want to convert
        # Get the last time dimension
        log_destination_attraction = log_destination_attraction.isel(time = -1).sel(
            **{dim: slice(None) for dim in iter_dims+['sweep','destination']}
        ).transpose(*iter_dims,'sweep','destination')
        alpha = alpha.sel(**{x: slice(None) for x in iter_dims+['sweep']})
        beta = beta.sel(**{x: slice(None) for x in iter_dims+['sweep']})

        # Convert the selected_data to torch tensor
        log_destination_attraction = torch.tensor(
            log_destination_attraction.values
        ).to(dtype = float32,device = device)
        alpha = torch.tensor(
            alpha.values
        ).to(dtype = float32,device = device)
        beta = torch.tensor(
            beta.values
        ).to(dtype = float32,device = device)
    
    # Reshape tensors to ensure operations are possible
    log_destination_attraction = torch.reshape(log_destination_attraction,(*iter_sizes,sweep,1,destination))
    cost_matrix = cost_matrix.repeat((*([1]*len(iter_sizes)),sweep,1))
    alpha = torch.reshape(alpha,(*iter_sizes,sweep,1,1))
    beta = torch.reshape(beta,(*iter_sizes,sweep,1,1))
    log_grand_total = torch.log(grand_total).to(device = device)

    # Compute log unnormalised expected flow
    # Compute log utility
    log_utility = log_destination_attraction*alpha - cost_matrix*beta
    # Compute log normalisation factor
    normalisation = torch.logsumexp(log_utility,dim=(-2,-1))
    # and reshape it
    normalisation = torch.reshape(normalisation,(*normalisation.shape,1,1))
    # Evaluate log flow scaled
    log_flow = log_utility - normalisation + log_grand_total
    
    if kwargs.get('torch',True):
        # Return torch tensor
        return log_flow
    else:
        # Create outputs xr data array
        return xr.DataArray(
            data = log_flow.detach().cpu().numpy(), 
            dims = dims,
            coords = {k:coords[k] for k in dims}
        ) 

def flow_matrix_expanded(log_destination_attraction,cost_matrix,grand_total,alpha,beta):
    return torch.exp(log_flow_matrix(
            **dict(
                log_destination_attraction = log_destination_attraction,
                cost_matrix = cost_matrix,
                grand_total = grand_total,
                alpha = alpha,
                beta = beta
            )
    ))

def flow_matrix_jacobian(**kwargs):
    # Calculate gradient of intensity
    return torch.autograd.functional.jacobian(
        flow_matrix_expanded, 
        inputs = tuple([kwargs[k] for k in FLOW_MATRIX_ARGS]), 
        create_graph = True
    )
