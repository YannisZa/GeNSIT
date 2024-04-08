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
    gamma = 2 / (sigma**2)
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
        potential *= gamma*epsilon
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
            N = log_destination_attraction.size(dim = 0)
            sweep = log_destination_attraction.size(dim = 2)
        elif log_destination_attraction.ndim > 1:
            N = 1
            sweep = 1
        else:
            N = 1
            sweep = 1
    # If input is xarray use the following code
    else:
        # Extract dimensions
        N = log_destination_attraction['id'].shape[0]
        sweep = len(log_destination_attraction.coords['sweep'].values.tolist())
        dims = ['id','origin','destination','sweep']
        # Merge all coordinates
        coords = kwargs['log_destination_attraction'].coords
        coords = coords.assign(origin = arange(1,origin+1,dtype='int32'))

        # Use the .sel() method to select the dimensions you want to convert
        # Get the last time dimension
        log_destination_attraction = log_destination_attraction.isel(time = -1).sel(
            **{dim: slice(None) for dim in ['id','destination','sweep']}
        ).transpose('id','sweep','destination')
        alpha = alpha.sel(id = slice(None),sweep = slice(None))
        beta = beta.sel(id = slice(None),sweep = slice(None))
        
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
    log_destination_attraction = torch.reshape(log_destination_attraction,(N,1,destination,sweep))
    cost_matrix = torch.reshape(cost_matrix.unsqueeze(1).repeat(1, 1, 1, sweep),(1,origin,destination,sweep))
    alpha = torch.reshape(alpha,(N,1,1,sweep))
    beta = torch.reshape(beta,(N,1,1,sweep))
    log_grand_total = torch.log(grand_total).to(device = device)

    # Compute log unnormalised expected flow
    # Compute log utility
    log_utility = log_destination_attraction*alpha - cost_matrix*beta
    # Compute log normalisation factor
    normalisation = torch.logsumexp(log_utility,dim=(1,2))
    # and reshape it
    normalisation = torch.reshape(normalisation,(N,1,1,sweep))
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
            coords={k:coords[k] for k in dims}
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
