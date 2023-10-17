import torch
import xarray as xr

from numpy import arange
from torch import float32


SDE_POT_ARGS = ['log_destination_attraction','origin_demand','cost_matrix','grand_total','alpha','beta','delta','gamma','kappa','epsilon']
FLOW_MATRIX_ARGS = ['log_destination_attraction','origin_demand','cost_matrix','grand_total','alpha','beta']

def sde_pot_and_jacobian(**kwargs):
    # Calculate potential
    potential = sde_pot(**kwargs)
    # Calculate gradient of potential
    jacobian = sde_pot_jacobian(**kwargs)
    return potential,jacobian

def sde_pot_expanded(log_destination_attraction,origin_demand,cost_matrix,grand_total,alpha,beta,delta,gamma,kappa,epsilon):
    return sde_pot(
            **dict(
                log_destination_attraction=log_destination_attraction,
                origin_demand=origin_demand,
                cost_matrix=cost_matrix,
                grand_total=grand_total,
                alpha=alpha,
                beta=beta,
                delta=delta,
                gamma=gamma,
                kappa=kappa,
                epsilon=epsilon
            )
    )

def sde_pot(**kwargs):
    
    # Get parameters
    grand_total = kwargs['grand_total']
    origin_demand = kwargs['origin_demand']
    cost_matrix = kwargs['cost_matrix']
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    delta = kwargs['delta']
    kappa = kwargs['kappa']
    gamma = kwargs['gamma']
    epsilon = kwargs['epsilon']
    log_destination_attraction = kwargs['log_destination_attraction']
    
    # Compute log intensity total
    log_total = torch.log(grand_total.sum())
    # Compute log unnormalised expected flow
    log_utility = alpha*log_destination_attraction - beta*cost_matrix
    # Compute log normalisation factor
    log_normalisation = torch.logsumexp(log_utility.ravel(),dim=0)
    
    # Extract dimensions and reshape quantities
    origin = cost_matrix.size(dim=0)
    origin_demand = torch.reshape(origin_demand,(origin,1))

    # Compute potential
    if alpha == 0:
        potential = -float('inf')
    else:
        potential = -(1./alpha)*torch.dot(origin_demand,log_normalisation)
        potential += kappa*(torch.exp(log_destination_attraction)).sum() - delta*torch.sum(log_destination_attraction)
        potential *= gamma*epsilon
    
    return potential

def sde_pot_jacobian(**kwargs):
    # Calculate gradient of potential
    return torch.autograd.functional.jacobian(
        sde_pot_expanded, 
        inputs=tuple([kwargs[k] for k in SDE_POT_ARGS]), 
        create_graph=True
    )

def sde_pot_hessian(**kwargs):

    # Calculate hessian of potential
    return torch.autograd.functional.hessian(
        sde_pot_expanded, 
        inputs=tuple([kwargs[k] for k in SDE_POT_ARGS]), 
        create_graph=True
    )


def log_flow_matrix(**kwargs):
    # Get parameters
    device = kwargs.get('device','cpu')
    tensor = kwargs.get('torch',True)
    
    origin_demand = kwargs['origin_demand']
    grand_total = kwargs.get('grand_total',torch.tensor(1.0,dtype=float32,device=device))
    cost_matrix = kwargs['cost_matrix']
    
    log_destination_attraction = kwargs['log_destination_attraction']
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    
    # Extract dimensions and reshape quantities
    origin,destination = cost_matrix.size(dim=0), cost_matrix.size(dim=1)
    origin_demand = torch.reshape(origin_demand,(origin,1))

    # If input is torch use the following code
    if tensor:
        if log_destination_attraction.ndim > 2:
            N = log_destination_attraction.size(dim=0)
            time = log_destination_attraction.size(dim=1)
        elif log_destination_attraction.ndim > 1:
            N = 1
            time = log_destination_attraction.size(dim=0)
        else:
            N = 1
            time = 1
    # If input is xarray use the following code
    else:
        # Extract dimensions
        N = log_destination_attraction['N'].shape[0]
        time = log_destination_attraction['time'].shape[0]
        # Use the .sel() method to select the dimensions you want to convert
        log_destination_attraction = log_destination_attraction.sel(
            **{dim: slice(None) for dim in ['N','time','destination']}
        )
        alpha = alpha.sel(N=slice(None))
        beta = beta.sel(N=slice(None))
        # Convert the selected_data to torch tensor
        log_destination_attraction = torch.tensor(
            log_destination_attraction.values
        ).to(dtype=float32,device=device)
        alpha = torch.tensor(
            alpha.values
        ).to(dtype=float32,device=device)
        beta = torch.tensor(
            beta.values
        ).to(dtype=float32,device=device)

    log_flow = torch.zeros((N,origin,destination)).to(dtype=float32,device=device)
    # Reshape tensors to ensure operations are possible
    log_destination_attraction = torch.reshape(log_destination_attraction,(N,time,destination))
    cost_matrix = torch.reshape(cost_matrix,(1,origin,destination))
    alpha = torch.reshape(alpha,(N,1,1))
    beta = torch.reshape(beta,(N,1,1))

    # Compute log unnormalised expected flow
    # Compute log utility
    log_utility = torch.log(origin_demand) + log_destination_attraction*alpha - cost_matrix*beta
    # Compute log normalisation factor
    normalisation = torch.logsumexp(log_utility,dim=(1,2))
    # and reshape it
    normalisation = torch.reshape(normalisation,(N,1,1))
    # Evaluate log flow scaled
    log_flow = log_utility - normalisation + torch.log(grand_total)
    
    if kwargs.get('torch',True):
        # Return torch tensor
        return log_flow
    else:
        group = {}
        group['N'] = kwargs['log_destination_attraction'].coords['N']
        group['origin'] = arange(1,origin+1,1,dtype='int32')
        group['destination'] = arange(1,destination+1,1,dtype='int32')
        # Create outputs xr data array
        return xr.DataArray(
            data=log_flow.detach().cpu().numpy(), 
            dims=list(group.keys()),
            coords=group
        ) 

def flow_matrix_expanded(log_destination_attraction,origin_demand,cost_matrix,grand_total,alpha,beta):
    return torch.exp(log_flow_matrix(
            **dict(
                log_destination_attraction=log_destination_attraction,
                origin_demand=origin_demand,
                cost_matrix=cost_matrix,
                grand_total=grand_total,
                alpha=alpha,
                beta=beta
            )
    ))

def flow_matrix_jacobian(**kwargs):
    # Calculate gradient of intensity
    return torch.autograd.functional.jacobian(
        flow_matrix_expanded, 
        inputs=tuple([kwargs[k] for k in FLOW_MATRIX_ARGS]), 
        create_graph=True
    )
