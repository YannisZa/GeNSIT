import torch

from torch import int8,float32,int32,float64,int64,bool

from multiresticodm.global_variables import NUMBA_PARALLELISE

# def sde_pot_and_jacobian(xx,theta,origin_demand,cost_matrix):
    
#     # Get parameters
#     alpha = theta[0]
#     beta = theta[1]
#     delta = theta[2]
#     gamma = theta[3]
#     kappa = theta[4]
#     epsilon = theta[5]
    
#     # Compute log intensity total
#     log_total = np.log(origin_demand.sum())
#     # Compute log unnormalised expected flow
#     log_utility = alpha*xx - beta*cost_matrix
#     # Compute log normalisation factor
#     log_normalisation = logsumexp(log_utility.ravel())
    
#     # Compute potential
#     if alpha == 0:
#         potential =  -np.infty
#     else:
#         potential = -(1./alpha)*np.exp(log_total)*log_normalisation
#         potential += kappa*np.sum(np.exp(xx)) - delta*np.sum(xx)
#         potential *= gamma*epsilon
    
#     # Compute intensity
#     intensity = np.exp(log_utility - log_normalisation + log_total)
    
#     # Compute gradient of potential
#     grad = -np.sum(intensity,axis=0)
#     grad += kappa*np.exp(xx) - delta
#     grad *= gamma*epsilon

#     return potential,grad

# def sde_pot(xx,theta,origin_demand,cost_matrix):

#     # Get parameters
#     alpha = theta[0]
#     beta = theta[1]
#     delta = theta[2]
#     gamma = theta[3]
#     kappa = theta[4]
#     epsilon = theta[5]
    
#     # Compute log intensity total
#     log_total = np.log(origin_demand.sum())
#     # Compute log unnormalised expected flow
#     log_utility = alpha*xx - beta*cost_matrix
#     # Compute log normalisation factor
#     log_normalisation = logsumexp(log_utility.ravel())
#     # Compute potential
#     if alpha == 0:
#         potential =  -np.infty
#     else:
#         potential = -(1./alpha)*np.exp(log_total)*log_normalisation
#         potential += kappa*np.sum(np.exp(xx)) - delta*np.sum(xx)
#         potential *= gamma*epsilon
    
#     return potential

# @njit(cache=CACHED)
# def sde_pot_jacobian(xx,theta,origin_demand,cost_matrix):

#     # Get parameters
#     alpha = theta[0]
#     beta = theta[1]
#     delta = theta[2]
#     gamma = theta[3]
#     kappa = theta[4]
#     epsilon = theta[5]
    
#     # Compute log intensity total
#     log_total = np.log(origin_demand.sum())
#     # Compute log unnormalised expected flow
#     log_utility = alpha*xx - beta*cost_matrix
#     # Compute log normalisation factor
#     log_normalisation = logsumexp(log_utility.ravel())
#     # Compute intensity
#     intensity = np.exp(log_utility - log_normalisation + log_total)
#     # Compute gradient of potential
#     grad = -np.sum(intensity,axis=0)
#     grad += kappa*np.exp(xx) - delta
#     grad *= gamma*epsilon
    
#     return grad



# @njit(parallel=NUMBA_PARALLELISE)
# def sde_pot_hessian(xx,theta,origin_demand,cost_matrix):

#     # Extract dimensions
#     _,ncols = np.shape(cost_matrix)

#     # Get first two parameters
#     alpha = theta[0]
#     beta = theta[1]
#     gamma = theta[3]
#     kappa = theta[4]
#     epsilon = theta[5]

#     # Compute intensity
#     log_total = np.log(origin_demand.sum())
#     # Compute log utility
#     log_utility = alpha*xx- beta*cost_matrix
#     # Compute log normalisation factor
#     log_normalisation = logsumexp(log_utility.ravel())
#     # Evaluate log flow scaled
#     log_intensity =  log_utility - log_normalisation + log_total
#     # Compute log intensity colsums
#     low_rowsums = np.asarray([[logsumexp(x) for x in log_intensity.T]],dtype='float32').flatten()
#     # Compute log colsum probabilities
#     low_rowsum_probabilities = np.exp(low_rowsums - logsumexp(low_rowsums))

#     # Compute hessian
#     hessian = np.zeros((ncols,ncols),dtype=np.float32)
#     for j in prange(ncols):
#         for l in range(j+1,ncols):
#             hessian[l,j] = alpha * np.exp(low_rowsums[j] + low_rowsums[l] - log_total)
#             hessian[j,l] = hessian[l,j]

#     # Update hessian
#     hessian += kappa * np.diag(np.exp(xx))
#     hessian -= alpha * np.diag(np.exp(low_rowsums) * (1-np.exp(low_rowsum_probabilities)))
#     hessian *= gamma*epsilon
#     return hessian


def _log_flow_matrix(**kwargs):

    # Get parameters
    log_destination_attraction = kwargs['log_destination_attraction']
    cost_matrix = kwargs['cost_matrix']
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    grand_total = kwargs.get('grand_total',None)
    device = kwargs.get('device','cpu')

    # Extract dimensions
    nrows,ncols = cost_matrix.size(dim=0), cost_matrix.size(dim=1)
    N = log_destination_attraction.size(dim=0) if log_destination_attraction.ndim > 2 else 1
    log_flow = torch.zeros((N,nrows,ncols)).to(dtype=float32,device=device)

    # Reshape tensors to ensure operations are possible
    log_destination_attraction = torch.reshape(log_destination_attraction,(N,1,ncols))
    cost_matrix = torch.reshape(cost_matrix,(1,nrows,ncols))
    alpha = torch.reshape(alpha,(N,1,1))
    beta = torch.reshape(beta,(N,1,1))

    # Compute log unnormalised expected flow
    # Compute log utility
    log_utility = log_destination_attraction*alpha - cost_matrix*beta
    # Compute log normalisation factor
    normalisation = torch.logsumexp(log_utility,dim=(1,2))
    # and reshape it
    normalisation = torch.reshape(normalisation,(N,1,1))
    # Evaluate log flow scaled
    log_flow = log_utility - normalisation + torch.log(grand_total).to(device=log_utility.device)

    return log_flow

def _destination_demand(**kwargs):
    # Compute log flow
    log_flow = _log_flow_matrix(
        **kwargs
    )
    # Squeeze output
    log_flow = torch.squeeze(log_flow)

    # Compute destination demand
    log_destination_demand = torch.logsumexp(log_flow,dim=0,keepdim=True)
    return torch.exp(log_destination_demand)

# @njit(cache=CACHED,parallel=NUMBA_PARALLELISE)
# def flow_matrix_jacobian(theta,log_intensity):
#     # Extract dimensions
#     nrows,ncols = np.shape(log_intensity)
#     # Compute intensity
#     intensity = np.exp(log_intensity)
#     intensity_total = np.sum(intensity)
#     # Compute intensity rowsums
#     intensity_rowsum_probs = np.array([np.sum(intensity[:,j])/intensity_total for j in range(ncols)])
#     # Get alpha parameter
#     alpha = theta[0]
#     # Create result array
#     res = np.zeros((nrows,ncols,ncols))
#     for i in prange(nrows):
#         for j in prange(ncols):
#             for l in prange(ncols):
#                 if l == j:
#                     res[i,j,l] += alpha * intensity[i,j] * (1 - intensity_rowsum_probs[j])
#                 else:
#                     res[i,j,l] -= alpha * intensity[i,j] * intensity_rowsum_probs[l]
#     return res
#     # # Compute outer product of vector in log space
#     # log_rowsum_prob_matrix = np.repeat(log_rowsum_probabilities[:,np.newaxis],ncols,axis=1)
#     # # Compute gradient terms
#     # # Diagonal elements only in real space
#     # gradient_term1 = np.array([np.diag(np.exp(log_intensity)[i,:]).T for i in range(nrows)],dtype='float32')
#     # # # Diagonal and off-diagonal elements in log space
#     # log_gradient_term2 = np.array([np.repeat(log_intensity[i,:][np.newaxis,:],ncols,axis=0).T + log_rowsum_prob_matrix.T for i in range(nrows)],dtype='float32')
#     # # Compute gradient in real space
#     # return theta[0] * (gradient_term1 - np.exp(log_gradient_term2))

# @njit(cache=CACHED)
# def sde_ais_pot_and_jacobian(xx,theta,J):
#     delta = theta[2]
#     gamma = theta[3]
#     kappa = theta[4]
#     # Note that lim_{beta->0, alpha->0} gamma*V_{theta}(x) = gamma*kappa*\sum_{j=1}^J \exp(x_j) - gamma*(delta+1/J) * \sum_{j=1}^J x_j
#     gamma_kk_exp_xx = gamma*kappa*np.exp(xx)
#     # Function proportional to the potential function in the limit of alpha -> 0, beta -> 0
#     V = -gamma*(delta+1./J)*xx.sum() + gamma_kk_exp_xx.sum()
#     # Gradient of function above
#     gradV = -gamma*(delta+1./J)*np.ones(J,dtype='float32') + gamma_kk_exp_xx

#     return V, gradV


# @njit(cache=CACHED)
# def destination_attraction_log_likelihood_and_jacobian(xx,log_destination_attraction,s2_inv):
#     """
#     Computes log data y likelihood and its gradient
#     """
#     # Compute difference
#     diff = (xx - log_destination_attraction).flatten()
#     # Compute gradient of log likelihood
#     grad = -s2_inv*diff
#     # Compute log likelihood (without constant factor)
#     potential = -0.5*s2_inv*(diff.dot(diff))

#     return potential, grad


# # @njit(cache=CACHED,parallel=True)
# def annealed_importance_sampling_log_z_parallel(
#     index,
#     theta,
#     ais_samples,
#     n_temperatures,
#     leapfrog_steps,
#     epsilon,
#     origin_demand,
#     cost_matrix,
#     progress_proxy=None
# ):
#     log_weights = np.zeros(index,dtype='float32')
#     for i in prange(index):
#         log_weights[i] = annealed_importance_sampling_log_z(
#             i,
#             theta,
#             ais_samples,
#             n_temperatures,
#             leapfrog_steps,
#             epsilon,
#             origin_demand,
#             cost_matrix
#         )
#         if progress_proxy is not None:
#             progress_proxy.update(1)
#     return log_weights


# # Note: seed is initalised to None by default
# # Random seed is only allowed to take integer inputs in numba's jit decorator
# @njit(cache=CACHED)
# def annealed_importance_sampling_log_z(
#         index,
#         theta,
#         ais_samples,
#         n_temperatures,
#         leapfrog_steps,
#         epsilon,
#         origin_demand,
#         cost_matrix
#     ):
#     # Initialize AIS
#     acceptance = 0
#     proposals = 0
    
#     # Get dimensions
#     I,J = np.shape(cost_matrix)
#     # Get parameters
#     delta = theta[2]
#     gamma = theta[3]
#     kappa = theta[4]
#     # Number of samples:ais_samples
#     # Number of bridging distributions:n_temperatures
#     # HMC leapfrog steps:leapfrog_steps
#     # HMC leapfrog stepsize:epsilon

#     # Initialise temperature schedule
#     temperatures = np.linspace(0, 1, n_temperatures)
#     negative_temperatures = 1. - temperatures
#     # Initialise importance weights for target distribution
#     # This initialisation corresponds to taking the mean of the particles corresponding to a given target distribution weight
#     log_weights = -np.log(ais_samples)*np.ones(ais_samples,dtype='float32')
#     # For each particle
#     for ip in np.arange(ais_samples):

#         # Initialize
#         # Log-gamma model with alpha,beta->0
#         xx = np.log(np.random.gamma(gamma*(delta+1./J), 1./(gamma*kappa), J))
#         # Compute potential of prior distribution (temperature = 0)
#         V0, gradV0 = sde_ais_pot_and_jacobian(xx,theta,J)

#         # Compute potential of target distribution (temperature = 1)
#         V1, gradV1 = sde_pot_and_jacobian(xx,theta,origin_demand,cost_matrix)
        
#         # Anneal
#         for it in np.arange(1, n_temperatures):
#             # Update log weights using AIS (special case of sequential importance sampling)
#             # log (w_j(x_1,...,x_j)) = log (w_{j-1}(x_1,...,x_{j-1})) + log ( p_j(x_{j-1}) ) - log ( p_{j-1}(x_{j-1}) ) where
#             # p_j(x_{j-1}) = p_0(x_{j-1}) ^ {1-t_j} p_M (x_{j-1}) ^ {t_j}
#             log_weights[ip] += (temperatures[it] - temperatures[it-1])*(V0 - V1)
            
#             # Initialize HMC kernel
#             # Sample momentum
#             p = np.random.normal(0., 1., J)
#             # Compute log tempered distribution log p_j(x_{j-1)) = (1-t_j) * log( p_0(x_{j-1}) ) + t_j * log( p_M(x_{j-1}) )
#             V, gradV = negative_temperatures[it]*V0 + temperatures[it]*V1, negative_temperatures[it]*gradV0 + temperatures[it]*gradV1
#             # Define Hamiltonian energy
#             H = 0.5*np.dot(p, p) + V
#             # HMC leapfrog integrator
#             x_p = xx
#             p_p = p
#             V_p, gradV_p = V, gradV
            
#             for j in np.arange(leapfrog_steps):
#                 # Make half a step in momentum space
#                 p_p = p_p - 0.5*epsilon*gradV_p
#                 # Make a full step in latent space
#                 x_p = x_p + epsilon*p_p
#                 # Compute potential of prior distribution
#                 V0_p, gradV0_p = sde_ais_pot_and_jacobian(x_p,theta,J)
#                 # Compute potential of target distribution
#                 V1_p, gradV1_p = sde_pot_and_jacobian(x_p,theta,origin_demand,cost_matrix)
#                 # Compute log tempered distribution log p_j(x_{j))
#                 V_p, gradV_p = negative_temperatures[it]*V0_p + temperatures[it]*V1_p, negative_temperatures[it]*gradV0_p + temperatures[it]*gradV1_p
#                 # Make another half step in momentum space
#                 p_p = p_p - 0.5*epsilon*gradV_p

#             # HMC accept/reject
#             proposals += 1
#             H_p = 0.5*np.dot(p_p, p_p) + V_p
#             # Accept/reject
#             if np.log(np.random.uniform(0, 1)) < H - H_p:
#                 xx = x_p
#                 V0, gradV0 = V0_p, gradV0_p
#                 V1, gradV1 = V1_p, gradV1_p
#                 acceptance += 1

#     # Take the mean of the particles corresponding to a given target distribution weight
#     # You can see this is the case by looking at the initialisation of log_weights
#     return logsumexp(log_weights)