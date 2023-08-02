import torch

CACHED = True

# @njit(cache=CACHED)
# def sde_pot_and_jacobian(xx,theta,origin_demand,cost_matrix):

#     # Extract dimensions
#     nrows,ncols = np.shape(cost_matrix)

#     # Get first two parameters
#     alpha = theta[0]
#     beta = theta[1]
#     delta = theta[2]
#     gamma = theta[3]
#     kappa = theta[4]
#     epsilon = theta[5]

#     # Compute log intensity rowsums
#     log_rowsums = np.log(origin_demand)
#     # Compute log unnormalised expected flow
#     log_utility = alpha*xx - beta*cost_matrix
#     # Compute log normalisation factor
#     log_normalisation = np.array([logsumexp(log_utility[i,:]) for i in prange(nrows)],dtype='float32')

#     # Compute potential
#     if alpha == 0:
#         potential =  -np.infty
#     else:
#         potential = -(1./alpha)*np.dot(origin_demand,log_normalisation)
#         potential += kappa*np.sum(np.exp(xx)) - delta*np.sum(xx)
#         potential *= gamma*epsilon

#     # Compute intensity
#     intensity = np.exp(log_utility - log_normalisation.reshape((nrows,1)) + log_rowsums.reshape((nrows,1)))

#     # Compute gradient of potential
#     grad = -np.sum(intensity,axis=0)
#     grad += kappa*np.exp(xx) - delta
#     grad *= gamma*epsilon
#     return potential,grad

# @njit(cache=CACHED)
# def sde_pot(xx,theta,origin_demand,cost_matrix):

#     # Extract dimensions
#     nrows,ncols = np.shape(cost_matrix)

#     # Get first two parameters
#     alpha = theta[0]
#     beta = theta[1]
#     delta = theta[2]
#     gamma = theta[3]
#     kappa = theta[4]
#     epsilon = theta[5]

#     # Compute log unnormalised expected flow
#     log_utility = alpha*xx - beta*cost_matrix
#     # Compute log normalisation factor
#     log_normalisation = np.array([logsumexp(log_utility[i,:]) for i in prange(nrows)],dtype='float32')
#     # Compute potential
#     if alpha == 0:
#         potential =  -np.infty
#     else:
#         potential = -(1./alpha)*np.dot(origin_demand,log_normalisation)
#         potential += kappa*np.sum(np.exp(xx)) - delta*np.sum(xx)
#         potential *= gamma*epsilon

#     return potential

# @njit(cache=CACHED)
# def sde_pot_jacobian(xx,theta,origin_demand,cost_matrix):

#     # Extract dimensions
#     nrows,ncols = np.shape(cost_matrix)

#     # Get first two parameters
#     alpha = theta[0]
#     beta = theta[1]
#     delta = theta[2]
#     gamma = theta[3]
#     kappa = theta[4]
#     epsilon = theta[5]

#     # Compute log unnormalised expected flow
#     wksp = alpha*xx - beta*cost_matrix
#     # Compute log normalisation factor
#     log_normalisation = np.array([logsumexp(wksp[i,:]) for i in prange(nrows)])
#     # Compute gradient of potential
#     grad = -np.sum(origin_demand.reshape((nrows,1))*np.exp(wksp-log_normalisation.reshape((nrows,1))),axis=0)
#     grad += kappa*np.exp(xx) - delta
#     grad *= gamma*epsilon
#     return grad

# @njit(parallel=True)
# def sde_pot_hessian(xx,theta,origin_demand,cost_matrix):

#     # Extract dimensions
#     nrows,ncols = np.shape(cost_matrix)

#     # Get first two parameters
#     alpha = theta[0]
#     beta = theta[1]
#     delta = theta[2]
#     gamma = theta[3]
#     kappa = theta[4]
#     epsilon = theta[5]

#     # Compute log unnormalised expected flow
#     wksp = alpha*xx - beta*cost_matrix
#     # Compute log normalisation factor
#     normalisation = np.array([logsumexp(wksp[i,:]) for i in range(nrows)])
#     # Compute utility potential
#     wksp = np.exp(wksp - normalisation.reshape((nrows,1)))

#     # Compute hessian
#     hessian = np.zeros((ncols,ncols))
#     for i in prange(nrows):
#         for j in prange(ncols):
#             for k in range(j+1,ncols):
#                 temp = alpha*origin_demand[i]*wksp[i,j]*wksp[i,k]
#                 hessian[k,j] += temp
#                 hessian[j,k] += temp

#         hessian += alpha*origin_demand[i]*np.diag(wksp[i,:]*(wksp[i,:]-1))

#     # Update hessian
#     hessian += kappa*np.diag(np.exp(xx))
#     hessian *= gamma*epsilon
#     return hessian


# @njit(cache=CACHED)
# def log_flow_matrix(xx,theta,origin_demand,cost_matrix,total_flow):
#     # Extract dimensions
#     nrows,ncols = np.shape(cost_matrix)
#     log_flow = np.zeros((nrows,ncols),dtype='float32')
#     # Get first two parameters
#     alpha = theta[0]
#     beta = theta[1]
#     # Compute log unnormalised expected flow
#     wksp = alpha*xx - beta*cost_matrix
#     log_normalisation = np.array([logsumexp(wksp[i,]) for i in range(nrows)])
#     # Compute log flow
#     log_flow = np.log(origin_demand).reshape((nrows,1)) + wksp - log_normalisation.reshape((nrows,1)) + np.log(total_flow)
#     return log_flow

# # @njit(cache=CACHED,parallel=True)
# def log_flow_matrix_vectorised(xxs,thetas,origin_demand,cost_matrix,total_flow,progress_proxy):
#     # Extract dimensions
#     N = np.shape(xxs)[0]
#     nrows,ncols = np.shape(cost_matrix)
#     log_flow = np.zeros((N,nrows,ncols),dtype='float32')
#     # Compute log unnormalised expected flow
#     for n in prange(N):
#         log_flow[n,:] =  log_flow_matrix(xxs[n,:],thetas[n,:],origin_demand,cost_matrix,total_flow)
#         # Update progress bar
#         if progress_proxy is not None:
#             progress_proxy.update(1)

#     return log_flow

# @njit(cache=CACHED,parallel=True)
# def flow_matrix_jacobian(theta,log_intensity):
#     # Extract dimensions
#     nrows,ncols = np.shape(log_intensity)
#     # Extract theta
#     alpha = theta[0]
#     # Compute log intensity rowsums
#     log_rowsums = np.asarray([logsumexp(log_intensity[i,:]) for i in range(nrows)],dtype='float32')
#     # Compute log rowsum probabilities
#     log_rowsum_probabilities = log_intensity - log_rowsums.reshape((nrows,1))
#     # Compute gradient
#     gradient = np.zeros((nrows,ncols,ncols),dtype='float32')
#     for i in prange(nrows):
#         gradient[i,:] = np.diag(np.exp(log_intensity[i,:]))
#         gradient[i,:] -= np.outer(np.exp(log_rowsum_probabilities[i,:]),np.exp(log_intensity[i,:]))
#     gradient *= alpha
#     return gradient.astype('float32')

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


# @njit
# def destination_attraction_log_likelihood_and_jacobian(xx,log_destination_attraction,s2_inv):
#     # Compute difference
#     diff = (xx - log_destination_attraction).flatten()
#     # Compute gradient of log likelihood
#     grad = -s2_inv*diff
#     # Compute log likelihood (without constant factor)
#     potential = -0.5*s2_inv*np.dot(diff, diff)

#     return potential, grad



# @njit(cache=CACHED,parallel=True)
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