from copy import deepcopy
import os
import sys
import torch
import numpy as np
import multiprocessing as mp


from os import path
from tqdm import tqdm
from typing import Union
from functools import partial
from torch import int32, float32
from pathlib import Path as PathLib
from joblib import Parallel, delayed
from multiresticodm.global_variables import TABLE_INFERENCE_EXPERIMENTS

import multiresticodm.probability_utils as ProbabilityUtils

from multiresticodm.config import Config
from multiresticodm.harris_wilson_model import HarrisWilson
from multiresticodm.math_utils import torch_optimize, logsumexp
from multiresticodm.utils import setup_logger, makedir, set_seed

AIS_SAMPLE_ARGS = ['alpha','beta','gamma','n_temperatures','ais_samples','leapfrog_steps','epsilon_step','semaphore','pbar']

def instantiate_harris_wilson_mcmc(config:Config,physics_model:HarrisWilson,**kwargs):
    if hasattr(sys.modules[__name__], ('HarrisWilsonMarkovChainMonteCarlo')):
        return getattr(sys.modules[__name__],f"HarrisWilson{len(config['inputs']['dims'].keys())}DMarkovChainMonteCarlo")(
            config = config,
            physics_model = physics_model,
            **kwargs
        )
    else:
        raise Exception(f"Input class HarrisWilsonMarkovChainMonteCarlo not found")

class HarrisWilsonMarkovChainMonteCarlo():

    def __init__(
            self, 
            config: Config,
            physics_model:HarrisWilson,
            **kwargs
        ):
        # Setup logger
        level = kwargs['logger'].level if 'logger' in kwargs else config.get('level','INFO').upper()
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            level=level,
            log_to_file=False,
            log_to_console=kwargs.get('log_to_console',True),
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevel(level)
        
        # Store sim model but not its config
        self.physics_model = physics_model

        # Cannot learn sigma with MCMC
        if 'sigma' in list(self.physics_model.params_to_learn.keys()):
            raise Exception(f"Cannot learn sigma using existing MCMC schemes")
        
        # Store config
        self.config = config

        # Device name
        self.device = self.config['inputs']['device']

        # Stopping times
        self.stopping_times = None
        self.stopping_times_directory = None
        # Number of parallelisation workers
        self.mcmc_workers = self.config.settings['mcmc'].get('mcmc_workers',1)

        self.logger.info(f'Building {self.physics_model.noise_regime} {self.physics_model.intensity_model._type} Markov Chain Monte Carlo Engine')

    def __repr__(self):
        return "MarkovChainMonteCarlo(SpatialInteraction)"
        
    def load_stopping_times(self):
        # Get total number of stopping times samples required
        N = int(self.config.settings['training']['N'])
        if self.config.settings['mcmc']['parameters'].get('theta_steps',1) is not None:
            N *= int(self.config.settings['mcmc']['parameters'].get('theta_steps',1))
        self.logger.note(f'Attempting to import {N} stopping times')
        # Try to import stopping times from file
        read_stopping_times = self.import_stopping_times(N=N)
        # If file is not provided
        if not read_stopping_times:
            # Generate stopping times
            self.logger.note('Generating stopping times')
            self.stopping_times = ProbabilityUtils.generate_stopping_times(N=N,k_power=self.k_power,seed=self.config['inputs']['seed'])
            # Reset random seed
            set_seed(None)
            # Export stopping times
            self.export_stopping_times()
        # Truncate based on samples required
        self.stopping_times = self.stopping_times[:N]
        self.logger.warning(f'Max stopping time is {torch.max(self.stopping_times)}')

    def import_stopping_times(self,N:int):
        # Get path to directory
        parent_directory = PathLib(self.config.settings['inputs']['in_directory'])
        # Make directory
        directory = path.join(str(parent_directory),'stopping_times')
        makedir(directory)
        # Extract filepath
        self.stopping_times_filepath = path.join(parent_directory,'stopping_times',self.config.settings['inputs']['data'].get('stopping_times',f'stopping_times_N_{N}.txt'))
        if path.isfile(self.stopping_times_filepath):
            # Get number of samples of stopping times in file
            self.stopping_times = np.loadtxt(self.stopping_times_filepath,dtype='int32')
            self.stopping_times = torch.tensor(self.stopping_times,dtype=float32,device=self.device)
            # If there are more than the number of samples needs load stopping times
            # Otherwise generate new stopping times
            if len(self.stopping_times) == N:
            # if len(self.stopping_times) >= N:
                return True
            else:
                self.stopping_times = None
        return False

    def export_stopping_times(self):
        # Check that filepath is provided
        if self.stopping_times_filepath is not None:
            self.logger.info('Exporting stopping times')
            np.savetxt(self.stopping_times_filepath,self.stopping_times.detach().cpu().numpy())


class HarrisWilson2DMarkovChainMonteCarlo(HarrisWilsonMarkovChainMonteCarlo):

    def __init__(
            self, 
            config:Config,
            physics_model:HarrisWilson,
            **kwargs
        ):
        
        # Instantiate superclass
        super().__init__(
            config = config,
            physics_model = physics_model,
            **kwargs
        )

        # self.logger.info((self.__str__()))

    def __str__(self):
        return f"""
            2D Spatial Interaction Model Markov Chain Monte Carlo algorithm
            Dataset: {self.config.settings['inputs']['dataset']}
            Cost matrix: {self.config.settings['inputs']['data']['cost_matrix']}
            Cost matrix sum: {self.physics_model.intensity_model.data.cost_matrix.ravel().sum()}
            Origin demand sum: {self.physics_model.intensity_model.data.origin_demand.sum()}
            Delta: {self.physics_model.params.delta}
            Kappa: {self.physics_model.params.kappa}
            Epsilon: {self.physics_model.params.epsilon}
            Gamma: {self.physics_model.params.gamma}
            Beta scaling: {self.physics_model.params.bmax}
            Table dimensions: {"x".join([str(x) for x in self.physics_model.intensity_model.dims])}
            Noise regime: {self.physics_model.noise_regime}
            Number of threads per core: {str(self.mcmc_workers)}
        """
    
    def build(self,**kwargs):
        # Store table distribution name
        try:
            self.table_distribution_name = self.config.settings['inputs']['contingency_table']['distribution_name']
        except:
            try:
                self.table_distribution_name = kwargs.get('table_distribution_name','multinomial')
            except:
                raise Exception("No distribution name provided in config.")

        if self.physics_model.noise_regime == 'low':
            # Update method for computing z inverse
            self.z_inverse = self.biased_z_inverse
        else:
            # Update method for computing z inverse
            self.z_inverse = self.unbiased_z_inverse

            # Store number of Annealed Importance Samples (AIS)
            self.destination_attraction_ais_samples = self.config.settings['mcmc']['destination_attraction']['ais_samples']
            # Store Leapfrog steps for AIS
            self.destination_attraction_leapfrog_steps_ais = self.config.settings['mcmc']['destination_attraction']['ais_leapfrog_steps']
            # Store Leapfrog step size for AIS
            self.destination_attraction_leapfrog_step_size_ais = self.config.settings['mcmc']['destination_attraction']['ais_leapfrog_step_size']
            # Store number of bridging distributions in temperature schedule of AIS
            self.destination_attraction_n_bridging_distributions = self.config.settings['mcmc']['destination_attraction']['n_bridging_distributions']
            # Store power of k for truncating infinite series of AIS samples
            self.k_power = torch.tensor(1.1,dtype=float32,device=self.device)#self.config.settings['mcmc']['destination_attraction']['k_power']

            # Get stopping times for Z inverse estimator
            self.load_stopping_times()
        if self.config.settings['experiments'][0]['type'].lower() in TABLE_INFERENCE_EXPERIMENTS:
            self.theta_step = self.theta_given_table_gibbs_step
            self.log_destination_attraction_step = self.log_destination_attraction_given_table_gibbs_step
            # Keep track of joint log destination attarction and theta steps
            self.theta_steps = int(self.config.settings['mcmc']['parameters'].get('theta_steps',1))
            self.log_destination_attraction_steps = int(self.config.settings['mcmc']['destination_attraction'].get('log_destination_attraction_steps',1))
        else:
            self.theta_steps = 1
            self.log_destination_attraction_steps = 1
            self.theta_step = self.theta_gibbs_step
            self.log_destination_attraction_step = self.log_destination_attraction_gibbs_step

        # Store proposal covariance
        self.theta_proposal_covariance = torch.tensor(self.config.settings['mcmc']['parameters']['covariance'],dtype=float32,device=self.device)
        # Store theta step size
        self.theta_step_size = self.config.settings['mcmc']['parameters']['step_size']

        # Store Leapfrog steps
        self.destination_attraction_leapfrog_steps = self.config.settings['mcmc']['destination_attraction']['leapfrog_steps']
        # Store Leapfrog step size
        self.destination_attraction_step_size = self.config.settings['mcmc']['destination_attraction']['leapfrog_step_size']

        # Get table distribution
        self.table_distribution = f"log_{self.table_distribution_name}_pmf"
        # Get log table pmf unnormalised jacobian
        self.log_table_likelihood_total_derivative_wrt_x = getattr(ProbabilityUtils, 'log_table_likelihood_total_derivative_wrt_x')
        if hasattr(ProbabilityUtils, (self.table_distribution+"_unnormalised")):
            self.table_unnormalised_log_likelihood = getattr(ProbabilityUtils, self.table_distribution+"_unnormalised")
            # print(self.table_unnormalised_log_likelihood)
        else:
            raise Exception(f"Input class ProbabilityUtils does not have distribution {(self.table_distribution+'_unnormalised')}")
        self.table_log_likelihood_jacobian = None
        if hasattr(ProbabilityUtils, (self.table_distribution+'_jacobian_wrt_intensity')):
            self.table_log_likelihood_jacobian = getattr(ProbabilityUtils, (self.table_distribution+'_jacobian_wrt_intensity'))
        else:
            raise Exception(f"Input class ProbabilityUtils does not have distribution {(self.table_distribution+'wrt')} either intensity")
        
    
    def negative_table_log_likelihood(
        self,
        log_intensity,
        table,
    ):
        """ Computes log likelihood of table (augemented latent variable)

        Parameters
        ----------
        xx : Union[np.array,np.ndarray]
            Log destination attraction
        theta : Union[np.array,np.ndarray]
            List of parameters (alpha)
        table : np.ndarray
            Table of integer flows

        Returns
        -------
        np.float,np.ndarray(float32)
            log table likelihood

        """
        self.logger.debug('negative_table_log_likelihood')
        
        log_likelihood = self.table_unnormalised_log_likelihood(
            log_intensity=log_intensity,
            table=table
        )
        return -log_likelihood
    
    def negative_table_log_likelihood_expanded(
        self,
        log_destination_attraction,
        table,
        alpha,
        beta
    ):
        """ Computes log likelihood of table (augemented latent variable)

        Parameters
        ----------
        xx : Union[np.array,np.ndarray]
            Log destination attraction
        theta : Union[np.array,np.ndarray]
            List of parameters (alpha)
        table : np.ndarray
            Table of integer flows

        Returns
        -------
        np.float,np.ndarray(float32)
            log table likelihood

        """
        self.logger.debug('negative_table_log_likelihood_expanded')
        
        log_intensity = self.physics_model.intensity_model.log_intensity(
            log_destination_attraction = log_destination_attraction,
            alpha = alpha,
            beta = beta
        )
        log_likelihood = self.table_unnormalised_log_likelihood(
            log_intensity=log_intensity,
            table=table
        )
        return -log_likelihood

    
    def negative_table_log_likelihood_gradient(
        self,
        **kwargs
    ):
        self.logger.debug('negative_table_log_likelihood_gradient \n')

        # Make sure to include fixed parameters if there are any
        kwargs.update(**vars(self.physics_model.params))

        # Get log table likelihood derivative with respect to inputs
        return -torch.autograd.functional.jacobian(
            self.negative_table_log_likelihood_expanded, 
            inputs=tuple([kwargs[k].to(dtype=float32) for k in ['log_destination_attraction','table','alpha','beta']]), 
            create_graph=False
        )[0]
    
    def negative_table_log_likelihood_and_gradient(
            self,
            **kwargs
    ):
        # Make sure to include fixed parameters if there are any
        kwargs.update(**vars(self.physics_model.params))
        negative_table_log_likelihood = self.negative_table_log_likelihood_expanded(
            log_destination_attraction = kwargs['log_destination_attraction'],
            table = kwargs['table'],
            alpha = kwargs['alpha'],
            beta = kwargs['beta']
        )
        negative_table_log_likelihood_gradient = self.negative_table_log_likelihood_gradient(**kwargs)
        return negative_table_log_likelihood,negative_table_log_likelihood_gradient
    
    def annealed_importance_sampling_log_z_expanded(
        self,
        index,
        alpha,
        beta,
        gamma,
        n_temperatures,
        ais_samples,
        leapfrog_steps,
        epsilon_step,
        semaphore,
        pbar
    ):
        return self.annealed_importance_sampling_log_z(
            index,
            alpha = alpha,
            beta = beta,
            gamma = gamma,
            n_temperatures = n_temperatures,
            ais_samples = ais_samples,
            leapfrog_steps = leapfrog_steps,
            epsilon_step = epsilon_step,
            semaphore = semaphore,
            pbar = pbar
        )


        
    # Note: seed is initalised to None by default
    # Random seed is only allowed to take integer inputs in numba's jit decorator
    def annealed_importance_sampling_log_z(
        self,
        index,
        **kwargs
    ):
        semaphore = kwargs.get('semaphore',None)
        pbar = kwargs.get('pbar',None)

        if semaphore is not None:
            semaphore.acquire()

        # Initialize AIS
        acceptance = 0
        proposals = 0
        # Get dimensions
        Ndestinations = self.physics_model.intensity_model.dims['destination']
        # Get parameters
        kwargs['alpha'] = kwargs['alpha'] if kwargs['alpha'] is not None else self.physics_model.params.alpha
        kwargs['beta'] = kwargs['beta'] if kwargs['beta'] is not None else self.physics_model.params.beta
        kwargs['gamma'] = kwargs['gamma'] if kwargs['gamma'] is not None else self.physics_model.params.gamma
        gamma = kwargs['gamma']
        kappa = self.physics_model.params.kappa
        delta = self.physics_model.params.delta
        n_temperatures = kwargs['n_temperatures']
        ais_samples = kwargs['ais_samples']
        epsilon_step = kwargs['epsilon_step']
        leapfrog_steps = kwargs['leapfrog_steps']

        # Number of samples:ais_samples
        # Number of bridging distributions:n_temperatures
        # HMC leapfrog steps:leapfrog_steps
        # HMC leapfrog stepsize:epsilon

        # Initialise temperature schedule
        temperatures = torch.linspace(0, 1, n_temperatures)
        negative_temperatures = 1. - temperatures
        # Initialise importance weights for target distribution
        # This initialisation corresponds to taking the mean of the particles corresponding to a given target distribution weight
        log_weights = -torch.log(
            torch.tensor(
                ais_samples,
                dtype=float32,
                device=self.device
            )
        ) * torch.ones(
            ais_samples,
            dtype=float32,
            device=self.device
        )
        # For each particle
        for ip in range(ais_samples):

            # Initialize
            # Log-gamma model with alpha,beta->0
            gamma_distr = torch.distributions.gamma.Gamma(gamma*(delta+1./Ndestinations), 1./(gamma*kappa))
            xx = torch.log(gamma_distr.sample(torch.tensor((Ndestinations,))))
            # Compute potential of prior distribution (temperature = 0)
            V0, gradV0 = self.physics_model.sde_ais_potential_and_jacobian(log_destination_attraction=xx,**kwargs)

            # Compute potential of target distribution (temperature = 1)
            V1, gradV1 = self.physics_model.sde_potential_and_gradient(log_destination_attraction=xx,**kwargs)
            
            # Anneal
            for it in range(1, n_temperatures):
                # Update log weights using AIS (special case of sequential importance sampling)
                # log (w_j(x_1,...,x_j)) = log (w_{j-1}(x_1,...,x_{j-1})) + log ( p_j(x_{j-1}) ) - log ( p_{j-1}(x_{j-1}) ) where
                # p_j(x_{j-1}) = p_0(x_{j-1}) ^ {1-t_j} p_M (x_{j-1}) ^ {t_j}
                log_weights[ip] += (temperatures[it] - temperatures[it-1])*(V0 - V1)
                
                # Initialize HMC kernel
                # Sample momentum
                p = torch.randn(Ndestinations)
                # Compute log tempered distribution log p_j(x_{j-1)) = (1-t_j) * log( p_0(x_{j-1}) ) + t_j * log( p_M(x_{j-1}) )
                V, gradV = negative_temperatures[it]*V0 + temperatures[it]*V1, negative_temperatures[it]*gradV0 + temperatures[it]*gradV1
                # Define Hamiltonian energy
                H = 0.5*torch.dot(p, p) + V
                # HMC leapfrog integrator
                x_p = xx
                p_p = p
                V_p, gradV_p = V, gradV
                
                for j in torch.arange(leapfrog_steps):
                    # Make half a step in momentum space
                    p_p = p_p - 0.5*epsilon_step*gradV_p
                    # Make a full step in latent space
                    x_p = x_p + epsilon_step*p_p
                    # Compute potential of prior distribution
                    V0_p, gradV0_p = self.physics_model.sde_ais_potential_and_jacobian(log_destination_attraction=x_p,**kwargs)
                    # Compute potential of target distribution
                    V1_p, gradV1_p = self.physics_model.sde_potential_and_gradient(log_destination_attraction=x_p,**kwargs)
                    # Compute log tempered distribution log p_j(x_{j))
                    V_p, gradV_p = negative_temperatures[it]*V0_p + temperatures[it]*V1_p, negative_temperatures[it]*gradV0_p + temperatures[it]*gradV1_p
                    # Make another half step in momentum space
                    p_p = p_p - 0.5*epsilon_step*gradV_p

                # HMC accept/reject
                proposals += 1
                H_p = 0.5*torch.dot(p_p, p_p) + V_p
                # Accept/reject
                if torch.log(torch.rand(1)) < H - H_p:
                    xx = x_p
                    V0, gradV0 = V0_p, gradV0_p
                    V1, gradV1 = V1_p, gradV1_p
                    acceptance += 1

        if semaphore is not None:
            semaphore.release()
        if pbar is not None:
            pbar.update(1)

        # Take the mean of the particles corresponding to a given target distribution weight
        # You can see this is the case by looking at the initialisation of log_weights
        return torch.logsumexp(log_weights.ravel(),dim=0)
    
    def annealed_importance_sampling_log_z_parallel(
        self,
        N,
        **kwargs
    ):
        # Run experiments in parallel
        ctx = mp.get_context('spawn')
        # pbar = tqdm(total=N, desc='Running AIS in parallel',leave=False)

        kwargs['semaphore'] = None
        # kwargs['pbar'] = pbar
        results = []
        # Create partial function by fixing all kwargs
        annealed_importance_sampling_log_z_expanded_partial = partial(
            self.annealed_importance_sampling_log_z_expanded,
            **{k:kwargs.get(k,None) for k in AIS_SAMPLE_ARGS}
        )

        with ctx.Pool(min(self.mcmc_workers,N)) as p:
            for res in p.imap_unordered(
                annealed_importance_sampling_log_z_expanded_partial,
                range(N)
            ):
                results.append(res)

        # pbar.close()

        return torch.tensor(results)

    def biased_z_inverse(self,index:int,theta:dict):
        self.logger.debug('Biased Z inverse')
        # compute 1/z(theta) using Saddle point approximation
        # Create partial function
        torch_optimize_partial = partial(
            torch_optimize,
            function = self.physics_model.sde_potential_and_gradient,
            method = 'L-BFGS-B',
            **theta
        )
        # Create initialisations
        Ndests = self.physics_model.intensity_model.dims['destination']
        g = np.log(self.physics_model.params.delta.item())*np.ones((Ndests,Ndests)) - np.log(self.physics_model.params.delta.item())*np.eye(Ndests) + np.log(1+self.physics_model.params.delta.item())*np.eye(Ndests)
        g = g.astype('float32')

        # Get minimum across different initialisations in parallel
        if self.mcmc_workers > 1:
            xs = list(Parallel(n_jobs = self.mcmc_workers)(delayed(torch_optimize_partial)(g[i,:]) for i in range(Ndests)))
        else:
            # Get minimum across different initialisations in parallel
            xs = [torch_optimize_partial(g[i,:]) for i in range(Ndests)]

        # Compute potential
        fs = np.asarray([
            self.physics_model.sde_potential(
                torch.tensor(xs[i]).to(
                    dtype=float32,
                    device=self.device
                ),
                **theta,
                **vars(self.physics_model.params),
            ).detach().cpu().numpy() for i in range(Ndests)
        ])
        # Get arg min
        arg_min = np.argmin(fs)
        minimum = torch.tensor(
            xs[arg_min],
            device = self.device,
            dtype = float32,
            requires_grad = True
        )
        minimum_potential = fs[arg_min]

        # Get Hessian matrix
        A = self.physics_model.sde_potential_hessian(
            minimum,
            **theta,
            **vars(self.physics_model.params)
        )
        # Find its cholesky decomposition Hessian = L*L^T for efficient computation
        L = torch.linalg.cholesky(A)
        # Compute the log determinant of the hessian
        # det(Hessian) = det(L)*det(L^T) = det(L)^2
        # det(L) = \prod_{j=1}^M L_{jj} and
        # \log(det(L)) = \sum_{j=1}^M \log(L_{jj})
        # So \log(det(Hessian)^(1/2)) = \log(det(L))
        half_log_det_A = torch.sum(torch.log(torch.diag(L)))

        # Compute log_normalising constant, i.e. \log(z(\theta))
        # -gamma*V(x_{minimum}) + (M/2) * \log(2\pi \gamma^{-1})
        # lap =  -si.sde_potential_and_gradient(minimum,theta)[0] + lap_c1 - half_log_det_A
        # Compute log-posterior
        # \log(p(x|\theta)) = -gamma*V(x) - \log(z(\theta))
        # log_likelihood_values[i, j] = -lap - si.sde_potential_and_gradient(xd,theta)[0]

        # Return array
        ret = torch.empty(2,dtype=float32,device=self.device)
        # Log z(\theta) without the constant (2\pi\gamma^{-1})^{M/2}
        ret[0] = minimum_potential + half_log_det_A
        # self.physics_model.sde_potential_and_gradient(minimum,theta)[0] +  half_log_det_A
        # Sign of log inverse of z(\theta)
        ret[1] = 1.

        return ret
    
    def unbiased_z_inverse(self,index:int,theta:Union[np.ndarray,list]):
        # Debiasing scheme - returns unbiased esimates of 1/z(theta)

        # Extract stopping time
        N = int(self.stopping_times[index])

        self.logger.debug(f"Unbiased Z inverse N = {N+1}")

        if min(self.mcmc_workers,N+1) > 1:
            # Multiprocessing
            # self.logger.debug(f"Multiprocessing with workers = {min(self.n_workers,N+1)}")
            # with joblib_progress(f"Multiprocessing {min(self.n_workers,N+1)}", total=(N+1)):
            # log_weights = np.asarray(
            #     Parallel(n_jobs=min(self.n_workers,N+1))(
            #         delayed(self.annealed_importance_sampling_log_z_partial)(i,theta) for i in range(N+1)
            #     )
            # )
            # Parallelise function using multiprocessing semaphore
            # with ProgressBar(total=(N+1)) as progress_bar:
            self.logger.debug(f"annealed_importance_sampling_log_z_parallel")
            log_weights = self.annealed_importance_sampling_log_z_parallel(
                (N+1),
                ais_samples=int(self.destination_attraction_ais_samples),
                n_temperatures=int(self.destination_attraction_n_bridging_distributions),
                leapfrog_steps=int(self.destination_attraction_leapfrog_steps_ais),
                epsilon_step=float(self.destination_attraction_leapfrog_step_size_ais),
                **{p:theta[p] if theta.get(p,None) is not None else getattr(self.physics_model.params,p) for p in ['alpha','beta','gamma']}
            )
            log_weights = log_weights.to(dtype=float32,device=self.device)
        else:
            log_weights = []
            self.logger.debug(f"annealed_importance_sampling_log_z_partial")
            for i in range(N+1):
                log_weights.append(self.annealed_importance_sampling_log_z(
                    i,
                    ais_samples=int(self.destination_attraction_ais_samples),
                    n_temperatures=int(self.destination_attraction_n_bridging_distributions),
                    leapfrog_steps=int(self.destination_attraction_leapfrog_steps_ais),
                    epsilon_step=float(self.destination_attraction_leapfrog_step_size_ais),
                    **{p:theta[p] if theta.get(p,None) is not None else getattr(self.physics_model.params,p) for p in ['alpha','beta','gamma']}
                ))
            log_weights = torch.tensor(log_weights,dtype=float32,device=self.device)
        
        if not torch.all(torch.isfinite(log_weights)):
            raise Exception('Nulls/NaNs found in annealed importance sampling')
        if len(log_weights) <= 0:
            raise Exception('No weights computed in annealed importance sampling')
        self.logger.debug(f"Truncating infinite series")
        # Leave this argument as is (it is not a bug - see function dfn)
        ret = ProbabilityUtils.compute_truncated_infinite_series(N,log_weights,self.k_power,self.device)
        # print(f"sum = {ret[0]}, sign = {ret[1]}")
        return ret


    def theta_gibbs_step(
            self,
            index:int,
            theta_prev:Union[list,np.array,np.ndarray],
            log_destination_attraction:Union[list,np.array,np.ndarray],
            values:list
    ):

        self.logger.debug('Theta Gibbs step')

        # Unpack values
        V, \
        gradV, \
        log_z_inverse, \
        sign = values

        ''' Theta update '''
        # Multiply beta by total cost
        # theta_scaled_and_expanded = np.concatenate([theta_prev,np.array([self.physics_model.intensity_model.data.delta,self.physics_model.intensity_model.gamma,self.physics_model.intensity_model.data.kappa,self.physics_model.intensity_model.data.epsilon])])
        # theta_scaled_and_expanded[1] *= self.physics_model.intensity_model.data.bmax
        # print('theta',theta_scaled_and_expanded)
        # print('xx',log_destination_attraction)
        # print('theta',theta_prev)
        # print('V',V)
        # print('gradV',gradV)
        # print('log_z_inverse',log_z_inverse)

        # Theta-proposal (random walk with reflecting boundaries
        rndm_walk = self.theta_step_size*torch.matmul(self.theta_proposal_covariance, torch.randn(2,dtype=float32,device=self.device))
        theta_new = deepcopy(theta_prev)
        for i,prev in enumerate(theta_prev):
            # Perform one step
            theta_new[i] = prev + rndm_walk[i]
            # Relfect the boundaries if theta proposal falls outside of [0,2]^2
            if theta_new[i] < 0.:
                # Reflect off boundary
                theta_new[i] = -theta_new[i]
            elif theta_new[i] > 2.:
                # Reflect off boundary
                theta_new[i] = 2. - (theta_new[i] - 2.)

        # Theta-accept/reject
        if theta_new.min() < 0 or theta_new.max() >= 2:
            # print(f'Parameters {theta_new} outside of [0,2]^2 range')
            # self.logger.debug("Rejected")
            return theta_prev,0,V,gradV,log_z_inverse,sign

        # try:
        # Multiply beta by total cost
        theta_new_scaled = deepcopy(theta_new)
        theta_new_scaled[1] *= self.physics_model.params.bmax
        theta_new_dict = dict(zip(list(self.physics_model.params_to_learn.keys()),theta_new_scaled))

        # Compute inverse of z(theta)
        log_z_inverse_new, sign_new = self.z_inverse(
            index=index,
            theta=theta_new_dict
        )

        # Evaluate log potential function for theta proposal
        V_new, gradV_new = self.physics_model.sde_potential_and_gradient(
            log_destination_attraction,
            **theta_new_dict,
        )
        # Compute log parameter posterior for choice of X and updated theta proposal
        log_target_new = log_z_inverse_new - V_new
        # Compute log parameter posterior for choice of X and initial theta
        log_target = log_z_inverse - V

        # print('log_destination_attraction',log_destination_attraction)
        # print('V_new',V_new)
        # print('log_z_inverse_new',log_z_inverse_new)
        # print(("Proposing " + str(theta_new) + " with " + str(log_target_new)))
        # print(("Current sample " + str(theta_prev) + " with " + str(log_target)))
        # print(("Difference log target " + str(log_target_new-log_target)))
        # print('\n')
        if torch.log(torch.rand(1)) < log_target_new - log_target:
            self.logger.debug("Accepted")
            return theta_new,1,V_new,gradV_new,log_z_inverse_new,sign_new
        else:
            self.logger.debug("Rejected")
            return theta_prev,0,V,gradV,log_z_inverse,sign
        # except:
        #     print("Exception raised in theta_gibbs_step")
        #     # self.logger.debug("Exception raised")
        #     return theta_prev,0,V,gradV,log_z_inverse,sign


    def theta_given_table_gibbs_step(
            self,
            index:int,
            theta_prev:Union[list,np.array,np.ndarray],
            log_destination_attraction:Union[list,np.array,np.ndarray],
            table:Union[dict,None], 
            values:list
    ):

        ''' Theta update '''

        self.logger.debug('Theta Gibbs step given table')

        # Unpack values
        V, \
        gradV, \
        log_z_inverse, \
        negative_log_table_likelihood, \
        sign = values

        # UNCOMMENT
        # print('theta',theta_prev)
        # print('xx',log_destination_attraction)
        # print('V',V)
        # print('gradV',gradV)
        # print('log_table_likelihood',-negative_log_table_likelihood)
        # print('log_z_inverse',log_z_inverse)

        # Theta-proposal (random walk with reflecting boundaries
        theta_new = theta_prev.unsqueeze(dim=1) + self.theta_step_size*torch.matmul(self.theta_proposal_covariance, torch.randn((2,1)))
        theta_new = theta_new.squeeze()
        # Relfect the boundaries if theta proposal falls outside of [0,2]^2
        for j in range(2):
            if theta_new[j] < 0.:
                # Reflect off boundary
                theta_new[j] = -theta_new[j]
            elif theta_new[j] > 2.:
                # Reflect off boundary
                theta_new[j] = 2. - (theta_new[j] - 2.)

        # Theta-accept/reject
        if theta_new.min() < 0 or theta_new.max() >= 2:
            # print(f'Parameters {theta_new} outside of [0,2]^2 range')
            # self.logger.debug("Rejected")
            return theta_prev, 0, V, gradV, log_z_inverse, negative_log_table_likelihood, sign

        # try:
        # Multiply beta by total cost
        theta_new_scaled = deepcopy(theta_new) 
        theta_new_scaled[1] *= self.physics_model.params.bmax
        theta_new_dict = dict(zip(list(self.physics_model.params_to_learn.keys()),theta_new_scaled))

        # Compute inverse of z(theta)
        log_z_inverse_new, sign_new = self.z_inverse(
            index=index,
            theta=theta_new_dict
        )

        # Evaluate log potential function for theta proposal
        V_new, gradV_new = self.physics_model.sde_potential_and_gradient(
            log_destination_attraction,
            **theta_new_dict,
            **vars(self.physics_model.params)
        )

        # Compute table likelihood and its gradient
        negative_log_table_likelihood_new = self.negative_table_log_likelihood_expanded(
            log_destination_attraction = log_destination_attraction,
            alpha = theta_new_dict['alpha'],
            beta = theta_new_dict['beta'],
            table = table
        )

        # Compute log parameter posterior for choice of X and updated theta proposal
        log_target_new = log_z_inverse_new - V_new - negative_log_table_likelihood_new
        # Compute log parameter posterior for choice of X and initial theta
        log_target = log_z_inverse - V - negative_log_table_likelihood

        if not torch.isfinite(log_target) or not torch.isfinite(log_target_new):
            print('log_target_new',torch.isfinite(log_target_new))
            print('log_z_inverse_new',log_z_inverse_new)
            raise Exception('Nulls appeared in theta_given_table_gibbs_step')
    
        # UNCOMMENT
        # Multiply beta by total cost
        # theta_scaled_and_expanded_prev = np.concatenate([theta_prev,np.array([self.physics_model.intensity_model.data.delta,self.physics_model.intensity_model.gamma,self.physics_model.intensity_model.data.kappa,self.physics_model.intensity_model.data.epsilon])])
        # theta_scaled_and_expanded_prev[1] *= self.physics_model.params.bmax
        # log_intensities_prev = self.physics_model.intensity_model.log_intensity(
        #                     log_destination_attraction,
        #                     theta_scaled_and_expanded_prev,
        #                     total_flow=np.sum(table.ravel())
        #                 )
        # theta_scaled_and_expanded_new = np.concatenate([theta_new,np.array([self.physics_model.intensity_model.data.delta,self.physics_model.intensity_model.gamma,self.physics_model.intensity_model.data.kappa,self.physics_model.intensity_model.data.epsilon])])
        # theta_scaled_and_expanded_new[1] *= self.physics_model.params.bmax
        # log_intensities_new = self.physics_model.intensity_model.log_intensity(
        #                     log_destination_attraction,
        #                     theta_scaled_and_expanded_new,
        #                     total_flow=np.sum(table.ravel())
        #                 )
        # print(("Proposing " + str(theta_new)))
        # print(("Current sample " + str(theta_prev)))
        # print('negative_log_table_likelihood',negative_log_table_likelihood)
        # print('-negative_log_table_likelihood_new',negative_log_table_likelihood_new)
        # print('log table likelihood difference',-negative_log_table_likelihood_new + negative_log_table_likelihood)
        # print('log target difference without table likelihood',log_z_inverse_new - V_new - (log_z_inverse - V))
        # print('log_target_new - log_target',log_target_new - log_target)
        # print('theta_new',theta_new)
        # print('V_new',V_new)
        # print('V',V)
        # print('log_z_inverse_new - V_new',log_z_inverse_new - V_new)
        # print('log_z_inverse - V',log_z_inverse - V)
        # print('log_z_inverse',log_z_inverse)
        # print('log_z_inverse_new',log_z_inverse_new)
        # print('\n')
        
        if np.log(np.random.uniform(0, 1)) < log_target_new - log_target:
            self.logger.debug("Accepted")
            return theta_new, \
                    1, \
                    V_new, \
                    gradV_new, \
                    log_z_inverse_new, \
                    negative_log_table_likelihood_new, \
                    sign_new
        else:
            self.logger.debug("Rejected")
            return theta_prev, \
                    0, \
                    V, \
                    gradV, \
                    log_z_inverse, \
                    negative_log_table_likelihood, \
                    sign
        # except:
        #     print("Exception raised in theta_given_table_gibbs_step")
        #     # self.logger.debug("Exception raised")
        #     return theta_prev, 0, V, gradV, log_z_inverse, negative_log_table_likelihood, negative_gradient_log_table_likelihood, sign


    def log_destination_attraction_gibbs_step(self,theta,log_destination_attraction_data,log_destination_attraction_prev,values:list):

        self.logger.debug('Log destination attraction Gibbs step')

        # Unpack values
        V, \
        gradV = values

        # Multiply beta by total cost
        theta_scaled = deepcopy(theta)
        theta_scaled[1] *= self.physics_model.params.bmax

        ''' Log destination demand update '''

        # Initialize leapfrog integrator for HMC proposal
        momentum = torch.randn(self.physics_model.intensity_model.dims['destination'],dtype=float32,device=self.device)
        # Compute log(\pi(y|x))
        negative_log_data_likelihood, \
        negative_gradient_log_data_likelihood = self.physics_model.negative_destination_attraction_log_likelihood_and_gradient(
                log_destination_attraction_data,
                log_destination_attraction_prev,
                1./self.physics_model.noise_var
        )
        # Compute log initial potential energy and its derivarive weighted by the likelihood function \pi(y|x)
        # \log(\exp(-\gamma)V_{\theta}(xx)) + \log(\pi(y|x)) + \log(p(T|x,\theta))
        # V is equal to \gamma*V_{\theta}(xx) + 1/(2*s^2)*(xx-xx_data)^2 +
        W, gradW = V + negative_log_data_likelihood, gradV + negative_gradient_log_data_likelihood
        # Initial total log Hamiltonian energy (kinetic + potential)
        H = 0.5*torch.dot(momentum, momentum) + W

        # print('V',V)
        # print('gradV',gradV)
        # print('log_destination_attraction',log_destination_attraction)
        # print('log_data_likelihood',log_data_likelihood)
        # print('gradient_log_data_likelihood',gradient_log_data_likelihood)
        # print('V',V)
        # print('VL',log_data_likelihood)
        # print('W',W)
        # print('H',H)
        # print('\n')

        # Momentum initialisation
        momentum_new = momentum
        # X-Proposal
        log_destination_attraction_new = log_destination_attraction_prev
        # Initial log potential energy and its gradient weighted by the likelihood function \pi(y|x)
        W_new, gradW_new = W, gradW

        # print('gradW_new',gradW_new)
        # print('momentum_new',momentum_new)
        # print('V',V)
        # print('log_data_likelihood',log_data_likelihood)
        # print('gradV',gradV)
        # print('gradient_log_data_likelihood',gradient_log_data_likelihood)
        # print('gradW_new',gradW_new)
        # print('gradV + negative_gradient_log_data_likelihood',gradV + negative_gradient_log_data_likelihood)
        # print(momentum_new -0.5*self.destination_attraction_step_size*gradW_new)

        # Leapfrog integrator
        for j in range(self.destination_attraction_leapfrog_steps):
            # Make a half step for momentum in the beginning
            # inverse_temps[j]*gradV_p = grad V(x|theta)*(1/T)
            momentum_new = momentum_new -0.5*self.destination_attraction_step_size*gradW_new
            # Make a full step for the position
            log_destination_attraction_new = log_destination_attraction_new + self.destination_attraction_step_size*momentum_new

            # Update log potential energy and its gradient
            # Compute updated log(\pi(y|x))
            negative_log_data_likelihood_new, negative_gradient_log_data_likelihood_new = self.physics_model.negative_destination_attraction_log_likelihood_and_gradient(
                    log_destination_attraction_data,
                    log_destination_attraction_new,
                    1./self.physics_model.noise_var
            )
            # Compute updated log potential function
            V_new, gradV_new = self.physics_model.sde_potential_and_gradient(
                                        log_destination_attraction_new,
                                        **dict(zip(list(self.physics_model.params_to_learn.keys()),theta_scaled)),
                                )
            # Compute log updated potential energy and its derivarive weighted by the likelihood function \pi(y|x)
            # \log(\exp(-\gamma)V_{\theta}(xx)) + \log(\pi(y|x)) + \log(p(T|x,\theta))
            W_new, gradW_new = V_new + negative_log_data_likelihood_new, \
                            gradV_new + negative_gradient_log_data_likelihood_new

            # print('',momentum_new.shape)
            # print('destination_attraction_new',np.exp(log_destination_attraction_new).sum())
            # print('V_new',V_new)
            # print('gradV_new',gradV_new)
            # print('------------')
            # print('W_new',W_new.shape)
            # print('gradW_new 2',gradW_new)
            # Make a full step for the momentum except at the end of trajectory
            momentum_new = momentum_new - 0.5*self.destination_attraction_step_size*gradW_new

            # UNCOMMENT
            # H_new = 0.5*np.dot(momentum_new, momentum_new) + W_new
            # print('log_destination_attraction_prev',log_destination_attraction_prev)
            # print('log_destination_attraction_new',log_destination_attraction_new)
            # print('V',V,'V_new',V_new)
            # print('negative_log_data_likelihood',negative_log_data_likelihood)
            # print('negative_log_data_likelihood_new',negative_log_data_likelihood_new)
            # print('0.5*np.dot(momentum, momentum)',0.5*np.dot(momentum, momentum))
            # print('0.5*np.dot(momentum_new, momentum_new)',0.5*np.dot(momentum_new, momentum_new))
            # print('H-H_new',H-H_new)
            # print('gradient without poisson likelihood new ',gradV_new + negative_gradient_log_data_likelihood_new)
            # print('gradient of poisson likelihood new ',negative_gradient_log_table_likelihood_new)
            # print('gradW_new',gradW_new)
            # print('\n')
            # print("Proposing " + str(log_destination_attraction_new))
            # print("Momentum " + str(momentum_new))
            # sys.exit()

        # Compute proposal log Hamiltonian energy
        H_new = 0.5*torch.dot(momentum_new, momentum_new) + W_new

        # print("Proposing " + str(log_destination_attraction_new) + ' with ' + str(H_new))
        # print(str(log_destination_attraction_prev) + ' vs ' + str(H))
        # print(("Difference log target " + str(H-H_new)))
        # print(np.exp(log_destination_attraction_new).sum())
        # print('H-H_new',H-H_new)
        # print('\n')

        # Accept/reject
        if torch.log(torch.rand(1)) < H - H_new:
            return log_destination_attraction_new, 1, V_new, gradV_new
        else:
            return log_destination_attraction_prev, 0, V, gradV


    def log_destination_attraction_given_table_gibbs_step(self,theta,log_destination_attraction_data,log_destination_attraction_prev,table,values:list):

        self.logger.debug('Log destination attraction Gibbs step given table')

        # Unpack values
        V, \
        gradV, \
        negative_log_table_likelihood = values

        # Multiply beta by total cost
        theta_scaled = deepcopy(theta)
        theta_scaled[1] *= self.physics_model.params.bmax
        theta_scaled_dict = dict(zip(list(self.physics_model.params_to_learn.keys()),theta_scaled))

        ''' Log destination demand update '''

        # print('total intensity',np.sum(np.exp(log_intensity)))
        # negative_log_table_likelihood_copy = self.negative_table_log_likelihood(log_intensity,table)
        # print('negative_log_table_likelihood',negative_log_table_likelihood)
        # print('negative_log_table_likelihood copy',negative_log_table_likelihood_copy)
        
        # Initialize leapfrog integrator for HMC proposal
        momentum = torch.randn(size=(self.physics_model.intensity_model.dims['destination'],))
        # Compute -log(\pi(y|x))
        negative_log_data_likelihood, \
        negative_gradient_log_data_likelihood = self.physics_model.negative_destination_attraction_log_likelihood_and_gradient(
                log_destination_attraction_data,
                log_destination_attraction_prev,
                1./self.physics_model.noise_var
        )
        # # Compute gradient of lambda
        # intensity_gradient = self.physics_model.intensity_model.intensity_gradient(
        #     log_destination_attraction = log_destination_attraction_prev,
        #     **theta_scaled_dict
        # )

        # Initialise gradient of log table likelihood
        negative_gradient_log_table_likelihood = self.negative_table_log_likelihood_gradient(
            log_destination_attraction = log_destination_attraction_prev,
            table = table,
            **theta_scaled_dict
        )

        # Compute log initial potential energy and its derivarive weighted by the likelihood function \pi(y|x)
        # \log(\exp(-\gamma)V_{\theta}(xx)) + \log(\pi(y|x)) + \log(p(T|x,\theta))
        # V is equal to \gamma*V_{\theta}(xx) + 1/(2*s^2)*(xx-xx_data)^2 +
        W = V + \
                negative_log_data_likelihood + \
                negative_log_table_likelihood
        gradW = gradV + \
            negative_gradient_log_data_likelihood + \
            negative_gradient_log_table_likelihood
        # Initial total log Hamiltonian energy (kinetic + potential)
        H = 0.5*np.dot(momentum, momentum) + W

        # UNCOMMENT
        # print('STARTING')
        # print('V',V)
        # print('gradV',gradV)
        # print('log_destination_attraction_prev',log_destination_attraction_prev)
        # print('log_table_likelihood',-negative_log_table_likelihood)
        # print('gradient_log_table_likelihood',negative_gradient_log_table_likelihood)
        # print('log_data_likelihood',negative_log_data_likelihood)
        # print('gradient_log_data_likelihood',negative_gradient_log_data_likelihood)
        # print('gradV + negative_gradient_log_data_likelihood',gradW+negative_gradient_log_data_likelihood)
        # print('gradW',gradW)
        # print('momentum',momentum)
        # print('H',H)
        # print('\n')

        # Momentum initialisation
        momentum_new = momentum
        # X-Proposal
        log_destination_attraction_new = log_destination_attraction_prev
        # Initial log potential energy and its gradient weighted by the likelihood function \pi(y|x)
        W_new, gradW_new = W, gradW

        # print('xx',np.exp(log_destination_attraction_prev).sum())
        # print('\n')

        # print('gradW_new',gradW_new.shape)
        # print('momentum_new',momentum_new.shape)
        # print('V',V)
        # print('destination_attraction_new',np.exp(log_destination_attraction_new).sum())
        # print('negative_log_data_likelihood',negative_log_data_likelihood)
        # print('gradV',gradV)
        # print('negative_gradient_log_data_likelihood',negative_gradient_log_data_likelihood)
        # print('gradW_new',gradW_new)
        # print('gradV + negative_gradient_log_data_likelihood',gradV + negative_gradient_log_data_likelihood)
        # print('negative_gradient_log_table_likelihood',negative_gradient_log_table_likelihood)
        # print('\n')
        # print(momentum_new -0.5*self.destination_attraction_step_size*gradW_new)

        # Leapfrog integrator
        for j in range(self.destination_attraction_leapfrog_steps):
            # Make a half step for momentum in the beginning
            # inverse_temps[j]*gradV_p = grad V(x|theta)*(1/T)
            momentum_new = momentum_new -0.5*self.destination_attraction_step_size*gradW_new
            # Make a full step for the position
            log_destination_attraction_new = log_destination_attraction_new + self.destination_attraction_step_size*momentum_new

            # Update log potential energy and its gradient
            # Compute updated -log(\pi(y|x))
            negative_log_data_likelihood_new, \
            negative_gradient_log_data_likelihood_new = self.physics_model.negative_destination_attraction_log_likelihood_and_gradient(
                log_destination_attraction_data,
                log_destination_attraction_new,
                1./self.physics_model.noise_var
            )
            # Compute updated log potential function
            V_new, gradV_new = self.physics_model.sde_potential_and_gradient(
                log_destination_attraction_new,
                **theta_scaled_dict
            )

            # print('xx',np.exp(log_destination_attraction_new).sum())
            # print('intensity')
            # print(log_intensity)
            # print('\n ')
            
            # Compute negative table likelihood
            negative_log_table_likelihood_new, \
            negative_gradient_log_table_likelihood_new = self.negative_table_log_likelihood_and_gradient(
                log_destination_attraction = log_destination_attraction_prev,
                table = table,
                **theta_scaled_dict
            )
                                                                                            
            # Compute log updated potential energy and its derivarive weighted by the likelihood function \pi(y|x)
            # \log(\exp(-\gamma)V_{\theta}(xx)) + \log(\pi(y|x)) + \log(p(T|constraints, x, \theta))
            W_new = V_new + \
                    negative_log_data_likelihood_new + \
                    negative_log_table_likelihood_new
            gradW_new = gradV_new + \
                        negative_gradient_log_data_likelihood_new + \
                        negative_gradient_log_table_likelihood_new

            # Make a full step for the momentum except at the end of trajectory
            momentum_new = momentum_new - 0.5*self.destination_attraction_step_size*gradW_new

            # UNCOMMENT
            H_new = 0.5*torch.dot(momentum_new, momentum_new) + W_new
            # print('destination_attraction_prev',np.exp(log_destination_attraction_prev),np.exp(log_destination_attraction_prev).sum())
            # print('destination_attraction_new',np.exp(log_destination_attraction_new),np.exp(log_destination_attraction_new).sum())
            # print('destination_attraction_prev',np.exp(log_destination_attraction_prev).sum())
            # print('destination_attraction_new',np.exp(log_destination_attraction_new).sum())
            # print('V',V,'V_new',V_new)            
            # print('gradV_new',gradV_new)
            # print('negative_gradient_log_data_likelihood_new',negative_gradient_log_data_likelihood_new)
            # print('negative_gradient_log_table_likelihood_new',negative_gradient_log_table_likelihood_new)
            # print('gradW_new',gradW_new)
            # print('W_new',np.shape(W_new))
            # print('momentum_new',np.shape(momentum_new))
            # print('negative_log_data_likelihood',negative_log_data_likelihood)
            # print('negative_log_data_likelihood_new',negative_log_data_likelihood_new)
            # print('gradient without table likelihood new ',gradV_new + negative_gradient_log_data_likelihood_new)
            # print('gradient of table likelihood new ',negative_gradient_log_table_likelihood_new)
            # print('V-V_new',V-V_new)
            # print('log data likelihood difference',negative_log_data_likelihood_new-negative_log_data_likelihood)
            # print('log table likelihood difference',negative_log_table_likelihood-negative_log_table_likelihood_new)
            # print('log target difference without table likelihood',(V + negative_log_data_likelihood + 0.5*np.dot(momentum, momentum))-(V_new + negative_log_data_likelihood_new + 0.5*np.dot(momentum_new, momentum_new)))
            # print('H-H_new',H-H_new)
            # print('Index',j)
            # print('W_new',W_new)
            # print('gradW_new',gradW_new)
            # print(np.sum(np.exp(self.destination_attraction_step_size*momentum_new)-1))
            # print('\n')

            # print("Proposing " + str(log_destination_attraction_new))
            # print("Momentum " + str(momentum_new))
            # sys.exit()

        # Compute proposal log Hamiltonian energy
        H_new = 0.5*torch.dot(momentum_new, momentum_new) + W_new
        
        # Make sure acceptance ratio is finite
        if not torch.isfinite(H_new) or not torch.isfinite(H):
            raise Exception('Nulls appeared in log_destination_attraction_given_table_gibbs_step')

        # Compute variances
        # print('Data variance',np.repeat(self.physics_model.noise_var,self.physics_model.intensity_model.dims['destination']))
        # table_rowsums = table.sum(axis=1).reshape((self.physics_model.intensity_model.dims[0],1))
        # intensity_rowsums = np.array([logsumexp(log_intensity[i,:]) for i in range(self.physics_model.intensity_model.dims[0])]).reshape((self.physics_model.intensity_model.dims[0],1))
        # intensity_probs = np.exp( log_intensity - intensity_rowsums )
        # table_variance = np.sum(table_rowsums*intensity_probs*(1-intensity_probs),axis=0)
        # print('Table variance',table_variance)

        # UNCOMMENT
        # print('log_destination_attraction_new',np.exp(log_destination_attraction_new).sum())
        # print('V-V_new',V-V_new)
        # print('log data likelihood difference',negative_log_data_likelihood-negative_log_data_likelihood_new)
        # print('negative_log_table_likelihood',negative_log_table_likelihood)
        # print('negative_log_table_likelihood_new',negative_log_table_likelihood_new)
        # print('log table likelihood difference',negative_log_table_likelihood-negative_log_table_likelihood_new)
        # print('log target difference without table likelihood',(V + negative_log_data_likelihood + 0.5*np.dot(momentum, momentum))-(V_new + negative_log_data_likelihood_new + 0.5*np.dot(momentum_new, momentum_new)))
        # print('H-H_new',H-H_new)
        # print('\n')
        
        # Accept/reject
        if np.log(np.random.uniform(0, 1)) < H - H_new:
            return log_destination_attraction_new,\
                    1,\
                    V_new,\
                    gradV_new,\
                    negative_log_table_likelihood_new
        else:
            return log_destination_attraction_prev,\
                    0,\
                    V,\
                    gradV,\
                    negative_log_table_likelihood