import time
import sys
import logging
import numpy as np

from os import path
from typing import Union
from functools import partial
from pathlib import Path as PathLib
from joblib import Parallel, delayed
# from numba_progress import ProgressBar
import multiresticodm.probability_utils as ProbabilityUtils

from multiresticodm.math_utils import scipy_optimize
from multiresticodm.spatial_interaction_model import SpatialInteraction
from multiresticodm.utils import setup_logger, str_in_list,makedir,set_seed,deep_get

def instantiate_spatial_interaction_mcmc(sim:SpatialInteraction,**kwargs):
    if hasattr(sys.modules[__name__], (sim.sim_name+'MarkovChainMonteCarlo')):
        return getattr(sys.modules[__name__], (sim.sim_name+'MarkovChainMonteCarlo'))(sim,**kwargs)
    else:
        raise Exception(f"Input class {(sim.sim_name+'MarkovChainMonteCarlo')} not found")

class SpatialInteractionMarkovChainMonteCarlo():

    def __init__(self, sim:SpatialInteraction,**kwargs):
        # Setup logger
        level = sim.config.level if hasattr(sim,'config') else kwargs.get('level','INFO')
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            level=level,
            log_to_file=False,
            log_to_console=kwargs.get('log_to_console',True),
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevel(level)
        # Store sim model
        self.sim = sim

        # Stopping times
        self.stopping_times = None
        self.stopping_times_directory = None
        # Number of parallelisation workers
        self.n_workers = int(next(deep_get(key='n_workers',value=self.sim.config.settings)))
        self.n_threads = next(deep_get(key='n_threads',value=self.sim.config.settings))

        # Get seed and number of cores for multiprocessing
        self.seed = int(self.sim.config.settings['inputs'].get('seed',None)) \
                    if self.sim.config.settings['inputs'].get('seed',None) is not None \
                    else None

        self.logger.info(f'Building {sim.noise_regime} {sim.sim_type} Markov Chain Monte Carlo Engine')

    def __repr__(self):
        return "MarkovChainMonteCarlo(SpatialInteraction)"
        
    def load_stopping_times(self):
        # Get total number of stopping times samples required
        n_samples_required = int(self.N)
        if hasattr(self,'theta_steps'):
            n_samples_required *= int(self.theta_steps)
        self.logger.info(f'Attempting to import {n_samples_required} stopping times')
        # Try to import stopping times from file
        read_stopping_times = self.import_stopping_times(N=n_samples_required)
        # If file is not provided
        if not read_stopping_times:
            # Generate stopping times
            self.logger.info('Generating stopping times')
            self.stopping_times = ProbabilityUtils.generate_stopping_times(N=n_samples_required,k_power=self.k_power,seed=self.seed)
            # Reset random seed
            set_seed(None)
            # Export stopping times
            self.export_stopping_times()

        # Truncate based on samples required
        self.stopping_times = self.stopping_times[:n_samples_required]
        self.logger.warning(f'Max stopping time is {np.max(self.stopping_times)}')

    def import_stopping_times(self,N:int):
        # Get path to directory
        parent_directory = PathLib(self.sim.config.settings['inputs']['dataset'])
        # Make directory
        directory = path.join(str(parent_directory),'stopping_times')
        makedir(directory)
        # Extract filepath
        self.stopping_times_filepath = path.join(parent_directory,'stopping_times',self.sim.config.settings['spatial_interaction_model']['import']['stopping_times'])
        if path.isfile(self.stopping_times_filepath):
            # Get number of samples of stopping times in file
            self.stopping_times = np.loadtxt(self.stopping_times_filepath,dtype='int32')
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
            np.savetxt(self.stopping_times_filepath,self.stopping_times)


class SpatialInteraction2DMarkovChainMonteCarlo(SpatialInteractionMarkovChainMonteCarlo):

    def __init__(self, sim:SpatialInteraction,**kwargs):
        
        # Instantiate superclass
        super().__init__(sim,**kwargs)

        # Store table distribution name
        try:
            self.table_distribution_name = self.sim.config.settings['inputs']['contingency_table']['distribution_name']
        except:
            raise Exception("No distribution name provided in config.")

        # Store total iterations
        self.N = self.sim.config.settings['mcmc']['N']

        # Store initial samples
        self.theta0 = np.array([self.sim.config.settings['mcmc']['spatial_interaction_model']['parameters']['alpha0'],
                                self.sim.config.settings['mcmc']['spatial_interaction_model']['parameters']['beta0']])

        # Read initialisation
        self.log_destination_attraction0 = None
        if str_in_list('log_destination_attraction0',self.sim.config.settings['mcmc']['spatial_interaction_model']['destination_attraction'].keys()):
            initialisation = self.sim.config.settings['mcmc']['spatial_interaction_model']['destination_attraction']['log_destination_attraction0']
            # If it is a path read file
            if isinstance(initialisation,str):
                # Construct path
                input_directory = self.sim.config.settings['inputs']['dataset']
                # Extract filepath
                if path.isfile(path.join(input_directory,initialisation)):                    
                    self.log_destination_attraction0 = np.loadtxt(path.join(input_directory,initialisation),dtype='float32')
            # If it is an array/list make sure it has the 
            elif isinstance(initialisation,(list,np.ndarray,np.generic)):
                if len(initialisation) == self.sim.dims[1]:
                    self.log_destination_attraction0 = initialisation
            else:
                self.logger.warning(f"Initial destination attraction {initialisation} is neither a list nor a valid filepath.")

        # If destination attraction has not been initialised
        if self.log_destination_attraction0 is None:
            self.logger.warning(f"Destination attraction is arbitrarily initialised.")
            # arbitrarily initialise destination attraction
            self.log_destination_attraction0 = np.log(np.repeat(1./self.sim.dims[1],self.sim.dims[1]).astype('float32'))

        # Decide on which sampler to use based on level of noise
        if self.sim.noise_regime == 'low':
            # Update method for computing z inverse
            self.z_inverse = self.biased_z_inverse
        else:
            # Update method for computing z inverse
            self.z_inverse = self.unbiased_z_inverse

            # Store number of Annealed Importance Samples (AIS)
            self.destination_attraction_ais_samples = self.sim.config.settings['mcmc']['spatial_interaction_model']['destination_attraction']['ais_samples']
            # Store Leapfrog steps for AIS
            self.destination_attraction_leapfrog_steps_ais = self.sim.config.settings['mcmc']['spatial_interaction_model']['destination_attraction']['ais_leapfrog_steps']
            # Store Leapfrog step size for AIS
            self.destination_attraction_leapfrog_step_size_ais = self.sim.config.settings['mcmc']['spatial_interaction_model']['destination_attraction']['ais_leapfrog_step_size']
            # Store number of bridging distributions in temperature schedule of AIS
            self.destination_attraction_n_bridging_distributions = self.sim.config.settings['mcmc']['spatial_interaction_model']['destination_attraction']['n_bridging_distributions']
            # Store power of k for truncating infinite series of AIS samples
            self.k_power = 1.1#self.sim.config.settings['mcmc']['spatial_interaction_model']['destination_attraction']['k_power']

            # Get stopping times for Z inverse estimator
            self.load_stopping_times()

            # Define sampling methods
            self.annealed_importance_sampling_log_z_partial = partial(
                self.sim.annealed_importance_sampling_log_z,
                ais_samples=self.destination_attraction_ais_samples,
                n_temperatures=self.destination_attraction_n_bridging_distributions,
                leapfrog_steps=self.destination_attraction_leapfrog_steps_ais,
                epsilon=self.destination_attraction_leapfrog_step_size_ais,
                origin_demand=self.sim.origin_demand,
                cost_matrix=self.sim.cost_matrix   
            )

        if self.sim.config.settings['mcmc'].get('table_inference',False):
            self.theta_step = self.theta_given_table_gibbs_step
            self.log_destination_attraction_step = self.log_destination_attraction_given_table_gibbs_step
            # Keep track of joint log destination attarction and theta steps
            self.theta_steps = int(self.sim.config.settings['mcmc']['spatial_interaction_model'].get('theta_steps',1))
            self.log_destination_attraction_steps = int(self.sim.config.settings['mcmc']['spatial_interaction_model'].get('log_destination_attraction_steps',1))
        else:
            self.theta_steps = 1
            self.log_destination_attraction_steps = 1
            self.theta_step = self.theta_gibbs_step
            self.log_destination_attraction_step = self.log_destination_attraction_gibbs_step

        # Store proposal covariance
        self.theta_proposal_covariance = np.array(self.sim.config.settings['mcmc']['spatial_interaction_model']['parameters']['covariance'])
        # Store theta step size
        self.theta_step_size = self.sim.config.settings['mcmc']['spatial_interaction_model']['parameters']['step_size']

        # Store Leapfrog steps
        self.destination_attraction_leapfrog_steps = self.sim.config.settings['mcmc']['spatial_interaction_model']['destination_attraction']['leapfrog_steps']
        # Store Leapfrog step size
        self.destination_attraction_step_size = self.sim.config.settings['mcmc']['spatial_interaction_model']['destination_attraction']['leapfrog_step_size']

        # Get table distribution
        self.table_distribution = f"log_{self.table_distribution_name}_pmf"
        # Get log table pmf unnormalised jacobian
        self.log_table_likelihood_total_derivative_wrt_x = getattr(ProbabilityUtils, 'log_table_likelihood_total_derivative_wrt_x')
        if hasattr(ProbabilityUtils, (self.table_distribution+"_unnormalised")):
            self.table_unnormalised_log_likelihood = getattr(ProbabilityUtils, self.table_distribution+"_unnormalised")
        else:
            raise Exception(f"Input class ProbabilityUtils does not have distribution {(self.table_distribution+'_unnormalised')}")
        self.table_log_likelihood_jacobian = None
        if hasattr(ProbabilityUtils, (self.table_distribution+'_jacobian_wrt_intensity')):
            self.table_log_likelihood_jacobian = getattr(ProbabilityUtils, (self.table_distribution+'_jacobian_wrt_intensity'))
        else:
            raise Exception(f"Input class ProbabilityUtils does not have distribution {(self.table_distribution+'wrt')} either intensity")

        # self.logger.info((self.__str__()))

    def __str__(self):
        return f"""
            2D Spatial Interaction Model Markov Chain Monte Carlo algorithm
            Dataset: {self.sim.config.settings['inputs']['dataset']}
            Cost matrix: {self.sim.config.settings['spatial_interaction_model']['import']['cost_matrix']}
            Destination attraction sum: {np.exp(self.sim.log_destination_attraction).sum()}
            Cost matrix sum: {np.sum(self.sim.cost_matrix.ravel())}
            Origin demand sum: {np.sum(self.sim.origin_demand)}
            Delta: {self.sim.delta}
            Kappa: {self.sim.kappa}
            Epsilon: {self.sim.epsilon}
            Gamma: {self.sim.gamma}
            Beta scaling: {self.sim.bmax}
            Table dimensions: {"x".join([str(x) for x in self.sim.dims])}
            Noise regime: {self.sim.noise_regime}
            Number of cores: {self.n_workers}
            Number of threads per core: {str(self.n_threads)}
            Random number generation seed: {self.seed}
        """

    def negative_table_log_likelihood(self,log_intensity:Union[np.array,np.ndarray],table:np.ndarray,normalised:bool=False):
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

    
    def negative_table_log_likelihood_gradient(self,log_intensity:Union[np.array,np.ndarray],table:np.ndarray,**kwargs):
        self.logger.debug('negative_table_log_likelihood_gradient \n')

        # Get log table likelihood derivative with respect to intensity
        self.logger.debug('table_log_likelihood_jacobian')
        log_likelihood_grad = self.table_log_likelihood_jacobian(log_intensity,table).astype('float32')
        self.logger.debug('log_table_likelihood_total_derivative_wrt_x')
        # Get total derivative by chain rule
        log_likelihood_total_gradient = self.log_table_likelihood_total_derivative_wrt_x(
                likelihood_grad = log_likelihood_grad,
                intensity_grad_x = kwargs.get('intensity_gradient',None)
        )
        return -log_likelihood_total_gradient
    
    def negative_table_log_likelihood_and_gradient(
            self,
            log_intensity:Union[np.array,np.ndarray],
            table:np.ndarray,
            **kwargs
    ):
        negative_table_log_likelihood = self.negative_table_log_likelihood(log_intensity,table)
        negative_table_log_likelihood_gradient = self.negative_table_log_likelihood_gradient(log_intensity,table,**kwargs)
        return negative_table_log_likelihood,negative_table_log_likelihood_gradient

    def biased_z_inverse(self,index:int,theta:Union[np.ndarray,list]):
        self.logger.debug('Biased Z inverse')
        # compute 1/z(theta) using Saddle point approximation

        # Get delta
        delta = theta[2]
        
        # Create partial function
        scipy_optimize_partial = partial(scipy_optimize,function=self.sim.sde_potential_and_gradient,method='L-BFGS-B',theta=theta)

        # Create initialisations
        g = np.log(delta)*np.ones((self.sim.dims[1],self.sim.dims[1])) - \
            np.log(delta)*np.eye(self.sim.dims[1]) +\
            np.log(1+delta)*np.eye(self.sim.dims[1])
        g = g.astype('float32')
        
        # Get minimum across different initialisations in parallel
        if self.n_workers > 1:
            xs = np.asarray(Parallel(n_jobs=self.n_workers)(delayed(scipy_optimize_partial)(g[i,:]) for i in range(self.sim.dims[1])))
        else:
            xs = np.asarray([scipy_optimize_partial(g[i,:]) for i in range(self.sim.dims[1])])
        # Compute potential
        fs = np.asarray([self.sim.sde_potential(xs[j],theta) for j in range(self.sim.dims[1])])
        
        # Get arg min
        arg_min = np.argmin(fs)
        minimum = xs[arg_min]
        minimum_potential = fs[arg_min]

        # Get Hessian matrix
        A = self.sim.sde_potential_hessian(minimum,theta)
        # Find its cholesky decomposition Hessian = L*L^T for efficient computation
        L = np.linalg.cholesky(A)
        # Compute the log determinant of the hessian
        # det(Hessian) = det(L)*det(L^T) = det(L)^2
        # det(L) = \prod_{j=1}^M L_{jj} and
        # \log(det(L)) = \sum_{j=1}^M \log(L_{jj})
        # So \log(det(Hessian)^(1/2)) = \log(det(L))
        half_log_det_A = np.sum(np.log(np.diag(L)))

        # Compute log_normalising constant, i.e. \log(z(\theta))
        # -gamma*V(x_{minimum}) + (M/2) * \log(2\pi \gamma^{-1})
        # lap =  -si.sde_potential_and_gradient(minimum,theta)[0] + lap_c1 - half_log_det_A
        # Compute log-posterior
        # \log(p(x|\theta)) = -gamma*V(x) - \log(z(\theta))
        # log_likelihood_values[i, j] = -lap - si.sde_potential_and_gradient(xd,theta)[0]

        # Return array
        ret = np.empty(2)
        # Log z(\theta) without the constant (2\pi\gamma^{-1})^{M/2}
        ret[0] = minimum_potential + half_log_det_A
        # self.sim.sde_potential_and_gradient(minimum,theta)[0] +  half_log_det_A
        # Sign of log inverse of z(\theta)
        ret[1] = 1.

        return ret

    
    def unbiased_z_inverse(self,index:int,theta:Union[np.ndarray,list]):
        # Debiasing scheme - returns unbiased esimates of 1/z(theta)

        # Extract stopping time
        N = int(self.stopping_times[index])

        self.logger.debug(f"Unbiased Z inverse N = {N+1}")

        # print(f"Debiasing with N = {N}")
        # if min(self.n_workers,N+1) > 3: 
        if self.n_workers > 1: 
            # Multiprocessing
            # self.logger.debug(f"Multiprocessing with workers = {min(self.n_workers,N+1)}")
            # with joblib_progress(f"Multiprocessing {min(self.n_workers,N+1)}", total=(N+1)):
            # log_weights = np.asarray(
            #     Parallel(n_jobs=min(self.n_workers,N+1))(
            #         delayed(self.annealed_importance_sampling_log_z_partial)(i,theta) for i in range(N+1)
            #     )
            # )
            # Parallelise function in numba
            # with ProgressBar(total=(N+1)) as progress_bar:
            self.logger.debug(f"annealed_importance_sampling_log_z_parallel")
            log_weights = self.sim.annealed_importance_sampling_log_z_parallel(
                index=(N+1),
                theta=theta,
                ais_samples=int(self.destination_attraction_ais_samples),
                n_temperatures=int(self.destination_attraction_n_bridging_distributions),
                leapfrog_steps=int(self.destination_attraction_leapfrog_steps_ais),
                epsilon=float(self.destination_attraction_leapfrog_step_size_ais),
                origin_demand=self.sim.origin_demand,
                cost_matrix=self.sim.cost_matrix,
                progress_proxy=None
            )

            if not np.all(np.isfinite(log_weights)):
                raise Exception('Nulls/NaNs found in annealed importance sampling')
        else:
            # print('For loop numba')
            log_weights = np.zeros(N+1)
            self.logger.debug(f"annealed_importance_sampling_log_z_partial")
            for i in range(N+1):
                log_weights[i] = self.annealed_importance_sampling_log_z_partial(index=i,theta=theta)
            
            # self.annealed_importance_sampling_log_z_partial.parallel_diagnostics(level=4)
            # sys.exit()
        self.logger.debug(f"Truncating infinite series")
        # Leave this argument as is (it is not a bug - see function dfn)
        ret = ProbabilityUtils.compute_truncated_infinite_series(N,log_weights,self.k_power)
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
        # theta_scaled_and_expanded = np.concatenate([theta_prev,np.array([self.sim.delta,self.sim.gamma,self.sim.kappa,self.sim.epsilon])])
        # theta_scaled_and_expanded[1] *= self.sim.bmax
        # print('theta',theta_scaled_and_expanded)
        # print('xx',log_destination_attraction)
        # print('theta',theta_prev)
        # print('V',V)
        # print('gradV',gradV)
        # print('log_z_inverse',log_z_inverse)

        # Theta-proposal (random walk with reflecting boundaries
        theta_new = theta_prev + self.theta_step_size*np.dot(self.theta_proposal_covariance, np.random.normal(0, 1, 2))

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
            return theta_prev,0,V,gradV,log_z_inverse,sign

        try:
            # Multiply beta by total cost
            theta_scaled_and_expanded_new = np.concatenate([theta_new,np.array([self.sim.delta,self.sim.gamma,self.sim.kappa,self.sim.epsilon])])
            theta_scaled_and_expanded_new[1] *= self.sim.bmax

            # Compute inverse of z(theta)
            log_z_inverse_new, sign_new = self.z_inverse(
                                                index=index,
                                                theta=theta_scaled_and_expanded_new
                                        )

            # Evaluate log potential function for theta proposal
            V_new, gradV_new = self.sim.sde_potential_and_gradient(
                                        log_destination_attraction,
                                        theta_scaled_and_expanded_new
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
            if np.log(np.random.uniform(0, 1)) < log_target_new - log_target:
                self.logger.debug("Accepted")
                return theta_new,1,V_new,gradV_new,log_z_inverse_new,sign_new
            else:
                self.logger.debug("Rejected")
                return theta_prev,0,V,gradV,log_z_inverse,sign
        except:
            print("Exception raised in theta_gibbs_step")
            # self.logger.debug("Exception raised")
            return theta_prev,0,V,gradV,log_z_inverse,sign


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
        log_intensity, \
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
        theta_new = theta_prev + self.theta_step_size*np.dot(self.theta_proposal_covariance, np.random.normal(0, 1, 2))
        # print('theta_new',theta_new)
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
            return theta_prev, 0, V, gradV, log_z_inverse, log_intensity, negative_log_table_likelihood, sign

        # try:
        # Multiply beta by total cost
        theta_scaled_and_expanded_new = np.concatenate([theta_new,np.array([self.sim.delta,self.sim.gamma,self.sim.kappa,self.sim.epsilon])])
        theta_scaled_and_expanded_new[1] *= self.sim.bmax

        # Compute inverse of z(theta)
        log_z_inverse_new, sign_new = self.z_inverse(
                                            index=index,
                                            theta=theta_scaled_and_expanded_new
                                        )

        # Evaluate log potential function for theta proposal
        V_new, gradV_new = self.sim.sde_potential_and_gradient(
                                    log_destination_attraction,
                                    theta_scaled_and_expanded_new
                        )

        # Compute intensity and its gradient with respect to log destination attraction
        log_intensity_new = self.sim.log_intensity(
                            log_destination_attraction,
                            theta_scaled_and_expanded_new,
                            total_flow=np.sum(table)
                        )
        
        # Compute table likelihood and its gradient
        negative_log_table_likelihood_new = self.negative_table_log_likelihood(
                                                        log_intensity_new,
                                                        table
                                        )

        # Compute log parameter posterior for choice of X and updated theta proposal
        log_target_new = log_z_inverse_new - V_new - negative_log_table_likelihood_new
        # Compute log parameter posterior for choice of X and initial theta
        log_target = log_z_inverse - V - negative_log_table_likelihood

        if not np.isfinite(log_target) or not np.isfinite(log_target_new):
            raise Exception('Nulls appeared in theta_given_table_gibbs_step')
    
        # UNCOMMENT
        # Multiply beta by total cost
        # theta_scaled_and_expanded_prev = np.concatenate([theta_prev,np.array([self.sim.delta,self.sim.gamma,self.sim.kappa,self.sim.epsilon])])
        # theta_scaled_and_expanded_prev[1] *= self.sim.bmax
        # log_intensities_prev = self.sim.log_intensity(
        #                     log_destination_attraction,
        #                     theta_scaled_and_expanded_prev,
        #                     total_flow=np.sum(table.ravel())
        #                 )
        # theta_scaled_and_expanded_new = np.concatenate([theta_new,np.array([self.sim.delta,self.sim.gamma,self.sim.kappa,self.sim.epsilon])])
        # theta_scaled_and_expanded_new[1] *= self.sim.bmax
        # log_intensities_new = self.sim.log_intensity(
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
                    log_intensity_new, \
                    negative_log_table_likelihood_new, \
                    sign_new
        else:
            self.logger.debug("Rejected")
            return theta_prev, \
                    0, \
                    V, \
                    gradV, \
                    log_z_inverse, \
                    log_intensity, \
                    negative_log_table_likelihood, \
                    sign
        # except:
        #     print("Exception raised in theta_given_table_gibbs_step")
        #     # self.logger.debug("Exception raised")
        #     return theta_prev, 0, V, gradV, log_z_inverse, negative_log_table_likelihood, negative_gradient_log_table_likelihood, sign


    def log_destination_attraction_gibbs_step(self,theta:Union[list,np.array,np.ndarray],log_destination_attraction_prev:Union[list,np.array,np.ndarray],values:list):

        self.logger.debug('Log destination attraction Gibbs step')

        # Unpack values
        V, \
        gradV = values

        # Multiply beta by total cost
        theta_scaled_and_expanded = np.concatenate([theta,np.array([self.sim.delta,self.sim.gamma,self.sim.kappa,self.sim.epsilon])])
        theta_scaled_and_expanded[1] *= self.sim.bmax

        ''' Log destination demand update '''

        # Initialize leapfrog integrator for HMC proposal
        momentum = np.random.normal(0., 1., self.sim.dims[1])
        # Compute log(\pi(y|x))
        negative_log_data_likelihood, negative_gradient_log_data_likelihood =  self.sim.negative_destination_attraction_log_likelihood_and_gradient(
                                                                    log_destination_attraction_prev,
                                                                    1./self.sim.noise_var
                                                            )
        # Compute log initial potential energy and its derivarive weighted by the likelihood function \pi(y|x)
        # \log(\exp(-\gamma)V_{\theta}(xx)) + \log(\pi(y|x)) + \log(p(T|x,\theta))
        # V is equal to \gamma*V_{\theta}(xx) + 1/(2*s^2)*(xx-xx_data)^2 +
        W, gradW = V + negative_log_data_likelihood, gradV + negative_gradient_log_data_likelihood
        # Initial total log Hamiltonian energy (kinetic + potential)
        H = 0.5*np.dot(momentum, momentum) + W

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
            negative_log_data_likelihood_new, negative_gradient_log_data_likelihood_new = self.sim.negative_destination_attraction_log_likelihood_and_gradient(
                                                                                log_destination_attraction_new,
                                                                                1./self.sim.noise_var
                                                                        )
            # Compute updated log potential function
            V_new, gradV_new = self.sim.sde_potential_and_gradient(
                                        log_destination_attraction_new,
                                        theta_scaled_and_expanded
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
        H_new = 0.5*np.dot(momentum_new, momentum_new) + W_new

        # print("Proposing " + str(log_destination_attraction_new) + ' with ' + str(H_new))
        # print(str(log_destination_attraction_prev) + ' vs ' + str(H))
        # print(("Difference log target " + str(H-H_new)))
        # print(np.exp(log_destination_attraction_new).sum())
        # print('H-H_new',H-H_new)
        # print('\n')

        # Accept/reject
        if np.log(np.random.uniform(0, 1)) < H - H_new:
            return log_destination_attraction_new, 1, V_new, gradV_new
        else:
            return log_destination_attraction_prev, 0, V, gradV


    def log_destination_attraction_given_table_gibbs_step(self,theta:Union[list,np.array,np.ndarray],log_destination_attraction_prev:Union[list,np.array,np.ndarray],table:Union[dict,None],values:list):

        self.logger.debug('Log destination attraction Gibbs step given table')

        # Unpack values
        V, \
        gradV, \
        log_intensity, \
        negative_log_table_likelihood = values

        # Multiply beta by total cost
        theta_scaled_and_expanded = np.concatenate([
                                        theta,
                                        np.array([
                                            self.sim.delta,
                                            self.sim.gamma,
                                            self.sim.kappa,
                                            self.sim.epsilon
                                        ])
                                    ])
        theta_scaled_and_expanded[1] *= self.sim.bmax

        ''' Log destination demand update '''

        # print('total intensity',np.sum(np.exp(log_intensity)))
        # negative_log_table_likelihood_copy = self.negative_table_log_likelihood(log_intensity,table)
        # print('negative_log_table_likelihood',negative_log_table_likelihood)
        # print('negative_log_table_likelihood copy',negative_log_table_likelihood_copy)
        
        # Initialize leapfrog integrator for HMC proposal
        momentum = np.random.normal(0., 1., self.sim.dims[1])
        # Compute -log(\pi(y|x))
        negative_log_data_likelihood, \
        negative_gradient_log_data_likelihood = self.sim.negative_destination_attraction_log_likelihood_and_gradient(
                                                        log_destination_attraction_prev,
                                                        1./self.sim.noise_var
                                                )
        # Compute gradient of lambda
        intensity_gradient = self.sim.intensity_gradient(
            theta_scaled_and_expanded,
            log_intensity
        )

        # Initialise gradient of log table likelihood
        negative_gradient_log_table_likelihood = self.negative_table_log_likelihood_gradient(
                                                        log_intensity,
                                                        table,
                                                        intensity_gradient=intensity_gradient
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
            negative_gradient_log_data_likelihood_new = self.sim.negative_destination_attraction_log_likelihood_and_gradient(
                                                                log_destination_attraction_new,
                                                                1./self.sim.noise_var
                                                        )
            # Compute updated log potential function
            V_new, gradV_new = self.sim.sde_potential_and_gradient(
                                            log_destination_attraction_new,
                                            theta_scaled_and_expanded
                                )

            # Compute new intensity
            log_intensity_new,intensity_gradient_new = self.sim.log_intensity_and_gradient(
                    xx=log_destination_attraction_new,
                    theta=theta_scaled_and_expanded,
                    total_flow=np.sum(table)
            )

            # print('xx',np.exp(log_destination_attraction_new).sum())
            # print('intensity')
            # print(log_intensity)
            # print('\n ')
            
            # Compute negative table likelihood
            negative_log_table_likelihood_new, \
            negative_gradient_log_table_likelihood_new = self.negative_table_log_likelihood_and_gradient(
                                                                log_intensity_new,
                                                                table,
                                                                intensity_gradient=intensity_gradient_new
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
            H_new = 0.5*np.dot(momentum_new, momentum_new) + W_new
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
        H_new = 0.5*np.dot(momentum_new, momentum_new) + W_new
        
        # Make sure acceptance ratio is finite
        if not np.isfinite(H_new) or not np.isfinite(H):
            raise Exception('Nulls appeared in log_destination_attraction_given_table_gibbs_step')

        # Compute variances
        # print('Data variance',np.repeat(self.sim.noise_var,self.sim.dims[1]))
        # table_rowsums = table.sum(axis=1).reshape((self.sim.dims[0],1))
        # intensity_rowsums = np.array([logsumexp(log_intensity[i,:]) for i in range(self.sim.dims[0])]).reshape((self.sim.dims[0],1))
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
                    log_intensity_new,\
                    negative_log_table_likelihood_new
        else:
            return log_destination_attraction_prev,\
                    0,\
                    V,\
                    gradV,\
                    log_intensity,\
                    negative_log_table_likelihood