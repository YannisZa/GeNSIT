import os
import sys
import logging
import numpy as np

from pathlib import Path

from ticodm.config import Config
from ticodm.sim_models import ProductionConstrained,TotalConstrained
from ticodm.probability_utils import log_odds_ratio_wrt_intensity
from ticodm.utils import str_in_list,write_txt,extract_config_parameters,numba_set_seed,makedir


def instantiate_sim(config:Config,disable_logger:bool=False): #-> Union[ContingencyTable,None]:
    
    sim_type = ''
    if isinstance(config,Config) or hasattr(config,'settings'):
        sim_type = config.settings['inputs']['spatial_interaction_model'].get('sim_type',None)
        if sim_type is not None and hasattr(sys.modules[__name__], sim_type):
            sim_type += 'SIM'
            return getattr(sys.modules[__name__], sim_type)(config,disable_logger)
    elif isinstance(config,dict):
        sim_type = config.get('sim_type',None)
        if sim_type is not None and hasattr(sys.modules[__name__], sim_type):
            sim_type += 'SIM'
            return getattr(sys.modules[__name__], sim_type)(config,disable_logger)
    
    raise ValueError(f"Input class '{sim_type}' not found")

class SpatialInteraction():
    def __init__(self,config:Config,disable_logger:bool=False):
        '''  Constructor '''

        # Import logger
        self.logger = logging.getLogger(__name__)
        self.logger.disabled = disable_logger
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)
        
        if isinstance(config,Config) or hasattr(config,'settings'):
            # Set configuration file
            self.config = extract_config_parameters(
                        config,
                        {"seed":"",
                        "inputs":{"spatial_interaction_model":"","seed":"","n_workers":"","n_threads":"","generate_data":"","dataset":"","contingency_table":""},
                        "mcmc":{"table_inference":"","N":"","spatial_interaction_model":""},
                        "outputs":""
                        }
            )
        elif not isinstance(config,dict):
            raise Exception(f"Config type {type(config)} not recognized")


    def export(self,dirpath:str='./synthetic_dummy',overwrite:bool=False) -> None:
        # Make directory if it does not exist
        makedir(dirpath)

        if hasattr(self,'origin_demand') and self.origin_demand is not None:
            # Get filepath experiment filepath
            filepath = os.path.join(dirpath,'origin_demand.txt')

            # Write experiment summaries to file
            if (not os.path.exists(filepath)) or overwrite:
                write_txt((self.origin_demand).astype('int32'),filepath)

        if hasattr(self,'log_true_destination_attraction') and self.log_true_destination_attraction is not None:
            # Get filepath experiment filepath
            filepath = os.path.join(dirpath,'true_log_destination_attraction.txt')

            # Write experiment summaries to file
            if (not os.path.exists(filepath)) or overwrite:
                write_txt(self.log_true_destination_attraction,filepath)

        if hasattr(self,'log_destination_attraction') and self.log_destination_attraction is not None:
            # Get filepath experiment filepath
            filepath = os.path.join(dirpath,'log_destination_attraction.txt')

            # Write experiment summaries to file
            if (not os.path.exists(filepath)) or overwrite:
                write_txt(self.log_destination_attraction,filepath)

        if hasattr(self,'alpha_true') and self.alpha_true is not None \
            and hasattr(self,'beta_true') and self.beta_true is not None:
            # Get filepath experiment filepath
            filepath = os.path.join(dirpath,'true_theta.txt')

            # Write experiment summaries to file
            if (not os.path.exists(filepath)) or overwrite:
                write_txt(np.array([self.alpha_true,self.beta_true,self.delta,self.gamma,self.kappa,self.epsilon,self.bmax]),filepath)

        if hasattr(self,'destination_demand') and self.normalised_destination_demand is not None:
            # Get filepath experiment filepath
            filepath = os.path.join(dirpath,'destination_demand.txt')

            # Write experiment summaries to file
            if (not os.path.exists(filepath)) or overwrite:
                write_txt(self.normalised_destination_demand,filepath)

        if hasattr(self,'cost_matrix') and self.cost_matrix is not None:
            # Get filepath experiment filepath
            filepath = os.path.join(dirpath,'cost_matrix.txt')

            # Write experiment summaries to file
            if (not os.path.exists(filepath)) or overwrite:
                write_txt(self.cost_matrix,filepath)

        self.logger.info(f'Spatial interaction model data successfully exported to {dirpath}')


class SpatialInteraction2D(SpatialInteraction):
    def __init__(self,config:Config,disable_logger:bool=False):
        '''  Constructor '''
        # Set configuration file
        super().__init__(config,disable_logger)
        # Dimensions
        self.dims = np.asarray([0,0])

        # Update name of SIM
        self.sim_name = 'SpatialInteraction2D'
        
        # Store parameter names
        self.parameter_names = ['alpha','beta','delta','gamma','kappa','epsilon']

        if isinstance(config,Config) or hasattr(config,'settings'):
        
            # Store parameters
            self.bmax = config.settings['inputs']['spatial_interaction_model']['beta_max']
            self.epsilon = float(config.settings['inputs']['spatial_interaction_model']['epsilon'])
            if str_in_list('delta',self.config.settings['inputs']['spatial_interaction_model'].keys()) and \
                self.config.settings['inputs']['spatial_interaction_model']['delta'] > 0:
                self.delta = float(config.settings['inputs']['spatial_interaction_model']['delta'])
            else:
                self.delta = None
            self.kappa = None
            if str_in_list('kappa',self.config.settings['inputs']['spatial_interaction_model'].keys()) and \
                config.settings['inputs']['spatial_interaction_model']['kappa'] > 0:
                self.kappa = float(config.settings['inputs']['spatial_interaction_model']['kappa'])
            else:
                self.kappa = None
            self.gamma = float(config.settings['inputs']['spatial_interaction_model']['gamma'])
            # Determine noise regime
            if self.gamma >= 10000:
                self.noise_regime = 'low'
            else:
                self.noise_regime = 'high'
            
            # Import data
            self.import_data()
        
        elif isinstance(config,dict):
            # Config
            self.config = config
            # Attributes
            attrs = {
                'sim_type':None,
                'log_destination_attraction':None,
                'origin_demand':None,
                'cost_matrix':None,
                'bmax':1,
                'delta':None,
                'kappa':None,
                'epsilon':1,
                'gamma':10000,
                'noise_percentage': 3.0,
            }
            # Parse config parameters
            for attr,attr_default in attrs.items():
                if not hasattr(self,attr):
                    setattr(self, attr, config.get(attr,attr_default))
            # Check that none of them are null
            for attr in attrs.keys():
                if (getattr(self, attr) is None) and (not attr in ['delta','kappa']):
                    raise Exception(f"{attr} is None")
            try:
                assert np.shape(self.origin_demand.flatten())[0] == np.shape(self.cost_matrix)[0]
            except Exception:
                raise ValueError(f"Inconsistent number of origins {np.shape(self.origin_demand.flatten())[0]} vs {np.shape(self.cost_matrix)[0]}")
            try:
                assert np.shape(self.log_destination_attraction.flatten())[0] == np.shape(self.cost_matrix)[1]
            except Exception:
                raise ValueError(f"Inconsistent number of destinations {np.shape(self.log_destination_attraction.flatten())[0]} vs {np.shape(self.cost_matrix)[1]}")
            # pass dimensions
            self.dims = np.asarray(np.shape(self.cost_matrix))
            # Set ground truth knowledge to false
            self.ground_truth_known = False

            # Delta and kappa 
            if self.delta is None and self.kappa is None:
                self.kappa = (np.sum(self.origin_demand))/(np.sum(np.exp(self.log_destination_attraction))-np.min(np.exp(self.log_destination_attraction))*self.dims[1])
                self.delta = np.min(np.exp(self.log_destination_attraction)) * self.kappa
            elif self.kappa is None and self.delta is not None:
                self.kappa = (np.sum(self.origin_demand) + self.delta*self.dims[1])/np.sum(np.exp(self.log_destination_attraction))
            elif self.kappa is not None and self.delta is None:
                self.delta = self.kappa * np.min(np.exp(self.log_destination_attraction))
            
            # Determine noise regime
            if self.gamma >= 10000:
                self.noise_regime = 'low'
            else:
                self.noise_regime = 'high'

        else:
            raise ValueError(f"Config type {type(config)} not recognized")
        
    def inherit_numba_functions(self,sim_model):
        # List of attributes to be inherited
        attributes = [
            "log_flow_matrix",
            "log_flow_matrix_vectorised",
            "flow_matrix_jacobian",
            "sde_pot",
            "sde_pot_jacobian",
            "sde_pot_hessian",
            "sde_pot_and_jacobian",
            "sde_ais_pot_and_jacobian",
            "annealed_importance_sampling_log_z",
            "annealed_importance_sampling_log_z_parallel",
            "destination_attraction_log_likelihood_and_jacobian",
        ]
        for attr in attributes:
            # inherit attribute
            if hasattr(sim_model,attr):
                setattr(self, attr, getattr(sim_model,attr))

    def __str__(self):
        # return (f"""{'x'.join([str(d) for d in self.dims])} Production Constrained Spatial Interaction Model""")
        if isinstance(self.config,dict):
            return f"""
                {'x'.join([str(d) for d in self.dims])} {self.sim_type} Spatial Interaction Model
                Destination attraction sum: {np.exp(self.log_destination_attraction).sum()}
                Cost matrix sum: {np.sum(self.cost_matrix.ravel())}
                Origin demand sum: {np.sum(self.origin_demand)}
                Delta: {self.delta}
                Kappa: {self.kappa}
                Epsilon: {self.epsilon}
                Gamma: {self.gamma}
                Beta scaling: {self.bmax}
            """
        else:
            return f"""
                {'x'.join([str(d) for d in self.dims])} {self.sim_type} Constrained Spatial Interaction Model
                Dataset: {Path(self.config.settings['inputs']['dataset']).stem}
                Cost matrix: {Path(self.config.settings['inputs']['spatial_interaction_model']['import']['cost_matrix']).stem}
                Destination attraction sum: {np.exp(self.log_destination_attraction).sum()}
                Cost matrix sum: {np.sum(self.cost_matrix.ravel())}
                Origin demand sum: {np.sum(self.origin_demand)}
                Delta: {self.delta}
                Kappa: {self.kappa}
                Epsilon: {self.epsilon}
                Gamma: {self.gamma}
                Beta scaling: {self.bmax}
            """

    def import_data(self):
        """ Stores important data for training and validation to global variables. """

        if not str_in_list('dataset',self.config.settings['inputs']):
            raise Exception('Input dataset not provided. SIM cannot be loaded.')

        if str_in_list('origin_demand',self.config.settings['inputs']['spatial_interaction_model']['import'].keys()):
            origin_demand_filepath = os.path.join(
                self.config.settings['inputs']['dataset'],
                self.config.settings['inputs']['spatial_interaction_model']['import']['origin_demand']
            )
            if os.path.isfile(origin_demand_filepath):
                # Import rowsums
                origin_demand = np.loadtxt(origin_demand_filepath,dtype='float32')
                # Store size of rows
                self.dims[0] = len(origin_demand)
                # Check to see see that they are all positive
                if (origin_demand <= 0).any():
                    raise Exception(f'Origin demand {origin_demand} are not strictly positive')
            else:
                raise Exception(f"Origin demand file {origin_demand_filepath} not found")
        else:
            raise Exception(f"Origin demand filepath not provided")

        if len(self.config.settings['inputs']['spatial_interaction_model']['import'].get('log_destination_attraction',[])) >= 0:
            log_destination_attraction_filepath = os.path.join(self.config.settings['inputs']['dataset'],self.config.settings['inputs']['spatial_interaction_model']['import']['log_destination_attraction'])
            if os.path.isfile(log_destination_attraction_filepath):
                # Import log destination attractions
                self.log_destination_attraction = np.loadtxt(log_destination_attraction_filepath,dtype='float32')
                # Store size of rows
                self.dims[1] = len(self.log_destination_attraction)
                # Check to see see that they are all positive
                # if (self.log_destination_attraction <= 0).any():
                    # raise Exception(f'Log destination attraction {self.log_destination_attraction} are not strictly positive')
            else:
                raise Exception(f"Log destination attraction file {log_destination_attraction_filepath} not found")
        else:
            raise Exception(f"Log destination attraction filepath not provided")

        if str_in_list('cost_matrix',self.config.settings['inputs']['spatial_interaction_model']['import'].keys()):
            cost_matrix_filepath = os.path.join(self.config.settings['inputs']['dataset'],self.config.settings['inputs']['spatial_interaction_model']['import']['cost_matrix'])
            if os.path.isfile(cost_matrix_filepath):
                # Import rowsums
                cost_matrix = np.loadtxt(os.path.join(self.config.settings['inputs']['dataset'],self.config.settings['inputs']['spatial_interaction_model']['import']['cost_matrix']),dtype='float32')
                # Make sure it has the right dimension
                try:
                    assert np.shape(cost_matrix) == tuple(self.dims)
                except:
                    self.dims = np.asarray(np.shape(cost_matrix))
                    raise Exception(f"Cost matrix does not have the required dimension {tuple(self.dims)} but has {self.dims}")
            else:
                raise Exception(f"Cost matrix file {cost_matrix_filepath} not found")

            # Check to see see that they are all positive
            if (cost_matrix < 0).any():
                raise Exception(f'Cost matrix {cost_matrix} is not non-negative.')
        else:
            raise Exception(f"Cost matrix filepath not provided")

        # Reshape cost matrix if necessary
        if self.dims[0] == 1:
            cost_matrix = np.reshape(cost_matrix[:,np.newaxis],tuple(self.dims))
        if self.dims[1] == 1:
            cost_matrix = np.reshape(cost_matrix[np.newaxis,:],tuple(self.dims))

        # Update noise level
        if str_in_list('noise_percentage',self.config.settings['inputs']['spatial_interaction_model'].keys()):
            self.noise_var = ((self.config.settings['inputs']['spatial_interaction_model']['noise_percentage']/100)*np.log(self.dims[1]))**2
        else:
            self.noise_var = (0.03*np.log(self.dims[1]))**2
            self.config.settings['inputs']['spatial_interaction_model']['noise_percentage'] = 3

        # Normalise origin demand
        self.origin_demand = origin_demand

        # Normalise cost matrix
        self.cost_matrix = cost_matrix

        # Determine if true data exists
        if (hasattr(self,'alpha_true') and self.alpha_true is not None) and (hasattr(self,'beta_true') and self.beta_true is not None):
            self.ground_truth_known = True
        else:
            self.ground_truth_known = False

        # Store additional sim-specific parameters
        # Update delta and kappa
        if self.delta is None and self.kappa is None:
            self.kappa = (np.sum(self.origin_demand))/(np.sum(np.exp(self.log_destination_attraction))-np.min(np.exp(self.log_destination_attraction))*self.dims[1])
            self.delta = np.min(np.exp(self.log_destination_attraction)) * self.kappa
            # Update config
            self.config.settings['inputs']['spatial_interaction_model']['delta'] = self.delta
            self.config.settings['inputs']['spatial_interaction_model']['kappa'] = self.kappa
        elif self.kappa is None and self.delta is not None:
            self.kappa = (np.sum(self.origin_demand) + self.delta*self.dims[1])/np.sum(np.exp(self.log_destination_attraction))
            # Update config
            self.config.settings['inputs']['spatial_interaction_model']['kappa'] = self.kappa
        elif self.kappa is not None and self.delta is None:
            self.delta = self.kappa * np.min(np.exp(self.log_destination_attraction))
            # Update config
            self.config.settings['inputs']['spatial_interaction_model']['delta'] = self.delta


    def negative_destination_attraction_log_likelihood_and_gradient(self,xx,s2_inv:float=100.):
        """ Log of potential function of the likelihood =  log(pi(y|x)).

        Parameters
        ----------
        xx : np.array
            Log destination sizes
        s2_inv : float
            Inverse sigma^2 where sigma is the noise of the observation model.

        Returns
        -------
        float,np.array
            log Likelihood value and log of its gradient with respect to xx

        """
        log_likelihood,\
        log_likelihood_gradient = self.destination_attraction_log_likelihood_and_jacobian(
            xx,
            self.log_destination_attraction,
            s2_inv
        )
        return -log_likelihood, -log_likelihood_gradient

    def sde_potential(self,xx,theta):
        """ Computes the potential function of singly constrained model

        Parameters
        ----------
        xx : np.array[Mx1]
            Log destination attraction
        theta : np.array[]
            List of parameters (alpha,beta,delta,kappa,epsilon)

        Returns
        -------
        float
            Potential function value at xx.
        np.array[Mx1]
            Potential function Jacobian at xx.

        """
        return self.sde_pot(xx,theta,self.origin_demand,self.cost_matrix)

    
    def sde_potential_gradient(self,xx,theta):
        return self.sde_pot_jacobian(xx,theta,self.origin_demand,self.cost_matrix)
    

    def sde_potential_and_gradient(self,xx,theta):
        return self.sde_pot_and_jacobian(xx,theta,self.origin_demand,self.cost_matrix)


    def sde_potential_hessian(self,xx,theta):
        """ Computes Hessian matrix of potential function for the singly constrained model.

        Parameters
        ----------
        xx : np.array
            Log destination attraction
        theta : np.array
            List of parameters (alpha,beta,delta,kappa,epsilon)

        Returns
        -------
        np.ndarray
            Hessian matrix of potential function

        """

        return self.sde_pot_hessian(xx,theta,self.origin_demand,self.cost_matrix)
        

    def sde_potential_and_gradient_annealed_importance_sampling(self,xx):
        """ Potential function for annealed importance sampling (no flows model)
        -gamma*V_{theta}'(x)
        where V_{theta}'(x) is equal to the potential function in the limit of alpha -> 1, beta -> 0

        Parameters
        ----------
        xx : np.array
            Log destination sizes
        theta : np.array
            List of parameters

        Returns
        -------
        float, np.array
            AIS potential value and its gradient

        """
        return self.sde_ais_pot_and_jacobian(xx=xx,theta=np.array([self.delta,self.gamma,self.kappa]),J=self.dims[1])

    def log_intensity(self,xx:np.ndarray,theta:np.ndarray,total_flow:float):
        """ Reconstruct expected flow matrices (intensity function)

        Parameters
        ----------
        xx : np.array
            Log destination attraction.
        theta : np.array
            Fitted parameters.

        Returns
        -------
        np.array
            Expected flow matrix (non-integer).

        """
        return self.log_flow_matrix(
            xx,
            theta,
            self.origin_demand,
            self.cost_matrix,
            total_flow
        ).astype('float32')
    
    def intensity_gradient(self,theta:np.ndarray,log_intensity:np.ndarray):
        """ Reconstruct gradient of intensity with respect to xx

        Parameters
        ----------
        xx : np.array
            Log destination attraction.
        theta : np.array
            Fitted parameters.

        Returns
        -------
        np.array
            Expected flow matrix (non-integer).

        """
        return self.flow_matrix_jacobian(
            theta,
            log_intensity
        )
    
    def log_odds_ratio(self,log_intensity:np.ndarray):
        """ Reconstruct log odds ratio of intensity function

        Parameters
        ----------
        xx : np.array
            Log destination attraction.
        theta : np.array
            Fitted parameters.

        Returns
        -------
        np.array
            Expected flow matrix (non-integer).

        """
        return log_odds_ratio_wrt_intensity(
            log_intensity
        )
    

    def log_intensity_and_gradient(self,xx:np.ndarray,theta:np.ndarray,total_flow:float):
        
        # Compute log intensity
        log_intensity = self.log_intensity(xx,theta,total_flow)
        # Pack gradient
        intensity_gradient = self.intensity_gradient(theta,log_intensity)
        return log_intensity, intensity_gradient

    def ode_log_stationary_points_update(self,xx,theta):

        # Get dimensions of cost matrix
        kappa = theta[4]
        delta = theta[2]

        # Compute lambdas
        log_intensity = self.log_intensity(xx,theta,1)
        # Get log stationary points
        log_stationary_points = np.log((np.sum(np.exp(log_intensity),axis=0) + delta)) - np.log(kappa)

        return log_stationary_points


    def ode_log_stationary_equations(self,xx,theta):

        # Get dimensions of cost matrix
        kappa = theta[4]
        delta = theta[2]

        # Compute lambdas
        log_intensity = self.log_intensity(xx,theta,1)
        # Solve equation for equilibrium points
        return np.sum(np.exp(log_intensity),axis=0) - kappa*np.exp(xx) + delta

    def ode_stationary_points_iterative_solver(self,xx,theta,convergence_threshold:float=1e-9):
        # Extract necessary data
        # Perform update for stationary points
        xx_new = self.ode_log_stationary_points_update(xx,theta)
        xxs = xx.reshape(1,np.shape(xx)[0])
        # Solve equation until successive solutions do not change signficantly (equilibrium point identified)
        while (np.absolute(xx_new-xx) > convergence_threshold).any():
            xx = xx_new
            xx_new = self.ode_log_stationary_points_update(xx,theta)
            xxs = np.append(xxs,xx_new.reshape(1,np.shape(xx_new)[0]),axis=0)

        return xxs


    def sde_solver(self,t0,t1,N,x0,theta,break_early:bool=True,break_threhshold:float=1e-7):
        """
        Return the result of one full simulation.
        """
        # Find discretisation time step size
        dt = float(t1 - t0) / N
        # Create timesteps
        timesteps = np.arange(t0, t1 + dt, dt)
        # Find SDE noise std
        sigma = np.sqrt(2/theta[3])
        # Get dimension of solution
        J = x0.shape[0]
        # Initialise solution vector
        x = np.zeros((N + 1,J))
        # Add initial solution
        x[0] = x0
        # Initialise mean log solution
        mean_x = np.zeros((N+1,J))
        mean_x[0] = x0
        for i in range(1, timesteps.size):
            x[i] = x[i - 1] - self.sde_pot_and_gradient(x[i - 1],theta)[1] * (1/theta[3]) * dt + \
                        sigma * np.random.normal(loc=0.0, scale=np.sqrt(dt), size=J)
            # Update mean solution
            mean_x[i] = ((i)/(i+1)) * mean_x[i-1] + 1/(i+1) * x[i]
            # If mean has not changed significantly
            if np.all((mean_x[i]-mean_x[i-1]) <= break_threhshold) and break_early:
                return timesteps[:(i+1)],x[:(i+1)]

        return timesteps, x



class ProductionConstrainedSIM(SpatialInteraction2D):
    """ Object including flow (O/D) matrix inference routines of singly constrained SIM. """

    def __init__(self,config:Config,disable_logger:bool=False):
        '''  Constructor '''
        # Define type of spatial interaction model
        self.sim_type = "ProductionConstrained"
        # Set configuration file
        super().__init__(config,disable_logger)
        # Inherit numba functions
        self.inherit_numba_functions(ProductionConstrained)

        # self.logger.info(('Building ' + self.__str__()))

    def __repr__(self):
        return "ProductionConstrained(SpatialInteraction2D)"


class TotalConstrainedSIM(SpatialInteraction2D):
    """ Object including flow (O/D) matrix inference routines of SIM with only total constrained. """

    def __init__(self,config:Config,disable_logger:bool=False):
        '''  Constructor '''
        # Define type of spatial interaction model
        self.sim_type = "TotalConstrained"
        # Set configuration file
        super().__init__(config,disable_logger)
        # Inherit numba functions
        self.inherit_numba_functions(TotalConstrained)

        # self.logger.info(('Building ' + self.__str__()))

    def __repr__(self):
        return "TotalConstrained(SpatialInteraction2D)"
    