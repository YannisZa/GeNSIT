""" The Harris and Wilson model numerical solver 
    Code extended from https://github.com/ThGaskin/NeuralABM
"""
import sys
import torch
import logging

from multiresticodm.config import Config
# from multiresticodm.utils import safe_delete,str_in_list
from multiresticodm.global_variables import PARAMETER_DEFAULTS
from multiresticodm.spatial_interaction_model import SpatialInteraction2D
from multiresticodm.utils import setup_logger, sigma_to_noise_regime

""" Load a dataset or generate synthetic data on which to train the neural net """


class HarrisWilson:
    def __init__(
        self,
        *,
        sim: SpatialInteraction2D = None,
        config: Config = None,
        dt: float = 0.001,
        true_parameters: dict = None,
        device: str = None,
        **kwargs
    ):
        """The Harris and Wilson model of economic activity.

        :param sim: the Spatial interaction model with all necessary data
        :param params_to_learn: (optional) the names of free parameters to learn
        :param dt: (optional) the time differential to use for the solver
        :param true_parameters: (optional) a dictionary of the true parameters
        :param device: the training device to use
        """

        # Setup logger
        self.level = config.level if hasattr(config,'level') else kwargs.get('level','INFO')
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            level=self.level,
            log_to_file=kwargs.get('log_to_file',True),
            log_to_console=kwargs.get('log_to_console',True),
        )

        # Store SIM and its config separately
        self.sim = sim

        if config is not None:
            self.config = config

        # Device name
        self.device = device

        # Model parameters
        self.aux_param_names = ['noise_var','epsilon']
        self.main_param_names = ['alpha','beta','kappa','sigma','delta']
        self.true_parameters = true_parameters

        try:
            assert set(self.config.settings['inputs']['to_learn']).issubset(set(self.main_param_names))
        except:
            self.logger.error(f"Some parameters in {','.join(self.config.settings['inputs']['to_learn'])} cannot be learned.")
            self.logger.error(f"Acceptable parameters are {','.join(self.main_param_names)}.")
            raise Exception('Cannot instantiate Harris Wilson Model.')
        
        params_to_learn = {}
        if true_parameters is not None:
            idx = 0
            for param in self.config.settings['inputs']['to_learn']:
                if param not in true_parameters.keys():
                    params_to_learn[param] = idx
                    idx += 1
        self.params_to_learn = params_to_learn

        # Auxiliary hyperparameters
        for param in PARAMETER_DEFAULTS.keys():
            setattr(
                self,
                param,
                torch.tensor(true_parameters.get(param,PARAMETER_DEFAULTS[param])).float().to(device)
            )
            if hasattr(self,'config'):
                self.config.settings['harris_wilson_model']['parameters'][param] = true_parameters.get(param,PARAMETER_DEFAULTS[param])
        self.dt = torch.tensor(dt).float().to(device)

        # Error noise on log destination attraction
        noise_percentage = true_parameters.get('noise_percentage',PARAMETER_DEFAULTS['noise_percentage'])
        self.noise_var = torch.pow(((noise_percentage/torch.tensor(100).float())*torch.log(self.sim.dims[1])),2)
        if hasattr(self,'config'):
            self.config.settings['harris_wilson_model']['noise_percentage'] = noise_percentage

        # Update noise regime
        if 'sigma' in self.params_to_learn:
            self.noise_regime = 'variable'
        else:
            # Get true sigma
            self.noise_regime = sigma_to_noise_regime(self.true_parameters['sigma'])
        self.config.settings['noise_regime'] = self.noise_regime

    
    def elicit_delta_and_kappa(self):
        # Update delta and kappa 
        delta = self.true_parameters.get('delta',None)
        kappa = self.true_parameters.get('kappa',None)
        if delta is None and kappa is None:
            self.kappa = (torch.sum(self.sim.origin_demand)).float()/(torch.sum(torch.exp(self.log_destination_attraction))-torch.min(torch.exp(self.log_destination_attraction)).float()*self.sim.dims[1])
            self.delta = torch.min(torch.exp(self.log_destination_attraction)).float() * self.kappa
        elif self.kappa is None and self.delta is not None:
            self.kappa = (torch.sum(self.sim.origin_demand) + self.delta*self.sim.dims[1])/torch.sum(torch.exp(self.sim.log_destination_attraction))
        elif self.kappa is not None and self.delta is None:
            self.delta = self.kappa * torch.min(torch.exp(self.sim.log_destination_attraction)).float()
        for param in ['delta','kappa']:
            if hasattr(self,'config'):
                self.config['harris_wilson_model']['parameters'][param] = getattr(self,param)
    


    # ... SDE Model variant stationary distribution

    # def negative_destination_attraction_log_likelihood_and_gradient(self,xx,s2_inv:float=100.):
    #     """ Log of potential function of the likelihood =  log(pi(y|x)).

    #     Parameters
    #     ----------
    #     xx : np.array
    #         Log destination sizes
    #     s2_inv : float
    #         Inverse sigma^2 where sigma is the noise of the observation model.

    #     Returns
    #     -------
    #     float,np.array
    #         log Likelihood value and log of its gradient with respect to xx

    #     """
    #     log_likelihood,\
    #     log_likelihood_gradient = self.destination_attraction_log_likelihood_and_jacobian(
    #         xx,
    #         self.log_destination_attraction,
    #         s2_inv
    #     )
    #     return -log_likelihood, -log_likelihood_gradient

    # def sde_potential(self,xx,theta):
    #     """ Computes the potential function of singly constrained model

    #     Parameters
    #     ----------
    #     xx : np.array[Mx1]
    #         Log destination attraction
    #     theta : np.array[]
    #         List of parameters (alpha,beta,delta,kappa,epsilon)

    #     Returns
    #     -------
    #     float
    #         Potential function value at xx.
    #     np.array[Mx1]
    #         Potential function Jacobian at xx.

    #     """
    #     return self.sde_pot(xx,theta,self.origin_demand,self.cost_matrix)

    
    # def sde_potential_gradient(self,xx,theta):
    #     return self.sde_pot_jacobian(xx,theta,self.origin_demand,self.cost_matrix)
    

    # def sde_potential_and_gradient(self,xx,theta):
    #     return self.sde_pot_and_jacobian(xx,theta,self.origin_demand,self.cost_matrix)


    # def sde_potential_hessian(self,xx,theta):
    #     """ Computes Hessian matrix of potential function for the singly constrained model.

    #     Parameters
    #     ----------
    #     xx : np.array
    #         Log destination attraction
    #     theta : np.array
    #         List of parameters (alpha,beta,delta,kappa,epsilon)

    #     Returns
    #     -------
    #     np.ndarray
    #         Hessian matrix of potential function

    #     """

    #     return self.sde_pot_hessian(xx,theta,self.origin_demand,self.cost_matrix)
        

    # def sde_potential_and_gradient_annealed_importance_sampling(self,xx):
    #     """ Potential function for annealed importance sampling (no flows model)
    #     -gamma*V_{theta}'(x)
    #     where V_{theta}'(x) is equal to the potential function in the limit of alpha -> 1, beta -> 0

    #     Parameters
    #     ----------
    #     xx : np.array
    #         Log destination sizes
    #     theta : np.array
    #         List of parameters

    #     Returns
    #     -------
    #     float, np.array
    #         AIS potential value and its gradient

    #     """
    #     return self.sde_ais_pot_and_jacobian(xx=xx,theta=np.array([self.delta,self.gamma,self.kappa]),J=self.dims[1])
    # def ode_log_stationary_points_update(self,xx,theta):

    #     # Get dimensions of cost matrix
    #     kappa = theta[4]
    #     delta = theta[2]

    #     # Compute lambdas
    #     log_intensity = self.log_intensity(xx,theta,1)
    #     # Get log stationary points
    #     log_stationary_points = np.log((np.sum(np.exp(log_intensity),axis=0) + delta)) - np.log(kappa)

    #     return log_stationary_points


    # def ode_log_stationary_equations(self,xx,theta):

    #     # Get dimensions of cost matrix
    #     kappa = theta[4]
    #     delta = theta[2]

    #     # Compute lambdas
    #     log_intensity = self.log_intensity(xx,theta,1)
    #     # Solve equation for equilibrium points
    #     return np.sum(np.exp(log_intensity),axis=0) - kappa*np.exp(xx) + delta

    # def ode_stationary_points_iterative_solver(self,xx,theta,convergence_threshold:float=1e-9):
    #     # Extract necessary data
    #     # Perform update for stationary points
    #     xx_new = self.ode_log_stationary_points_update(xx,theta)
    #     xxs = xx.reshape(1,np.shape(xx)[0])
    #     # Solve equation until successive solutions do not change signficantly (equilibrium point identified)
    #     while (np.absolute(xx_new-xx) > convergence_threshold).any():
    #         xx = xx_new
    #         xx_new = self.ode_log_stationary_points_update(xx,theta)
    #         xxs = np.append(xxs,xx_new.reshape(1,np.shape(xx_new)[0]),axis=0)

    #     return xxs


    # ... Model run functions ..........................................................................................
    
    def run_single(
        self,
        *,
        curr_destination_attractions,
        free_parameters=None,
        dt: float = None,
        requires_grad: bool = True
    ):

        """Runs the model for a single iteration.

        :param curr_destination_attractions: the current values which to take as initial data.
        :param free_parameters: the input parameters (to learn). Defaults to the model defaults.
        :param dt: (optional) the time differential to use. Defaults to the model default.
        :param requires_grad: whether the resulting values require differentiation
        :return: the updated values

        """

        # Parameters to learn
        alpha = (
            self.true_parameters["alpha"]
            if 'alpha' not in self.params_to_learn.keys()
            else free_parameters[self.params_to_learn["alpha"]]
        )
        beta = (
            self.true_parameters["beta"]
            if "beta" not in self.params_to_learn.keys()
            else free_parameters[self.params_to_learn["beta"]]
        )
        kappa = (
            self.true_parameters["kappa"]
            if "kappa" not in self.params_to_learn.keys()
            else free_parameters[self.params_to_learn["kappa"]]
        )
        delta = (
            self.true_parameters["delta"]
            if "delta" not in self.params_to_learn.keys()
            else free_parameters[self.params_to_learn["delta"]]
        )
        sigma = (
            self.true_parameters["sigma"]
            if "sigma" not in self.params_to_learn.keys()
            else free_parameters[self.params_to_learn["sigma"]]
        )

        # Training parameters
        dt = self.dt if dt is None else dt

        new_sizes = curr_destination_attractions.clone()
        new_sizes.requires_grad = requires_grad

        # Calculate the weight matrix C^beta
        C_beta = torch.pow(self.sim.cost_matrix, beta)

        # Calculate the exponential sizes W_j^alpha
        W_alpha = torch.pow(curr_destination_attractions, alpha)

        # Calculate the vector of demands
        demand = self.sim.intensity_demand(
            W_alpha,
            C_beta
        )

        # Update the current values
        new_sizes = (
            new_sizes
            + +torch.mul(
                curr_destination_attractions,
                self.epsilon * (demand - kappa * curr_destination_attractions + delta)
                + sigma
                * 1
                / torch.sqrt(torch.tensor(2, dtype=torch.float) * torch.pi * dt).to(
                    self.device
                )
                * torch.normal(0, 1, size=(self.sim.dims[1], 1)).to(self.device),
            )
            * dt
        )

        return new_sizes

    def run(
        self,
        *,
        init_destination_attraction,
        n_iterations: int,
        dt: float = None,
        requires_grad: bool = True,
        generate_time_series: bool = False
    ) -> torch.tensor:

        """Runs the model for n_iterations.

        :param init_destination_attraction: the initial destination zone size values
        :param n_iterations: the number of iteration steps.
        :param dt: (optional) the time differential to use. Defaults to the model default.
        :param requires_grad: (optional) whether the calculated values require differentiation
        :param generate_time_series: whether to generate a complete time series or only return the final value
        :return: the time series data

        """

        if not generate_time_series:
            sizes = init_destination_attraction.clone()
            for _ in range(n_iterations):
                sizes = self.run_single(
                    curr_destination_attractions=sizes,
                    free_parameters=free_parameters,
                    dt=dt,
                    requires_grad=requires_grad,
                )
                return torch.stack(sizes)

        else:
            sizes = [init_destination_attraction.clone()]
            for _ in range(n_iterations):
                sizes.append(
                    self.run_single(
                        curr_destination_attractions=sizes[-1],
                        free_parameters=free_parameters,
                        dt=dt,
                        requires_grad=requires_grad,
                    )
                )
            sizes = torch.stack(sizes)
            return torch.reshape(sizes, (sizes.shape[0], sizes.shape[1], 1))


# def destination_demand(self,W_alpha,weights):

#     # Calculate the normalisations sum_{k,m} W_m^alpha exp(-beta * c_km)
#     normalisation = torch.sum(
#         torch.mul(W_alpha, torch.transpose(weights, 0, 1))
#     )

#     # Calculate the vector of demands
#     return torch.mul(
#         W_alpha,
#         torch.reshape(
#             torch.sum(
#                 torch.div(torch.mul(torch.sum(self.or_sizes), weights), normalisation),
#                 dim=0,
#                 keepdim=True,
#             ),
#             (self.M, 1),
#         ),
#     )

    def __repr__(self):
        return f"HarrisWilson( {self.sim.sim_type}(SpatialInteraction2D) )"

    def __str__(self):

        return f"""
            {'x'.join([str(d.cpu().detach().numpy()) for d in self.sim.dims])} Harris Wilson model using {self.sim.sim_type} Constrained Spatial Interaction Model
            Learned parameters: {', '.join(self.params_to_learn.keys())}
            Epsilon: {self.epsilon}
            Kappa: {self.kappa}
            Delta: {self.delta}
            Sigma: {self.sigma}
        """