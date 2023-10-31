""" The Harris and Wilson model numerical solver 
    Code extended from https://github.com/ThGaskin/NeuralABM
"""
import sys
import torch

from torch import float32

from multiresticodm.config import Config
from multiresticodm.global_variables import PARAMETER_DEFAULTS,Dataset
from multiresticodm.spatial_interaction_model import SpatialInteraction2D
from multiresticodm.utils import set_seed, setup_logger, to_json_format

""" Load a dataset or generate synthetic data on which to train the neural net """


class HarrisWilson:
    def __init__(
        self,
        *,
        intensity_model: SpatialInteraction2D = None,
        config: Config = None,
        dt: float = 0.001,
        true_parameters: dict = None,
        device: str = None,
        **kwargs
    ):
        """The Harris and Wilson model of economic activity.

        :param sim: the Spatial interaction model with all necessary data
        :param dt: (optional) the time differential to use for the solver
        :param true_parameters: (optional) a dictionary of the true parameters
        :param device: the training device to use
        """

        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            console_level = level,
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevels(
            console_level = level
        )

        # Store SIM and its config separately
        self.intensity_model = intensity_model
        self.config = config
        self.name = 'HarrisWilson'

        # Device name
        self.device = self.config['inputs']['device']

        # Model parameters
        self.param_names = ['alpha','beta','kappa','sigma','delta','epsilon','bmax']
        self.true_parameters = {k:v for k,v in true_parameters.items() if v is not None}
        # Add parameters to learn if None is provided as true parameter
        for key,value in true_parameters.items():
            if value is None and key not in self.config.settings['inputs']['to_learn']:
                self.config.settings['inputs']['to_learn'].append(key)
        
        # Check that learnable parameters are in valid set
        try:
            assert set(self.config.settings['inputs']['to_learn']).issubset(set(self.param_names))
        except:
            self.logger.error(f"Some parameters in {','.join(self.config.settings['inputs']['to_learn'])} cannot be learned.")
            self.logger.error(f"Acceptable parameters are {','.join(self.param_names)}.")
            raise Exception('Cannot instantiate Harris Wilson Model.') 
        
        # Set parameters to learn based on
        # kwargs or parameter defaults
        self.params_to_learn = {}
        for i,param in enumerate(self.config.settings['inputs']['to_learn']):
            self.params_to_learn[param] = i

        # Fixed hyperparameters
        self.params = Dataset()
        for param,default in PARAMETER_DEFAULTS.items():
            if not param in list(self.params_to_learn.keys()):
                true_param = self.true_parameters.get(param,default)
                setattr(
                    self.params,
                    param,
                    torch.tensor(true_param).to(device=device,dtype=float32)
                )
                if self.config is not None:
                    self.config.settings['harris_wilson_model']['parameters'][param] = to_json_format(
                        self.true_parameters.get(param,default)
                    )
        # Update gamma for MCMC
        if hasattr(self.params,'sigma'):
            self.params.gamma = 2/(self.params.sigma)**2
        
        # Time discretisation step size
        self.dt = torch.tensor(dt).float().to(device)

        # Error noise on log destination attraction
        self.noise_percentage = torch.tensor(
            self.true_parameters.get('noise_percentage',PARAMETER_DEFAULTS['noise_percentage'])
            ).to(
                dtype=float32,
                device=self.device
            )
        self.noise_var = torch.pow(
            (
                self.noise_percentage / \
                torch.tensor(100).float()
            ) * \
            torch.log(torch.tensor(self.intensity_model.dims['destination']).float()),
            2
        ).to(
            dtype=float32,
            device=self.device
        )
        if hasattr(self,'config'):
            self.config.settings['harris_wilson_model']['parameters']['noise_var'] = to_json_format(self.noise_var)

        # Update noise regime
        self.noise_regime = self.config['harris_wilson_model']['parameters'].get('sigma',None)
        if 'sigma' in list(self.params_to_learn.keys()) or self.noise_regime is None:
            self.noise_regime = 'variable'
        elif isinstance(self.noise_regime,list):
            self.noise_regime = 'sweeped'
        elif self.noise_regime >= 0.1:
            self.noise_regime = 'high'
        else:
            self.noise_regime = 'low'
        self.config.settings['noise_regime'] = self.noise_regime


    def sde_potential(self,log_destination_attraction,**kwargs):

        # Update input kwargs if required
        kwargs['log_destination_attraction'] = log_destination_attraction
        updated_kwargs = self.intensity_model.get_input_kwargs(kwargs)
        updated_kwargs.update(**vars(self.params))
        
        required = (self.intensity_model.REQUIRED_OUTPUTS+self.intensity_model.REQUIRED_INPUTS+['gamma'])

        self.intensity_model.check_sample_availability(
            required,
            updated_kwargs
        )
        return self.intensity_model.sde_pot(**updated_kwargs)
    
    def sde_potential_and_gradient(self,log_destination_attraction,**kwargs):

        # Update input kwargs if required
        kwargs['log_destination_attraction'] = log_destination_attraction
        updated_kwargs = self.intensity_model.get_input_kwargs(kwargs)
        updated_kwargs.update(**vars(self.params))

        required = (self.intensity_model.REQUIRED_OUTPUTS+self.intensity_model.REQUIRED_INPUTS+['gamma'])

        self.intensity_model.check_sample_availability(
            required,
            updated_kwargs
        )
        potential,jacobian = self.intensity_model.sde_pot_and_jacobian(**updated_kwargs)
        return potential,jacobian[0]
        

    def sde_potential_jacobian(self,log_destination_attraction,**kwargs):

        # Update input kwargs if required
        kwargs['log_destination_attraction'] = log_destination_attraction
        updated_kwargs = self.intensity_model.get_input_kwargs(kwargs)
        updated_kwargs.update(**vars(self.params))

        required = (self.intensity_model.REQUIRED_OUTPUTS+self.intensity_model.REQUIRED_INPUTS+['gamma'])

        self.intensity_model.check_sample_availability(
            required,
            updated_kwargs
        )
        return self.intensity_model.sde_pot_jacobian(**updated_kwargs)[0]

    
    def sde_potential_hessian(self,log_destination_attraction,**kwargs):

        # Update input kwargs if required
        kwargs['log_destination_attraction'] = log_destination_attraction
        updated_kwargs = self.intensity_model.get_input_kwargs(kwargs)
        updated_kwargs.update(**vars(self.params))

        required = (self.intensity_model.REQUIRED_OUTPUTS+self.intensity_model.REQUIRED_INPUTS+['gamma'])

        self.intensity_model.check_sample_availability(
            required,
            updated_kwargs
        )
        return self.intensity_model.sde_pot_hessian(**updated_kwargs)[0][0]
    
    def negative_destination_attraction_log_likelihood_and_gradient(self,xx_data,xx,s2_inv:float=100.):
        """ Log of potential function of the likelihood =  log(pi(y|x)).

        Parameters
        ----------
        xx : torch.tensor
            Log destination sizes
        s2_inv : float
            Inverse sigma^2 where sigma is the noise of the observation model.

        Returns
        -------
        float,torch.tensor
            log Likelihood value and log of its gradient with respect to xx

        """
        # Compute difference
        diff = (xx - xx_data).flatten()
        # Compute gradient of log likelihood
        negative_log_likelihood_gradient = -s2_inv*diff
        # Compute log likelihood (without constant factor)
        negative_log_likelihood = -0.5*s2_inv*(diff.dot(diff))
        return -negative_log_likelihood, -negative_log_likelihood_gradient
    
    def sde_ais_potential_and_jacobian(self,**kwargs):
        
        Ndestinations = self.intensity_model.dims['destination']
        delta = self.params.delta
        kappa = self.params.kappa
        xx = kwargs['log_destination_attraction']
        gamma = kwargs['gamma'] if kwargs['gamma'] is not None else self.params.gamma
        # Note that lim_{beta->0, alpha->0} gamma*V_{theta}(x) = gamma*kappa*\sum_{j=1}^J \exp(x_j) - gamma*(delta+1/J) * \sum_{j=1}^J x_j
        gamma_kk_exp_xx = gamma*kappa*torch.exp(xx)
        # Function proportional to the potential function in the limit of alpha -> 0, beta -> 0
        V = -gamma*(delta+1./Ndestinations)*xx.sum() + gamma_kk_exp_xx.sum()
        # Gradient of function above
        gradV = -gamma*(delta+1./Ndestinations)*torch.ones(Ndestinations,dtype=float32,device=self.device) + gamma_kk_exp_xx

        return V, gradV

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
            self.params.alpha
            if 'alpha' not in list(self.params_to_learn.keys())
            else free_parameters[self.params_to_learn["alpha"]]
        )
        beta = (
            self.params.beta
            if "beta" not in list(self.params_to_learn.keys())
            else free_parameters[self.params_to_learn["beta"]]
        )
        kappa = (
            self.params.kappa
            if "kappa" not in list(self.params_to_learn.keys())
            else free_parameters[self.params_to_learn["kappa"]]
        )
        delta = (
            self.params.delta
            if "delta" not in list(self.params_to_learn.keys())
            else free_parameters[self.params_to_learn["delta"]]
        )
        epsilon = (
            self.params.epsilon
            if "epsilon" not in list(self.params_to_learn.keys())
            else free_parameters[self.params_to_learn["epsilon"]]
        )
        sigma = (
            self.params.sigma
            if "sigma" not in list(self.params_to_learn.keys())
            else free_parameters[self.params_to_learn["sigma"]]
        )

        # Training parameters
        dt = self.dt if dt is None else dt

        new_sizes = curr_destination_attractions.clone()
        new_sizes.requires_grad = requires_grad


        # Calculate the vector of demands
        demand = self.intensity_model.intensity_demand(
            alpha = alpha,
            beta = beta,
            log_destination_attraction = torch.log(new_sizes),
            grand_total = torch.tensor(1.)
        )
        # Update the current values
        new_sizes = (
            new_sizes + \
            +torch.mul(
                curr_destination_attractions,
                epsilon * (demand - kappa * curr_destination_attractions + delta)
                + sigma
                * 1
                / torch.sqrt(torch.tensor(2, dtype=torch.float) * torch.pi * dt).to(
                    self.device
                )
                * torch.normal(0, 1, size=(1, self.intensity_model.dims['destination'])).to(self.device),
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
        free_parameters=None,
        requires_grad: bool = True,
        generate_time_series: bool = False,
        seed: int = None,
        semaphore = None,
        samples = None,
        pbar = None
    ) -> torch.tensor:

        """Runs the model for n_iterations.

        :param init_destination_attraction: the initial destination zone size values
        :param n_iterations: the number of iteration steps.
        :param dt: (optional) the time differential to use. Defaults to the model default.
        :param requires_grad: (optional) whether the calculated values require differentiation
        :param generate_time_series: whether to generate a complete time series or only return the final value
        :return: the time series data

        """
        if semaphore is not None:
            semaphore.acquire()
        if seed is not None:
            set_seed(seed)
        
        # Training parameters
        dt = self.dt if dt is None else dt

        if not generate_time_series:
            sizes = init_destination_attraction.clone()
            for _ in range(n_iterations):
                sizes = self.run_single(
                    curr_destination_attractions=sizes,
                    free_parameters=free_parameters,
                    dt=dt,
                    requires_grad=requires_grad,
                )
                sizes = torch.stack(sizes)

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
            sizes = torch.squeeze(torch.stack(tuple(sizes)))

        if semaphore is not None:
            semaphore.release()
        if samples is not None:
            samples[seed] = sizes
        if pbar is not None:
            pbar.update(1)
        return sizes


    def __repr__(self):
        return f"HarrisWilson( {self.intensity_model.sim_type}(SpatialInteraction2D) )"

    def __str__(self):

        return f"""
            {'x'.join([str(d.cpu().detach().numpy()) for d in self.intensity_model.dims])} Harris Wilson model using {self.intensity_model.sim_type} Constrained Spatial Interaction Model
            Learned parameters: {', '.join(self.params_to_learn.keys())}
            Epsilon: {self.main_params.epsilon}
            Kappa: {self.main_params.kappa}
            Delta: {self.main_params.delta}
        """