""" The Harris and Wilson model numerical solver 
    Code extended from https://github.com/ThGaskin/NeuralABM
"""
import sys
import torch
import optuna
import xarray as xr

from torch import float32

from gensit.config import Config
from gensit.utils.misc_utils import set_seed, setup_logger, to_json_format
from gensit.static.global_variables import PARAMETER_DEFAULTS,DATA_SCHEMA
from gensit.intensity_models.spatial_interaction_models import SpatialInteraction2D

""" Load a dataset or generate synthetic data on which to train the neural net """

MODEL_TYPE = 'HarrisWilson'
MODEL_PREFIX = ''


class HarrisWilson:
    def __init__(
        self,
        *,
        config: Config,
        trial:optuna.trial,
        intensity_model: SpatialInteraction2D = None,
        **kwargs
    ):
        """The Harris and Wilson model of economic activity.

        :param sim: the Spatial interaction model with all necessary data
        :param true_parameters: (optional) a dictionary of the true parameters
        """

        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            console_level = level,
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevels( console_level = level )

        # Store important quantities separately
        self.intensity_model = intensity_model
        self.config = config
        self.trial = trial
        self.model_type = MODEL_TYPE
        self.model_prefix = MODEL_PREFIX
        self.device = self.config['inputs']['device']

        # Model parameters
        learnable_param_names = ['alpha','beta','kappa','sigma','delta','epsilon','bmax']
        default_true_parameters = dict(zip(learnable_param_names,[None]*len(learnable_param_names)))
        true_params = {k:v for k,v in kwargs.get('true_parameters',default_true_parameters).items() if v is not None}
        # Add parameters to learn if None is provided as true parameter
        for key,value in true_params.items():
            if value is None and key not in self.config.settings['training']['to_learn']:
                self.config.settings['training']['to_learn'].append(key)
        
        # Check that learnable parameters are in valid set
        try:
            assert set(self.config.settings['training']['to_learn']).issubset(set(learnable_param_names))
        except:
            self.logger.error(f"Some parameters in {','.join(self.config.settings['training']['to_learn'])} cannot be learned.")
            self.logger.error(f"Acceptable parameters are {','.join(learnable_param_names)}.")
            raise Exception('Cannot instantiate Harris Wilson Model.') 
        
        # Set parameters to learn based on
        # kwargs or parameter defaults
        self.params_to_learn = {}
        for i,param in enumerate(self.config.settings['training']['to_learn']):
            self.params_to_learn[param] = i

        # Update hyperparameters
        self.update_hyperparameters(**true_params)

        # Store noise variance
        self.hyperparams['noise_var'] = self.obs_noise_percentage_to_var(self.hyperparams['noise_percentage'])

        # Update noise regime
        self.noise_regime = self.hyperparams['sigma']
        if 'sigma' in list(self.params_to_learn.keys()) or self.noise_regime is None:
            self.noise_regime = 'variable'
        elif isinstance(self.noise_regime,list):
            self.noise_regime = 'sweeped'
        elif self.noise_regime >= 0.1:
            self.noise_regime = 'high'
        else:
            self.noise_regime = 'low'
        self.config.settings['noise_regime'] = self.noise_regime


    def update_hyperparameters(self,**kwargs):
        # Set hyperparams
        self.hyperparams = {}
        if self.trial is not None:
            OPTUNA_HYPERPARAMS = {
                'epsilon': self.trial.suggest_float('epsilon', 0.01, 4.0), 
                'noise_percentage': self.trial.suggest_float('noise_percentage', 0.0001, 3.0),
                'dt': self.trial.suggest_float('dt', 1e-4, 1e-1)
                # 'alpha': self.trial.suggest_float('alpha', 0.01, 10.0), 
                # 'beta': self.trial.suggest_float('beta', 0.0, 10.0), 
                # 'kappa': self.trial.suggest_float('kappa', 0.01, 10.0), 
                # 'bmax': self.trial.suggest_float('bmax', 0.01, 1000.0, log=True)
            }
        
        for pname in PARAMETER_DEFAULTS.keys():
            if pname in list(self.params_to_learn.keys()):
                continue
            
            # Try to read from optuna trial
            if self.trial is not None and pname in OPTUNA_HYPERPARAMS:
                self.hyperparams[pname] = OPTUNA_HYPERPARAMS[pname]
            else:
                # Try to read from config
                paths_found = list(self.config.path_find(
                    key = pname,
                    settings = self.config.settings,
                    current_key_path = [],
                    all_key_paths = []
                ))
                # First try to read from kwargs
                if kwargs.get(pname,None) is not None:
                    self.hyperparams[pname] = kwargs[pname]
                # Second try to read from config
                elif len(paths_found) > 0 and len(paths_found[0]) > 0:
                    # Return sweepable flag for first instance of variable
                    value,_ = self.path_get(
                        key_path = paths_found[0],
                        settings = self.config.settings
                    )
                    self.hyperparams[pname] = torch.tensor(value)
                # Third read from defaults
                else:
                    self.hyperparams[pname] = PARAMETER_DEFAULTS[pname]

                if self.config is not None and len(paths_found) > 0 and len(paths_found[0]) > 0:
                    self.config.path_set(
                        self.config.settings,
                        to_json_format(self.hyperparams[pname]),
                        paths_found[0]
                    )

                # Pass to device
                self.hyperparams[pname] = self.hyperparams[pname].to(self.device)


    def obs_noise_percentage_to_var(self,noise_percentage:float):
        return torch.pow(
            (
                torch.tensor(noise_percentage,dtype = float32,device = self.device) / \
                torch.tensor(100).float()
            ) * \
            torch.log(torch.tensor(self.intensity_model.dims['destination']).float()),
            2
        ).to(
            dtype = float32,
            device = self.device
        )

    def sde_potential(self,log_destination_attraction,**kwargs):

        # Update input kwargs if required
        kwargs['log_destination_attraction'] = log_destination_attraction
        updated_kwargs = self.intensity_model.get_input_kwargs(kwargs)
        updated_kwargs.update(**self.hyperparams)
        
        required = (self.intensity_model.REQUIRED_OUTPUTS+self.intensity_model.REQUIRED_INPUTS+['sigma'])

        self.intensity_model.check_sample_availability(
            required,
            updated_kwargs
        )
        return self.intensity_model.sde_pot(**updated_kwargs)
    
    def sde_potential_and_gradient(self,log_destination_attraction,**kwargs):

        # Update input kwargs if required
        kwargs['log_destination_attraction'] = log_destination_attraction
        updated_kwargs = self.intensity_model.get_input_kwargs(kwargs)
        updated_kwargs.update(**self.hyperparams)

        required = (self.intensity_model.REQUIRED_OUTPUTS+self.intensity_model.REQUIRED_INPUTS+['sigma'])

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
        updated_kwargs.update(**self.hyperparams)

        required = (self.intensity_model.REQUIRED_OUTPUTS+self.intensity_model.REQUIRED_INPUTS+['sigma'])

        self.intensity_model.check_sample_availability(
            required,
            updated_kwargs
        )
        return self.intensity_model.sde_pot_jacobian(**updated_kwargs)[0]

    
    def sde_potential_hessian(self,log_destination_attraction,**kwargs):

        # Update input kwargs if required
        kwargs['log_destination_attraction'] = log_destination_attraction
        updated_kwargs = self.intensity_model.get_input_kwargs(kwargs)
        updated_kwargs.update(**self.hyperparams)

        required = (self.intensity_model.REQUIRED_OUTPUTS+self.intensity_model.REQUIRED_INPUTS+['sigma'])

        self.intensity_model.check_sample_availability(
            required,
            updated_kwargs
        )
        return self.intensity_model.sde_pot_hessian(**updated_kwargs)[0][0]
    
    def negative_destination_attraction_log_likelihood_and_gradient(self,**kwargs):
        """ Log of potential function of the likelihood =  log(pi(y|x)).
        """
        # Update input kwargs if required
        self.intensity_model.check_sample_availability(
            ['log_destination_attraction_ts','log_destination_attraction_pred'],
            kwargs
        )

        log_destination_attraction_ts = kwargs['log_destination_attraction_ts']
        log_destination_attraction_pred = kwargs['log_destination_attraction_pred']

        # Compute difference
        diff = (log_destination_attraction_pred.flatten() - log_destination_attraction_ts.flatten())
        # Compute log likelihood (without constant factor) and its gradient
        temp = 0.5*(1./self.hyperparams['noise_var'])*(diff.dot(diff)), (1./self.hyperparams['noise_var'])*diff
        return temp
    
    
    def negative_destination_attraction_log_likelihood(self,**kwargs):
        """ Log of potential function of the likelihood =  log(pi(y|x)).
        """
        # Update input kwargs if required
        self.intensity_model.check_sample_availability(
            ['log_destination_attraction_ts','log_destination_attraction_pred'],
            kwargs
        )
        log_destination_attraction_ts = kwargs['log_destination_attraction_ts']
        log_destination_attraction_pred = kwargs['log_destination_attraction_pred']
        noise_var = self.obs_noise_percentage_to_var(kwargs['noise_percentage']) \
                    if kwargs.get('noise_percentage',None) is not None \
                    else self.hyperparams['noise_var']

        if torch.is_tensor(log_destination_attraction_pred) and \
            torch.is_tensor(log_destination_attraction_ts):
            # Compute difference
            diff = (log_destination_attraction_pred.flatten() - log_destination_attraction_ts.flatten())
            # Compute log likelihood (without constant factor)
            return 0.5*(1./noise_var)*(diff.dot(diff))
        elif isinstance(log_destination_attraction_pred,(xr.DataArray,xr.Dataset)) or \
            isinstance(log_destination_attraction_ts,(xr.DataArray,xr.Dataset)):
            # Compute difference
            # log_destination_attraction_pred,
            # log_destination_attraction_ts = xr.align(
            #     log_destination_attraction_pred,
            #     log_destination_attraction_ts
            # )
            diff = (log_destination_attraction_pred - log_destination_attraction_ts)
            # Compute log likelihood (without constant factor)
            return 0.5*(1./noise_var.cpu().detach())*xr.dot(diff,diff,dims=['destination','time'])
        else:
            raise Exception(f"Did not recognise types {type(log_destination_attraction_pred)} and/or {type(log_destination_attraction_ts)}")


    def negative_destination_attraction_log_likelihood_gradient(self,**kwargs):
        """ Log of potential function of the likelihood =  log(pi(y|x)).
        """
        # Update input kwargs if required
        self.intensity_model.check_sample_availability(
            ['log_destination_attraction_ts','log_destination_attraction_pred'],
            kwargs
        )

        log_destination_attraction_ts = kwargs['log_destination_attraction_ts']
        log_destination_attraction_pred = kwargs['log_destination_attraction_pred']

        # Compute difference
        diff = (log_destination_attraction_pred.flatten() - log_destination_attraction_ts.flatten())
        # Compute gradient of log likelihood
        return (1./self.hyperparams['noise_var']) * diff
    
    def dest_attraction_ts_likelihood_loss(
            self,
            *args,
            **kwargs
        ):
            return self.negative_destination_attraction_log_likelihood(
                **dict(
                    log_destination_attraction_pred = torch.log(args[0]),
                    log_destination_attraction_ts = torch.log(args[1])
                ),
                **kwargs
            )
    
    def sde_ais_potential_and_jacobian(self,**kwargs):
        
        Ndestinations = self.intensity_model.dims['destination']
        delta = self.hyperparams['delta']
        kappa = self.hyperparams['kappa']
        xx = kwargs['log_destination_attraction']
        sigma = kwargs['sigma'] if kwargs['sigma'] is not None else self.hyperparams['sigma']
        gamma = 2/(sigma**2)
        # Note that lim_{beta->0, alpha->0} gamma*V_{theta}(x) = gamma*kappa*\sum_{j = 1}^J \exp(x_j) - gamma*(delta+1/J) * \sum_{j = 1}^J x_j
        gamma_kk_exp_xx = gamma*kappa*torch.exp(xx)
        # Function proportional to the potential function in the limit of alpha -> 0, beta -> 0
        V = -gamma*(delta+1./Ndestinations)*xx.sum() + gamma_kk_exp_xx.sum()
        # Gradient of function above
        gradV = -gamma*(delta+1./Ndestinations)*torch.ones(Ndestinations,dtype = float32,device = self.device) + gamma_kk_exp_xx

        return V, gradV

    # ... Model run functions ..........................................................................................
    
    def run_single(
        self,
        *,
        curr_destination_attractions,
        log_intensity_normalised,
        free_parameters = None,
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
        self.logger.debug(f"Forward solving SDE")
        self.logger.trace('Parsing parameters')

        # Parameters to learn
        kappa = (
            self.hyperparams['kappa']
            if "kappa" not in list(self.params_to_learn.keys())
            else free_parameters[self.params_to_learn["kappa"]]
        )
        delta = (
            self.hyperparams['delta']
            if "delta" not in list(self.params_to_learn.keys())
            else free_parameters[self.params_to_learn["delta"]]
        )
        epsilon = (
            self.hyperparams['epsilon']
            if "epsilon" not in list(self.params_to_learn.keys())
            else free_parameters[self.params_to_learn["epsilon"]]
        )
        sigma = (
            self.hyperparams['sigma']
            if "sigma" not in list(self.params_to_learn.keys())
            else free_parameters[self.params_to_learn["sigma"]]
        )

        # Training parameters
        dt = self.hyperparams['dt'] if dt is None else dt
        self.logger.trace('Cloning dest attractions')
        new_sizes = curr_destination_attractions.clone()
        new_sizes.requires_grad = requires_grad

        # Compute normalised demand
        demand_normalised = torch.exp(
            torch.logsumexp(
                log_intensity_normalised,
                dim = DATA_SCHEMA['log_destination_attraction']['dims'].index('destination')
            )
        )
        
        # Reshape demand to match rest of objects
        demand_normalised = demand_normalised.reshape(new_sizes.shape)
        
        self.logger.trace('Time update')
        new_sizes = (
            new_sizes + \
            +torch.mul(
                curr_destination_attractions,
                epsilon * (demand_normalised - kappa * curr_destination_attractions + delta)
                + sigma
                * 1
                / torch.sqrt(torch.tensor(2, dtype = torch.float) * torch.pi * dt).to(
                    self.device
                )
                * torch.normal(0, 1, size=(self.intensity_model.dims['destination'],1)).to(self.device),
            )
            * dt
        )
        return new_sizes

    def run(
        self,
        *,
        init_destination_attraction,
        free_parameters:dict,
        n_iterations: int,
        dt: float = None,
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
        dt = self.hyperparams['dt'] if dt is None else dt

        if not generate_time_series:
            sizes = init_destination_attraction.clone().unsqueeze(1)


            for _ in range(n_iterations):
                # Compute log intensity
                log_intensity_sample = self.intensity_model.log_intensity(
                    log_destination_attraction = torch.log(sizes),
                    grand_total = torch.tensor(1.0),
                    **free_parameters
                ).squeeze()

                sizes = self.run_single(
                    curr_destination_attractions = sizes,
                    log_intensity_normalised = log_intensity_sample,
                    dt = dt,
                    requires_grad = requires_grad,
                )
        else:
            sizes = [init_destination_attraction.clone()]
            for _ in range(n_iterations):
                # Compute log intensity
                log_intensity_sample = self.intensity_model.log_intensity(
                    log_destination_attraction = torch.log(sizes[-1]),
                    **free_parameters
                ).squeeze()

                sizes.append(
                    self.run_single(
                        curr_destination_attractions = sizes[-1],
                        log_intensity_normalised = log_intensity_sample,
                        dt = dt,
                        requires_grad = requires_grad,
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
        return f"HarrisWilson({self.intensity_model.__repr__()})"

    def __str__(self):

        return f"""
            {'x'.join([str(d) for d in self.intensity_model.dims.values()])} Harris Wilson model using {self.intensity_model}
            Learned parameters: {', '.join(self.params_to_learn.keys())}
            Epsilon: {self.hyperparams['epsilon']}
            Kappa: {self.hyperparams['kappa']}
            Delta: {self.hyperparams['delta']}
        """