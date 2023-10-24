import gc
import sys
import time
import warnings
import torch.multiprocessing as mp

from os import path
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from torch import float32, uint8
from scipy.optimize import minimize
from joblib import Parallel, delayed

from multiresticodm.utils import *
from multiresticodm.config import Config
from multiresticodm.inputs import Inputs
from multiresticodm.outputs import Outputs
from multiresticodm.global_variables import *
from multiresticodm.math_utils import apply_norm
from multiresticodm.markov_basis import MarkovBasis
from multiresticodm.contingency_table import instantiate_ct
from multiresticodm.harris_wilson_model import HarrisWilson
from multiresticodm.spatial_interaction_model import instantiate_sim
from multiresticodm.harris_wilson_model_neural_net import NeuralNet, HarrisWilson_NN
from multiresticodm.harris_wilson_model_mcmc import instantiate_harris_wilson_mcmc
from multiresticodm.contingency_table_mcmc import ContingencyTableMarkovChainMonteCarlo

# Suppress scientific notation
np.set_printoptions(suppress=True)

def instantiate_experiment(experiment_type:str,config:Config,**kwargs):
    # Get whether sweep is active and its settings available
    has_coupled_sweep_paths = len(config.coupled_sweep_paths.values()) > 0
    has_isolated_sweep_paths = len(config.isolated_sweep_paths.values()) > 0
    if hasattr(sys.modules[__name__], experiment_type):
        if config.settings.get("sweep_mode",False) and \
            (has_isolated_sweep_paths or has_coupled_sweep_paths):
            return ExperimentSweep(
                config=config,
                **kwargs
            )
        else:
            return getattr(sys.modules[__name__], experiment_type)(config=config,**kwargs)
    else:
        raise Exception(f'Experiment class {experiment_type} not found')

class ExperimentHandler(object):

    def __init__(self, config:Config, **kwargs):
        # Import logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__,
            console_handler_level = level, 
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevels(
            console_level = level
        )
        
        # Get configuration
        self.config = config

        # Setup experiments
        self.setup_experiments(**kwargs)

    def setup_experiments(self,**kwargs):
        
        # Dictionary of experiment ids to experiment objects
        self.experiments = {}
        
        # Only run experiments specified in command line
        for experiment_type,experiment_index in self.config.settings['run_experiments'].items():
            # Construct sub-config with only data relevant for experiment
            experiment_config = deepcopy(self.config)
            # experiment_config.logger = self.logger
            # Store one experiment
            experiment_config.settings['experiments'] = [
                self.config.settings['experiments'][experiment_index]
            ]
            # Update id, seed and logging detail
            experiment_config.settings['experiment_type'] = experiment_type
            if self.config.settings['inputs'].get('dataset',None) is None:
                raise Exception(f'No dataset found for experiment type {experiment_type}')
            # Instatiate new experiment
            experiment = instantiate_experiment(
                experiment_type=experiment_type,
                config=experiment_config,
                logger=self.logger
            )
            # Append it to list of experiments
            self.experiments[experiment_type] = experiment

    def run_and_write_experiments_sequentially(self):
        # Run all experiments sequential
        for _,experiment in self.experiments.items():
            # Run experiment
            experiment.run()

            # Reset
            try:
                experiment.reset()
            except:
                pass
        
class Experiment(object):
    def __init__(self, config:Config, **kwargs):
        # Create logger
        level = kwargs.get('console_level',None)
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            console_handler_level = level, 
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger lever
        self.logger.setLevels(
            console_level = level
        )
        
        self.logger.debug(f"{self}")
        # Make sure you are reading a config
        if isinstance(config,dict):
            config = Config(
                settings=config,
                logger = self.logger
            )
        elif not isinstance(config,Config):
            raise Exception(f'config provided has invalid type {type(config)}')

        # Store config
        self.config = config
        if len(self.config['inputs'].get('load_experiment',[])) > 0:
            # Get path
            filepath = self.config['inputs'].get('load_experiment','')
            # Load metadata
            settings = read_json(path.join(filepath,path.basename(filepath)+'_metadata.json'))
            # Deep update config settings based on metadata
            settings_flattened = deep_flatten(settings,parent_key='',sep='')
            # Remove load experiment 
            del settings_flattened['load_experiment']
            deep_updates(self.config.settings,settings_flattened,overwrite=True)
            # Merge settings to config
            self.config = Config(
                settings={**self.config, **settings_flattened},
                logger = self.logger
            )
        
        # Update config with current timestamp ( but do not overwrite)
        datetime_results = list(deep_get(key='datetime',value=self.config.settings))
        if len(datetime_results) > 0:
            deep_update(self.config.settings, key='datetime', val=datetime.now().strftime("%d_%m_%Y_%H_%M_%S"), overwrite=False)
        else:
            self.config['datetime'] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        # Inherit experiment id from parameter sweep (if it exists)
        # This will be used to create a unique output directory for every sweep
        self.sweep_experiment_id = kwargs.get('experiment_id',None)

        # Update current config
        # self.config = self.sim.config.update_recursively(self.config,updated_config,overwrite=True)
        # print(self.config)
        # Decide how often to print statemtents
        self.print_statements = self.config.get('print_statements',True)
        self.store_progress = self.config.get('store_progress',1.0)
        self.print_percentage = min(0.05,self.store_progress)*int(self.print_statements)

        # Update seed if specified
        self.seed = None
        if "seed" in self.config['inputs'].keys():
            self.seed = int(self.config['inputs']["seed"])
            self.logger.info(f"Updated seed to {self.seed}")
        # Get experiment data
        self.logger.info(f"Experiment {self.config['experiment_type']} has been set up.")

        # Get device name
        self.device = self.config['inputs']['device']
        # Get device id
        self.device_id = kwargs.get('device_id',0)
        # print('current_device',torch.cuda.current_device())

        # Disable tqdm if needed
        self.tqdm_disabled = kwargs.get('tqdm_disabled',False)

        # Count the number of gradient descent steps
        self._time = 0
        self._write_every = self.config['outputs'].get('write_every',1)
        self._write_start = self.config['outputs'].get('write_start',1)
        self.n_processed_steps = 0

    def run(self,**kwargs) -> None:
        pass

    def reset(self,metadata:bool=False) -> None:
        self.logger.note(f"Resetting experimental results to release memory.")
        
        # Get shapes 
        theta_shape = deepcopy(np.shape(self.thetas[-1])[0] if hasattr(self,'thetas') and self.thetas is not None else (2))
        log_destination_attraction_shape = np.shape(self.log_destination_attractions[-1])[0] if hasattr(self,'log_destination_attraction') and self.log_destination_attractions is not None else (self.sim.dims['destination'])
        table_shape = np.shape(self.tables[-1]) if hasattr(self,'tables') and self.tables is not None else tuple(self.sim.dims)

        # Reset tables and columns sums to release memory
        safe_delete(deep_call(self,'tables',None))
        safe_delete(deep_call(self,'thetas',None))
        safe_delete(deep_call(self,'log_destination_attractions',None))
        safe_delete(deep_call(self,'signs',None))
        safe_delete(deep_call(self,'results',None))

        if metadata:
            safe_delete(self.config)
            self.config = Config(
                settings={},
                logger = self.logger
            )

        # Run garbage collector
        gc.collect()

        # Reinitialise objects
        self.results = []
        try:
            self.thetas = np.zeros((0,theta_shape),dtype='float32')
        except:
            pass
        try:
            self.signs = np.zeros((0,1),dtype='int8')
        except:
            pass
        try:
            self.log_destination_attractions = np.zeros((0,log_destination_attraction_shape),dtype='float32')
        except:
            pass
        try:
            self.tables = np.zeros((0,*table_shape),dtype='int32')
        except:
            pass

    def define_sample_batch_sizes(self):
        N = self.harris_wilson_mcmc.config['training']['N']
        # Define sample batch sizes
        sample_sizes = np.repeat(int(self.store_progress*N),np.floor(1/self.store_progress))
        if sample_sizes.sum() < N:
            sample_sizes = np.append(sample_sizes,[N-sample_sizes.sum()])
        sample_sizes = np.cumsum(sample_sizes)
        return sample_sizes
    

    def initialise_parameters(self,param_names:list=[]):
        
        initialisations = {}
        for param in param_names:

            if param == 'table':
                # Get initial margins
                # try:
                initialisations['table_sample'] = deep_call(self,'.ct_mcmc.initialise_table()',None)
                # except:
                    # self.logger.warning("Table could not be initialised.")
            
            elif param == 'theta':
                try:
                    theta_sample = self.harris_wilson_model.params_to_learn
                except:
                    try:
                        theta_sample = self.harris_wilson_mcmc.physics_model.params_to_learn
                    except:
                        self.logger.warning("Theta could not be initialised.")
                self.params_to_learn = list(theta_sample.keys())
                initialisations['theta_sample'] = torch.tensor(list(theta_sample.values()),dtype=float32,device=self.device)
                
            elif param == 'log_destination_attraction':
                # Arbitrarily initialise destination attraction
                initialisations['log_destination_attraction_sample'] = torch.log(
                    torch.repeat_interleave(
                        torch.tensor(1./self.harris_wilson_mcmc.physics_model.intensity_model.dims['destination']),
                        self.harris_wilson_mcmc.physics_model.intensity_model.dims['destination']
                    )
                ).to(
                    dtype=float32,
                    device=self.device
                )
                initialisations['log_destination_attraction_sample'].requires_grad = True

            elif param == 'sign':
                initialisations['sign_sample'] = torch.tensor(1,dtype=uint8,device=self.device)
            
            elif param == 'loss':
                initialisations['loss_sample'] = torch.tensor(0.0,dtype=float32,device=self.device,requires_grad=True)
            
            elif param == 'log_target':
                initialisations['log_target_sample'] = torch.tensor(0.0,dtype=float32,device=self.device)

            elif param == 'computation_time':
                initialisations['computation_time'] = 0

        return initialisations
        
    def initialise_data_structures(self):

        if self.config.settings.get('export_samples',True):

            # Get dimensions
            dims = self.config['inputs']['dims']
            
            # Setup neural net loss
            if 'loss' in self.output_names:
                # Setup chunked dataset to store the state data in
                self.losses = self.outputs.h5group.create_dataset(
                    'loss',
                    (0,),
                    maxshape=(None,),
                    chunks=True,
                    compression=3,
                )
                self.losses.attrs['dim_names'] = XARRAY_SCHEMA['loss']['coords']
                self.losses.attrs['coords_mode__time'] = 'start_and_step'
                self.losses.attrs['coords__time'] = [self._write_start, self._write_every]
            
            # Setup sampled/predicted log destination attractions
            if 'log_destination_attraction' in self.output_names:
                self.log_destination_attractions = self.outputs.h5group.create_dataset(
                    "log_destination_attraction",
                    (0,self.inputs.data.destination_attraction_ts.shape[0],dims['destination']),
                    maxshape=(None,self.inputs.data.destination_attraction_ts.shape[0],dims['destination']),
                    chunks=True,
                    compression=3,
                )
                self.log_destination_attractions.attrs["dim_names"] = XARRAY_SCHEMA['log_destination_attraction']['coords']
                self.log_destination_attractions.attrs["coords_mode__time"] = "start_and_step"
                self.log_destination_attractions.attrs["coords__time"] = [self._write_start, self._write_every]
            
            # Setup computation time
            if 'computation_time' in self.output_names:
                self.compute_time = self.outputs.h5group.create_dataset(
                    'computation_time',
                    (0,),
                    maxshape=(None,),
                    chunks=True,
                    compression=3,
                )
                self.compute_time.attrs['dim_names'] = XARRAY_SCHEMA['computation_time']['coords']
                self.compute_time.attrs['coords_mode__epoch'] = 'trivial'

            # Setup sampled/predicted theta
            if 'theta' in self.output_names:
                predicted_thetas = []
                for p_name in self.theta_names:
                    dset = self.outputs.h5group.create_dataset(
                        p_name, 
                        (0,), 
                        maxshape=(None,), 
                        chunks=True, 
                        compression=3
                    )
                    dset.attrs['dim_names'] = XARRAY_SCHEMA[p_name]['coords']
                    dset.attrs['coords_mode__time'] = 'start_and_step'
                    dset.attrs['coords__time'] = [self._write_start, self._write_every]

                    predicted_thetas.append(dset)
                self.thetas = predicted_thetas
            
            # Setup sampled signs
            if 'sign' in self.output_names:
                self.signs = self.outputs.h5group.create_dataset(
                    "sign",
                    (0,),
                    maxshape=(None,),
                    chunks=True,
                    compression=3,
                )
                self.signs.attrs["dim_names"] = XARRAY_SCHEMA['sign']['coords']
                self.signs.attrs["coords_mode__time"] = "start_and_step"
                self.signs.attrs["coords__time"] = [self._write_start, self._write_every]

            # Setup sampled tables
            if 'table' in self.output_names:
                self.tables = self.outputs.h5group.create_dataset(
                    "table",
                    (0,*dims),
                    maxshape=(None,*dims),
                    chunks=True,
                    compression=3,
                )
                self.tables.attrs["dim_names"] = ["origin","destination","iter"]
                self.tables.attrs["coords_mode__time"] = "start_and_step"
                
            # Setup acceptances
            if 'theta_acc' in self.output_names:
                # Setup chunked dataset to store the state data in
                self.theta_acc = self.outputs.h5group.create_dataset(
                    'theta_acceptance',
                    (0,),
                    maxshape=(None,),
                    chunks=True,
                    compression=3,
                )
                self.losses.attrs['dim_names'] = XARRAY_SCHEMA['theta_acceptance']['coords']
                self.losses.attrs['coords_mode__time'] = 'start_and_step'
                self.losses.attrs['coords__time'] = [self._write_start, self._write_every]
            if 'log_destination_attraction_acc' in self.output_names:
                # Setup chunked dataset to store the state data in
                self.log_destination_attraction_acc = self.outputs.h5group.create_dataset(
                    'log_destination_attraction_acceptance',
                    (0,),
                    maxshape=(None,),
                    chunks=True,
                    compression=3,
                )
                self.losses.attrs['dim_names'] = XARRAY_SCHEMA['log_destination_attraction_acc']['coords']
                self.losses.attrs['coords_mode__time'] = 'start_and_step'
                self.losses.attrs['coords__time'] = [self._write_start, self._write_every]
            if 'table_acc' in self.output_names:
                # Setup chunked dataset to store the state data in
                self.table_acc = self.outputs.h5group.create_dataset(
                    'table_acceptance',
                    (0,),
                    maxshape=(None,),
                    chunks=True,
                    compression=3,
                )
                self.losses.attrs['dim_names'] = XARRAY_SCHEMA['table_acc']['coords']
                self.losses.attrs['coords_mode__time'] = 'start_and_step'
                self.losses.attrs['coords__time'] = [self._write_start, self._write_every]
        
    def update_and_export(
            self,
            batch_size: int,
            data_size: int,
            t: int,
            **kwargs
        ):
        self.logger.progress('Update and export')
        # Update the model parameters after every batch and clear the loss
        if t % batch_size == 0 or t == data_size - 1:
            # Update time
            self._time += 1

            # Update gradients
            loss = kwargs.get('loss',None)
            if loss is not None:
                loss.backward()
                self.harris_wilson_nn._neural_net.optimizer.step()
                self.harris_wilson_nn._neural_net.optimizer.zero_grad()

            # Write to file
            self.write_data(
                **kwargs
            )
            # Delete loss
            del loss
            loss = torch.tensor(0.0, requires_grad=True)
            self.n_processed_steps = 0
        
        return loss

    def write_data(self,**kwargs):
        '''Write the current state into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        '''
        self.logger.debug('Writing data')
        if self._time >= self._write_start and self._time % self._write_every == 0:
            if 'loss' in self.output_names:
                # Store samples
                _loss_sample = kwargs.get('loss',None) 
                _loss_sample = _loss_sample.clone().detach().cpu().numpy().item() / self.n_processed_steps if _loss_sample is not None else None
                self.losses.resize(self.losses.shape[0] + 1, axis=0)
                self.losses[-1] = _loss_sample

            if 'table' in self.output_names:
                _table_sample = kwargs.get('table',None)
                _table_sample = _table_sample.clone().detach().cpu() if _table_sample is not None else None
                self.tables.resize(self.tables.shape[0] + 1, axis=0)
                self.tables[-1,...] = _table_sample

            if 'theta' in self.output_names:
                _theta_sample = kwargs.get('theta',[None]*len(self.thetas))
                _theta_sample = _theta_sample.clone().detach().cpu() if _theta_sample is not None else None
                for idx, dset in enumerate(self.thetas):
                    dset.resize(dset.shape[0] + 1, axis=0)
                    dset[-1] = _theta_sample[idx]
            
            if 'sign' in self.output_names:
                _sign_sample = kwargs.get('sign',None)
                _sign_sample = _sign_sample.clone().detach().cpu() if _sign_sample is not None else None
                self.signs.resize(self.signs.shape[0] + 1, axis=0)
                self.signs[-1] = _sign_sample
            
            if 'log_destination_attraction' in self.output_names:
                _log_destination_attraction_sample = kwargs.get('log_destination_attraction',np.array([None]))
                _log_destination_attraction_sample = _log_destination_attraction_sample.clone().detach().cpu() if _log_destination_attraction_sample is not None else None
                self.log_destination_attractions.resize(self.log_destination_attractions.shape[0] + 1, axis=0)
                self.log_destination_attractions[-1,...] = _log_destination_attraction_sample

            if 'theta_acc' in self.output_names:
                _theta_acc = kwargs.get('theta_acc',None)
                _theta_acc = _theta_acc / self.n_processed_steps if _theta_acc is not None else None
                self.theta_acc.resize(self.theta_acc.shape[0] + 1, axis=0)
                self.theta_acc[-1] = _theta_acc

            if 'log_destination_attraction_acc' in self.output_names:
                _log_dest_attract_acc = kwargs.get('log_destination_attraction_acc',None)
                _log_dest_attract_acc = _log_dest_attract_acc / self.n_processed_steps if _log_dest_attract_acc is not None else None
                self.log_destination_attraction_acc.resize(self.log_destination_attraction_acc.shape[0] + 1, axis=0)
                self.log_destination_attraction_acc[-1] = _log_dest_attract_acc

            if 'table_acc' in self.output_names:
                _table_acc = kwargs.get('table_acc',None)
                _table_acc = _table_acc / self.n_processed_steps if _table_acc is not None else None
                self.table_acc.resize(self.table_acc.shape[0] + 1, axis=0)
                self.table_acc[-1] = _table_acc

            # Update metadata
            self.update_metadata()

    
    def print_initialisations(self,parameter_inits,print_lengths:bool=True,print_values:bool=False):
        for p,v in parameter_inits.items():
            if isinstance(v,(list,np.ndarray)):
                if print_lengths and print_values:
                    print(p,len(v),v)
                elif print_values:
                    print(p,v)
                else:
                    print(p,len(v))
            else:
                print(p,v)

    
    def update_metadata(self):
        if hasattr(self,'theta_acc'):
            self.config['theta_acceptance'] = int(100*self.theta_acc[:].mean(axis=0))
            self.logger.progress('Theta acceptance:',self.config['theta_acceptance'])
        if hasattr(self,'signs'):
            self.config['positives_percentage'] = int(100*self.signs[:].sum(axis=0))
            self.logger.progress('Positives %:',self.config['positives_percentage'])
        if hasattr(self,'log_destination_attraction_acc'):
            self.config['log_destination_attraction_acceptance'] = int(100*self.log_destination_attraction_acc[:].mean(axis=0))
            self.logger.progress('Log destination attraction acceptance:',self.config['log_destination_attraction_acceptance'])
        if hasattr(self,'table_acc'):
            self.config['table_acceptance'] = int(100*self.table_acc[:].mean(axis=0))
            self.logger.progress('Table acceptance:',self.config['table_acceptance'])
        if hasattr(self,'losses'):
            self.logger.progress('Average loss:',self.losses[:].mean(axis=0))

class DataGeneration(Experiment):

    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)
        
        self.config = config
        self.instance = int(kwargs['instance'])

    def run(self,**kwargs) -> None:
        # Generate inputs
        Inputs(
            config = self.config,
            synthetic_data = True,
            logger = self.logger,
            instance = self.instance
        )

class RSquaredAnalysis(Experiment):

    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)
        # Build spatial interaction model
        self.sim = instantiate_sim(self.config)

        self.grid_size = config['grid_size']
        self.amin,self.amax = config['a_range']
        self.bmin,self.bmax = config['b_range']

    def run(self,**kwargs) -> None:
        
        # Initialize search grid
        alpha_values = np.linspace(self.amin, self.amax, self.grid_size,endpoint=True)
        beta_values = np.linspace(self.bmin, self.bmax, self.grid_size,endpoint=True)
        r2_values = np.zeros((self.grid_size, self.grid_size),dtype='float16')

        # Define theta parameters
        theta = np.array([
                        0.0, 
                        0.0,
                        self.sim.delta,
                        self.sim.gamma,
                        self.sim.kappa,
                        self.sim.epsilon
                ],dtype='float64')

        # Search values
        max_r2 = 0 
        max_w_prediction = np.exp(np.ones(self.sim.dims['destination'])*(1/self.sim.dims['destination']))

        # Total sum squares
        w_data = np.exp(self.sim.log_destination_attraction)
        w_data_centred = w_data - np.mean(w_data)
        ss_tot = np.dot(w_data_centred, w_data_centred)

        # Perform grid evaluations
        for i in tqdm(range(self.grid_size)):
            for j in range(self.grid_size):
                try:
                    theta[0] = alpha_values[j]
                    theta[1] = beta_values[i]*self.sim.bmax
                    
                    # Minimise potential function
                    potential_func = minimize(
                        self.sim.sde_potential_and_gradient,
                        self.sim.log_destination_attraction,
                        method='L-BFGS-B',
                        jac=True,
                        args=(theta),
                        options={'disp': False}
                    )
                    w_pred = np.exp(potential_func.x,dtype='float32')
                    # Residiual sum squares
                    res = w_pred - w_data
                    ss_res = np.dot(res, res)

                    # Regression sum squares
                    r2_values[i, j] = 1. - ss_res/ss_tot
                    if r2_values[i, j] > max_r2:
                        max_w_prediction = deepcopy(w_pred)
                        max_r2 = r2_values[i, j]
                    last_r2 = r2_values[i, j]
                except:
                    r2_values[i, j] = last_r2

        # Output results
        idx = np.unravel_index(r2_values.argmax(), r2_values.shape)

        if self.config['print_statements']:
            print("Fitted alpha, beta and scaled beta values:")
            print(alpha_values[idx[1]],beta_values[idx[0]], beta_values[idx[0]]*self.sim.bmax)
            print("R^2 value:")
            print(r2_values[idx],np.max(r2_values.ravel()))
            print('Destination attraction prediction')
            print(max_w_prediction)
            print('True destination attraction')
            print(np.exp(self.sim.log_destination_attraction))
            if self.sim.ground_truth_known:
                print('True theta')
                print(self.sim.alpha_true,self.sim.beta_true)

        # Save fitted values to parameters
        self.config['noise_regime'] = self.sim.noise_regime
        self.config['fitted_alpha'] = alpha_values[idx[1]]
        self.config['fitted_beta'] = beta_values[idx[0]]
        self.config['fitted_scaled_beta'] = beta_values[idx[0]]*self.sim.bmax
        self.config['R^2'] = float(r2_values[idx])
        self.config['predicted_w'] = max_w_prediction.tolist()

        # Append to result array
        self.results = [{"samples":{"r2":r2_values}}]


class LogTargetAnalysis(Experiment):

    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)
        # Build spatial interaction model
        sim = instantiate_sim(self.config)

        self.grid_size = config['grid_size']
        self.amin,self.amax = config['a_range']
        self.bmin,self.bmax = config['b_range']

        # Spatial interaction model MCMC
        self.harris_wilson_mcmc = instantiate_harris_wilson_mcmc(sim)
        self.harris_wilson_mcmc.build(**kwargs)

    def run(self,**kwargs) -> None:
        # Initialize search grid
        self.bmin *= self.sim.bmax
        self.bmax *= self.sim.bmax

        alpha_values = np.linspace(self.amin, self.amax, self.grid_size,endpoint=True)
        beta_values = np.linspace(self.bmin, self.bmax, self.grid_size,endpoint=True)
        XX, YY = np.meshgrid(alpha_values, beta_values)
        log_targets = np.empty((self.grid_size, self.grid_size))
        log_targets[:] = np.nan

        # Define theta parameters
        theta = np.concatenate([np.array([alpha_values[0], beta_values[0]]),np.array([self.sim.delta,self.sim.gamma,self.sim.kappa,self.sim.epsilon])],dtype='float64')

        # Normalise initial log destination sizes
        xd = self.sim.log_destination_attraction

        # Search values
        max_target = -np.infty
        argmax_index = None
        argmax_theta = None
        lap_c1 = 0.5*self.sim.dims['destination']*np.log(2.*np.pi)

        # Perform grid evaluations
        for i in tqdm(range(self.grid_size)):
            for j in tqdm(range(self.grid_size),leave=False):
                try:
                    # Residiual sum squares
                    theta[0] = XX[i, j]
                    theta[1] = YY[i, j]

                    # Minimise potential function
                    log_z_inverse,_ = self.harris_wilson_mcmc.biased_z_inverse(0,theta)

                    # Compute potential function
                    potential_func,_ = self.sim.sde_potential_and_gradient(xd,theta)
                    log_target = log_z_inverse-potential_func-lap_c1

                    # Store log_target
                    log_targets[i,j] = log_target

                    if log_target > max_target:
                        argmax_index = deepcopy((i,j))
                        argmax_theta = deepcopy(theta)
                        max_target = deepcopy(log_target)
                except Exception as e:
                    print(e)
                    None

        if self.config['print_statements']:
            print("Fitted alpha, beta and scaled beta values:")
            print(XX[argmax_index],YY[argmax_index]*self.amax/(self.bmax), YY[argmax_index])
            print("Log target:")
            print(log_targets[argmax_index])

        # Compute estimated flows
        theta[0] = XX[argmax_index]
        theta[1] = YY[argmax_index]

        # Save fitted values to parameters
        self.config['fitted_alpha'] = XX[argmax_index]
        self.config['fitted_scaled_beta'] = YY[argmax_index]*self.amax/(self.bmax)
        self.config['fitted_beta'] = YY[argmax_index]
        self.config['kappa'] = self.sim.kappa
        self.config['log_target'] = log_targets[argmax_index]
        self.config['noise_regime'] = self.sim.noise_regime

        # Append to result array
        self.results = [{"samples":{"log_target":log_targets}}]


class SIM_MCMC(Experiment):
    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)

        # Fix random seed
        set_seed(self.seed)

        # Enable garbage collections
        gc.enable()

        # Prepare inputs
        self.inputs = Inputs(
            config=config,
            synthetic_data = False,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )

        # Pass inputs to device
        self.inputs.pass_to_device()

        # Instantiate Spatial Interaction Model
        self.logger.note("Initializing the spatial interaction model ...")

        sim = instantiate_sim(
            name = config['spatial_interaction_model']['name'],
            config = config,
            true_parameters = config['spatial_interaction_model']['parameters'],
            instance = kwargs.get('instance',''),
            **vars(self.inputs.data),
            logger=self.logger
        )
        # Get and remove config
        config = pop_variable(sim,'config')

        # Build the Harris Wilson model
        self.logger.note("Initializing the Harris Wilson physics model ...")
        harris_wilson_model = HarrisWilson(
            intensity_model = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )

        # Spatial interaction model MCMC
        self.harris_wilson_mcmc = instantiate_harris_wilson_mcmc(
            config = config,
            physics_model = harris_wilson_model,
            logger = self.logger
        )
        self.harris_wilson_mcmc.build(**kwargs)
        
        # Get config
        config = getattr(self.harris_wilson_mcmc,'config') if hasattr(self.harris_wilson_mcmc,'config') else None

        # Update config
        self.config = config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep_params=kwargs.get('sweep_params',{}),
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep_params=kwargs.get('sweep_params',{}))
        # Write metadata
        if self.config.settings.get('export_metadata',True):
            self.logger.debug("Writing metadata ...")
            if self.config.settings["sweep_mode"]:
                dir_path = os.path.join("samples",self.outputs.sweep_id)
                filename = 'metadata'
            else:
                dir_path = ""
                filename = 'config'

            self.outputs.write_metadata(
                dir_path=dir_path,
                filename=filename
            )
        
        self.logger.note(f"{self.harris_wilson_mcmc}")
        self.logger.info(f"Experiment: {self.outputs.experiment_id}")
        # self.logger.critical(f"{json.dumps(kwargs.get('sweep_params',{}),indent=2)}")

        self.output_names = ['log_destination_attraction','theta','sign','log_target','computation_time']
        self.theta_names = config['inputs']['to_learn']
        
    def run(self,**kwargs) -> None:

        self.logger.info(f"Running MCMC inference of {self.harris_wilson_mcmc.physics_model.noise_regime} noise SpatialInteraction.")

        # Fix random seed
        set_seed(self.seed)

        # Initialise parameters
        initial_params = self.initialise_parameters(self.output_names)
        # globals().update(**initial_params)
        theta_sample = initial_params['theta_sample']
        log_destination_attraction_sample = initial_params['log_destination_attraction_sample']
        
        # Print initialisations
        # self.print_initialisations(parameter_inits,print_lengths=False,print_values=True)
        
        # Expand theta
        theta_sample_scaled = deepcopy(theta_sample)
        theta_sample_scaled[1] *= self.harris_wilson_mcmc.physics_model.params.bmax

        # Compute initial log inverse z(\theta)
        log_z_inverse, sign_sample = self.harris_wilson_mcmc.z_inverse(
                0,
                dict(zip(self.params_to_learn,theta_sample_scaled))
        )
        # Evaluate log potential function for initial choice of \theta
        V, gradV = self.harris_wilson_mcmc.physics_model.sde_potential_and_gradient(
                log_destination_attraction_sample,
                **dict(zip(self.params_to_learn,theta_sample_scaled)),
                **vars(self.harris_wilson_mcmc.physics_model.params)
        )

        # Store number of samples
        N = self.config.settings['training']['N']
        # Total samples for table,theta,x posteriors, respectively
        M = self.harris_wilson_mcmc.theta_steps
        L = self.harris_wilson_mcmc.log_destination_attraction_steps

        for i in tqdm(
            range(N),
            disable=self.config.settings['mcmc']['disable_tqdm'],
            leave=False,
            position=(self.device_id+1),
            desc = f"SIM_MCMC device id: {self.device_id}"
        ):

            # Track the epoch training time
            start_time = time.time()
        
            # Run theta sampling
            for j in tqdm(
                range(M),
                disable=True,
                leave=False
            ):

                # Gather all additional values
                auxiliary_values = [V,
                                gradV,
                                log_z_inverse,
                                sign_sample]
            
                # Take step
                theta_sample, \
                theta_acc, \
                V, \
                gradV, \
                log_z_inverse, \
                sign_sample = self.harris_wilson_mcmc.theta_step(
                    i,
                    theta_sample,
                    log_destination_attraction_sample,
                    auxiliary_values
                )

                # Write to file
                self.write_data(
                    theta_sample = theta_sample,
                    theta_acc = theta_acc,
                    sign_sample = sign_sample
                )
            
            # Run x sampling
            for l in tqdm(
                range(L),
                disable=True,
                leave=False
            ):
                
                # Gather all additional values
                auxiliary_values = [V, 
                                gradV]
                # Take step
                log_destination_attraction_sample, \
                log_dest_attract_acc, \
                V, \
                gradV = self.harris_wilson_mcmc.log_destination_attraction_step(
                    theta_sample,
                    self.inputs.data.log_destination_attraction,
                    log_destination_attraction_sample,
                    auxiliary_values
                )
                # Write to data
                self.write_data(
                    log_destination_attraction_sample = log_destination_attraction_sample,
                    log_destination_attraction_acc = log_dest_attract_acc,
                )

                self.logger.progress(f"Completed epoch {i+1} / {N}.")

            # Write the epoch training time (wall clock time)
            if hasattr(self,'compute_time'):
                self.compute_time.resize(self.compute_time.shape[0] + 1, axis=0)
                self.compute_time[-1] = time.time() - start_time
        
        if self.config.settings.get('export_metadata',True):
            # Write metadata
            dir_path = os.path.join("samples",self.outputs.sweep_id)
            self.outputs.write_metadata(
                dir_path=dir_path,
                filename=f"metadata"
            )
        if self.config.settings.get('export_samples',True):
            # Close h5 data file
            self.outputs.h5file.close()
            # Write log file
            self.outputs.write_log(self.logger)
        
        self.logger.note("Simulation run finished.")

class JointTableSIM_MCMC(Experiment):

    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)

        # Fix random seed
        rng = set_seed(self.seed)

        # Enable garbage collections
        gc.enable()

        # Prepare inputs
        self.inputs = Inputs(
            config=config,
            synthetic_data = False,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )

        # Pass inputs to device
        self.inputs.pass_to_device()

        # Instantiate Spatial Interaction Model
        self.logger.note("Initializing the spatial interaction model ...")

        sim = instantiate_sim(
            name = config['spatial_interaction_model']['name'],
            config = config,
            true_parameters = config['spatial_interaction_model']['parameters'],
            instance = kwargs.get('instance',''),
            **vars(self.inputs.data),
            logger=self.logger
        )
        # Get and remove config
        config = pop_variable(sim,'config')

        # Build the Harris Wilson model
        self.logger.note("Initializing the Harris Wilson physics model ...")
        harris_wilson_model = HarrisWilson(
            intensity_model = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        
        # Spatial interaction model MCMC
        self.harris_wilson_mcmc = instantiate_harris_wilson_mcmc(
            config = config,
            physics_model = harris_wilson_model,
            logger = self.logger
        )
        self.harris_wilson_mcmc.build(**kwargs)
        
        # Get config
        config = getattr(self.harris_wilson_mcmc,'config') if hasattr(self.harris_wilson_mcmc,'config') else None

        # Build contingency table
        ct = instantiate_ct(
            config = config,
            logger = self.logger,
            **vars(self.inputs.data)
        )
        
        # Build contingency table MCMC
        self.ct_mcmc = ContingencyTableMarkovChainMonteCarlo(
            ct = ct,
            rng = rng,
            log_to_console = False,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )

        # Update config
        self.config = config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep_params=kwargs.get('sweep_params',{}),
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep_params=kwargs.get('sweep_params',{}))

        # Write metadata
        if self.config.settings.get('export_metadata',True):
            self.logger.debug("Writing metadata ...")
            if self.config.settings["sweep_mode"]:
                dir_path = os.path.join("samples",self.outputs.sweep_id)
                filename = 'metadata'
            else:
                dir_path = ""
                filename = 'config'

            self.outputs.write_metadata(
                dir_path=dir_path,
                filename=filename
            )
        
        self.logger.note(f"{self.harris_wilson_mcmc}")
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")

        self.output_names = ['log_destination_attraction','theta','sign','table','log_target','computation_time']
        self.theta_names = config['inputs']['to_learn']
        
    def run(self,**kwargs) -> None:

        self.logger.info(f"Running MCMC inference of {self.harris_wilson_mcmc.physics_model.noise_regime} noise {self.harris_wilson_mcmc.physics_model.name}.")

        # Fix random seed
        set_seed(self.seed)

        # Initialise parameters
        initial_params = self.initialise_parameters(self.output_names)
        # globals().update(**initial_params)
        theta_sample = initial_params['theta_sample']
        log_destination_attraction_sample = initial_params['log_destination_attraction_sample']
        table_sample = initial_params['table_sample']
        
        # Print initialisations
        # self.print_initialisations(parameter_inits,print_lengths=False,print_values=True)
        
        # Expand theta
        theta_sample_scaled = deepcopy(theta_sample)
        theta_sample_scaled[1] *= self.harris_wilson_mcmc.physics_model.params.bmax

        # Compute table likelihood and its gradient
        negative_log_table_likelihood = self.harris_wilson_mcmc.negative_table_log_likelihood_expanded(
            log_destination_attraction = log_destination_attraction_sample,
            alpha = theta_sample_scaled[0],
            beta = theta_sample_scaled[1],
            table = table_sample
        )

        # Compute initial log inverse z(\theta)
        log_z_inverse, sign_sample = self.harris_wilson_mcmc.z_inverse(
            0,
            dict(zip(self.params_to_learn,theta_sample_scaled))
        )

        # Evaluate log potential function for initial choice of \theta
        V, gradV = self.harris_wilson_mcmc.physics_model.sde_potential_and_gradient(
            log_destination_attraction_sample,
            **dict(zip(self.params_to_learn,theta_sample_scaled)),
            **vars(self.harris_wilson_mcmc.physics_model.params)
        )

        # Store number of samples
        N = self.config.settings['training']['N']
        # Total samples for table,theta,x posteriors, respectively
        M = self.harris_wilson_mcmc.theta_steps
        L = self.harris_wilson_mcmc.log_destination_attraction_steps

        for i in tqdm(
            range(N),
            disable=self.config.settings['mcmc']['disable_tqdm'],
            leave=False,
            position=(self.device_id+1),
            desc = f"JointTableSIM_MCMC device id: {self.device_id}"
        ):

            # Track the epoch training time
            start_time = time.time()
        
            # Run theta sampling
            for j in tqdm(
                range(M),
                disable=True,
                leave=False
            ):

                # Gather all additional values
                auxiliary_values = [V, 
                                gradV, 
                                log_z_inverse, 
                                negative_log_table_likelihood, 
                                sign_sample]
            
                # Take step
                theta_sample, \
                theta_acc, \
                V, \
                gradV, \
                log_z_inverse, \
                negative_log_table_likelihood, \
                sign_sample = self.harris_wilson_mcmc.theta_step(
                            i,
                            theta_sample,
                            log_destination_attraction_sample,
                            table_sample,
                            auxiliary_values
                        )

                # Write to file
                self.write_data(
                    theta_sample = theta_sample,
                    theta_acc = theta_acc,
                    sign_sample = sign_sample
                )
            
            # Run x sampling
            for l in tqdm(
                range(L),
                disable=True,
                leave=False
            ):
                
                # Gather all additional values
                auxiliary_values = [V,
                                gradV,
                                negative_log_table_likelihood]
                # Take step
                log_destination_attraction_sample, \
                log_dest_attract_acc, \
                V, \
                gradV, \
                negative_log_table_likelihood = self.harris_wilson_mcmc.log_destination_attraction_step(
                    theta_sample,
                    self.inputs.data.log_destination_attraction,
                    log_destination_attraction_sample,
                    table_sample,
                    auxiliary_values
                )
                # Write to data
                self.write_data(
                    log_destination_attraction_sample = log_destination_attraction_sample,
                    log_destination_attraction_acc = log_dest_attract_acc,
                )

                self.logger.progress(f"Completed epoch {i+1} / {N}.")

            # Compute new intensity
            log_intensity = self.harris_wilson_mcmc.physics_model.intensity_model.log_intensity(
                log_destination_attraction = log_destination_attraction_sample,
                **dict(zip(self.params_to_learn,theta_sample)),
            )

            # Run table sampling
            for k in tqdm(
                range(self.config.settings['mcmc']['contingency_table']['table_steps']),
                disable=True,
                leave=False
            ):
                
                # Take step
                table_sample, table_accepted = self.ct_mcmc.table_gibbs_step(
                    table_sample,
                    log_intensity.squeeze()
                )

                # Write to file
                self.write_data(
                    table_sample = table_sample,
                    table_acc = table_accepted
                )
            
            # Compute table likelihood for updated table
            negative_log_table_likelihood = self.harris_wilson_mcmc.negative_table_log_likelihood_expanded(
                log_destination_attraction = log_destination_attraction_sample,
                table = table_sample,
                **dict(zip(self.params_to_learn,theta_sample))
            )

            # Write the epoch training time (wall clock time)
            if hasattr(self,'compute_time'):
                self.compute_time.resize(self.compute_time.shape[0] + 1, axis=0)
                self.compute_time[-1] = time.time() - start_time
            
        if self.config.settings.get('export_metadata',True):
            # Write metadata
            dir_path = os.path.join("samples",self.outputs.sweep_id)
            self.outputs.write_metadata(
                dir_path=dir_path,
                filename=f"metadata"
            )
        if self.config.settings.get('export_samples',True):
            # Close h5 data file
            self.outputs.h5file.close()
            # Write log file
            self.outputs.write_log(self.logger)
        
        self.logger.note("Simulation run finished.")

class Table_MCMC(Experiment):
    def __init__(self, config:Config, **kwargs):
        
        # Initalise superclass
        super().__init__(config,**kwargs)

        # Fix random seed
        rng = set_seed(self.seed)

        # Enable garbage collections
        gc.enable()

        # Prepare inputs
        self.inputs = Inputs(
            config=config,
            synthetic_data = False,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )

        # Pass inputs to device
        self.inputs.pass_to_device()

        # Build contingency table
        ct = instantiate_ct(
            config = config,
            logger = self.logger,
            **vars(self.inputs.data)
        )
        # Update table distribution
        config.settings['contingency_table']['distribution_name'] = ct.distribution_name

        # Build contingency table MCMC
        self.ct_mcmc = ContingencyTableMarkovChainMonteCarlo(
            ct = ct,
            rng = rng,
            log_to_console = False,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Set table steps to 1
        self.ct_mcmc.table_steps = 1

        # Get config
        config = getattr(self.ct_mcmc,'config') if hasattr(self.ct_mcmc,'config') else None
        # Initialise intensity
        if (ct is not None) and (ct.ground_truth_table is not None):
            self.logger.info("Using table as ground truth intensity")
            # Use ground truth table to construct intensity
            with np.errstate(invalid='ignore',divide='ignore'):
                self.log_intensity = torch.log(
                    ct.ground_truth_table
                ).to(float32)
            
        else:
            try:
                # Instantiate Spatial Interaction Model
                sim = instantiate_sim(
                    name = config['spatial_interaction_model']['name'],
                    config = config,
                    true_parameters = config['spatial_interaction_model']['parameters'],
                    instance = kwargs.get('instance',''),
                    **vars(self.inputs.data),
                    logger=self.logger
                )
                # Get and remove config
                config = pop_variable(sim,'config')
                self.logger.note("Using SIM model as ground truth intensity")
                
                # Spatial interaction model for intensity
                self.log_intensity = sim.log_intensity(
                    log_true_destination_attraction = sim.log_destination_attraction,
                    alpha = sim.alpha,
                    beta = sim.beta*sim.bmax,
                    grand_total = ct.margins[tuplize(range(ct.ndims()))].item()
                )
            except:
                raise Exception('No ground truth or table provided to construct table intensities.')

        # Update config
        if config is not None:
            self.config = config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep_params=kwargs.get('sweep_params',{}),
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep_params=kwargs.get('sweep_params',{}))

        # Write metadata
        if self.config.settings.get('export_metadata',True):
            self.logger.debug("Writing metadata ...")
            if self.config.settings["sweep_mode"]:
                dir_path = os.path.join("samples",self.outputs.sweep_id)
                filename = 'metadata'
            else:
                dir_path = ""
                filename = 'config'
            
            self.outputs.write_metadata(
                dir_path=dir_path,
                filename=filename
            )
        
        self.logger.note(f"{self.ct_mcmc}")
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")

        self.output_names = ['table']

    def run(self,**kwargs) -> None:

        self.logger.note(f"Running Table MCMC.")

        # Initialise data structures
        self.initialise_data_structures()

        # Store number of samples
        num_epochs = self.config['training']['N']

        # For each epoch
        for e in tqdm(
            range(num_epochs),
            disable=self.tqdm_disabled,
            leave=False,
            position=(self.device_id+1),
            desc = f"NonJointTableSIM_NN device id: {self.device_id}"
        ):

            # Track the epoch training time
            self.start_time = time.time()

            # Count the number of batch items processed
            self.n_processed_steps = 0

            # Sample table
            if e == 0:
                table_sample = self.ct_mcmc.initialise_table(
                    intensity = torch.exp(self.log_intensity)
                )
                accepted = 1
            else:
                table_sample,accepted = self.ct_mcmc.table_gibbs_step(
                    table_prev = table_sample,
                    log_intensity = self.log_intensity
                )
            # Clean and write to file
            _ = self.update_and_export(
                table = table_sample,
                table_acceptance = accepted,
                # Batch size is in training settings
                t = 0,
                data_size = 1,
                **self.config['training']
            )

            self.logger.progress(f"Completed epoch {e+1} / {num_epochs}.")

            # Write the epoch training time (wall clock time)
            if hasattr(self,'compute_time'):
                self.compute_time.resize(self.compute_time.shape[0] + 1, axis=0)
                self.compute_time[-1] = time.time() - self.start_time

        if self.config.settings.get('export_metadata',True):
            # Write metadata
            dir_path = os.path.join("samples",self.outputs.sweep_id)
            self.outputs.write_metadata(
                dir_path=dir_path,
                filename=f"metadata"
            )
        if self.config.settings.get('export_samples',True):
            # Close h5 data file
            self.outputs.h5file.close()
            # Write log file
            self.outputs.write_log(self.logger)
        
        self.logger.note("Simulation run finished.")

class TableSummaries_MCMCConvergence(Experiment):
    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)

        # Setup table
        ct = instantiate_ct(
            config = self.config,
            logger = self.logger,
            **vars(self.inputs.data)
        )
        # Update table distribution
        self.config.settings['inputs']['contingency_table']['distribution_name'] = ct.distribution_name
        # Build spatial interaction model
        sim = instantiate_sim(self.config)
        
        # Cell constraints cannot be handled here due to the different margins generated
        try:
            assert len(self.ct.config.settings['inputs']['contingency_table']['constraints'].get('cells',[])) == 0
        except:
            self.logger.error(self.ct.config.settings['inputs']['contingency_table']['constraints'].get('cells',[]))
            self.logger.error("Cell constraints found in config.")
            raise Exception('TableSummariesMCMCConvergence cannot handle cell constraints due to the different margins generated')
        # Initialise intensities at ground truths
        if (ct is not None) and (ct.table is not None):
            self.logger.info("Using table as ground truth intensity")
            # Use true table to construct intensities
            # with np.errstate(invalid='ignore',divide='ignore'):
            self.true_log_intensities = np.log(
                                            ct.table,
                                            dtype=np.float64
                                        )
            # np.log(ct.table,out=np.ones(np.shape(ct.table),dtype='float32')*(-1e10),where=(ct.table!=0),dtype='float32')
        elif (sim is not None) and (sim.ground_truth_known):
            self.logger.info("Using SIM model as ground truth intensity")
            # Spatial interaction model MCMC
            # Compute intensities
            self.true_log_intensities = sim.log_intensity(
                                            sim.log_true_destination_attraction,
                                            np.array([sim.alpha_true,sim.beta_true*sim.bmax]),
                                            total_flow=ct.margins[tuplize(range(ct.ndims()))].item()
                                        )
        else:
            raise Exception('No ground truth or table provided to construct table intensities.')
        # Replace infinities with very large values
        self.true_log_intensities[np.isinf(self.true_log_intensities) & (self.true_log_intensities < 0) ] = -1e6
        self.true_log_intensities[np.isinf(self.true_log_intensities) & (self.true_log_intensities > 0) ] = 1e6

        # Store number of copies of MCMC sampler to run
        self.K = self.config['K']
        # Create K different margins (data) for each MCMC sampler
        self.samplers = {}

        # Get table probabilities from which 
        # margin probs will be elicited
        self.samplers['0'] = ContingencyTableMarkovChainMonteCarlo(
            ct,
            table_mb=None,
            log_to_console=kwargs.get('log_to_console',True),
            logger = self.logger
        )
        self.samplers['0'].table_steps = 1
        if self.K > 1:
            for k in tqdm(range(1,self.K)):
                # Take copy of table
                ct_copy = deepcopy(ct)
                # Set table steps to 1
                ct_copy.config.settings['mcmc']['contingency_table'].setdefault('table_steps',1)
                # Append sampler to samplers
                self.samplers[str(k)] = ContingencyTableMarkovChainMonteCarlo(
                    ct_copy,
                    table_mb=self.samplers['0'].markov_basis,
                    log_to_console=True,
                    logger = self.logger
                )
                # Initialise fixed margins
                self.samplers[str(k)].sample_constrained_margins(np.exp(self.true_log_intensities))
        
        # Delete duplicate of contingency table and spatial interaction model
        safe_delete(self.ct)
        safe_delete(self.sim)
        # Run garbage collector to release memory
        gc.collect()

        self.output_names = ['table']

    def initialise_parameters(self):
            
        # Initialise table
        tables0 = []
        for k in range(self.K):
            # Randomly sample initial table
            tables0.append(self.samplers[str(k)].initialise_table(np.exp(self.true_log_intensities)))

        # Update metadata initially
        self.update_metadata(
            0,
            batch_counter=0,
            print_flag=False,
            update_flag=True
        )
        return tables0

    
    def run(self,**kwargs) -> None:

        # Time run
        self.start_time = time.time()

        # Fix random seed
        set_seed(self.seed)

        # Initialise table samples
        self.tables = self.initialise_parameters()

        # Initiliase means by MCMC iteration
        table_mean = np.mean(np.array(self.tables,dtype='float32'),axis=0)
        
        # Initialise error norms
        table_norm = apply_norm(
            tab=table_mean[np.newaxis,:],
            tab0=np.exp(self.true_log_intensities,dtype='float32'),
            name=self.config['norm'],
            **self.config
        )
        
        
        # Store number of samples
        # Total samples for joint posterior
        N = self.samplers['0'].ct.config.settings['mcmc']['N']

        print('Running MCMC')
        for i in tqdm(
            range(1,N),
            disable=self.config['disable_tqdm'],
            leave=False,
            position=(self.device_id+1),
            desc = f"TableSummaries_MCMCConvergence device id: {self.device_id}"
        ):
            # Run MCMC for one step in all chains in ensemble
            # Do it in parallel
            if self.samplers['0'].n_workers > 1:
                self.tables = Parallel(n_jobs=self.samplers['0'].n_workers)(
                    delayed(self.samplers[str(k)].table_gibbs_step)(
                        self.tables[k],self.true_log_intensities
                    )[0] for k in range(self.K)
                )
            # Do it in sequence
            else:
                for k in range(self.K):
                    self.tables[k] = self.samplers[str(k)].table_gibbs_step(
                            self.tables[k],
                            self.true_log_intensities
                    )[0].astype('float32')
                    # try:
                    #     assert self.samplers[str(k)].ct.table_admissible(res[k][2])
                    # except:
                    #     raise Exception(f'Inadmissible table for sampler {k}')

            # Take ensemble mean
            ensemble_table_mean = np.mean(self.tables,axis=0)

            # Update MCMC running table mean
            table_mean = running_average_multivariate(
                ensemble_table_mean,
                table_mean,
                i
            )
            # Compute error norm
            table_norm = np.concatenate(
                (table_norm,
                apply_norm(
                    tab=table_mean[np.newaxis,:],
                    tab0=np.exp(self.true_log_intensities,dtype='float32'),
                    name=self.config['norm'],
                    **self.config
                )), 
                axis=0
            )

            if (self.config['print_statements']) and (i in [int(p*N) for p in np.arange(0,1,0.1)]):
                print('norm')
                print(table_norm[-1])

        # Update metadata
        self.update_metadata(
            (i+1),
            batch_counter=0,
            print_flag=True,
            update_flag=True
        )

        # Unfix random seed
        set_seed(None)

        self.logger.info(f"Experimental results have been compiled.")

        self.results = [{
            "samples":{
                "tableerror":table_norm,
            }
        }]

        return self.results[-1]

class SIM_NN(Experiment):
    def __init__(self, config:Config, **kwargs):
        
        # Initalise superclass
        super().__init__(config,**kwargs)

        # Fix random seed
        rng = set_seed(self.seed)

        # Enable garbage collections
        gc.enable()

        # Prepare inputs
        self.inputs = Inputs(
            config=config,
            synthetic_data = False,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )

        # Pass inputs to device
        self.inputs.pass_to_device()

        # Instantiate Spatial Interaction Model
        self.logger.note("Initializing the spatial interaction model ...")

        sim = instantiate_sim(
            name = config['spatial_interaction_model']['name'],
            config = config,
            true_parameters = config['spatial_interaction_model']['parameters'],
            instance = kwargs.get('instance',''),
            **vars(self.inputs.data),
            logger=self.logger
        )
        # Get and remove config
        config = pop_variable(sim,'config')

        # Build Harris Wilson model
        self.logger.note("Initializing the Harris Wilson physics model ...")
        harris_wilson_model = HarrisWilson(
            intensity_model = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get and remove config
        config = pop_variable(harris_wilson_model,'config')
        
        # Set up the neural net
        self.logger.note("Initializing the neural net ...")
        neural_network = NeuralNet(
            input_size=self.inputs.data.destination_attraction_ts.shape[-1],
            output_size=len(config['inputs']['to_learn']),
            **config['neural_network']['hyperparameters'],
            logger = self.logger
        ).to(self.device)

        # Instantiate harris and wilson neural network model
        self.logger.note("Initializing the Harris Wilson Neural Network model ...")
        self.harris_wilson_nn = HarrisWilson_NN(
            rng = rng,
            config = config,
            neural_net = neural_network,
            loss_function = config['neural_network'].pop('loss_function'),
            physics_model = harris_wilson_model,
            write_every = self._write_every,
            write_start = self._write_start,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get config
        config = getattr(self.harris_wilson_nn,'config') if hasattr(self.harris_wilson_nn,'config') else None

        # Update config
        if config is not None:
            self.config = config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep_params=kwargs.get('sweep_params',{}),
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep_params=kwargs.get('sweep_params',{}))

        # Write metadata
        if self.config.settings.get('export_metadata',True):
            self.logger.debug("Writing metadata ...")
            if self.config.settings["sweep_mode"]:
                dir_path = os.path.join("samples",self.outputs.sweep_id)
                filename = 'metadata'
            else:
                dir_path = ""
                filename = 'config'

            self.outputs.write_metadata(
                dir_path=dir_path,
                filename=filename
            )
        
        self.logger.note(f"{self.harris_wilson_nn}")
        
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")

        self.output_names = ['log_destination_attraction','theta','loss']
        self.theta_names = config['inputs']['to_learn']
        
    def run(self,**kwargs) -> None:

        self.logger.note(f"Running Neural Network training of Harris Wilson model.")

        # Initialise data structures
        self.initialise_data_structures()
        
        # Store number of samples
        num_epochs = self.config['training']['N']
        # For each epoch
        for e in tqdm(
            range(num_epochs),
            disable=self.tqdm_disabled,
            leave=False,
            position=(self.device_id+1),
            desc = f"SIM_NN device id: {self.device_id}"
        ):

            # Track the epoch training time
            start_time = time.time()

            # Track the training loss
            loss_sample = torch.tensor(0.0, requires_grad=True)

            # Count the number of batch items processed
            self.n_processed_steps = 0
            
            # For each epoch
            for t, training_data in enumerate(self.inputs.data.destination_attraction_ts):
                
                # Perform neural net training
                loss_sample, \
                theta_sample, \
                log_destination_attraction_sample = self.harris_wilson_nn.epoch_time_step(
                    loss = loss_sample,
                    experiment = self,
                    nn_data = training_data,
                    dt = self.config['harris_wilson_model'].get('dt',0.001),
                )
                
                # Add axis to every sample to ensure compatibility 
                # with the functions used below
                # theta_sample_expanded = torch.unsqueeze(theta_sample,0)
                # log_destination_attraction_sample_expanded = log_destination_attraction_sample.unsqueeze(0).unsqueeze(0)

                # Clean and write to file
                loss_sample = self.update_and_export(
                    loss = loss_sample,
                    theta = theta_sample,
                    log_destination_attraction = log_destination_attraction_sample,
                    # Batch size is in training settings
                    t = t,
                    data_size = len(training_data),
                    **self.config['training']
                )

                self.logger.progress(f"Completed epoch {e+1} / {num_epochs}.")

                # Write the epoch training time (wall clock time)
                if hasattr(self,'compute_time'):
                    self.compute_time.resize(self.compute_time.shape[0] + 1, axis=0)
                    self.compute_time[-1] = time.time() - start_time
        
        if self.config.settings.get('export_metadata',True):
            # Write metadata
            dir_path = os.path.join("samples",self.outputs.sweep_id)
            self.outputs.write_metadata(
                dir_path=dir_path,
                filename=f"metadata"
            )
        if self.config.settings.get('export_samples',True):
            # Close h5 data file
            self.outputs.h5file.close()
            # Write log file
            self.outputs.write_log(self.logger)
        
        self.logger.note("Simulation run finished.")

class NonJointTableSIM_NN(Experiment):
    def __init__(self, config:Config, **kwargs):
        
        # Initalise superclass
        super().__init__(config,**kwargs)

        # Fix random seed
        rng = set_seed(self.seed)

        # Enable garbage collections
        gc.enable()

        # Prepare inputs
        self.inputs = Inputs(
            config=config,
            synthetic_data = False,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        
        # Pass inputs to device
        self.inputs.pass_to_device()

        # Instantiate Spatial Interaction Model
        self.logger.note("Initializing the spatial interaction model ...")

        sim = instantiate_sim(
            name = config['spatial_interaction_model']['name'],
            config = config,
            true_parameters = config['spatial_interaction_model']['parameters'],
            instance = kwargs.get('instance',''),
            **vars(self.inputs.data),
            logger=self.logger
        )
        # Get and remove config
        config = pop_variable(sim,'config')

        # Build Harris Wilson model
        self.logger.note("Initializing the Harris Wilson physics model ...")
        harris_wilson_model = HarrisWilson(
            intensity_model = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get and remove config
        config = pop_variable(harris_wilson_model,'config')
        
        # Set up the neural net
        self.logger.note("Initializing the neural net ...")
        neural_network = NeuralNet(
            input_size=self.inputs.data.destination_attraction_ts.shape[-1],
            output_size=len(config['inputs']['to_learn']),
            **config['neural_network']['hyperparameters'],
            logger = self.logger
        ).to(self.device)

        # Instantiate harris and wilson neural network model
        self.logger.note("Initializing the Harris Wilson Neural Network model ...")
        self.harris_wilson_nn = HarrisWilson_NN(
            rng = rng,
            config = config,
            neural_net = neural_network,
            loss_function = config['neural_network'].pop('loss_function'),
            physics_model = harris_wilson_model,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get config
        config = getattr(self.harris_wilson_nn,'config') if hasattr(self.harris_wilson_nn,'config') else None

        # Build contingency table
        ct = instantiate_ct(
            config = config,
            logger = self.logger,
            **vars(self.inputs.data)
        )
        # Build contingency table MCMC
        self.ct_mcmc = ContingencyTableMarkovChainMonteCarlo(
            ct = ct,
            rng = rng,
            log_to_console = False,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )

        # Update config
        self.config = config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep_params=kwargs.get('sweep_params',{}),
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep_params=kwargs.get('sweep_params',{}))

        # Write metadata
        if self.config.settings.get('export_metadata',True):
            self.logger.debug("Writing metadata ...")
            if self.config.settings["sweep_mode"]:
                dir_path = os.path.join("samples",self.outputs.sweep_id)
                filename = 'metadata'
            else:
                dir_path = ""
                filename = 'config'
            
            self.outputs.write_metadata(
                dir_path=dir_path,
                filename=filename
            )
        
        self.logger.note(f"{self.harris_wilson_nn}")
        self.logger.note(f"{self.ct_mcmc}")
        
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")

        self.output_names = ['log_destination_attraction','theta','loss', 'table']
        self.theta_names = config['inputs']['to_learn']
        
    def run(self,**kwargs) -> None:

        self.logger.note(f"Running Disjoint Table Inference and Neural Network training of Harris Wilson model.")

        # Initialise data structures
        self.initialise_data_structures()
        
        # Store number of samples
        num_epochs = self.config['training']['N']

        # For each epoch
        for e in tqdm(
            range(num_epochs),
            disable=self.tqdm_disabled,
            leave=False,
            position=(self.device_id+1),
            desc = f"Table MCMC device id: {self.device_id}"
        ):

            # Track the epoch training time
            start_time = time.time()
            
            # Track the training loss
            loss_sample = torch.tensor(0.0, requires_grad=True)

            # Count the number of batch items processed
            self.n_processed_steps = 0
            
            # Process the training set elementwise, updating the loss after batch_size steps
            for t, training_data in enumerate(self.inputs.data.destination_attraction_ts):

                # Perform neural net training
                loss_sample, \
                theta_sample, \
                log_destination_attraction_sample = self.harris_wilson_nn.epoch_time_step(
                    loss = loss_sample,
                    experiment = self,
                    nn_data = training_data,
                    dt = self.config['harris_wilson_model'].get('dt',0.001),
                )
            
                # Add axis to every sample to ensure compatibility 
                # with the functions used below
                theta_sample_expanded = torch.unsqueeze(theta_sample,0)
                log_destination_attraction_sample_expanded = log_destination_attraction_sample.unsqueeze(0).unsqueeze(0)

                # Compute log intensity
                log_intensity = self.harris_wilson_nn.physics_model.intensity_model.log_intensity(
                    log_destination_attraction = log_destination_attraction_sample_expanded,
                    grand_total = self.ct_mcmc.ct.data.margins[tuplize(range(ndims(self.ct_mcmc.ct)))],
                    **dict(zip(self.harris_wilson_nn.parameters_to_learn,theta_sample_expanded.split(1,dim=1)))
                ).squeeze()

                # Sample table
                if e == 0:
                    table_sample = self.ct_mcmc.initialise_table(
                        intensity = torch.exp(log_intensity)
                    )
                    accepted = 1
                else:
                    table_sample,accepted = self.ct_mcmc.table_gibbs_step(
                        table_prev = table_sample,
                        log_intensity = log_intensity
                    )

                # Clean and write to file
                loss_sample = self.update_and_export(
                    loss = loss_sample,
                    theta = theta_sample,
                    log_destination_attraction = log_destination_attraction_sample,
                    table = table_sample,
                    table_acceptance = accepted,
                    # Batch size is in training settings
                    t = t,
                    data_size = len(training_data),
                    **self.config['training']
                )
            
            self.logger.progress(f"Completed epoch {e+1} / {num_epochs}.")

            # Write the epoch training time (wall clock time)
            if hasattr(self,'compute_time'):
                self.compute_time.resize(self.compute_time.shape[0] + 1, axis=0)
                self.compute_time[-1] = time.time() - start_time

        if self.config.settings.get('export_metadata',True):
            # Write metadata
            dir_path = os.path.join("samples",self.outputs.sweep_id)
            self.outputs.write_metadata(
                dir_path=dir_path,
                filename=f"metadata"
            )
        if self.config.settings.get('export_samples',True):
            # Close h5 data file
            self.outputs.h5file.close()
            # Write log file
            self.outputs.write_log(self.logger)
        
        self.logger.note("Simulation run finished.")

class JointTableSIM_NN(Experiment):
    def __init__(self, config:Config, **kwargs):

        # Initalise superclass
        super().__init__(config,**kwargs)

        # Fix random seed
        rng = set_seed(self.seed)

        # Enable garbage collections
        gc.enable()

        # Prepare inputs
        self.inputs = Inputs(
            config=config,
            synthetic_data = False,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )

        # Pass inputs to device
        self.inputs.pass_to_device()

        # Instantiate Spatial Interaction Model
        self.logger.note("Initializing the spatial interaction model ...")

        sim = instantiate_sim(
            name = config['spatial_interaction_model']['name'],
            config = config,
            true_parameters = config['spatial_interaction_model']['parameters'],
            instance = kwargs.get('instance',''),
            **vars(self.inputs.data),
            logger=self.logger
        )
        # Get and remove config
        config = pop_variable(sim,'config')

        # Build Harris Wilson model
        self.logger.note("Initializing the Harris Wilson physics model ...")
        harris_wilson_model = HarrisWilson(
            intensity_model = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get and remove config
        config = pop_variable(harris_wilson_model,'config')
        
        # Set up the neural net
        self.logger.note("Initializing the neural net ...")
        neural_network = NeuralNet(
            input_size=self.inputs.data.destination_attraction_ts.shape[-1],
            output_size=len(config['inputs']['to_learn']),
            **config['neural_network']['hyperparameters'],
            logger = self.logger
        ).to(self.device)

        # Instantiate harris and wilson neural network model
        self.logger.note("Initializing the Harris Wilson Neural Network model ...")
        self.harris_wilson_nn = HarrisWilson_NN(
            rng = rng,
            config = config,
            neural_net = neural_network,
            loss_function = config['neural_network'].pop('loss_function'),
            physics_model = harris_wilson_model,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get config
        config = getattr(self.harris_wilson_nn,'config') if hasattr(self.harris_wilson_nn,'config') else None

        # Build contingency table
        ct = instantiate_ct(
            config = config,
            logger = self.logger,
            **vars(self.inputs.data)
        )
        # Build contingency table MCMC
        self.ct_mcmc = ContingencyTableMarkovChainMonteCarlo(
            ct = ct,
            rng = rng,
            log_to_console = False,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        
        # Update config
        self.config = config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep_params=kwargs.get('sweep_params',{}),
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep_params=kwargs.get('sweep_params',{}))

        # Write metadata
        if self.config.settings.get('export_metadata',True):
            self.logger.debug("Writing metadata ...")
            if self.config.settings["sweep_mode"]:
                dir_path = os.path.join("samples",self.outputs.sweep_id)
                filename = 'metadata'
            else:
                dir_path = ""
                filename = 'config'
            
            self.outputs.write_metadata(
                dir_path=dir_path,
                filename=filename
            )
        
        self.logger.note(f"{self.harris_wilson_nn}")
        self.logger.info(f"{self.ct_mcmc}")
        
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")

        self.output_names = ['log_destination_attraction','theta','loss', 'table']
        self.theta_names = config['inputs']['to_learn']
        
    def run(self,**kwargs) -> None:

        self.logger.note(f"Running Joint Table Inference and Neural Network training of Harris Wilson model.")

        # Initialise data structures
        self.initialise_data_structures()
        
        # Store number of samples
        num_epochs = self.config['training']['N']

        # For each epoch
        for e in tqdm(
            range(num_epochs),
            disable=self.tqdm_disabled,
            leave=False,
            position=(self.device_id+1),
            desc = f"JointTableSIM_NN device id: {self.device_id}"
        ):

            # Track the epoch training time
            start_time = time.time()
            
            # Track the training loss
            loss_sample = torch.tensor(0.0, requires_grad=True)

            # Count the number of batch items processed
            self.n_processed_steps = 0

            # Process the training set elementwise, updating the loss after batch_size steps
            for t, training_data in enumerate(self.inputs.data.destination_attraction_ts):

                # Perform neural net training
                loss_sample, \
                theta_sample, \
                log_destination_attraction_sample = self.harris_wilson_nn.epoch_time_step(
                    loss = loss_sample,
                    experiment = self,
                    nn_data = training_data,
                    dt = self.config['harris_wilson_model'].get('dt',0.001),
                )
            
                # Add axis to every sample to ensure compatibility 
                # with the functions used below
                theta_sample_expanded = torch.unsqueeze(theta_sample,0)
                log_destination_attraction_sample_expanded = log_destination_attraction_sample.unsqueeze(0).unsqueeze(0)

                # Compute log intensity
                log_intensity = self.harris_wilson_nn.physics_model.intensity_model.log_intensity(
                    log_destination_attraction = log_destination_attraction_sample_expanded,
                    grand_total = self.ct_mcmc.ct.data.margins[tuplize(range(ndims(self.ct_mcmc.ct)))],
                    **dict(zip(self.harris_wilson_nn.parameters_to_learn,theta_sample_expanded.split(1,dim=1)))
                ).squeeze()
                
                # Sample table
                if e == 0:
                    table_sample = self.ct_mcmc.initialise_table(
                        intensity = torch.exp(log_intensity)
                    )
                    accepted = 1
                else:
                    table_sample,accepted = self.ct_mcmc.table_gibbs_step(
                        table_prev = table_sample,
                        log_intensity = log_intensity
                    )
                self.logger.progress('table loss update')
                # Update table loss
                loss_sample += self.ct_mcmc.table_loss_function(
                    log_intensity = log_intensity,
                    table = table_sample
                )

                # Clean and write to file
                loss_sample = self.update_and_export(
                    loss = loss_sample,
                    theta = theta_sample,
                    log_destination_attraction = log_destination_attraction_sample,
                    table = table_sample,
                    table_acceptance = accepted,
                    # Batch size is in training settings
                    t = t,
                    data_size = len(training_data),
                    **self.config['training']
                )
            self.logger.progress(f"Completed epoch {e+1} / {num_epochs}.")

            # Write the epoch training time (wall clock time)
            if hasattr(self,'compute_time'):
                self.compute_time.resize(self.compute_time.shape[0] + 1, axis=0)
                self.compute_time[-1] = time.time() - start_time
                
        if self.config.settings.get('export_metadata',True):
            # Write metadata
            dir_path = os.path.join("samples",self.outputs.sweep_id)
            self.outputs.write_metadata(
                dir_path=dir_path,
                filename=f"metadata"
            )
        if self.config.settings.get('export_samples',True):
            # Close h5 data file
            self.outputs.h5file.close()
            # Write log file
            self.outputs.write_log(self.logger)
        
        self.logger.note("Simulation run finished.")

class ExperimentSweep():

    def __init__(
            self,
            config:Config,
            **kwargs
        ):

        # Setup logger
        self.logger = setup_logger(
            __name__,
            console_handler_level = config.level,
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']

        self.logger.info(f"Performing parameter sweep")

        # Get config
        self.config = config

        # Store one datetime
        # for all sweeps
        self.config.settings['datetime'] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        # Load schema
        self.config.load_schema()

        # Initialise experiments 
        # (each one corresponds to one parameter sweep)
        self.experiment_configs = []

        # Store number of workers
        self.n_workers = self.config.settings['inputs'].get("n_workers",1)

        # Parse sweep configurations
        self.sweep_params = self.config.parse_sweep_params()

        # Temporarily disable sample output writing
        export_samples = deepcopy(self.config['experiments'][0]['export_samples'])
        deep_update(self.config.settings,'export_samples',False)

        dir_range = []
        if isinstance(self.config['inputs']['dataset'],dict):
            dir_range = deepcopy(self.config['inputs']['dataset']['sweep']['range'])
            self.config['inputs']['dataset'] = dir_range[0]

        self.outputs = Outputs(
            self.config,
            sweep_params=kwargs.get('sweep_params',{}),
            logger = self.logger
        )
        # Prepare writing to file
        self.outputs.open_output_file(kwargs.get('sweep_params',{}))
        # Enable it again
        deep_updates(self.config.settings,{'export_samples':export_samples})

        # Write metadata
        if self.config.settings['experiments'][0].get('export_metadata',True):
            self.outputs.write_metadata(
                dir_path='',
                filename=f"config"
            )

        if len(dir_range) > 0:
            self.config['inputs']['dataset'] = dir_range
        self.logger.note(f"ExperimentSweep: {self.outputs.experiment_id} prepared")

    
    def __repr__(self) -> str:
        return "ParameterSweep("+(self.experiment.__repr__())+")"

    def __str__(self) -> str:
        return f"""
            Sweep key paths: {self.sweep_key_paths}
        """


    def run(self,**kwargs):
        # Create sweep configurations
        sweep_configurations, \
        param_sizes_str, \
        total_size_str = self.config.prepare_sweep_configurations(self.sweep_params)

        self.logger.info(f"{self.outputs.experiment_id}")
        self.logger.info(f"Parameter space size: {param_sizes_str}. Total = {total_size_str}.")
        self.logger.info(f"Preparing configs...")
        
        # For each configuration update experiment config 
        # and instantiate new experiment
        self.prepare_experiments_sequential(sweep_configurations)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Decide whether to run sweeps in parallel or not
            if self.n_workers > 1:
                self.run_parallel()
                # self.run_concurrent()
            else:
                self.run_sequential()
        
    def prepare_experiments_sequential(self,sweep_configurations):
        for sval in tqdm(
            sweep_configurations,
            total=len(sweep_configurations),
            leave=False,
            desc='Preparing experiments'
        ):
            # Create new config
            new_config = deepcopy(self.config)
            # Deactivate sweep             
            new_config.settings["sweep_mode"] = False
            # Deactivate logging
            self.logger.setLevels(
                console_level='ERROR',
                file_level='DEBUG'
            )
            
            # Activate sample exports
            new_config.settings['export_samples'] = True
            # Create sweep dictionary
            sweep = {}
            # Update config
            i = 0
            for value in self.sweep_params['isolated'].values():
                new_config.path_set(
                    new_config,
                    sval[i],
                    value['path']
                )
                # Update current sweep
                sweep[value['var']] = sval[i]
                i += 1
            for sweep_group in self.sweep_params['coupled'].values():
                for value in sweep_group.values():
                    new_config.path_set(
                        new_config,
                        sval[i],
                        value['path']
                    )
                    # Update current sweep
                    sweep[value['var']] = sval[i]
                    i += 1
            # Append to experiments
            self.experiment_configs.append({"config":new_config,"sweep":sweep})
    
    def instantiate_and_run(self,instance_num:int,config_and_sweep:dict,semaphore=None,counter=None):
    
        self.logger.info(f'Instance = {str(instance_num)} START')
        if semaphore is not None:
            semaphore.acquire()
        # Create new experiment
        new_experiment = instantiate_experiment(
            experiment_type=config_and_sweep['config'].settings['experiment_type'],
            config=config_and_sweep['config'],
            sweep_params=config_and_sweep['sweep'],
            instance=str(instance_num),
            experiment_id=self.outputs.experiment_id,
            tqdm_disabled=False,
            device_id=(instance_num%self.n_workers),
            logger=self.logger,
        )
        self.logger.debug('New experiment set up')
        new_experiment.run()
        if semaphore is not None:
            semaphore.release()
        if counter is not None:
            with counter.get_lock():
                counter.value += 1
        self.logger.info(f'Instance = {str(instance_num)} DONE')

    def run_sequential(self):
        for instance,conf_and_sweep in tqdm(
            enumerate(self.experiment_configs),
            total=len(self.experiment_configs),
            desc='Running sweeps in sequence',
            leave=False,
            position=0
        ):
            self.instantiate_and_run(
                instance_num=instance,
                config_and_sweep=conf_and_sweep,
                semaphore=None,
                counter=None
            )
    
    def run_parallel(self):
        # Run experiments in parallel
        semaphore = mp.Semaphore(self.n_workers)
        counter = mp.Value('i', 0, lock=True)
        processes = []
        with tqdm(
            total=len(self.experiment_configs), 
            desc='Running sweeps in parallel',
            leave=False,
            position=0
        ) as pbar:
            for instance,conf_and_sweep in enumerate(self.experiment_configs):
                p = mp.Process(
                    target=self.instantiate_and_run, 
                    args=(
                        instance, 
                        conf_and_sweep, 
                        semaphore,
                        counter
                    )
                )
                processes.append(p)

            for p in processes:
                p.start()

            while counter.value < len(self.experiment_configs):
                pbar.update(counter.value - pbar.n)

            for p in processes:
                p.join()

            for p in processes:
                p.close()
            
    def run_process(self,process):
        process.run()
        return True
