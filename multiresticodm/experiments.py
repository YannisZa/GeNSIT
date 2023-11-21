import sys
import time
import warnings
import concurrent.futures as concurrency

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
from multiresticodm.contingency_table import instantiate_ct
from multiresticodm.harris_wilson_model import HarrisWilson
from multiresticodm.spatial_interaction_model import instantiate_sim
from multiresticodm.harris_wilson_model_neural_net import NeuralNet, HarrisWilson_NN
from multiresticodm.harris_wilson_model_mcmc import instantiate_harris_wilson_mcmc
from multiresticodm.contingency_table_mcmc import ContingencyTableMarkovChainMonteCarlo

# Suppress scientific notation
np.set_printoptions(suppress=True)

def instantiate_experiment(experiment_type:str,config:Config,**kwargs):
    if hasattr(sys.modules[__name__], experiment_type):
        # Get whether sweep is active
        if config.sweep_mode(settings = config.settings):
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
            console_level = level, 
            
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
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            console_level = level, 
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger lever
        self.logger.setLevels(
            console_level = level
        )
        
        # Enable garbage collections
        # gc.enable()

        self.logger.debug(f"{self}")
        # Make sure you are reading a config
        if not isinstance(config,Config):
            raise Exception(f'config provided has invalid type {type(config)}')

        # Store config
        self.config = config
        
        # Update config with current timestamp ( but do not overwrite)
        datetime_results = list(deep_get(key='datetime',value=self.config.settings))
        if len(datetime_results) > 0:
            deep_update(self.config.settings, key='datetime', val=datetime.now().strftime("%d_%m_%Y_%H_%M_%S"), overwrite=False)
        else:
            self.config['datetime'] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        # Inherit experiment id from parameter sweep (if it exists)
        # This will be used to create a unique output directory for every sweep
        self.sweep_experiment_id = kwargs.get('experiment_id',None)
        self.outputs_base_dir = kwargs.get('base_dir',None)

        # Update current config
        # self.config = self.sim.config.update_recursively(self.config,updated_config,overwrite=True)
        # print(self.config)
        # Decide how often to print statemtents
        self.store_progress = self.config.get('store_progress',1.0)
        self.print_percentage = min(0.05,self.store_progress)

        # Update seed if specified
        self.seed = None
        if "seed" in self.config['inputs'].keys():
            self.seed = int(self.config['inputs']["seed"])
            self.logger.info(f"Updated seed to {self.seed}")

        # Get device name
        self.device = self.config['inputs']['device']
        # Get device id
        self.device_id = kwargs.get('device_id',0)
        # print('current_device',torch.cuda.current_device())

        # Disable tqdm if needed
        self.tqdm_disabled = self.config['experiments'][0].get('disable_tqdm',True)

        # Count the number of gradient descent steps
        self._time = 0
        self._write_every = self.config['outputs'].get('write_every',1)
        self._write_start = self.config['outputs'].get('write_start',1)
        
        # Get experiment data
        self.logger.info(f"Experiment {self.config['experiment_type']} has been set up.")

    def run(self,**kwargs) -> None:
        pass

    def load(self):
        if self.config['inputs'].get('load_experiment',''):
            try:
                # Load config
                config = Config(
                    path = self.config['inputs'].get('load_experiment',''),
                    logger = self.logger
                )
            except:
                return config
            return None
        return None            

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
    
    def write_metadata(self):
        if self.config.settings.get('export_metadata',True):
            self.logger.debug("Writing metadata ...")
            dir_path = ""
            if self.config.settings["sweep_mode"] or len(self.outputs.sweep_id) == 0:
                filename='config'
            else:
                filename='metadata'
            
            if len(self.outputs.sweep_id) > 0:
                dir_path = os.path.join("samples",self.outputs.sweep_id)
            
            self.outputs.write_metadata(
                dir_path=dir_path,
                filename=filename
            )
    
    def close_outputs(self):
        if self.config.settings.get('export_samples',True):
            # Write log file
            self.outputs.write_log()
        # Close h5 data file
        self.outputs.h5file.close()

    def initialise_parameters(self,param_names:list=[]):
        
        initialisations = {}
        # Get dimensions
        dims = self.config['inputs']['dims']
        
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
                        try:
                            theta_sample = self.harris_wilson_nn.physics_model.params_to_learn
                        except:
                            self.logger.warning("Theta could not be initialised.")
                self.params_to_learn = list(theta_sample.keys())
                initialisations['theta_sample'] = torch.tensor(list(theta_sample.values()),dtype=float32,device=self.device)
                
            elif param == 'log_destination_attraction':
                # Arbitrarily initialise destination attraction
                initialisations['log_destination_attraction_sample'] = torch.log(
                    torch.repeat_interleave(
                        torch.tensor(1./dims['destination']),
                        dims['destination']
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
            # Flag for loading experiment data
            load_experiment = self.config.settings['load_data']

            # Get dimensions
            dims = self.config['inputs']['dims']
            
            # Setup neural net loss
            if 'loss' in self.output_names:
                for loss_name in list(self.harris_wilson_nn.loss_functions.keys())+['total']:
                    # Setup chunked dataset to store the state data in
                    if loss_name in self.outputs.h5group and not load_experiment:
                        # Delete current dataset
                        safe_delete(getattr(self,loss_name))
                    if loss_name not in self.outputs.h5group:
                        setattr(
                            self,
                            loss_name,
                            self.outputs.h5group.create_dataset(
                                loss_name,
                                (0,),
                                maxshape=(None,),
                                chunks=True,
                                compression=3,
                            )
                        )
                        getattr(self,loss_name).attrs['dim_names'] = XARRAY_SCHEMA['loss']['coords']
                        getattr(self,loss_name).attrs['coords_mode__time'] = 'start_and_step'
                        getattr(self,loss_name).attrs['coords__time'] = [self._write_start, self._write_every]
                    else:
                        setattr(
                            self,
                            loss_name,
                            self.outputs.h5group[loss_name]
                        )



            # Setup sampled/predicted log destination attractions
            if 'log_destination_attraction' in self.output_names:
                if 'log_destination_attraction' in self.outputs.h5group and not load_experiment:
                    # Delete current dataset
                    safe_delete(getattr(self,'log_destination_attraction'))
                if 'log_destination_attraction' not in self.outputs.h5group:
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
                else:
                    setattr(
                        self,
                        'log_destination_attraction',
                        self.outputs.h5group['log_destination_attraction']
                    )
            # Setup computation time
            if 'computation_time' in self.output_names:
                if 'compute_time' in self.outputs.h5group and not load_experiment:
                    # Delete current dataset
                    safe_delete(getattr(self,'compute_time'))
                if 'compute_time' not in self.outputs.h5group:
                    self.compute_time = self.outputs.h5group.create_dataset(
                        'computation_time',
                        (0,),
                        maxshape=(None,),
                        chunks=True,
                        compression=3,
                    )
                    self.compute_time.attrs['dim_names'] = XARRAY_SCHEMA['computation_time']['coords']
                    self.compute_time.attrs['coords_mode__epoch'] = 'trivial'
                else:
                    setattr(
                        self,
                        'computation_time',
                        self.outputs.h5group['computation_time']
                    )

            # Setup sampled/predicted theta
            if 'theta' in self.output_names:
                self.thetas = []
                for p_name in self.config['inputs']['to_learn']:
                    if p_name in self.outputs.h5group and not load_experiment:
                        # Delete current dataset
                        safe_delete(getattr(self,p_name))
                    if p_name not in self.outputs.h5group:
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
                    else:
                        dset = self.outputs.h5group[p_name]
                    # Append to thetas
                    self.thetas.append(dset)
            
            # Setup sampled signs
            if 'sign' in self.output_names:
                if 'sign' in self.outputs.h5group and not load_experiment:
                    # Delete current dataset
                    safe_delete(getattr(self,'sign'))
                if 'sign' not in self.outputs.h5group:
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
                else:
                    setattr(
                        self,
                        'sign',
                        self.outputs.h5group['sign']
                    )
            # Setup sampled tables
            if 'table' in self.output_names:
                if 'table' in self.outputs.h5group and not load_experiment:
                    # Delete current dataset
                    safe_delete(getattr(self,'table'))
                if 'table' not in self.outputs.h5group:
                    self.tables = self.outputs.h5group.create_dataset(
                        "table",
                        (0,*unpack_dims(dims,time_dims=False)),
                        maxshape=(None,*unpack_dims(dims,time_dims=False)),
                        chunks=True,
                        compression=3,
                    )
                    self.tables.attrs["dim_names"] = ["origin","destination","N"]
                    self.tables.attrs["coords_mode__time"] = "start_and_step"
                else:
                    setattr(
                        self,
                        'table',
                        self.outputs.h5group['table']
                    )
            # Setup acceptances
            if 'theta_acc' in self.output_names:
                if 'theta_acc' in self.outputs.h5group and not load_experiment:
                    # Delete current dataset
                    safe_delete(getattr(self,'theta_acc'))
                if 'theta_acc' not in self.outputs.h5group:
                    # Setup chunked dataset to store the state data in
                    self.theta_acc = self.outputs.h5group.create_dataset(
                        'theta_acc',
                        (0,),
                        maxshape=(None,),
                        chunks=True,
                        compression=3,
                    )
                    self.theta_acc.attrs['dim_names'] = XARRAY_SCHEMA['theta_acc']['coords']
                    self.theta_acc.attrs['coords_mode__time'] = 'start_and_step'
                    self.theta_acc.attrs['coords__time'] = [self._write_start, self._write_every]
                else:
                    setattr(
                        self,
                        'theta_acc',
                        self.outputs.h5group['theta_acc']
                    )
            if 'log_destination_attraction_acc' in self.output_names:
                if 'log_destination_attraction_acc' in self.outputs.h5group and not load_experiment:
                    # Delete current dataset
                    safe_delete(getattr(self,'log_destination_attraction_acc'))
                if 'log_destination_attraction_acc' not in self.outputs.h5group:
                    # Setup chunked dataset to store the state data in
                    self.log_destination_attraction_acc = self.outputs.h5group.create_dataset(
                        'log_destination_attraction_acc',
                        (0,),
                        maxshape=(None,),
                        chunks=True,
                        compression=3,
                    )
                    self.log_destination_attraction_acc.attrs['dim_names'] = XARRAY_SCHEMA['log_destination_attraction_acc']['coords']
                    self.log_destination_attraction_acc.attrs['coords_mode__time'] = 'start_and_step'
                    self.log_destination_attraction_acc.attrs['coords__time'] = [self._write_start, self._write_every]
                else:
                    setattr(
                        self,
                        'log_destination_attraction_acc',
                        self.outputs.h5group['log_destination_attraction_acc']
                    )
            if 'table_acc' in self.output_names:
                if 'table_acc' in self.outputs.h5group and not load_experiment:
                    # Delete current dataset
                    safe_delete(getattr(self,'table_acc'))
                if 'table_acc' not in self.outputs.h5group:
                    # Setup chunked dataset to store the state data in
                    self.table_acc = self.outputs.h5group.create_dataset(
                        'table_acceptance',
                        (0,),
                        maxshape=(None,),
                        chunks=True,
                        compression=3,
                    )
                    self.table_acc.attrs['dim_names'] = XARRAY_SCHEMA['table_acc']['coords']
                    self.table_acc.attrs['coords_mode__time'] = 'start_and_step'
                    self.table_acc.attrs['coords__time'] = [self._write_start, self._write_every]
                else:
                    setattr(
                        self,
                        'table_acc',
                        self.outputs.h5group['table_acc']
                    )
    def update_and_export(
            self,
            batch_size: int,
            data_size: int,
            t: int,
            **kwargs
        ):
        self.logger.note('Update and export')
        # Update the model parameters after every batch and clear the loss
        if t % batch_size == 0 or t == data_size - 1:
            # Update time
            self._time += 1

            # Update gradients
            loss = kwargs.pop('loss',None)
            if loss is not None:
                # Extract values from each sub-loss
                loss_values = sum([val for val in loss.values()])
                # Perform gradient update
                loss_values.backward()
                self.harris_wilson_nn._neural_net.optimizer.step()
                self.harris_wilson_nn._neural_net.optimizer.zero_grad()

                # Compute average losses here
                n_processed_steps = kwargs.pop('n_processed_steps',None)
                if n_processed_steps is not None:
                    for name in loss.keys():
                        loss[name] = loss[name] / n_processed_steps[name]
                # Store total loss too
                loss['total'] = sum([val for val in loss.values()])
            # Write to file
            self.write_data(
                **kwargs,
                **loss
            )
            # Delete loss
            del loss
        
        return {},{}

    def write_data(self,**kwargs):
        '''Write the current state into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        '''
        self.logger.debug('Writing data')
        if self._time >= self._write_start and self._time % self._write_every == 0:
            if 'loss' in self.output_names:
                for loss_name in list(self.harris_wilson_nn.loss_functions.keys())+['total']:
                    # Store samples
                    _loss_sample = kwargs.get(loss_name,None)
                    _loss_sample = _loss_sample.clone().detach().cpu().numpy().item() if _loss_sample is not None else None
                    getattr(self,loss_name).resize(getattr(self,loss_name).shape[0] + 1, axis=0)
                    getattr(self,loss_name)[-1] = _loss_sample

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
                _theta_acc = _theta_acc if _theta_acc is not None else None
                self.theta_acc.resize(self.theta_acc.shape[0] + 1, axis=0)
                self.theta_acc[-1] = _theta_acc

            if 'log_destination_attraction_acc' in self.output_names:
                _log_dest_attract_acc = kwargs.get('log_destination_attraction_acc',None)
                _log_dest_attract_acc = _log_dest_attract_acc if _log_dest_attract_acc is not None else None
                self.log_destination_attraction_acc.resize(self.log_destination_attraction_acc.shape[0] + 1, axis=0)
                self.log_destination_attraction_acc[-1] = _log_dest_attract_acc

            if 'table_acc' in self.output_names:
                _table_acc = kwargs.get('table_acc',None)
                _table_acc = _table_acc if _table_acc is not None else None
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
            self.logger.progress(f"Theta acceptance: {self.config['theta_acceptance']}")
        if hasattr(self,'signs'):
            self.config['positives_percentage'] = int(100*self.signs[:].sum(axis=0))
            self.logger.progress(f"Positives %: {self.config['positives_percentage']}")
        if hasattr(self,'log_destination_attraction_acc'):
            self.config['log_destination_attraction_acceptance'] = int(100*self.log_destination_attraction_acc[:].mean(axis=0))
            self.logger.progress(f"Log destination attraction acceptance: {self.config['log_destination_attraction_acceptance']}")
        if hasattr(self,'table_acc'):
            self.config['table_acceptance'] = int(100*self.table_acc[:].mean(axis=0))
            self.logger.progress(f"Table acceptance: {self.config['table_acceptance']}")
        if hasattr(self,'harris_wilson_nn'):
            for loss_name in list(self.harris_wilson_nn.loss_functions.keys())+['total']:
                self.logger.progress(f'{loss_name} loss: {getattr(self,loss_name)[:].mean(axis=0)}')


class DataGeneration(Experiment):

    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)

        # Perform experiment-specific validation check
        config.experiment_validate_config()
        
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
        
        # Perform experiment-specific validation check
        config.experiment_validate_config()

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

        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Simulation run finished.")

class LogTargetAnalysis(Experiment):

    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)

        # Perform experiment-specific validation check
        config.experiment_validate_config()

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

        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Simulation run finished.")

class SIM_MCMC(Experiment):
    def __init__(self, config:Config, **kwargs):

        # Initalise superclass
        super().__init__(config,**kwargs)

        # Perform experiment-specific validation check
        config.experiment_validate_config()

        self.output_names = ['log_destination_attraction','theta','sign','log_target','computation_time']

        # Fix random seed
        set_seed(self.seed)

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
        self.config = getattr(self.harris_wilson_mcmc,'config') if hasattr(self.harris_wilson_mcmc,'config') else config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep_params=kwargs.get('sweep_params',{}),
            base_dir=self.outputs_base_dir,
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep_params=kwargs.get('sweep_params',{}))
        # Write metadata
        self.write_metadata()
        
        self.logger.note(f"{self.harris_wilson_mcmc}")
        self.logger.info(f"Experiment: {self.outputs.experiment_id}")
        # self.logger.critical(f"{json.dumps(kwargs.get('sweep_params',{}),indent=2)}")

        
    def run(self,**kwargs) -> None:

        self.logger.info(f"Running MCMC inference of {self.harris_wilson_mcmc.physics_model.noise_regime} noise SpatialInteraction.")

        # Fix random seed
        set_seed(self.seed)

        # Initialise parameters
        initial_params = self.initialise_parameters(self.output_names)
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
        
        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Simulation run finished.")

class JointTableSIM_MCMC(Experiment):
    def __init__(self, config:Config, **kwargs):

        # Initalise superclass
        super().__init__(config,**kwargs)

        # Perform experiment-specific validation check
        config.experiment_validate_config()

        self.output_names = ['log_destination_attraction','theta','sign','table','log_target','computation_time']

        # Fix random seed
        rng = set_seed(self.seed)

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
        config = getattr(self.harris_wilson_mcmc,'config') if hasattr(self.harris_wilson_mcmc,'config') else config

        # Build contingency table
        self.logger.note("Initializing the contingency table ...")
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

        # Get config
        self.config = getattr(self.ct_mcmc.ct,'config') if isinstance(self.ct_mcmc.ct,Config) else config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep_params=kwargs.get('sweep_params',{}),
            base_dir=self.outputs_base_dir,
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep_params=kwargs.get('sweep_params',{}))

        # Write metadata
        self.write_metadata()
        
        self.logger.note(f"{self.harris_wilson_mcmc}")
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")
        
    def run(self,**kwargs) -> None:

        self.logger.info(f"Running MCMC inference of {self.harris_wilson_mcmc.physics_model.noise_regime} noise {self.harris_wilson_mcmc.physics_model.name}.")

        # Fix random seed
        set_seed(self.seed)

        # Initialise parameters
        initial_params = self.initialise_parameters(self.output_names)
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
            log_intensity_sample = self.harris_wilson_mcmc.physics_model.intensity_model.log_intensity(
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
                    log_intensity_sample.squeeze()
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
            
        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Simulation run finished.")

class Table_MCMC(Experiment):
    def __init__(self, config:Config, **kwargs):
        
        # Initalise superclass
        super().__init__(config,**kwargs)

        # Perform experiment-specific validation check
        config.experiment_validate_config()

        self.output_names = ['table']

        # Fix random seed
        rng = set_seed(self.seed)

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
        self.logger.note("Initializing the contingency table ...")
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
        config = getattr(self.ct_mcmc.ct,'config') if isinstance(self.ct_mcmc.ct,Config) else config

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

        # Get config
        self.config = config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep_params=kwargs.get('sweep_params',{}),
            base_dir=self.outputs_base_dir,
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep_params=kwargs.get('sweep_params',{}))

        # Write metadata
        self.write_metadata()
        
        self.logger.note(f"{self.ct_mcmc}")
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")

    def run(self,**kwargs) -> None:

        self.logger.note(f"Running Table MCMC.")

        # Initialise data structures
        self.initialise_data_structures()

        # Initialise parameters
        initial_params = self.initialise_parameters(self.output_names)
        table_sample = initial_params['table_sample']

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
            self.start_time = time.time()

            # Sample table
            table_sample,accepted = self.ct_mcmc.table_gibbs_step(
                table_prev = table_sample,
                log_intensity = self.log_intensity
            )

            # Clean and write to file
            _,_ = self.update_and_export(
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

        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Simulation run finished.")

class TableSummaries_MCMCConvergence(Experiment):
    def __init__(self, config:Config, **kwargs):

        # Initalise superclass
        super().__init__(config,**kwargs)

        # Perform experiment-specific validation check
        config.experiment_validate_config()

        self.output_names = ['table']

        # Setup table
        self.logger.note("Initializing the contingency table ...")
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

        self.logger.info('Running MCMC')
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

            if (i in [int(p*N) for p in np.arange(0,1,0.1)]):
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
        
        # Perform experiment-specific validation check
        config.experiment_validate_config()

        self.output_names = ['log_destination_attraction','theta','loss']

        # Fix random seed
        rng = set_seed(self.seed)

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
            loss = config['neural_network'].pop('loss'),
            physics_model = harris_wilson_model,
            write_every = self._write_every,
            write_start = self._write_start,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get config
        self.config = getattr(self.harris_wilson_nn,'config') if hasattr(self.harris_wilson_nn,'config') else config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep_params=kwargs.get('sweep_params',{}),
            base_dir=self.outputs_base_dir,
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep_params=kwargs.get('sweep_params',{}))

        # Write metadata
        self.write_metadata()
        
        self.logger.note(f"{self.harris_wilson_nn}")
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")
        

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
            loss_sample = {}
            # Track number of elements in each loss function
            n_processed_steps = {}
            
            # Process the training set elementwise, updating the loss after batch_size steps
            for t, training_data in enumerate(self.inputs.data.destination_attraction_ts):
                
                # Perform neural net training
                theta_sample, \
                destination_attraction_sample = self.harris_wilson_nn.epoch_time_step(
                    experiment = self,
                    validation_data = dict(
                        destination_attraction_ts = training_data,
                    ),
                    dt = self.config['harris_wilson_model'].get('dt',0.001)
                )
                log_destination_attraction_sample = torch.log(destination_attraction_sample)
                
                # Update losses
                loss_sample,n_processed_steps = self.harris_wilson_nn.update_loss(
                    previous_loss = loss_sample,
                    n_processed_steps = n_processed_steps,
                    validation_data = dict(
                        destination_attraction_ts = training_data
                    ),
                    prediction_data = dict(
                        destination_attraction_ts = torch.flatten(destination_attraction_sample)
                    )
                )

                # Clean and write to file
                loss_sample,n_processed_steps = self.update_and_export(
                    loss = loss_sample,
                    n_processed_steps = n_processed_steps,
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
        
        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Simulation run finished.")

class NonJointTableSIM_NN(Experiment):
    def __init__(self, config:Config, **kwargs):
        
        # Initalise superclass
        super().__init__(config,**kwargs)

        # Perform experiment-specific validation check
        config.experiment_validate_config()

        self.output_names = ['log_destination_attraction','theta','loss', 'table']

        # Fix random seed
        rng = set_seed(self.seed)

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
        
        # Build contingency table
        self.logger.note("Initializing the contingency table ...")
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
        # Get config
        config = getattr(self.ct_mcmc.cxt,'config') if isinstance(self.ct_mcmc.ct,Config) else config

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
            loss = config['neural_network'].pop('loss'),
            physics_model = harris_wilson_model,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get config
        self.config = getattr(self.harris_wilson_nn,'config') if hasattr(self.harris_wilson_nn,'config') else config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep_params=kwargs.get('sweep_params',{}),
            base_dir=self.outputs_base_dir,
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep_params=kwargs.get('sweep_params',{}))

        # Write metadata
        self.write_metadata()
        
        self.logger.note(f"{self.harris_wilson_nn}")
        self.logger.note(f"{self.ct_mcmc}")
        
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")
        
    def run(self,**kwargs) -> None:

        self.logger.note(f"Running Disjoint Table Inference and Neural Network training of Harris Wilson model.")

        # Initialise data structures
        self.initialise_data_structures()

        # Initialise parameters
        initial_params = self.initialise_parameters(self.output_names)
        theta_sample = initial_params['theta_sample']
        log_destination_attraction_sample = initial_params['log_destination_attraction_sample']
        table_sample = initial_params['table_sample']
        
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
            start_time = time.time()
            
            # Track the training loss
            loss_sample = {}
            # Track number of elements in each loss function
            n_processed_steps = {}
            
            # Process the training set elementwise, updating the loss after batch_size steps
            for t, training_data in enumerate(self.inputs.data.destination_attraction_ts):

                # Perform neural net training
                theta_sample, \
                destination_attraction_sample = self.harris_wilson_nn.epoch_time_step(
                    experiment = self,
                    validation_data = dict(
                        destination_attraction_ts = training_data
                    ),
                    dt = self.config['harris_wilson_model'].get('dt',0.001)
                )
                log_destination_attraction_sample = torch.log(destination_attraction_sample)
            
                # Add axis to every sample to ensure compatibility 
                # with the functions used below
                theta_sample_expanded = torch.unsqueeze(theta_sample,0)
                log_destination_attraction_sample_expanded = log_destination_attraction_sample.unsqueeze(0).unsqueeze(0)

                # Compute log intensity
                log_intensity_sample = self.harris_wilson_nn.physics_model.intensity_model.log_intensity(
                    log_destination_attraction = log_destination_attraction_sample_expanded,
                    grand_total = self.ct_mcmc.ct.data.margins[tuplize(range(ndims(self.ct_mcmc.ct)))],
                    **dict(zip(self.harris_wilson_nn.physics_model.params_to_learn,theta_sample_expanded.split(1,dim=1)))
                ).squeeze()

                # Sample table
                table_sample,accepted = self.ct_mcmc.table_gibbs_step(
                    table_prev = table_sample,
                    log_intensity = log_intensity_sample
                )

                # Update losses
                loss_sample,n_processed_steps = self.harris_wilson_nn.update_loss(
                    previous_loss = loss_sample,
                    validation_data = dict(
                        destination_attraction_ts = training_data
                    ),
                    prediction_data = dict(
                        destination_attraction_ts = torch.flatten(destination_attraction_sample)
                    )
                )

                # Clean and write to file
                loss_sample,n_processed_steps = self.update_and_export(
                    loss = loss_sample,
                    n_processed_steps = n_processed_steps,
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

        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Simulation run finished.")

class JointTableSIM_NN(Experiment):
    
    def __init__(self, config:Config, **kwargs):

        # Initalise superclass
        super().__init__(config,**kwargs)

        # Perform experiment-specific validation check
        config.experiment_validate_config()

        self.output_names = ['log_destination_attraction','theta','loss', 'table']

        # Fix random seed
        rng = set_seed(self.seed)

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

        # Build contingency table
        self.logger.note("Initializing the contingency table ...")
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
        # Get config
        config = getattr(self.ct_mcmc.cxt,'config') if isinstance(self.ct_mcmc.ct,Config) else config
        
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
            loss = dict(
                **config['neural_network'].pop('loss'),
                table_likelihood = self.ct_mcmc.table_loss_function
            ),
            physics_model = harris_wilson_model,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get config
        self.config = getattr(self.harris_wilson_nn,'config') if hasattr(self.harris_wilson_nn,'config') else config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep_params=kwargs.get('sweep_params',{}),
            base_dir=self.outputs_base_dir,
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep_params=kwargs.get('sweep_params',{}))

        # Write metadata
        self.write_metadata()
        
        self.logger.note(f"{self.harris_wilson_nn}")
        self.logger.info(f"{self.ct_mcmc}")
        
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")

    def run(self,**kwargs) -> None:

        self.logger.note(f"Running Joint Table Inference and Neural Network training of Harris Wilson model.")

        # Initialise data structures
        self.initialise_data_structures()

        # Initialise parameters
        initial_params = self.initialise_parameters(self.output_names)
        theta_sample = initial_params['theta_sample']
        log_destination_attraction_sample = initial_params['log_destination_attraction_sample']
        table_sample = initial_params['table_sample']
        
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
            loss_sample = {}
            # Track number of elements in each loss function
            n_processed_steps = {}

            # Process the training set elementwise, updating the loss after batch_size steps
            for t, training_data in enumerate(self.inputs.data.destination_attraction_ts):

                # Perform neural net training
                theta_sample, \
                destination_attraction_sample = self.harris_wilson_nn.epoch_time_step(
                    experiment = self,
                    validation_data = dict(
                        destination_attraction_ts = training_data
                    ),
                    dt = self.config['harris_wilson_model'].get('dt',0.001)
                )
                log_destination_attraction_sample = torch.log(destination_attraction_sample)

                # Add axis to every sample to ensure compatibility 
                # with the functions used below
                theta_sample_expanded = torch.unsqueeze(theta_sample,0)
                log_destination_attraction_sample_expanded = log_destination_attraction_sample.unsqueeze(0).unsqueeze(0)

                # Compute log intensity
                log_intensity_sample = self.harris_wilson_nn.physics_model.intensity_model.log_intensity(
                    log_destination_attraction = log_destination_attraction_sample_expanded,
                    grand_total = self.ct_mcmc.ct.data.margins[tuplize(range(ndims(self.ct_mcmc.ct)))],
                    **dict(zip(self.harris_wilson_nn.physics_model.params_to_learn,theta_sample_expanded.split(1,dim=1)))
                ).squeeze()
                
                # Update destination_attraction loss
                loss_sample,n_processed_steps = self.harris_wilson_nn.update_loss(
                    previous_loss = loss_sample,
                    n_processed_steps = n_processed_steps,
                    validation_data = dict(
                        destination_attraction_ts = training_data,
                    ),
                    prediction_data = dict(
                        destination_attraction_ts = torch.flatten(destination_attraction_sample)
                    ),
                    loss_function_names = ['dest_attraction_ts'],
                    aux_inputs = vars(self.inputs.data)
                )
                
                # Sample table
                for _ in range(self.config['mcmc']['contingency_table'].get('table_steps',1)):
                    
                    # Perform table step
                    table_sample,accepted = self.ct_mcmc.table_gibbs_step(
                        table_prev = table_sample,
                        log_intensity = log_intensity_sample
                    )

                    # Update losses
                    loss_sample,n_processed_steps = self.harris_wilson_nn.update_loss(
                        previous_loss = loss_sample,
                        n_processed_steps = n_processed_steps,
                        validation_data = dict(
                            log_intensity = log_intensity_sample
                        ),
                        prediction_data = dict(
                            table = table_sample,
                        ),
                        loss_function_names = [lf for lf in self.harris_wilson_nn.loss_functions.keys() if lf != 'dest_attraction_ts'],
                        aux_inputs = vars(self.inputs.data)
                    )

                # Clean loss and write to file
                # This will only store the last table sample
                loss_sample,n_processed_steps = self.update_and_export(
                    loss = loss_sample,
                    n_processed_steps = n_processed_steps,
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
                
        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
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
            console_level = config.level,
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']
    
        # Try to pre load config from previous unfinished experiment
        self.config = config
        preloaded_config = self.load()
        # Flag for appending experiment outputs
        self.config.settings['load_data'] = False
        if preloaded_config is not None:
            self.logger.info(f"Resuming experiment in {config['inputs']['load_experiment']}")
            # Update config to the one that is loaded
            self.config = preloaded_config
            # Use these settings from the new config
            # Note that these parameters relate only to the computational/hardware
            # aspect of the experiment and not any actual parameters of the experiment
            for key in ['n_workers','n_threads','device', 'log_level']:
                # Get settings from new config
                new_setting = list(deep_get(key,config.settings))[0]
                # Update preloaded config
                deep_update(self.config.settings,key,new_setting)
            # Load existing experiment data
            self.config.settings['load_data'] = True
        del preloaded_config

        # Load schema
        self.config.load_schemas()

        # Store number of workers
        self.n_workers = self.config.settings['inputs'].get("n_workers",1)

        self.logger.info(f"Performing parameter sweep")

        # Parse sweep configurations
        self.sweep_params = self.config.parse_sweep_params()

        # Create sweep configurations
        sweep_configurations, \
        self.param_sizes_str, \
        self.total_size_str = self.config.prepare_sweep_configurations(self.sweep_params)
        
        # If outputs should be loaded and appended
        if not self.config.settings['load_data']:
            # Store one datetime
            # for all sweeps
            self.config.settings['datetime'] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            # Store all sweep configurations
            self.sweep_configurations = sweep_configurations

        # Temporarily disable sample output writing
        export_samples = deepcopy(self.config['experiments'][0]['export_samples'])
        deep_update(self.config.settings,'export_samples',False)

        # Keep only first dataset just to instantiate outputs
        dir_range = []
        if isinstance(self.config['inputs']['dataset'],dict):
            dir_range = deepcopy(self.config['inputs']['dataset']['sweep']['range'])
            self.config['inputs']['dataset'] = dir_range[0]
        
        self.outputs = Outputs(
            self.config,
            sweep_params=kwargs.get('sweep_params',{}),
            logger = self.logger
        )

        # Make output home directory
        self.outputs_base_dir = self.outputs.outputs_path
        self.outputs_experiment_id = self.outputs.experiment_id

        # Check if outputs exist 
        # and remove them from sweep configurations
        if self.config.settings['load_data']:
            # Update sweep configurations
            self.sweep_configurations = list(
                self.outputs.trim_sweep_configurations(
                    sweep_configurations = sweep_configurations,
                    sweep_params = self.sweep_params
                )
            )
        
        # Prepare writing to file
        self.outputs.open_output_file(sweep_params={})

        # Enable it again
        deep_updates(self.config.settings,{'export_samples':export_samples})

        # # Write metadata
        if self.config.settings['experiments'][0].get('export_metadata',True):
            self.outputs.write_metadata(
                dir_path='',
                filename=f"config"
            )
        
        # Restore dataset config entries
        if len(dir_range) > 0:
            self.config['inputs']['dataset'] = dir_range
        self.logger.note(f"ExperimentSweep: {self.outputs.experiment_id} prepared")

    
    def __repr__(self) -> str:
        return "ParameterSweep("+(self.experiment.__repr__())+")"

    def __str__(self) -> str:
        return f"""
            Sweep key paths: {self.sweep_key_paths}
        """

    def load(self):
        if self.config['inputs'].get('load_experiment',''):
            try:
                # Load config
                config = Config(
                    path = self.config['inputs'].get('load_experiment',''),
                    logger = self.logger
                )
                # Validate preloaded config
                # This does a bunch of useful stuff
                config.validate_config()
                return config
            except:
                return None
        return None

    def run(self,**kwargs):

        self.logger.info(f"{self.outputs.experiment_id}")
        self.logger.info(f"Parameter space size: {self.param_sizes_str}")
        self.logger.info(f"Total = {self.total_size_str}")
        self.logger.info(f"Of which unfinished = {len(self.sweep_configurations)}.")
        self.logger.info(f"Preparing configs...")
        # For each configuration update experiment config 
        # and instantiate new experiment
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Decide whether to run sweeps in parallel or not
            if self.n_workers > 1:
                self.run_concurrent(self.sweep_configurations)
            else:
                self.run_sequential(self.sweep_configurations)
        
    def prepare_experiment(self,sweep_configuration):
        # Deactivate logging
        self.logger.setLevels(
            console_level='ERROR',#'ERROR','DEBUG'
            file_level='DEBUG'
        )
        
        return self.config.prepare_experiment_config(
            self.sweep_params,
            sweep_configuration
        )
        
    
    def prepare_instantiate_and_run(self,instance_num:int,sweep_configuration:dict,semaphore=None,counter=None,pbar=None):
        try:
            if semaphore is not None:
                semaphore.acquire()
            
            # Prepare experiment
            config,sweep = self.prepare_experiment(sweep_configuration)
            
            self.logger.info(f'Instance = {str(instance_num)} START')

            # Create new experiment
            new_experiment = instantiate_experiment(
                experiment_type=config.settings['experiment_type'],
                config=config,
                sweep_params=sweep,
                instance=str(instance_num),
                base_dir=self.outputs_base_dir,
                experiment_id=self.outputs_experiment_id,
                device_id=(instance_num%self.n_workers),
                logger=self.logger,
            )
            self.logger.debug('New experiment set up')
            # Running experiment
            new_experiment.run()
            if counter is not None and pbar is not None:
                with counter.get_lock():
                    counter.value += 1
                    pbar.n = counter.value
                    pbar.refresh()
                
            if semaphore is not None:
                semaphore.release()
            # gc.collect()
            self.logger.info(f'Instance = {str(instance_num)} DONE')
        except Exception as e:
            raise Exception(f'failed running instance {instance_num}')

    def run_sequential(self,sweep_configurations):
        sweep_configurations = [
            (100, '_row_constrained', [[1]], '', ['dest_attraction_ts', 'table_likelihood'], ['mseloss', 'custom']),
            (100, '_doubly_constrained', [[0], [1]], '', ['table_likelihood'], ['custom']),
            (100, '_doubly_constrained', [[0], [1]], '', ['dest_attraction_ts', 'table_likelihood'], ['mseloss', 'custom']),
            (100, '_doubly_10%_cell_constrained', [[0], [1]], 'cell_constraints_permuted_size_90_cell_percentage_10_constrained_axes_0_1_seed_1234.txt', ['table_likelihood'], ['custom']),
            (100, '_doubly_10%_cell_constrained', [[0], [1]], 'cell_constraints_permuted_size_90_cell_percentage_10_constrained_axes_0_1_seed_1234.txt', ['dest_attraction_ts', 'table_likelihood'], ['mseloss', 'custom']),
            (100, '_doubly_20%_cell_constrained', [[0], [1]], 'cell_constraints_permuted_size_179_cell_percentage_20_constrained_axes_0_1_seed_1234.txt', ['table_likelihood'], ['custom']),
            (100, '_doubly_20%_cell_constrained', [[0], [1]], 'cell_constraints_permuted_size_179_cell_percentage_20_constrained_axes_0_1_seed_1234.txt', ['dest_attraction_ts', 'table_likelihood'], ['mseloss', 'custom'])
        ]
        for instance,sweep_config in tqdm(
            enumerate(sweep_configurations),
            total=len(sweep_configurations),
            desc='Running sweeps in sequence',
            leave=False,
            position=0
        ):
            self.prepare_instantiate_and_run(
                instance_num = instance,
                sweep_configuration = sweep_config,
                semaphore = None,
                counter = None,
                pbar = None
            )
    
    def run_concurrent(self,sweep_configurations):

        # Split the sweep configurations into chunks
        sweep_config_chunks = list(divide_chunks(
            sweep_configurations,
            self.config['outputs']['chunk_size']
        ))
        
        for chunk_id, sweep_config_chunk in enumerate(sweep_config_chunks):
            # Initialise progress bar
            pbar = tqdm(
                total=len(sweep_config_chunk), 
                desc=f'Running sweeps concurrently: Batch {chunk_id+1}/{len(sweep_config_chunks)}',
                leave=False,
                position=0
            )
            with concurrency.ProcessPoolExecutor(self.n_workers*2) as executor:
                # Start the processes and ignore the results
                futures = [executor.submit(
                    self.prepare_instantiate_and_run,
                    instance_num = instance,
                    sweep_configuration = sweep_config
                ) for instance,sweep_config in enumerate(sweep_config_chunk)]

                # Wait for all processes to finish
                for index,fut in enumerate(concurrency.as_completed(futures)):
                    try:
                        fut.result()
                    except:
                        self.logger.error(f""" Sweep config 
                            {sweep_config_chunk[index]}
                        """)
                        raise Exception(f"Future {index} failed")
                    pbar.update(1)
            # Close progress bar
            pbar.close()