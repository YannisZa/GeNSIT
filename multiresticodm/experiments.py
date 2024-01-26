import sys
import time
import warnings
import traceback

from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from torch import float32, uint8
from multiprocessing import Manager

from multiresticodm.config import Config
from multiresticodm.inputs import Inputs
from multiresticodm.outputs import Outputs
from multiresticodm.utils.misc_utils import *
from multiresticodm.static.global_variables import *
from multiresticodm.utils.math_utils import torch_optimize
from multiresticodm.contingency_table import instantiate_ct
from multiresticodm.harris_wilson_model import HarrisWilson
from multiresticodm.spatial_interaction_model import instantiate_sim
from multiresticodm.utils.multiprocessor import BoundedQueueProcessPoolExecutor
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

def experiment_output_names(experiment_type:str):
    getattr(sys.modules[__name__], experiment_type)


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
        for experiment_type,experiment_index in self.config.settings['experiment_type'].items():
            # Construct sub-config with only data relevant for experiment
            experiment_config = deepcopy(self.config)
            # experiment_config.logger = self.logger
            # Store one experiment
            experiment_config.settings['experiments'] = [
                self.config.settings['experiments'][experiment_index]
            ]
            # Update id, seed and logging detail
            experiment_config.settings['experiment_type'] = experiment_type

            # Reset config variables
            experiment_config.reset()
            # Validate experiment-specific config
            experiment_config.experiment_validate()

            if self.config.settings['inputs'].get('dataset',None) is None:
                raise Exception(f'No dataset found for experiment type {experiment_type}')
            # Instatiate new experiment
            experiment = instantiate_experiment(
                experiment_type = experiment_type,
                config = experiment_config,
                logger = self.logger
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

        # Flag for appending experiment outputs
        self.config.settings['load_data'] = False
        
        # Update config with current timestamp ( but do not overwrite)
        datetime_results = list(deep_get(key='datetime',value=self.config.settings))
        if len(datetime_results) > 0:
            deep_update(
                self.config.settings, 
                key='datetime', 
                val=datetime.now().strftime("%d_%m_%Y_%H_%M_%S"), 
                overwrite=False
            )
        else:
            self.config['datetime'] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        # Inherit experiment id from parameter sweep (if it exists)
        # This will be used to create a unique output directory for every sweep
        self.sweep_experiment_id = kwargs.get('experiment_id',None)
        self.outputs_base_dir = kwargs.get('base_dir',None)

        # Update current config
        # self.config = self.sim.config.update_recursively(self.config,updated_config,overwrite=True)
        # Decide how often to print statemtents
        self.store_progress = self.config.get('store_progress',1.0)
        self.print_percentage = min(0.05,self.store_progress)

        # Update seed if specified
        self.seed = None
        if "seed" in self.config['inputs'] and not isinstance(self.config['inputs']["seed"],dict):
            self.seed = int(self.config['inputs']["seed"])
            self.logger.info(f"Updated seed to {self.seed}")

        # Get device name
        self.device = self.config['inputs']['device']
        # Get tqdm position
        self.position = kwargs.get('position',0)

        # Disable tqdm if needed
        self.tqdm_disabled = self.config['experiments'][0].get('disable_tqdm',True)

        # Flag for validating samples
        self.samples_validated = self.config['experiments'][0].get('validate_samples',False)

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
                # Get sweep-related data
                config.get_sweep_data()
                return config
            except:
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
        N = self.learning_model.config['training']['N']
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
            
            # Remove load experiment setting
            self.config.settings['inputs']['load_experiment'] = ''
            
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
                initialisations['table'] = deep_call(self,'.ct_mcmc.initialise_table()',None)
                # except:
                    # self.logger.warning("Table could not be initialised.")
            
            elif param == 'theta':
                try:
                    theta = self.physics_model.params_to_learn
                except:
                    try:
                        theta = self.learning_model.physics_model.params_to_learn
                    except:
                        try:
                            theta = self.learning_model.physics_model.params_to_learn
                        except:
                            self.logger.warning("Theta could not be initialised.")
                self.params_to_learn = list(theta.keys())
                initialisations['theta'] = torch.tensor(list(theta.values()),dtype=float32,device=self.device)
                
            elif param == 'log_destination_attraction':
                # Arbitrarily initialise destination attraction
                initialisations['log_destination_attraction'] = torch.log(
                    torch.repeat_interleave(
                        torch.tensor(1./dims['destination']),
                        dims['destination']
                    )
                ).to(
                    dtype=float32,
                    device=self.device
                )
                initialisations['log_destination_attraction'].requires_grad = True

            elif param == 'sign':
                initialisations['sign'] = torch.tensor(1,dtype=uint8,device=self.device)
            
            elif param == 'loss':
                initialisations['loss'] = {
                    nm:torch.tensor(0.0,dtype=float32,device=self.device,requires_grad=True) \
                    for nm in self.learning_model.loss_functions.keys()
                }
            
            elif param == 'log_target':
                initialisations['log_target'] = torch.tensor(0.0,dtype=float32,device=self.device)

        return initialisations
        
    def initialise_data_structures(self):

        if self.config.settings.get('export_samples',True):
            # Flag for loading experiment data
            load_experiment = self.config.settings['load_data']

            # Get dimensions
            dims = self.config['inputs']['dims']
            
            # Setup neural net loss
            if 'loss' in self.output_names:
                for loss_name in list(self.learning_model.loss_functions.keys())+['total_loss']:
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
                        getattr(self,loss_name).attrs['dim_names'] = (
                            (['iter'] if DATA_SCHEMA['loss'].get('is_iterable',False) else [])+
                            DATA_SCHEMA['loss'].get('dims',[])
                        )
                        getattr(self,loss_name).attrs['dims_mode__time'] = 'start_and_step'
                        getattr(self,loss_name).attrs['dims__time'] = [self._write_start, self._write_every]
                    else:
                        setattr(
                            self,
                            loss_name,
                            self.outputs.h5group[loss_name]
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
                        dset.attrs['dim_names'] = (
                            (['iter'] if DATA_SCHEMA[p_name].get('is_iterable',False) else [])+
                            DATA_SCHEMA[p_name].get('dims',[])
                        )
                        dset.attrs['dims_mode__time'] = 'start_and_step'
                        dset.attrs['dims__time'] = [self._write_start, self._write_every]
                    else:
                        dset = self.outputs.h5group[p_name]
                    # Append to thetas
                    self.thetas.append(dset)

            # Setup chunked dataset to store the state data in
            for sample in ['r2','log_posterior_approximation']:
                if sample in self.output_names:
                    if sample in self.outputs.h5group and not load_experiment:
                        # Delete current dataset
                        safe_delete(getattr(self,sample))
                    if sample not in self.outputs.h5group:
                        setattr(
                            self,
                            sample,
                            self.outputs.h5group.create_dataset(
                                sample,
                                shape = tuple([grange['n'] for grange in self.config['experiments'][0]['grid_ranges'].values()]),
                            )
                        )
                        getattr(self,sample).attrs[sample] = (
                            (['iter'] if DATA_SCHEMA[sample].get('is_iterable',False) else [])+
                            DATA_SCHEMA[sample].get('dims',[])
                        )
                        getattr(self,sample).attrs['dims_mode__time'] = 'start_and_step'
                        getattr(self,sample).attrs['dims__time'] = [self._write_start, self._write_every]
                    else:
                        setattr(
                            self,
                            sample,
                            self.outputs.h5group[sample]
                        )

            for sample in [
                'log_destination_attraction','table','log_target','compute_time',
                'sign','theta_acc','log_destination_attraction_acc','table_acc'
            ]:
                if sample in self.output_names:
                    # Setup chunked dataset to store the state data in
                    if sample in self.outputs.h5group and not load_experiment:
                        # Delete current dataset
                        safe_delete(getattr(self,sample))
                    if sample not in self.outputs.h5group:
                        all_dims = (
                            (['iter'] if DATA_SCHEMA[sample].get('is_iterable',False) else [])+
                            DATA_SCHEMA[sample].get('dims',[])
                        )
                        setattr(
                            self,
                            sample,
                            self.outputs.h5group.create_dataset(
                                sample,
                                (0,*[dims[d] for d in DATA_SCHEMA[sample].get('dims',[])]),
                                maxshape=(None,*[dims[d] for d in DATA_SCHEMA[sample].get('dims',[])]),
                                chunks=True,
                                compression=3
                            )
                        )
                        getattr(self,sample).attrs[sample] = all_dims
                        getattr(self,sample).attrs['dims_mode__time'] = 'start_and_step'
                        getattr(self,sample).attrs['dims__time'] = [self._write_start, self._write_every]
                    else:
                        setattr(
                            self,
                            sample,
                            self.outputs.h5group[sample]
                        )
            
            
    def model_update_and_export(
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
                self.learning_model._neural_net.optimizer.step()
                self.learning_model._neural_net.optimizer.zero_grad()

                # Compute average losses here
                n_processed_steps = kwargs.pop('n_processed_steps',None)

                if n_processed_steps is not None:
                    for name in loss.keys():
                        loss[name] = loss[name] / n_processed_steps[name]
                # Store total loss
                loss['total_loss'] = loss_values

            # Write to file
            self.write_data(
                **kwargs,
                **loss
            )
            # Delete loss
            del loss

        
        # Reset loss
        loss = {
            nm:torch.tensor(0.0,requires_grad=True) \
            for nm in self.learning_model.loss_functions.keys()    
        }
        # Reset number of epoch steps for loss calculation
        n_processed_steps = {nm : 0 for nm in self.learning_model.loss_functions.keys()}
        return loss,n_processed_steps

    def write_data(self,**kwargs):
        '''Write the current state into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        '''
        self.logger.debug('Writing data')
        if self._time >= self._write_start and self._time % self._write_every == 0:
            if 'loss' in self.output_names:
                for loss_name in list(self.learning_model.loss_functions.keys())+['total_loss']:
                    # Store samples
                    _loss_sample = kwargs.get(loss_name,None)
                    _loss_sample = _loss_sample.clone().detach().cpu().numpy().item() if _loss_sample is not None else None
                    getattr(self,loss_name).resize(getattr(self,loss_name).shape[0] + 1, axis=0)
                    getattr(self,loss_name)[-1] = _loss_sample

            if 'theta' in self.output_names:
                _theta_sample = kwargs.get('theta',[None]*len(self.thetas))
                _theta_sample = _theta_sample.clone().detach().cpu() if _theta_sample is not None else None
                for idx, dset in enumerate(self.thetas):
                    dset.resize(dset.shape[0] + 1, axis=0)
                    dset[-1] = _theta_sample[idx]

            for sample in ['r2','log_posterior_approximation']:
                if sample in self.output_names:
                    sample_value = kwargs.get(sample,None)
                    sample_value = sample_value.clone().detach().cpu() if sample_value is not None else None
                    getattr(self,sample)[kwargs['index']] = sample_value

            for sample in [
                'log_destination_attraction','sign','table',
                'theta_acc','log_destination_attraction_acc','table_acc', 'compute_time'
            ]:
                if sample in self.output_names:
                    # Get sample value
                    sample_value = kwargs.get(sample,None)
                    sample_value = sample_value.clone().detach().cpu() \
                        if torch.is_tensor(sample_value) \
                        else sample_value
                    # Resize h5 data
                    getattr(self,sample).resize(getattr(self,sample).shape[0] + 1, axis=0)
                    # Store latest sample
                    getattr(self,sample)[-1] = sample_value


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

    
    def show_progress(self):
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
        if hasattr(self,'learning_model') and self.learning_model.model_type == 'neural_network':
            loss_names = list(self.learning_model.loss_functions.keys())
            loss_names = loss_names if len(loss_names) <= 1 else loss_names+['total_loss']
            for loss_name in loss_names:
                self.logger.progress(f'{loss_name.capitalize()}: {getattr(self,loss_name)[:][-1]}')

    def validate_samples(self,**kwargs):
        if 'r2' in self.output_names:
            try:
                assert getattr(self,'r2')[kwargs['index']] >= 0 \
                    and getattr(self,'r2')[kwargs['index']] <= 1
            except:
                raise InvalidDataRange(data = getattr(self,'r2')[kwargs['index']], rang = 'in [0,1]')
            
        if 'log_destination_attraction' in self.output_names:
            try:
                assert np.absolute(np.exp(getattr(self,'log_destination_attraction')[-1]).sum() - 1.0) <= 1e-3
            except:
                raise InvalidDataRange(data = np.exp(getattr(self,'log_destination_attraction')[-1].squeeze()).sum(), rang = 'equal to 1')
            
        if 'sign' in self.output_names:
            try:
                assert np.absolute(getattr(self,'sign')[-1]) <= 1e-3 or np.absolute(getattr(self,'sign')[-1] - 1.0) <= 1e-3
            except:
                raise InvalidDataRange(data = getattr(self,'sign')[-1], rang = 'equal to 0 or 1')
            
        if 'table' in self.output_names:
            table_sample = torch.tensor(getattr(self,'table')[-1].squeeze())
            try:
                assert self.ct_mcmc.ct.table_admissible(table_sample)
            except:
                self.logger.error(f"Margins admissible {self.ct_mcmc.ct.table_margins_admissible(table_sample)}")
                self.logger.error(f"Cells admissible {self.ct_mcmc.ct.table_cells_admissible(table_sample)}")
                self.logger.error(f"Table margins {self.ct_mcmc.ct.table_constrained_margins_summary_statistic(table_sample)}")
                self.logger.error(f"""
                    Fixed margins {torch.cat([self.ct_mcmc.ct.data.margins[tuplize(ax)] 
                    for ax in sorted(self.ct_mcmc.ct.constraints['constrained_axes'])],dim=0)}
                """)
                raise InvalidDataRange(data = getattr(self,'table')[-1], rang = 'not admissible')
            
            self.logger.info(f"""
                Table admissible using {self.ct_mcmc.proposal_type} proposal
                axes: {self.ct_mcmc.ct.constraints['constrained_axes']},
                # cells: {len(self.ct_mcmc.ct.constraints['cells'])}
            """)
            

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

class RSquared_Analysis(Experiment):

    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)
    
        self.output_names = EXPERIMENT_OUTPUT_NAMES['RSquared_Analysis']

        # Fix random seed
        set_seed(self.seed)

        # Get config parameters
        self.grid_ranges = self.config.settings['experiments'][0]['grid_ranges']

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
        self.physics_model = HarrisWilson(
            intensity_model = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get and remove config
        self.config = pop_variable(self.physics_model,'config')

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep=kwargs.get('sweep',{}),
            base_dir=self.outputs_base_dir,
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep=kwargs.get('sweep',{}))

        # Write metadata
        self.write_metadata()
        
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")
    

    def run(self,**kwargs) -> None:
        
        self.logger.info(f"Running RSquared Analysis of {self.physics_model.noise_regime} noise SpatialInteraction.")
        
        # Initialise data structures
        self.initialise_data_structures()

        # Initialise parameters
        initial_params = self.initialise_parameters(['theta'])
        theta_sample = initial_params['theta']

        # reset these values so that outputs can be written to file
        self._write_start = 0
        self._write_every = 1
        
        # Initialize search grid
        alpha_values = torch.linspace(
            *[self.grid_ranges['alpha'][k] for k in ['min','max','n']],
            dtype=float32,
            device=self.device
        )
        beta_values = torch.linspace(
            *[self.grid_ranges['beta'][k] for k in ['min','max','n']],
            dtype=float32,
            device=self.device
        )

        # Print initialisations
        # self.print_initialisations(parameter_inits,print_lengths=False,print_values=True)

        # Search values
        max_r2 = 0 
        max_w_prediction = torch.exp(
            torch.ones(
                self.inputs.data.dims['destination'],dtype=float32,device=self.device
            ) * torch.tensor(
                1./self.inputs.data.dims['destination'],dtype=float32,device=self.device
            )
        )
        # Get destination attraction for last time dimension
        time_index = DATA_SCHEMA['destination_attraction_ts']['dims'].index('time')
        time_axis = DATA_SCHEMA['destination_attraction_ts']['axes'][time_index]
        w_data = deepcopy(self.inputs.data.destination_attraction_ts)
        w_data = w_data.select(dim = time_axis,index = -1)

        x_data = torch.log(w_data)
        # Total sum squares
        w_data_centred = w_data - torch.mean(w_data)
        ss_tot = torch.dot(w_data_centred, w_data_centred)
        
        # Progress bar
        progress = tqdm(
            total = len(alpha_values)*len(beta_values),
            disable = self.tqdm_disabled,
            position=self.position,
            desc = f"RSquared_Analysis instance: {self.position}",
            leave = False
        )

        # Perform grid evaluations
        for i,alpha_val in enumerate(alpha_values):
            for j,beta_val in enumerate(beta_values):
                try:
                    theta_sample[0] = alpha_val
                    theta_sample[1] = beta_val*self.physics_model.params.bmax
                    
                    # Get minimum
                    x_pred = torch_optimize(
                        x_data.clone().detach().cpu().numpy(),
                        function = self.physics_model.sde_potential_and_gradient,
                        method = 'L-BFGS-B',
                        **dict(zip(self.params_to_learn,theta_sample)),
                        device = self.device
                    )
                    # If predictions are null 
                    # it means that optimisation failed
                    if x_pred is None:
                        # Write data
                        self.write_data(
                            r2 = None,
                            index = (i,j)
                        )
                        continue
                    
                    # Get predictions
                    w_pred = torch.tensor(
                        np.exp(x_pred),
                        dtype = float32,
                        device = self.device
                    )
                    # Residiual sum squares
                    res = w_pred - w_data
                    ss_res = torch.dot(res, res)

                    # Regression sum squares
                    r2 = 1. - ss_res/ss_tot
                    
                    # Write data
                    self.write_data(
                        r2 = r2,
                        index = (i,j)
                    )

                    if r2 > max_r2:
                        max_w_prediction = deepcopy(w_pred)
                        max_r2 = r2
                except:
                    pass
                progress.update(1)

        # Output results
        r2 = self.r2[:]
        idx = np.unravel_index(r2.argmax(), np.shape(r2))
        self.logger.info(f"R^2: {r2[idx]}")
        self.logger.info(f"""
        alpha = {alpha_values[idx[1]]},
        beta = {beta_values[idx[0]]}, 
        beta_scaled = {beta_values[idx[0]]*self.physics_model.params.bmax}
        """)
        self.logger.note('Destination attraction prediction')
        self.logger.note(max_w_prediction)
        self.logger.note('True destination attraction')
        self.logger.note(w_data)
        if self.physics_model.intensity_model.ground_truth_known:
            self.logger.debug('True theta')
            self.logger.debug(f"alpha = {self.physics_model.intensity_model.alpha}, \
                              beta = {self.physics_model.intensity_model.beta}")

        # Save fitted values to parameters
        self.config.settings['fitted_alpha'] = to_json_format(alpha_values[idx[1]])
        self.config.settings['fitted_beta'] = to_json_format(beta_values[idx[0]])
        self.config.settings['fitted_scaled_beta'] = to_json_format(beta_values[idx[0]]*self.physics_model.params.bmax)
        self.config.settings['R^2'] = to_json_format(float(r2[idx]))
        self.config.settings['predicted_w'] = to_json_format(max_w_prediction)

        # Update metadata
        self.show_progress()

        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Experiment finished.")

class LogTarget_Analysis(Experiment):

    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)
    
        self.output_names = EXPERIMENT_OUTPUT_NAMES['LogTarget_Analysis']

        # Fix random seed
        set_seed(self.seed)

        # Get config parameters
        self.grid_ranges = self.config.settings['experiments'][0]['grid_ranges']

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
        physics_model = HarrisWilson(
            intensity_model = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get and remove config
        config = pop_variable(physics_model,'config')

        # Spatial interaction model MCMC
        self.logger.note("Initializing the Harris Wilson model MCMC")
        self.learning_model = instantiate_harris_wilson_mcmc(
            config = config,
            physics_model = physics_model,
            logger = self.logger
        )
        self.learning_model.build(**kwargs)

        # Store config
        self.config = config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep=kwargs.get('sweep',{}),
            base_dir=self.outputs_base_dir,
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep=kwargs.get('sweep',{}))

        # Write metadata
        self.write_metadata()
        
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")

    def run(self,**kwargs) -> None:
        
        self.logger.info(f"Running LogTarget Analysis of {self.learning_model.physics_model.noise_regime} noise SpatialInteraction.")
        
        # Initialise data structures
        self.initialise_data_structures()

        # Initialise parameters
        initial_params = self.initialise_parameters(['theta'])
        theta_sample = initial_params['theta']

        # reset these values so that outputs can be written to file
        self._write_start = 0
        self._write_every = 1
        
        # Initialize search grid
        alpha_values = torch.linspace(
            *[self.grid_ranges['alpha'][k] for k in ['min','max','n']],
            dtype=float32,
            device=self.device
        )
        beta_values = torch.linspace(
            *[self.grid_ranges['beta'][k] for k in ['min','max','n']],
            dtype=float32,
            device=self.device
        )

        # Print initialisations
        # self.print_initialisations(parameter_inits,print_lengths=False,print_values=True)

        # Search values
        max_target = -np.infty
        argmax_theta = None
        lap_c1 = torch.tensor(
            0.5*self.inputs.data.dims['destination']*np.log(2.*np.pi),
            dtype = float32,
            device = self.device
        )

        # Get destination attraction for last time dimension
        time_index = DATA_SCHEMA['destination_attraction_ts']['dims'].index('time')
        time_axis = DATA_SCHEMA['destination_attraction_ts']['axes'][time_index]
        w_data = deepcopy(self.inputs.data.destination_attraction_ts)
        w_data = w_data.select(dim = time_axis,index = -1)
        x_data = torch.log(w_data)
        
        # Progress bar
        progress = tqdm(
            total = len(alpha_values)*len(beta_values),
            disable = self.tqdm_disabled,
            position=self.position,
            desc = f"LogTarget Analysis instance: {self.position}",
            leave = False
        )

        # Perform grid evaluations
        for alpha_val in alpha_values:
            for beta_val in beta_values:
                try:
                    theta_sample[0] = alpha_val
                    theta_sample[1] = beta_val*self.learning_model.physics_model.params.bmax

                    # Minimise potential function
                    log_z_inverse,_ = self.learning_model.biased_z_inverse(
                        0,
                        dict(zip(self.params_to_learn,theta_sample))
                    )

                    # Evaluate log potential function for initial choice of \theta
                    potential_func = self.learning_model.physics_model.sde_potential(
                        x_data,
                        **dict(zip(self.params_to_learn,theta_sample)),
                        **vars(self.learning_model.physics_model.params)
                    )
                    # Store log_target
                    log_target = log_z_inverse - potential_func - lap_c1

                    if log_target > max_target:
                        argmax_theta = dict(zip(self.params_to_learn,theta_sample.detach().clone()))
                        max_target = log_target.detach().clone()
                
                except Exception:
                    traceback.print_exc()
                    sys.exit()
                # Update progress
                progress.update(1)

        self.logger.info(f"Log target: {max_target}")
        self.logger.info(f"""
        alpha = {argmax_theta['alpha']},
        beta = {argmax_theta['beta']/self.physics_model.params.bmax}, 
        beta_scaled = {argmax_theta['beta']}
        """)
        
        # Save fitted values to parameters
        self.config['fitted_alpha'] = to_json_format(argmax_theta['alpha'])
        self.config['fitted_scaled_beta'] = to_json_format(argmax_theta['beta']/self.learning_model.physics_model.params.bmax)
        self.config['fitted_beta'] = to_json_format(argmax_theta['beta'])
        self.config['log_target'] = to_json_format(max_target)

        # Update metadata
        self.show_progress()

        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Experiment finished.")

class SIM_MCMC(Experiment):
    def __init__(self, config:Config, **kwargs):

        # Initalise superclass
        super().__init__(config,**kwargs)

        self.output_names = EXPERIMENT_OUTPUT_NAMES['SIM_MCMC']

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
        physics_model = HarrisWilson(
            intensity_model = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )

        # Get and remove config
        config = pop_variable(sim,'config')

        # Spatial interaction model MCMC
        self.logger.note("Initializing the Harris Wilson model MCMC")
        self.learning_model = instantiate_harris_wilson_mcmc(
            config = config,
            physics_model = physics_model,
            logger = self.logger
        )
        self.learning_model.build(**kwargs)
        
        # Get config
        self.config = getattr(self.learning_model,'config') if hasattr(self.learning_model,'config') else config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep=kwargs.get('sweep',{}),
            base_dir=self.outputs_base_dir,
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep=kwargs.get('sweep',{}))

        # Write metadata
        self.write_metadata()
        
        self.logger.note(f"{self.learning_model}")
        self.logger.info(f"Experiment: {self.outputs.experiment_id}")
        # self.logger.critical(f"{json.dumps(kwargs.get('sweep',{}),indent=2)}")

        
    def run(self,**kwargs) -> None:

        self.logger.info(f"Running MCMC inference of {self.learning_model.physics_model.noise_regime} noise SpatialInteraction.")

        # Fix random seed
        set_seed(self.seed)

        # Initialise parameters
        initial_params = self.initialise_parameters(self.output_names)
        theta_sample = initial_params['theta']
        log_destination_attraction_sample = initial_params['log_destination_attraction']
        
        # Print initialisations
        # self.print_initialisations(parameter_inits,print_lengths=False,print_values=True)
        
        # Expand theta
        theta_sample_scaled = deepcopy(theta_sample)
        theta_sample_scaled[1] *= self.learning_model.physics_model.params.bmax

        # Compute initial log inverse z(\theta)
        log_z_inverse, sign_sample = self.learning_model.z_inverse(
                0,
                dict(zip(self.params_to_learn,theta_sample_scaled))
        )
        # Evaluate log potential function for initial choice of \theta
        V, gradV = self.learning_model.physics_model.sde_potential_and_gradient(
                log_destination_attraction_sample,
                **dict(zip(self.params_to_learn,theta_sample_scaled)),
                **vars(self.learning_model.physics_model.params)
        )

        # Store number of samples
        N = self.config.settings['training']['N']
        # Total samples for table,theta,x posteriors, respectively
        M = self.learning_model.theta_steps
        L = self.learning_model.log_destination_attraction_steps

        for i in tqdm(
            range(N),
            disable=self.tqdm_disabled,
            leave=False,
            position=self.position,
            desc = f"SIM_MCMC instance: {self.position}"
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
                sign_sample = self.learning_model.theta_step(
                    i,
                    theta_sample,
                    log_destination_attraction_sample,
                    auxiliary_values
                )

                # Write to file
                self.write_data(
                    theta = theta_sample,
                    theta_acc = theta_acc,
                    sign = sign_sample
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
                gradV = self.learning_model.log_destination_attraction_step(
                    theta_sample,
                    self.inputs.data.log_destination_attraction,
                    log_destination_attraction_sample,
                    auxiliary_values
                )
                # Write to data
                self.write_data(
                    log_destination_attraction = log_destination_attraction_sample,
                    log_destination_attraction_acc = log_dest_attract_acc
                )

            # print statements
            self.show_progress()
            self.logger.iteration(f"Completed iteration {i+1} / {N}.")
            if self.logger.console.isEnabledFor(PROGRESS):
                print('\n')

            # Write the epoch training time (wall clock time)
            self.write_data(compute_time = time.time() - start_time)
        
        # Update metadata
        self.show_progress()

        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Experiment finished.")

class JointTableSIM_MCMC(Experiment):
    def __init__(self, config:Config, **kwargs):

        # Initalise superclass
        super().__init__(config,**kwargs)

        self.output_names = EXPERIMENT_OUTPUT_NAMES['JointTableSIM_MCMC']

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
        physics_model = HarrisWilson(
            intensity_model = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get and remove config
        config = pop_variable(physics_model,'config')
        
        # Spatial interaction model MCMC
        self.logger.note("Initializing the Harris Wilson model MCMC")
        self.learning_model = instantiate_harris_wilson_mcmc(
            config = config,
            physics_model = physics_model,
            logger = self.logger
        )
        self.learning_model.build(**kwargs)
        
        # Get config
        config = getattr(self.learning_model,'config') if hasattr(self.learning_model,'config') else config

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
            sweep=kwargs.get('sweep',{}),
            base_dir=self.outputs_base_dir,
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep=kwargs.get('sweep',{}))

        # Write metadata
        self.write_metadata()
        
        self.logger.note(f"{self.learning_model}")
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")
        
    def run(self,**kwargs) -> None:

        self.logger.info(f"Running MCMC inference of {self.learning_model.physics_model.noise_regime} noise {self.learning_model.physics_model.name}.")

        # Fix random seed
        set_seed(self.seed)

        # Initialise parameters
        initial_params = self.initialise_parameters(self.output_names)
        theta_sample = initial_params['theta']
        table_sample = initial_params['table']
        log_destination_attraction_sample = initial_params['log_destination_attraction']
        
        # Print initialisations
        # self.print_initialisations(parameter_inits,print_lengths=False,print_values=True)
        
        # Expand theta
        theta_sample_scaled = deepcopy(theta_sample)
        theta_sample_scaled[1] *= self.learning_model.physics_model.params.bmax

        # Compute table likelihood and its gradient
        negative_log_table_likelihood = self.learning_model.negative_table_log_likelihood_expanded(
            log_destination_attraction = log_destination_attraction_sample,
            alpha = theta_sample_scaled[0],
            beta = theta_sample_scaled[1],
            table = table_sample
        )

        # Compute initial log inverse z(\theta)
        log_z_inverse, sign_sample = self.learning_model.z_inverse(
            0,
            dict(zip(self.params_to_learn,theta_sample_scaled))
        )

        # Evaluate log potential function for initial choice of \theta
        V, gradV = self.learning_model.physics_model.sde_potential_and_gradient(
            log_destination_attraction_sample,
            **dict(zip(self.params_to_learn,theta_sample_scaled)),
            **vars(self.learning_model.physics_model.params)
        )

        # Store number of samples
        N = self.config.settings['training']['N']
        # Total samples for table,theta,x posteriors, respectively
        M = self.learning_model.theta_steps
        L = self.learning_model.log_destination_attraction_steps

        for i in tqdm(
            range(N),
            disable=self.tqdm_disabled,
            leave=False,
            position=self.position,
            desc = f"JointTableSIM_MCMC instance: {self.position}"
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
                sign_sample = self.learning_model.theta_step(
                            i,
                            theta_sample,
                            log_destination_attraction_sample,
                            table_sample,
                            auxiliary_values
                        )

                # Write to file
                self.write_data(
                    theta = theta_sample,
                    theta_acc = theta_acc,
                    sign = sign_sample
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
                negative_log_table_likelihood = self.learning_model.log_destination_attraction_step(
                    theta_sample,
                    self.inputs.data.log_destination_attraction,
                    log_destination_attraction_sample,
                    table_sample,
                    auxiliary_values
                )
                # Write to data
                self.write_data(
                    log_destination_attraction = log_destination_attraction_sample,
                    log_destination_attraction_acc = log_dest_attract_acc,
                )

            # Compute new intensity
            log_intensity_sample = self.learning_model.physics_model.intensity_model.log_intensity(
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
                    table = table_sample,
                    table_acc = table_accepted
                )
            
            # Compute table likelihood for updated table
            negative_log_table_likelihood = self.learning_model.negative_table_log_likelihood_expanded(
                log_destination_attraction = log_destination_attraction_sample,
                table = table_sample,
                **dict(zip(self.params_to_learn,theta_sample))
            )

            # print statements
            self.show_progress()
            self.logger.iteration(f"Completed iteration {i+1} / {N}.")
            if self.logger.console.isEnabledFor(PROGRESS):
                print('\n')
            
            # Write the epoch training time (wall clock time)
            self.write_data(compute_time = time.time() - start_time)
            
        # Update metadata
        self.show_progress()

        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Experiment finished.")

class Table_MCMC(Experiment):
    def __init__(self, config:Config, **kwargs):
        
        # Initalise superclass
        super().__init__(config,**kwargs)

        self.output_names = EXPERIMENT_OUTPUT_NAMES['table']

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
                    grand_total = ct.margins[tuplize(range(ndims(ct)))].item()
                )
            except:
                raise Exception('No ground truth or table provided to construct table intensities.')

        # Get config
        self.config = config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep=kwargs.get('sweep',{}),
            base_dir=self.outputs_base_dir,
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep=kwargs.get('sweep',{}))

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
        table_sample = initial_params['table']

        # Store number of samples
        N = self.config['training']['N']

        # For each epoch
        for i in tqdm(
            range(N),
            disable=self.tqdm_disabled,
            leave=False,
            position=self.position,
            desc = f"Table MCMC instance: {self.position}"
        ):

            # Track the epoch training time
            start_time = time.time()

            # Sample table
            table_sample,accepted = self.ct_mcmc.table_gibbs_step(
                table_prev = table_sample,
                log_intensity = self.log_intensity
            )

            # Clean and write to file
            _,_ = self.model_update_and_export(
                table = table_sample,
                table_acceptance = accepted,
                compute_time = time.time() - start_time,
                # Batch size is in training settings
                t = 0,
                data_size = 1,
                **self.config['training']
            )

            # print statements
            self.show_progress()
            self.logger.iteration(f"Completed iteration {i+1} / {N}.")
            if self.logger.console.isEnabledFor(PROGRESS):
                print('\n')

        # Update metadata
        self.show_progress()

        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Experiment finished.")

class SIM_NN(Experiment):
    def __init__(self, config:Config, **kwargs):
        
        # Initalise superclass
        super().__init__(config,**kwargs)
    
        self.output_names = EXPERIMENT_OUTPUT_NAMES['SIM_NN']

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

        # Build Harris Wilson model
        self.logger.note("Initializing the Harris Wilson physics model ...")
        physics_model = HarrisWilson(
            intensity_model = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get and remove config
        config = pop_variable(physics_model,'config')
        
        # Set up the neural net
        self.logger.note("Initializing the neural net ...")
        neural_network = NeuralNet(
            input_size = self.inputs.data.dims['destination'],
            output_size = len(config['inputs']['to_learn']),
            **config['neural_network']['hyperparameters'],
            logger = self.logger
        ).to(self.device)

        # Instantiate harris and wilson neural network model
        self.logger.note("Initializing the Harris Wilson Neural Network model ...")
        self.learning_model = HarrisWilson_NN(
            config = config,
            neural_net = neural_network,
            loss = config['neural_network'].pop('loss'),
            physics_model = physics_model,
            write_every = self._write_every,
            write_start = self._write_start,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get config
        self.config = getattr(self.learning_model,'config') if hasattr(self.learning_model,'config') else config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep=kwargs.get('sweep',{}),
            base_dir=self.outputs_base_dir,
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep=kwargs.get('sweep',{}))

        # Write metadata
        self.write_metadata()
        
        self.logger.note(f"{self.learning_model}")
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")
        

    def run(self,**kwargs) -> None:

        self.logger.note(f"Running Neural Network training of Harris Wilson model.")

        # Initialise data structures
        self.initialise_data_structures()
        
        # Store number of samples
        N = self.config['training']['N']

        # Initialise samples
        log_intensity_sample = None
        grand_total = torch.tensor(1.0,dtype=float32,device=self.device)

        # Track the training loss
        loss_sample = {
            nm : torch.tensor(
                0.0,
                dtype = float32,
                requires_grad = True
            ) \
            for nm in self.learning_model.loss_functions.keys()
        }

        # Track number of elements in each loss function
        n_processed_steps = {nm : 0 for nm in self.learning_model.loss_functions.keys()}

        # For each epoch
        for i in tqdm(
            range(N),
            disable=self.tqdm_disabled,
            leave=False,
            position=self.position,
            desc = f"SIM_NN instance: {self.position}"
        ):

            # Track the epoch training time
            start_time = time.time()
            
            # Process the training set elementwise, updating the loss after batch_size steps
            for t, training_data in enumerate(torch.unsqueeze(self.inputs.data.destination_attraction_ts,0)):
                
                # Learn parameters by solving neural net
                self.logger.debug('Solving neural net')
                theta_sample = self.learning_model._neural_net(
                    torch.flatten(training_data)
                )            
                # Add axis to every sample to ensure compatibility 
                # with the functions used below
                theta_sample_expanded = torch.unsqueeze(theta_sample,0).split(1,dim=1)
                training_data_expanded = training_data.clone()
                training_data_expanded.requires_grad = True
                
                # Compute log intensity
                log_intensity_sample = self.learning_model.physics_model.intensity_model.log_intensity(
                    log_destination_attraction = torch.log(training_data_expanded),
                    grand_total = grand_total,
                    **dict(zip(
                        self.learning_model.physics_model.params_to_learn,
                        theta_sample_expanded
                    ))
                ).squeeze()

                # Solve SDE
                destination_attraction_sample = self.learning_model.physics_model.run_single(
                    curr_destination_attractions = training_data,
                    free_parameters = theta_sample,
                    log_intensity_normalised = (log_intensity_sample - torch.log(grand_total)),
                    dt = self.config['harris_wilson_model']['dt'],
                    requires_grad = True
                )

                # Update losses
                loss_sample,n_processed_steps = self.learning_model.update_loss(
                    previous_loss = loss_sample,
                    n_processed_steps = n_processed_steps,
                    validation_data = dict(
                        destination_attraction_ts = training_data
                    ),
                    prediction_data = dict(
                        destination_attraction_ts = [destination_attraction_sample]
                    )
                )
                
                # Clean and write to file
                loss_sample,n_processed_steps = self.model_update_and_export(
                    loss = loss_sample,
                    n_processed_steps = n_processed_steps,
                    theta = theta_sample,
                    log_destination_attraction = torch.log(destination_attraction_sample),
                    compute_time = time.time() - start_time,
                    # Batch size is in training settings
                    t = t,
                    data_size = len(training_data),
                    **self.config['training']
                )
            
            # print statements
            self.show_progress()
            self.logger.iteration(f"Completed iteration {i+1} / {N}.")
            if self.logger.console.isEnabledFor(PROGRESS):
                print('\n')
        
        # Update metadata
        self.show_progress()

        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Experiment finished.")

class NonJointTableSIM_NN(Experiment):
    def __init__(self, config:Config, **kwargs):
        
        # Initalise superclass
        super().__init__(config,**kwargs)

        self.output_names = EXPERIMENT_OUTPUT_NAMES['NonJointTableSIM_NN']

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

        # Build Harris Wilson model
        self.logger.note("Initializing the Harris Wilson physics model ...")
        physics_model = HarrisWilson(
            intensity_model = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get and remove config
        config = pop_variable(physics_model,'config')
        
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
            log_to_console = False,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get config
        config = getattr(self.ct_mcmc.ct,'config') if isinstance(self.ct_mcmc.ct,Config) else config

        # Set up the neural net
        self.logger.note("Initializing the neural net ...")
        neural_network = NeuralNet(
            input_size = self.inputs.data.dims['destination'],
            output_size = len(config['inputs']['to_learn']),
            **config['neural_network']['hyperparameters'],
            logger = self.logger
        ).to(self.device)

        # Instantiate harris and wilson neural network model
        self.logger.note("Initializing the Harris Wilson Neural Network model ...")
        self.learning_model = HarrisWilson_NN(
            config = config,
            neural_net = neural_network,
            loss = config['neural_network'].pop('loss'),
            physics_model = physics_model,
            write_every = self._write_every,
            write_start = self._write_start,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get config
        self.config = getattr(self.learning_model,'config') if hasattr(self.learning_model,'config') else config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep=kwargs.get('sweep',{}),
            base_dir=self.outputs_base_dir,
            experiment_id=self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep=kwargs.get('sweep',{}))

        # Write metadata
        self.write_metadata()
        
        self.logger.note(f"{self.learning_model}")
        self.logger.note(f"{self.ct_mcmc}")
        self.logger.note(f"Experiment: {self.outputs.experiment_id}")

        
    def run(self,**kwargs) -> None:

        self.logger.note(f"Running Disjoint Table Inference and Neural Network training of Harris Wilson model.")

        # Initialise data structures
        self.initialise_data_structures()

        # Initialise parameters
        initial_params = self.initialise_parameters(self.output_names)
        table_sample = initial_params['table']
        grand_total = self.ct_mcmc.ct.data.margins[tuplize(range(ndims(self.ct_mcmc.ct)))].to(float32)
        
        # Store number of samples
        N = self.config['training']['N']

        # Track the training loss
        loss_sample = {
            nm : torch.tensor(
                0.0,
                dtype = float32,
                requires_grad = True
            ) \
            for nm in self.learning_model.loss_functions.keys()
        }

        # Track number of elements in each loss function
        n_processed_steps = {nm : 0 for nm in self.learning_model.loss_functions.keys()}

        # For each epoch
        for i in tqdm(
            range(N),
            disable=self.tqdm_disabled,
            leave=False,
            position=self.position,
            desc = f"NonJointTableSIM_NN instance: {self.position}"
        ):

            # Track the epoch training time
            start_time = time.time()
            
            # Process the training set elementwise, updating the loss after batch_size steps
            for t, training_data in enumerate(
                torch.unsqueeze(self.inputs.data.destination_attraction_ts,0)
            ):

                # Learn parameters by solving neural net
                self.logger.debug('Solving neural net')
                theta_sample = self.learning_model._neural_net(
                    torch.flatten(training_data)
                )            
                # Add axis to every sample to ensure compatibility 
                # with the functions used below
                theta_sample_expanded = torch.unsqueeze(theta_sample,0).split(1,dim=1)
                training_data_expanded = training_data.clone()
                training_data_expanded.requires_grad = True
                
                # Compute log intensity
                log_intensity_sample = self.learning_model.physics_model.intensity_model.log_intensity(
                    log_destination_attraction = torch.log(training_data_expanded),
                    grand_total = grand_total,
                    **dict(zip(
                        self.learning_model.physics_model.params_to_learn,
                        theta_sample_expanded
                    ))
                ).squeeze()

                # Solve SDE
                destination_attraction_sample = self.learning_model.physics_model.run_single(
                    curr_destination_attractions = training_data,
                    free_parameters = theta_sample,
                    log_intensity_normalised = (log_intensity_sample - torch.log(grand_total)),
                    dt = self.config['harris_wilson_model']['dt'],
                    requires_grad = True
                )

                # Sample table
                table_sample,accepted = self.ct_mcmc.table_gibbs_step(
                    table_prev = table_sample,
                    log_intensity = log_intensity_sample
                )

                # Update losses
                loss_sample,n_processed_steps = self.learning_model.update_loss(
                    previous_loss = loss_sample,
                    n_processed_steps = n_processed_steps,
                    validation_data = dict(
                        destination_attraction_ts = training_data
                    ),
                    prediction_data = dict(
                        destination_attraction_ts = [destination_attraction_sample]
                    )
                )

                # Clean and write to file
                loss_sample,n_processed_steps = self.model_update_and_export(
                    loss = loss_sample,
                    n_processed_steps = n_processed_steps,
                    theta = theta_sample,
                    log_destination_attraction = torch.log(destination_attraction_sample),
                    table = table_sample,
                    table_acceptance = accepted,
                    compute_time = time.time() - start_time,
                    # Batch size is in training settings
                    t = t,
                    data_size = len(training_data),
                    **self.config['training']
                )
                
                if self.samples_validated:
                    self.validate_samples()
            
            # print statements
            self.show_progress()
            self.logger.iteration(f"Completed iteration {i+1} / {N}.")
            if self.logger.console.isEnabledFor(PROGRESS):
                print('\n')

        # Update metadata
        self.show_progress()

        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Experiment finished.")

class JointTableSIM_NN(Experiment):
    
    def __init__(self, config:Config, **kwargs):

        # Initalise superclass
        super().__init__(config,**kwargs)

        self.output_names = EXPERIMENT_OUTPUT_NAMES['JointTableSIM_NN']

        # Fix random seed
        set_seed(self.seed)

        # Prepare inputs
        self.inputs = Inputs(
            config = config,
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
        physics_model = HarrisWilson(
            intensity_model = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get and remove config
        config = pop_variable(physics_model,'config')

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
            log_to_console = False,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get config
        config = getattr(self.ct_mcmc.ct,'config') if isinstance(self.ct_mcmc.ct,Config) else config
        
        # Set up the neural net
        self.logger.note("Initializing the neural net ...")
        neural_network = NeuralNet(
            input_size = self.inputs.data.dims['destination'],
            output_size = len(config['inputs']['to_learn']),
            **config['neural_network']['hyperparameters'],
            logger = self.logger
        ).to(self.device)

        # Instantiate harris and wilson neural network model
        self.logger.note("Initializing the Harris Wilson Neural Network model ...")
        self.learning_model = HarrisWilson_NN(
            config = config,
            neural_net = neural_network,
            loss = dict(
                **config['neural_network'].pop('loss'),
                table_likelihood_loss = self.ct_mcmc.table_likelihood_loss
            ),
            physics_model = physics_model,
            write_every = self._write_every,
            write_start = self._write_start,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )
        # Get config
        self.config = getattr(self.learning_model,'config') if hasattr(self.learning_model,'config') else config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module = __name__+kwargs.get('instance',''),
            sweep = kwargs.get('sweep',{}),
            base_dir = self.outputs_base_dir,
            experiment_id = self.sweep_experiment_id,
            logger = self.logger
        )

        # Prepare writing to file
        self.outputs.open_output_file(sweep = kwargs.get('sweep',{}))

        # Write metadata
        self.write_metadata()
        
        self.logger.note(f"{self.learning_model}")
        self.logger.info(f"{self.ct_mcmc}")
        self.logger.note(f"Experiment: {self.outputs.experiment_id}. Sweep id: {self.outputs.sweep_id}")

    def run(self,**kwargs) -> None:

        self.logger.note(f"Running Joint Table Inference and Neural Network training of Harris Wilson model.")
        
        # Initialise data structures
        self.initialise_data_structures()

        # Initialise parameters
        initial_params = self.initialise_parameters(self.output_names)
        table_sample = initial_params['table']
        grand_total = self.ct_mcmc.ct.data.margins[tuplize(range(ndims(self.ct_mcmc.ct)))].to(float32)

        # Store number of samples
        N = self.config['training']['N']

        # Track the training loss
        loss_sample = {
            nm : torch.tensor(
                0.0,
                dtype = float32,
                requires_grad = True
            ) \
            for nm in self.learning_model.loss_functions.keys()
        }
        
        # Track number of elements in each loss function
        n_processed_steps = {nm : 0 for nm in self.learning_model.loss_functions.keys()}
        
        # For each epoch
        for i in tqdm(
            range(N),
            disable=self.tqdm_disabled,
            leave=False,
            position=self.position,
            desc = f"JointTableSIM_NN instance: {self.position}"
        ):

            # Track the epoch training time
            start_time = time.time()

            # Process the training set elementwise, updating the loss after batch_size steps
            for t, training_data in enumerate(
                torch.unsqueeze(self.inputs.data.destination_attraction_ts,0)
            ):

                # Learn parameters by solving neural net
                self.logger.debug('Solving neural net')
                theta_sample = self.learning_model._neural_net(
                    torch.flatten(training_data)
                )            
                # Add axis to every sample to ensure compatibility 
                # with the functions used below
                theta_sample_expanded = torch.unsqueeze(theta_sample,0).split(1,dim=1)
                training_data_expanded = training_data.clone()
                training_data_expanded.requires_grad = True
                
                # Compute log intensity
                log_intensity_sample = self.learning_model.physics_model.intensity_model.log_intensity(
                    log_destination_attraction = torch.log(training_data_expanded),
                    grand_total = grand_total,
                    **dict(zip(
                        self.learning_model.physics_model.params_to_learn,
                        theta_sample_expanded
                    ))
                ).squeeze()

                # Solve SDE
                destination_attraction_sample = self.learning_model.physics_model.run_single(
                    curr_destination_attractions = training_data,
                    free_parameters = theta_sample,
                    log_intensity_normalised = (log_intensity_sample - torch.log(grand_total)),
                    dt = self.config['harris_wilson_model']['dt'],
                    requires_grad = True
                )
                
                # Sample table(s)
                table_samples = []
                for _ in range(self.config['mcmc']['contingency_table'].get('table_steps',1)):
                    # Perform table step
                    table_sample,accepted = self.ct_mcmc.table_gibbs_step(
                        table_prev = table_sample,
                        log_intensity = log_intensity_sample
                    )
                    table_samples.append(table_sample/table_sample.sum())

                # Update losses
                loss_sample,n_processed_steps = self.learning_model.update_loss(
                    previous_loss = loss_sample,
                    n_processed_steps = n_processed_steps,
                    validation_data = dict(
                        destination_attraction_ts = training_data,
                        log_intensity = log_intensity_sample - torch.log(grand_total)
                    ),
                    prediction_data = dict(
                        destination_attraction_ts = [destination_attraction_sample],
                        table = table_samples
                    ),
                    aux_inputs = vars(self.inputs.data)
                )

                # Clean loss and write to file
                # This will only store the last table sample
                loss_sample,n_processed_steps = self.model_update_and_export(
                    loss = loss_sample,
                    n_processed_steps = n_processed_steps,
                    theta = theta_sample,
                    log_destination_attraction = torch.log(destination_attraction_sample),
                    table = table_sample,
                    table_acceptance = accepted,
                    compute_time = time.time() - start_time,
                    # Batch size is in training settings
                    t = t,
                    data_size = len(training_data),
                    **self.config['training']
                )
            
            # print statements
            self.show_progress()
            self.logger.iteration(f"Completed iteration {i+1} / {N}.")
            if self.logger.console.isEnabledFor(PROGRESS):
                print('\n')

                
        # Update metadata
        self.show_progress()

        # Write metadata
        self.write_metadata()

        # Write log and close outputs
        self.close_outputs()
        
        self.logger.note("Experiment finished.")

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
        preloaded_config = self.load_config()
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

        # Store number of workers
        self.n_workers = self.config.settings['inputs'].get("n_workers",1)
        
        self.logger.info(f"Performing parameter sweep for {self.config.settings['experiment_type']}")

        # Get sweep-related data
        self.config.get_sweep_data()

        # If outputs should be loaded and appended
        if not self.config.settings['load_data']:
            # Store one datetime
            # for all sweeps
            self.config.settings['datetime'] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        # Temporarily disable sample output writing
        export_samples = deepcopy(self.config['experiments'][0]['export_samples'])
        deep_update(self.config.settings,'export_samples',False)

        # Keep only first dataset just to instantiate outputs
        dir_range = []
        if isinstance(self.config['inputs']['dataset'],dict):
            dir_range = deepcopy(self.config['inputs']['dataset']['sweep']['range'])
            self.config['inputs']['dataset'] = dir_range[0]
        
        self.outputs = Outputs(
            config = self.config,
            sweep = kwargs.get('sweep',{}),
            logger = self.logger
        )
        # Make output home directory
        self.outputs_base_dir = self.outputs.outputs_path
        self.outputs_experiment_id = self.outputs.experiment_id

        # Check if outputs exist 
        # and remove them from sweep configurations
        if self.config.settings['load_data']:
            # Update sweep configurations
            self.outputs.config = list(
                self.outputs.config.trim_sweep_configurations()
            )
        
        # Prepare writing to file
        self.outputs.open_output_file(sweep={})

        # Enable it again
        deep_updates(self.config.settings,{'export_samples':export_samples})

        # Write metadata
        if self.config.settings['experiments'][0].get('export_metadata',True):
            # Write to file
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
            Sweep key paths: {self.config.sweep_key_paths}
        """

    def load_config(self):
        if self.config['inputs'].get('load_experiment',''):
            try:
                # Load config
                config = Config(
                    path = self.config['inputs'].get('load_experiment',''),
                    logger = self.logger
                )
                # Validate preloaded config
                # This does a bunch of useful stuff
                config.validate()
                # Get sweep-related data
                config.get_sweep_data()
                return config
            except:
                return None
        return None

    def run(self,**kwargs):

        self.logger.info(f"{self.outputs.experiment_id}")
        self.logger.info(f"Parameter space size: {self.config.param_sizes_str}")
        self.logger.info(f"Total = {self.config.total_size_str}")
        self.logger.info(f"Total to be run = {len(self.config.sweep_configurations)}.")
        self.logger.info(f"Preparing configs...")
        # For each configuration update experiment config 
        # and instantiate new experiment
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Decide whether to run sweeps in parallel or not
            if self.n_workers > 1:
                self.run_concurrent(self.config.sweep_configurations)
            else:
                self.run_sequential(self.config.sweep_configurations)
        
    def prepare_experiment(self,sweep_configuration):
        # Deactivate logging
        self.logger.setLevels(
            console_level='ERROR',#'ERROR','DEBUG'
            file_level='DEBUG'
        )
        
        return self.config.prepare_experiment_config(sweep_configuration)
        
    
    def prepare_instantiate_and_run(self,instance_num:int,sweep_configuration:dict,active_positions=None):
        try:
            # Prepare experiment
            config,sweep = self.prepare_experiment(sweep_configuration)

            self.logger.info(f'Instance = {str(instance_num)} START')
            
            # Find tqdm position
            if active_positions is not None:
                position_id = position_index(active_positions)
                # Activate it
                active_positions[position_id] = True
                # Sleep for a second so that active_positions
                # can sync with other instances
                self.logger.debug(f"{instance_num},{active_positions},{position_id}")
                time.sleep(2)
            else:
                position_id = 0
            
            # Create new experiment
            new_experiment = instantiate_experiment(
                experiment_type = config.settings['experiment_type'],
                config = config,
                sweep = sweep,
                instance = str(instance_num),
                base_dir = self.outputs_base_dir,
                experiment_id = self.outputs_experiment_id,
                position = (position_id+1),
                logger = self.logger
            )
            self.logger.debug('New experiment set up')
            # print(f"{new_experiment.outputs.sweep_id} set up")

            # Running experiment
            new_experiment.run()

            self.logger.info(f'Instance = {str(instance_num)} DONE')

            return position_id
        
        except Exception:
            raise Exception(f'failed running instance {instance_num}')

    def run_sequential(self,sweep_configurations):
        for instance,sweep_config in tqdm(
            enumerate(sweep_configurations),
            total=len(sweep_configurations),
            desc='Running sweeps in sequence',
            leave=False,
            position=0
        ):
            try:
                _ = self.prepare_instantiate_and_run(
                    instance_num = instance,
                    sweep_configuration = sweep_config,
                    active_positions = None
                )
            except Exception as exc:
                raise exc
    
    def run_concurrent(self,sweep_configurations):

        # Split the sweep configurations into chunks
        sweep_config_chunks = list(divide_chunks(
            sweep_configurations,
            self.config['outputs']['chunk_size']
        ))
        
        for chunk_id, sweep_config_chunk in enumerate(sweep_config_chunks):
            # Initialise progress bar
            progress = tqdm(
                total=len(sweep_config_chunk), 
                desc=f'Running sweeps concurrently: Batch {chunk_id+1}/{len(sweep_config_chunks)}',
                leave=True,
                position=0
            )
            
            with Manager() as manager:
                # Process active flag by tqdm position
                active_positions = manager.list([False]*self.n_workers)

                def my_callback(fut):
                    progress.update()
                    position_id = fut.result()
                    active_positions[position_id] = False

                with BoundedQueueProcessPoolExecutor(self.n_workers) as executor:
                    # Start the processes and ignore the results
                    for instance,sweep_config in enumerate(sweep_config_chunk):
                        future = executor.submit(
                            self.prepare_instantiate_and_run,
                            instance_num = instance,
                            sweep_configuration = sweep_config,
                            active_positions = active_positions
                        )
                        future.add_done_callback(my_callback)

                # Delete executor and progress bar
                progress.close()
                safe_delete(progress)
                executor.shutdown(wait=True)
                safe_delete(executor)