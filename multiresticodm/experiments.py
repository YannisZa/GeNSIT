import gc
import sys
import time
import logging
import itertools


from os import path
from tqdm import tqdm
from glob import glob
from copy import deepcopy
from datetime import datetime
from scipy.optimize import minimize
from joblib import Parallel, delayed
from multiprocessing.pool import Pool

from multiresticodm.utils import *
from multiresticodm.config import Config
from multiresticodm.inputs import Inputs
from multiresticodm.outputs import Outputs
from multiresticodm.global_variables import *
from multiresticodm.markov_basis import MarkovBasis
from multiresticodm.contingency_table import ContingencyTable,instantiate_ct
from multiresticodm.math_utils import apply_norm
from multiresticodm.contingency_table_mcmc import ContingencyTableMarkovChainMonteCarlo
from multiresticodm.harris_wilson_model import HarrisWilson
from multiresticodm.harris_wilson_model_neural_net import NeuralNet, HarrisWilson_NN
from multiresticodm.spatial_interaction_model import SpatialInteraction,instantiate_sim
from multiresticodm.spatial_interaction_model_mcmc import instantiate_spatial_interaction_mcmc

# Suppress scientific notation
np.set_printoptions(suppress=True)

def instantiate_experiment(experiment_type:str,config:Config,**kwargs):
    # Get whether sweep is active and its settings available
    conf_setts_copy = deepcopy(config.settings)
    sweep_param_key_paths = get_keys_in_path(
        conf_setts_copy,
        "sweep",
        path = []
    )
    sweep_param_key_paths = [] if sweep_param_key_paths is None else sweep_param_key_paths
    if hasattr(sys.modules[__name__], experiment_type):
        if config.settings.get("sweep_mode",False) and \
            len(sweep_param_key_paths) > 0:
            return ExperimentSweep(
                config=config,
                sweep_key_paths=sweep_param_key_paths,
                **kwargs
            )
        else:
            return getattr(sys.modules[__name__], experiment_type)(config=config,**kwargs)
    else:
        raise Exception(f'Experiment class {experiment_type} not found')

class ExperimentHandler(object):

    def __init__(self, config:Config, **kwargs):
        # Import logger
        self.level = config.level if hasattr(config,'level') else kwargs.get('level','INFO')
        self.logger = setup_logger(
            __name__,
            level = self.level,
            log_to_console = kwargs.get('log_to_console',False),
            log_to_file = kwargs.get('log_to_file',False),
        )
        
        # Get configuration
        self.config = config
        # Store experiment name to list index dictionary
        self.avail_experiments = self.config.settings['available_experiments']

        # Setup experiments
        self.setup_experiments(**kwargs)

    def setup_experiments(self,**kwargs):
        
        # Dictionary of experiment ids to experiment objects
        self.experiments = {}

        # Only run experiments specified in command line
        for experiment_type in self.config.settings['run_experiments']:
            # Check that such experiment already exists in the config file
            if experiment_type in self.avail_experiments.keys():
                # Construct sub-config with only data relevant for experiment
                experiment_config = Config(
                    settings=deepcopy(self.config.settings),
                    level=self.level
                )
                # Store one experiment
                experiment_config.settings['experiments'] = [
                    self.config.settings['experiments'][self.avail_experiments[experiment_type]]
                ]
                # Update id, seed and logging detail
                experiment_config.settings['experiment_type'] = experiment_type
                if self.config.settings['inputs'].get('dataset',None) is not None:
                    experiment_config['experiment_data'] = path.basename(path.normpath(self.config.settings['inputs']['dataset']))
                else:
                    raise Exception(f'No dataset found for experiment type {experiment_type}')
                # Instatiate new experiment
                experiment = instantiate_experiment(
                    experiment_type=experiment_type,
                    config=experiment_config,
                    log_to_file=True,
                    log_to_console=False,
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
        self.level = config.level if hasattr(config,'level') else kwargs.get('level','INFO')
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            level = self.level,
            log_to_console = kwargs.get('log_to_console',True),
            log_to_file = kwargs.get('log_to_file',True),
        )
        
        self.logger.debug(f"{self}")
        # Make sure you are reading a config
        if isinstance(config,dict):
            config = Config(
                settings=config,
                level=self.level
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
                level=self.level
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

    def run(self) -> None:
        pass

    def reset(self,metadata:bool=False) -> None:
        self.logger.note(f"Resetting experimental results to release memory.")
        
        # Get shapes 
        theta_shape = deepcopy(np.shape(self.thetas[-1])[0] if hasattr(self,'thetas') and self.thetas is not None else (2))
        log_destination_attraction_shape = np.shape(self.log_destination_attractions[-1])[0] if hasattr(self,'log_destination_attraction') and self.log_destination_attractions is not None else (self.sim.dims[1])
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
                level=self.level
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
        N = self.sim_mcmc.N
        # Define sample batch sizes
        sample_sizes = np.repeat(int(self.store_progress*N),np.floor(1/self.store_progress))
        if sample_sizes.sum() < N:
            sample_sizes = np.append(sample_sizes,[N-sample_sizes.sum()])
        sample_sizes = np.cumsum(sample_sizes)
        return sample_sizes
    
    def initialise_parameters(self):
        # Get batch sizes
        batch_sizes = self.define_sample_batch_sizes()
        self.config['n_batches'] = len(batch_sizes)

        # Load last sample
        if str_in_list('load_experiment',self.config['inputs']) and len(self.config['inputs']['load_experiment']) > 0:
            # Read last batch of experiments
            outputs = Outputs(
                self.config
            )
            # All parameter initialisations
            parameter_inits = dict(zip(self.output_names,[0]*len(self.output_names)))
            parameter_acceptances = dict(zip(self.output_names,[0]*len(self.output_names)))
            # Total samples for table,theta,x posteriors, respectively
            K = deep_call(self,'.od_mcmc.table_steps',1)
            M = deep_call(self,'.sim_mcmc.theta_steps',1)
            L = deep_call(self,'.sim_mcmc.log_destination_attraction_steps',1)
            assert K == 1
            assert M == 1
            assert L == 1
            for sample_name in self.output_names:
                # Find last batch of samples and load it
                filenames = glob(path.join(outputs.outputs_path,f'samples/{sample_name}*.npy'))
                # Sort by batch number if it exists
                if str_in_list('batch',filenames):
                    filenames = sorted(filenames) 
                else:
                    filenames = sorted(filenames,key=lambda f: int(f.split(f"batch_")[1].split(f"_samples.npy")[0]))
                if len(filenames) > 0:
                    # Get last batch
                    filepath = filenames[-1]
                    # Extrach batch number
                    parameter_inits['batch_counter'] = int(filepath.split(f"batch_")[1].split(f"_samples.npy")[0]) + 1
                    # Total number of samples taken across all batches except current
                    total_samples = batch_sizes[parameter_inits['batch_counter']-1]
                    # Load samples
                    samples = read_npy(filepath)
                    # Initialise parameter
                    parameter_inits[sample_name] = samples[-1].astype(DATA_TYPES[sample_name]) if hasattr(samples[-1],'__len__') else samples[-1]
                    # Extract number of iterations
                    parameter_inits['N0'] = int(total_samples//(K*M*L))
                    # Store acceptance rate
                    parameter_acceptances[sample_name] = self.config.get((f"{sample_name}_acceptance"),0)*total_samples
                else:
                    raise Exception(f'Failed tried loading experiment {self.experiment_id}')
        # Initialise parameters
        else:
            parameter_inits = {}
            if hasattr(self,'od_mcmc'):
                # Get initial margins
                try:
                    _ = deep_call(
                        self,
                        '.od_mcmc.sample_unconstrained_margins()',
                        None
                    )
                except:
                    self.logger.warning("Unconstrained margins could not be sampled.")
                
                try:
                    parameter_inits['table'] = deep_call(self,'.od_mcmc.initialise_table()',None)
                except:
                    parameter_inits['table'] = None
                    self.logger.warning("Table could not be initialised.")
            
            if hasattr(self,'sim_mcmc'):
                # Parameter values
                parameter_inits = {
                    **parameter_inits, 
                    **{
                        "theta":deep_call(self,'.sim_mcmc.theta0',None),
                        "sign":np.int8(1),
                        "log_destination_attraction":deep_call(self,'.sim_mcmc.log_destination_attraction0',None),
                        "batch_counter":0,
                        "N0":0
                    }
                }
            # Parameter acceptances
            parameter_acceptances = dict(zip(list(parameter_inits.keys()),[0]*len(parameter_inits.keys())))
        
        if parameter_inits['batch_counter'] == (len(batch_sizes) - 1):
            self.logger.warning("Experiment cannot be resumed.")

        # Update metadata initially
        self.update_metadata(
            0,
            parameter_inits.get('batch_counter',0),
            print_flag=False,
            update_flag=True
        )
        return parameter_inits,parameter_acceptances

    def initialise_data_structures(self):
        # Count the number of gradient descent steps
        self._time = 0
        self._write_every = self.config['outputs'].get('write_every',1)
        self._write_start = self.config['outputs'].get('write_start',1)
        # Get dimensions
        dims = self.config['inputs']['dims']
        
        # Setup neural net loss
        if str_in_list('loss',self.output_names):
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
        if str_in_list('log_destination_attraction',self.output_names):
            self.log_destination_attractions = self.outputs.h5group.create_dataset(
                "log_destination_attraction",
                (dims[1],0),
                maxshape=(dims[1],None),
                chunks=True,
                compression=3,
            )
            self.log_destination_attractions.attrs["dim_names"] = XARRAY_SCHEMA['log_destination_attraction']['coords']
            self.log_destination_attractions.attrs["coords_mode__time"] = "start_and_step"
            self.log_destination_attractions.attrs["coords__time"] = [self._write_start, self._write_every]
        
        # Setup computation time
        if str_in_list('computation_time',self.output_names):
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
        if str_in_list('theta',self.output_names):
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
        if str_in_list('sign',self.output_names):
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
        if str_in_list('table',self.output_names):
            self.tables = self.outputs.h5group.create_dataset(
                "table",
                (*dims,0),
                maxshape=(*dims,None),
                chunks=True,
                compression=3,
            )
            self.tables.attrs["dim_names"] = ["origin","destination","iter"]
            self.tables.attrs["coords_mode__time"] = "start_and_step"
            self.tables.attrs["coords__time"] = [self._write_start, self._write_every]

    def write_data(self,**kwargs):
        '''Write the current state into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        '''
        if kwargs.get('time',1) >= kwargs.get('write_start',1) and (kwargs.get('time',1) % kwargs.get('write_every',1) == 0):

            if 'loss' in self.output_names:
                self.losses.resize(self.losses.shape[0] + 1, axis=0)
                self.losses[-1] = kwargs.get('loss',None)

            if 'table' in self.output_names:
                self.tables.resize(self.tables.shape[-1] + 1, axis=len(self.tables.shape)-1)
                self.tables[...,-1] = kwargs.get('table',None)

            if 'theta' in self.output_names:
                for idx, dset in enumerate(self.thetas):
                    dset.resize(dset.shape[0] + 1, axis=0)
                    dset[-1] = kwargs.get('table',[None]*len(self.thetas))[idx]
            
            if 'log_destination_attraction' in self.output_names:
                self.log_destination_attractions.resize(self.log_destination_attractions.shape[-1] + 1, axis=len(self.log_destination_attractions.shape)-1)
                self.log_destination_attractions[...,-1] = kwargs.get('log_destination_attraction',np.array([None])).flatten()

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

    
    def update_metadata(self,N,batch_counter,print_flag:bool=False,update_flag:bool=True):
        
        # Total samples for table,theta,x posteriors, respectively
        K = self.od_mcmc.table_steps if hasattr(self,'od_mcmc') else 1
        M = self.sim_mcmc.theta_steps if hasattr(self,'sim_mcmc') else 1
        L = self.sim_mcmc.log_destination_attraction_steps if hasattr(self,'sim_mcmc') else 1
        # Update batch counter
        self.config['batch_counter'] = batch_counter

        if batch_counter == 0 or N <= 1:
            if hasattr(self,'sim_mcmc'):
                # Compute total sum of squares for r2 computation
                self.w_data = np.exp(self.sim_mcmc.sim.log_destination_attraction)
                w_data_centred = self.w_data - np.mean(self.w_data)
                self.ss_tot = np.dot(w_data_centred, w_data_centred)
                
        if update_flag:
            if hasattr(self,'od_mcmc'):
                # Get dims from table
                self.config['table_dim'] = 'x'.join(map(str,deep_call(self,'.od_mcmc.ct.dims',defaults=None)))
                # Get total from table
                self.config['table_total'] = int(deep_call(
                    input=self,
                    expressions='.od_mcmc.ct.margins[args1]',
                    defaults=-1,
                    args1=tuplize(range(deep_call(self,'.od_mcmc.ct.ndims()',0)))
                ))
                # Get markov basis length from table mcmc
                if isinstance(deep_call(self,'.od_mcmc.table_mb',None),MarkovBasis):
                    self.config['markov_basis_len'] = int(len(self.od_mcmc.table_mb))
            elif hasattr(self,'sim_mcmc'):
                # Compute total sum of squares for r2 computation
                self.w_data = np.exp(self.sim_mcmc.sim.log_destination_attraction)
                w_data_centred = self.w_data - np.mean(self.w_data)
                self.ss_tot = np.dot(w_data_centred, w_data_centred)
                # Get dims from sim
                self.config['table_dim'] = 'x'.join(map(str,deep_call(self,'.sim_mcmc.sim.dims',defaults=None)))
                # Get sim auxiliary params
                self.config['auxiliary_parameters'] = deep_call(
                    input=self,
                    expressions=[f'.sim_mcmc.sim.{param}' for param in ['delta','gamma','kappa','epsilon']],
                    defaults=[0.0,1.0,1.0,1.0]
                )
                self.config['noise_regime'] = deep_call(self,'.sim_mcmc.sim.noise_regime','undefined')
            elif hasattr(self,'ct'):
                # Get dims from table
                self.config['table_dim'] = 'x'.join(map(str,deep_call(self,'.ct.dims',defaults=None)))
                # Get total from table
                self.config['table_total'] = int(deep_call(
                    input=self,
                    expressions='.ct.margins[args1]',
                    defaults=-1,
                    args1=tuplize(range(deep_call(self,'.ct.ndims()',0)))
                ))
            elif hasattr(self,'harris_wilson_nn') and hasattr(self.harris_wilson_nn,'physics_model'):
                model_str = '.harris_wilson_nn.physics_model'
                # Get dims from sim
                self.config['table_dim'] = 'x'.join(map(str,deep_call(self,f'{model_str}.sim.dims',defaults=None)))
                self.config['noise_regime'] = deep_call(self,f'{model_str}.noise_regime','unknown')
            elif hasattr(self,'sim'):
                # Get dims from sim
                self.config['table_dim'] = 'x'.join(map(str,deep_call(self,'.sim.dims',defaults=None)))
            else:
                self.config['table_dim'] = [None,None]
                self.config['table_total'] = None
                self.config['noise_regime'] = None


        if hasattr(self,'signs') and hasattr(self,'thetas'):
            mu_theta = np.dot(self.signs.T,self.thetas)/np.sum(self.signs)
            std_theta = np.dot(self.signs.T,np.power(self.thetas,2))/np.sum(self.signs) - np.power(mu_theta,2)
            if update_flag:
                self.config['theta_mu'] = mu_theta.tolist()
                self.config['theta_sd'] = std_theta.tolist()
        if hasattr(self,'signs') and hasattr(self,'log_destination_attractions'):
            mu_x = np.dot(self.signs.T,self.log_destination_attractions)/np.sum(self.signs)
            # Compute R2
            w_pred = np.exp(mu_x.flatten(),dtype='float32')
            # Residiual sum squares
            res = w_pred - self.w_data
            ss_res = np.dot(res, res)
            # Regression sum squares
            r2 = 1. - ss_res/self.ss_tot
            if update_flag:
                self.config['log_destination_attraction_r2'] = r2
        
        if update_flag:
            self.config['execution_time'] = (time.time() - self.start_time)
            if hasattr(self,'theta_acceptance') and (M*N) > 0:
                self.config['theta_acceptance'] = int(100*getattr(self,'theta_acceptance',0)/(M*N))
            if hasattr(self,'total_signs') and (M*N) > 0:
                self.config['positives_percentage'] = int(100*getattr(self,'total_signs',0)/(M*N))
            if hasattr(self,'log_destination_attraction_acceptance') and (L*N) > 0:
                self.config['log_destination_attraction_acceptance'] = int(100*getattr(self,'log_destination_attraction_acceptance',0)/(L*N))
            if hasattr(self,'table_acceptance') and (K*N) > 0:
                self.config['table_acceptance'] = int(100*(getattr(self,'table_acceptance',0)/(K*N)))
            if hasattr(self,'total_signs') and (K*N) > 0:
                self.config['total_signs'] = int(100*(getattr(self,'total_signs',0)/(K*N)))

        if print_flag:
            print('Iteration',N,'batch',f"{batch_counter+1}/{self.config.get('n_batches',1)}")
            if hasattr(self,'total_signs') and (M*N) > 0:
                print('Positives %',int(100*(M*self.total_signs/(M*N))))
            if hasattr(self,'theta_acceptance') and (M*N) > 0:
                print('Theta acceptance', int(100*self.theta_acceptance/(M*N)))
            if hasattr(self,'log_destination_attraction_acceptance') and (L*N) > 0:
                print('Log destination attraction mean acceptance', int(100*self.log_destination_attraction_acceptance/(L*N)))
            if hasattr(self,'table_acceptance') and (K*N) > 0:
                print('Table acceptance',int(100*self.table_acceptance/(K*N)))
            if hasattr(self,'signs') and hasattr(self,'thetas'):
                print('Theta mu +/- 2*std')
                print(f"{str(mu_theta)} +/- {str(2*std_theta)}")
            if hasattr(self,'signs') and hasattr(self,'log_destination_attractions'):
                print(f"Log destination attraction R^2 = {r2}")

class RSquaredAnalysis(Experiment):

    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)
        # Build spatial interaction model
        self.sim = instantiate_sim(self.config)

        self.grid_size = config['grid_size']
        self.amin,self.amax = config['a_range']
        self.bmin,self.bmax = config['b_range']

    def run(self) -> None:
        
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
        max_w_prediction = np.exp(np.ones(self.sim.dims[1])*(1/self.sim.dims[1]))

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

class RSquaredAnalysisGridSearch(Experiment):

    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)
        # Build spatial interaction model
        self.sim = instantiate_sim(self.config)
        
        self.grid_size = config['grid_size']
        self.amin,self.amax = config['a_range']
        self.bmin,self.bmax = config['b_range']


    def run(self) -> None:

        # Beta max
        beta_scale_min = 100
        beta_scale_max = 10000
        beta_maxs = np.array([self.sim.bmax])
        # beta_maxs = np.append(beta_maxs,np.linspace(beta_scale_min,beta_scale_max,9,endpoint=True))

        # Alpha, beta
        alpha_values = np.linspace(self.amin, self.amax, self.grid_size,endpoint=True)
        beta_values = np.linspace(self.bmin, self.bmax, self.grid_size,endpoint=True)

        
        # Delta
        delta_min = 0.001642710997442455#self.sim.delta*0.85
        delta_max = 0.01642710997442455#self.sim.delta*1.15
        deltas = np.array([self.sim.delta])
        deltas = np.append(deltas,np.linspace(delta_min,delta_max,99,endpoint=True))
        # deltas = np.linspace(delta_min,delta_max,10,endpoint=True) #[self.sim.delta]

        def kappa_from_delta(d):
            return (np.sum(self.sim.origin_demand)+d*self.sim.dims[1])/np.exp(self.sim.log_destination_attraction).sum()
        print()

        # Kappa
        # kappa_min = self.sim.kappa-0.1
        # kappa_max = self.sim.kappa+0.1
        # kappas = np.linspace(kappa_min,kappa_max,10,endpoint=True) #[self.sim.kappa] #
        kappas = np.array([self.sim.kappa])

        # Total sum squares
        w_data = np.exp(self.sim.log_destination_attraction)
        w_data_centred = w_data - np.mean(w_data)
        ss_tot = np.dot(w_data_centred, w_data_centred) 

        # Initialise
        last_r2 = 0
        max_r2 = 0
        argmax_w_prediction = np.exp(np.ones(self.sim.dims[1])*(1/self.sim.dims[1]))
        # Define theta
        theta = np.array([0.0,
                        0.0,
                        self.sim.delta,
                        self.sim.gamma,
                        self.sim.kappa,
                        self.sim.epsilon],dtype='float64')
        argmax_theta = deepcopy(theta)
        argmax_beta_scale = deepcopy(self.sim.bmax)
        r2_values = np.zeros((self.grid_size, self.grid_size,len(beta_maxs),len(kappas),len(deltas)),dtype='float16')

        for bi,beta_scale in tqdm(enumerate(beta_maxs),total=len(beta_maxs)):
            # Perform grid evaluations
            # for ki,kappa in tqdm(enumerate(kappas),total=len(kappas),leave=False):
                # theta[4] = kappa
            for di,delta in tqdm(enumerate(deltas),total=len(deltas),leave=False):
                theta[2] = delta
                theta[4] = kappa_from_delta(delta)
                ki = 0
                for i in tqdm(range(self.grid_size),leave=False):
                    for j in range(self.grid_size):
                        try:
                            # Get parameter theta
                            theta[0] = alpha_values[j]
                            theta[1] = beta_values[i]*beta_scale

                            # Minimise potential function
                            potential_func = minimize(self.sim.sde_potential_and_gradient,
                                                        self.sim.log_destination_attraction, 
                                                        method='L-BFGS-B', 
                                                        jac=True, 
                                                        args=(theta), 
                                                        options={'disp': False})
                            w_pred = np.exp(potential_func.x,dtype='float64')
                            
                            # Residiual sum squares
                            res = w_pred - w_data
                            ss_res = np.dot(res, res)

                            # Regression sum squares
                            r2_values[i,j,bi,ki,di] = 1. - ss_res/ss_tot

                            # Update max
                            if r2_values[i,j,bi,ki,di] > max_r2:
                                max_r2 = deepcopy(r2_values[i,j,bi,ki,di])
                                argmax_w_prediction = deepcopy(w_pred)
                                argmax_theta = deepcopy(theta)
                                argmax_beta_scale = deepcopy(beta_scale)
                            last_r2 = r2_values[i,j,bi,ki,di]

                        except Exception as e:
                            r2_values[i,j,bi,ki,di] = last_r2
                            
                
                if self.config['print_statements']:
                    print(f'kappa = {argmax_theta[4]}, delta = {argmax_theta[2]}, beta_max = {argmax_beta_scale}')
                    print("Fitted alpha, beta and scaled beta values:")
                    print(argmax_theta[0],argmax_theta[1]*1/(argmax_beta_scale), argmax_theta[1])
                    print("R^2 value:")
                    print(max_r2)
                    print('\n')
        
        # Output results
        print('\n')
        print('\n')
        print('GLOBAL OPTIMISATION PROCEDURE ENDED')
        print(f'kappa = {argmax_theta[4]}, delta = {argmax_theta[2]}, beta_max = {argmax_beta_scale}')
        print("Fitted alpha, beta and scaled beta values:")
        print(argmax_theta[0],argmax_theta[1]*1/(argmax_beta_scale), argmax_theta[1])
        print("R^2 value:")
        print(max_r2)
        print(np.max(r2_values.ravel()))
        print('\n')

        # Save fitted values to parameters
        self.config['fitted_alpha'] = argmax_theta[0]
        self.config['fitted_scaled_beta'] = argmax_theta[1]/(argmax_beta_scale)
        self.config['fitted_beta_scaling_factor'] = argmax_beta_scale
        self.config['fitted_beta'] = argmax_theta[1]
        self.config['fitted_kappa'] = kappa_from_delta(argmax_theta[2])
        self.config['fitted_delta'] = argmax_theta[2]
        self.config['kappa_min'] = kappa_from_delta(delta_min)#kappa_min
        self.config['kappa_max'] = kappa_from_delta(delta_max)#kappa_max
        self.config['delta_min'] = delta_min
        self.config['delta_max'] = delta_max
        self.config['beta_scale_min'] = beta_scale_min
        self.config['beta_scale_max'] = beta_scale_max
        self.config['R^2'] = float(max_r2)
        self.config['noise_regime'] = self.sim.noise_regime
        self.config['predicted_w'] = argmax_w_prediction.tolist()

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
        self.sim_mcmc = instantiate_spatial_interaction_mcmc(sim)

    def run(self) -> None:
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
        lap_c1 = 0.5*self.sim.dims[1]*np.log(2.*np.pi)

        # Perform grid evaluations
        for i in tqdm(range(self.grid_size)):
            for j in tqdm(range(self.grid_size),leave=False):
                try:
                    # Residiual sum squares
                    theta[0] = XX[i, j]
                    theta[1] = YY[i, j]

                    # Minimise potential function
                    log_z_inverse,_ = self.sim_mcmc.biased_z_inverse(0,theta)

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

        
class LogTargetAnalysisGridSearch(Experiment):

    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)
        
        # Build spatial interaction model
        sim = instantiate_sim(self.config)

        self.grid_size = config['grid_size']
        self.amin,self.amax = config['a_range']
        self.bmin,self.bmax = config['b_range']

        # Spatial interaction model MCMC
        self.sim_mcmc = instantiate_spatial_interaction_mcmc(sim)

    def run(self) -> None:

        # Initialise
        self.beta_scale_min = 1000
        self.beta_scale_max = 10000
        beta_maxs = np.linspace(self.beta_scale_min,self.beta_scale_max,self.grid_size,endpoint=True)

        max_target = -1e10
        argmax_theta = np.array([0,
                        0, 
                        self.sim.delta,
                        self.sim.gamma,
                        self.sim.kappa,
                        self.sim.epsilon],dtype='float64')
        argmax_beta_scale = self.sim.bmax
        log_targets = np.empty((self.grid_size,self.grid_size,self.grid_size,self.grid_size, self.grid_size))
        log_targets[:] = np.nan

        # Normalise initial log destination sizes
        xd = self.sim.log_destination_attraction

        # Alpha,beta
        alpha_values = np.linspace(self.amin, self.amax, self.grid_size,endpoint=True)
        beta_values = np.linspace(self.bmin, self.bmax, self.grid_size,endpoint=True)
        XX, YY = np.meshgrid(alpha_values, beta_values)

        # Kappa
        self.kappa_min = self.sim.kappa-0.2
        self.kappa_max = self.sim.kappa+0.2
        kappas = np.linspace(self.kappa_min,self.kappa_max,self.grid_size,endpoint=True)

        # Delta 
        self.delta_min = self.sim.delta*0.5
        self.delta_max = self.sim.delta+1.5
        deltas = np.linspace(self.delta_min,self.delta_max,self.grid_size,endpoint=True)

        # Normalisation factor
        lap_c1 = 0.5*self.sim.dims[1]*np.log(2.*np.pi)

        for bi,beta in tqdm(enumerate(beta_maxs),total=len(beta_maxs)):
            # Multiply by beta scaling factor
            self.bmin *= beta
            self.bmax *= beta
            # Perform grid evaluations
            for di,delta in tqdm(enumerate(deltas),total=len(deltas),leave=False):
                for ki,kappa in tqdm(enumerate(kappas),total=len(kappas),leave=False):
                    for i in tqdm(range(self.grid_size),leave=False):
                        for j in range(self.grid_size):
                            try:
                                # Define theta
                                theta = np.array([XX[i, j], 
                                                YY[i, j], 
                                                delta,
                                                self.sim.gamma,
                                                kappa,
                                                self.sim.epsilon],dtype='float64')

                                # Minimise potential function
                                log_z_inverse,_ = self.sim_mcmc.biased_z_inverse(0,theta)

                                # Compute potential function
                                potential_func,_ = self.sim.sde_potential_and_gradient(xd,theta)
                                log_target = log_z_inverse-potential_func-lap_c1

                                # Store log_target
                                log_targets[bi,di,ki,i,j] = log_target

                                if log_target > max_target:
                                    argmax_beta_scale = deepcopy(beta)
                                    argmax_theta = deepcopy(theta)
                                    max_target = log_target

                            except Exception as e:
                                None


                if self.config['print_statements']:
                    print("Fitted alpha, beta and scaled beta values:")
                    print(argmax_theta[0],argmax_theta[1]*self.amax/(argmax_beta_scale), argmax_theta[1])
                    print("Log target:")
                    print(max_target)
                    print('kappa',argmax_theta[4])
                    print('delta',argmax_theta[2])
                    print('beta scaele',argmax_beta_scale)
                    print('\n')
            # Reset beta scaling
            self.bmin /= beta
            self.bmax /= beta

        print('\n')
        print('\n')
        print('GLOBAL OPTIMISATION PROCEDURE ENDED')
        print("Fitted alpha, beta and scaled beta values:")
        print(argmax_theta[0],argmax_theta[1]*self.amax/(argmax_beta_scale), argmax_theta[1])
        print("Log target:")
        print(max_target)
        print('kappa',argmax_theta[4])
        print('delta',argmax_theta[2])
        print('beta scaele',argmax_beta_scale)
        print('\n')

        # Save fitted values to parameters
        self.config['fitted_alpha'] = argmax_theta[0]
        self.config['fitted_scaled_beta'] = argmax_theta[1]*1/(argmax_beta_scale)
        self.config['fitted_beta'] = argmax_theta[1]
        self.config['fitted_kappa'] = argmax_theta[4]
        self.config['fitted_delta'] = argmax_theta[2]
        self.config['kappa_min'] = self.kappa_min
        self.config['kappa_max'] = self.kappa_max
        self.config['delta_min'] = self.delta_min
        self.config['delta_max'] = self.delta_max
        self.config['beta_scale_min'] = self.beta_scale_min
        self.config['beta_scale_max'] = self.beta_scale_max
        self.config['log_target'] = max_target
        self.config['noise_regime'] = self.sim.noise_regime

        # Append to result array
        self.results = [{"samples":{"log_target":log_targets}}]



class SIM_MCMC(Experiment):
    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)

        # Build spatial interaction model
        sim = instantiate_sim(self.config)

        # Spatial interaction model MCMC
        self.sim_mcmc = instantiate_spatial_interaction_mcmc(sim,kwargs.get('log_to_console',True))

        # Delete duplicate of spatial interaction model
        safe_delete(self.sim)
        
        # Run garbage collector
        gc.collect()

        self.output_names = ['log_destination_attraction','theta','sign']
        
    def run(self) -> None:

        self.logger.info(f"Running MCMC inference of {self.sim_mcmc.sim.noise_regime} noise SpatialInteraction.")

        # Time run
        self.start_time = time.time()

        # Fix random seed
        set_seed(self.seed)

        # Initialise parameters
        parameter_inits,parameter_acceptances = self.initialise_parameters()

        # If experiment is complete return
        if parameter_inits['N0'] >= self.sim_mcmc.N:
            return
        
        # Print initialisations
        # self.print_initialisations(parameter_inits,print_lengths=False,print_values=True)

        # Initialise signs
        sign_sample = parameter_inits['sign'].astype('int8')
        self.signs = np.zeros((0,1),dtype='int8')
        self.total_signs = parameter_acceptances['sign']

        # Initialise theta
        theta_sample = parameter_inits['theta'].astype('float32')
        self.thetas = np.zeros((0,*np.shape(theta_sample)),dtype='float32')
        self.theta_acceptance = parameter_acceptances['theta']

        # Multiply beta by total cost
        theta_sample_scaled_and_expanded = np.concatenate([
                                                theta_sample,
                                                np.array([
                                                    self.sim_mcmc.sim.delta,
                                                    self.sim_mcmc.sim.gamma,
                                                    self.sim_mcmc.sim.kappa,
                                                    self.sim_mcmc.sim.epsilon
                                                ])
                                        ])
        theta_sample_scaled_and_expanded[1] *= self.sim_mcmc.sim.bmax

        # Initialise destination attraction
        log_destination_attraction_sample = parameter_inits['log_destination_attraction'].astype('float32')
        self.log_destination_attractions = np.zeros((0,*np.shape(log_destination_attraction_sample)),dtype='float32')
        self.log_destination_attraction_acceptance = parameter_acceptances['log_destination_attraction']

        # Compute initial log inverse z(\theta)
        log_z_inverse, sign_sample = self.sim_mcmc.z_inverse(
                                    0,
                                    theta_sample_scaled_and_expanded
                            )
        # Evaluate log potential function for initial choice of \theta
        V, gradV = self.sim_mcmc.sim.sde_potential_and_gradient(
                                    log_destination_attraction_sample,
                                    theta_sample_scaled_and_expanded
                            )

        # Store number of samples
        N = self.sim_mcmc.sim.config.settings['mcmc']['N']
        # Total samples for table,theta,x posteriors, respectively
        M = self.sim_mcmc.theta_steps
        L = self.sim_mcmc.log_destination_attraction_steps

        # Define sample batch sizes
        sample_sizes = self.define_sample_batch_sizes()
        # Initialise batch counter
        batch_counter = parameter_inits['batch_counter']

        if self.print_statements:
            print('theta',theta_sample_scaled_and_expanded)
            print('destination_attraction_prev',np.exp(log_destination_attraction_sample))
            print('\n')

        for i in tqdm(range(parameter_inits['N0'],N),disable=self.config['disable_tqdm']):
        
            # Run theta sampling
            for j in tqdm(
                        range(M),
                        disable=self.sim_mcmc.sim.config.settings['mcmc']['spatial_interaction_model']['disable_tqdm'],
                        leave=(not self.sim_mcmc.logger.disabled)
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
                sign_sample = self.sim_mcmc.theta_step(
                            i,
                            theta_sample,
                            log_destination_attraction_sample,
                            auxiliary_values
                        )
                # Append theta new to vector
                self.thetas =np.append(self.thetas,theta_sample[np.newaxis,:],axis=0)
                # Increment acceptance
                self.theta_acceptance+= theta_acc
                # Append sign to vector
                self.signs =np.append(self.signs,np.array([sign_sample]).reshape((1,1)).astype('int8'),axis=0)
                # Increment total signs
                self.total_signs += sign_sample
            
            # Run x sampling
            for l in tqdm(
                        range(L),
                        disable=self.sim_mcmc.sim.config.settings['mcmc']['spatial_interaction_model']['disable_tqdm'],
                        leave=(not self.sim_mcmc.logger.disabled)
                ):
                # Gather all additional values
                auxiliary_values = [V, 
                                gradV]
                # Take step
                log_destination_attraction_sample, \
                log_dest_attract_acc, \
                V, \
                gradV = self.sim_mcmc.log_destination_attraction_step(
                    theta_sample,
                    log_destination_attraction_sample,
                    auxiliary_values
                )
                # Append log destination attraction new to vector
                self.log_destination_attractions =np.append(self.log_destination_attractions,log_destination_attraction_sample[np.newaxis,:],axis=0)
                # Increment acceptance
                self.log_destination_attraction_acceptance += log_dest_attract_acc

            # Print metadata
            if ((int(self.print_percentage*N) > 0) and (i % int(self.print_percentage*N) == 0) or i == (N-1)):
                self.update_metadata(
                    i+1,
                    batch_counter,
                    print_flag=True,
                    update_flag=False
                )
            
            # Export batch and reset
            if ((i+1) >= sample_sizes[batch_counter]):
                # Append to result array
                self.results = [{"samples":{
                                    f"sign_batch_{batch_counter}":np.asarray(self.signs,dtype='int8'),
                                    f"theta_batch_{batch_counter}":np.asarray(self.thetas,dtype='float32'),
                                    f"log_destination_attraction_batch_{batch_counter}":np.asarray(self.log_destination_attractions,dtype='float32')
                                }}]
                
                self.update_metadata(
                    i+1,
                    batch_counter,
                    print_flag=False,
                    update_flag=True
                )
                
                # Write samples and metadata
                # self.write(metadata=True)

                # Reset tables and columns sums to release memory
                self.reset(metadata=False)

                # Increment batch counter
                batch_counter += 1
        
        # Unfix random seed
        set_seed(None)

        self.logger.info(f"Experimental results have been compiled.")

        # Append to result array
        self.results.append({"samples":{"log_destination_attraction":self.log_destination_attractions,
                                        "theta":self.thetas,
                                        "sign":self.signs}})


class JointTableSIM_MCMC(Experiment):

    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)
        
        # Setup table
        ct = instantiate_ct(table=None,config=self.config)
        # Update table distribution
        self.config.settings['inputs']['contingency_table']['distribution_name'] = ct.distribution_name
        # Build spatial interaction model
        sim = instantiate_sim(self.config)

        # Spatial interaction model MCMC
        self.sim_mcmc = instantiate_spatial_interaction_mcmc(
            sim,
            log_to_console=kwargs.get('log_to_console',True)
        )
        # Contingency Table mcmc
        self.od_mcmc = ContingencyTableMarkovChainMonteCarlo(
            ct,
            table_mb=None,
            log_to_console=kwargs.get('log_to_console',True)
        )

        # Delete duplicate of contingency table and spatial interaction model
        safe_delete(self.ct)
        safe_delete(self.sim)
        # Run garbage collector
        gc.collect()

        self.output_names = ['table','log_destination_attraction','theta','sign']

        # print_json(self.config)
        print(self.sim_mcmc)
        print(self.od_mcmc)

    def run(self) -> None:

        self.logger.info(f"Running joint MCMC inference of contingency tables and {self.sim_mcmc.sim.noise_regime} noise SpatialInteraction.")

        # Time run
        self.start_time = time.time()

        # Fix random seed
        set_seed(self.seed)

        # Initialise parameters
        parameter_inits,parameter_acceptances = self.initialise_parameters()
        # If experiment is complete return
        if parameter_inits['N0'] >= self.sim_mcmc.N:
            return
 
        # Initialise signs
        sign_sample = parameter_inits['sign'].astype('int8')
        self.signs = np.zeros((0,1),dtype='int8')
        self.total_signs = parameter_acceptances['sign']

        # Initialise theta
        theta_sample = parameter_inits['theta'].astype('float32')
        self.thetas = np.zeros((0,*np.shape(theta_sample)),dtype='float32')
        self.theta_acceptance = parameter_acceptances['theta']

        # Multiply beta by total cost
        theta_sample_scaled_and_expanded = np.concatenate([
                                                theta_sample,
                                                np.array([
                                                    self.sim_mcmc.sim.delta,
                                                    self.sim_mcmc.sim.gamma,
                                                    self.sim_mcmc.sim.kappa,
                                                    self.sim_mcmc.sim.epsilon,
                                                ])
                                        ]).astype('float32')
        theta_sample_scaled_and_expanded[1] *= self.sim_mcmc.sim.bmax

        # Initialise destination attraction
        log_destination_attraction_sample = parameter_inits['log_destination_attraction'].astype('float32')
        self.log_destination_attractions = np.zeros((0,*np.shape(log_destination_attraction_sample)),dtype='float32')
        self.log_destination_attraction_acceptance = parameter_acceptances['log_destination_attraction']
        
        # Initialise table
        table_sample = parameter_inits['table'].astype('int32')
        self.tables = np.zeros((0,*np.shape(table_sample)),dtype='int32')
        self.table_acceptance = parameter_acceptances['table']
        
        # Compute intensity
        log_intensity = self.sim_mcmc.sim.log_intensity(
                            log_destination_attraction_sample,
                            theta_sample_scaled_and_expanded,
                            total_flow=self.od_mcmc.ct.margins[tuplize(range(self.od_mcmc.ct.ndims()))].item()
                        )
        # Compute table likelihood and its gradient
        negative_log_table_likelihood = self.sim_mcmc.negative_table_log_likelihood(
                log_intensity,
                table_sample
        )
        # Compute initial log inverse z(\theta)
        log_z_inverse, sign_sample = self.sim_mcmc.z_inverse(
                                    0,
                                    theta_sample_scaled_and_expanded
                            )
        # Evaluate log potential function for initial choice of \theta
        V, gradV = self.sim_mcmc.sim.sde_potential_and_gradient(
                                    log_destination_attraction_sample,
                                    theta_sample_scaled_and_expanded
                            )
        
        # Store number of samples
        # Total samples for joint posterior
        N = self.sim_mcmc.sim.config.settings['mcmc']['N']
        # Total samples for table,theta,x posteriors, respectively
        K = self.od_mcmc.table_steps
        M = self.sim_mcmc.theta_steps
        L = self.sim_mcmc.log_destination_attraction_steps

        # Define sample batch sizes
        sample_sizes = self.define_sample_batch_sizes()
        # Initialise batch counter
        batch_counter = parameter_inits['batch_counter']

        if self.print_statements:
            print('theta',theta_sample_scaled_and_expanded)
            print('destination_attraction_prev',np.exp(log_destination_attraction_sample))
            print('\n')

        for i in tqdm(range(parameter_inits['N0'],N),disable=self.config['disable_tqdm']):
            
            # Run theta sampling
            for j in tqdm(
                        range(M),
                        disable=self.sim_mcmc.sim.config.settings['mcmc']['spatial_interaction_model']['disable_tqdm'],
                        leave=(not self.sim_mcmc.logger.disabled)
                ):

                # Gather all additional values
                auxiliary_values = [V, 
                                gradV, 
                                log_z_inverse, 
                                log_intensity,
                                negative_log_table_likelihood, 
                                sign_sample]
            
                # Take step in theta space
                theta_sample, \
                theta_acc, \
                V, \
                gradV, \
                log_z_inverse, \
                log_intensity, \
                negative_log_table_likelihood, \
                sign_sample = self.sim_mcmc.theta_step(
                            i,
                            theta_sample,
                            log_destination_attraction_sample,
                            table_sample,
                            auxiliary_values
                )
                # Append theta new to vector
                self.thetas = np.append(self.thetas,theta_sample[np.newaxis,:],axis=0)
                # Increment acceptance
                self.theta_acceptance += theta_acc
                # Append sign to vector
                self.signs = np.append(self.signs,np.array([sign_sample]).reshape((1,1)).astype('int8'),axis=0)
                # Increment total signs
                self.total_signs += sign_sample
            
            # Run x sampling
            for l in tqdm(
                        range(L),
                        disable=self.sim_mcmc.sim.config.settings['mcmc']['spatial_interaction_model']['disable_tqdm'],
                        leave=(not self.sim_mcmc.logger.disabled)
                ):
                # Gather all additional values
                auxiliary_values = [V,
                                gradV,
                                log_intensity,
                                negative_log_table_likelihood]
            
                # Take step in X space
                log_destination_attraction_sample, \
                log_dest_attract_acc, \
                V, \
                gradV, \
                log_intensity, \
                negative_log_table_likelihood = self.sim_mcmc.log_destination_attraction_step(
                    theta_sample,
                    log_destination_attraction_sample,
                    table_sample,
                    auxiliary_values
                )
                # Append log destination attraction new to vector
                self.log_destination_attractions = np.append(self.log_destination_attractions,log_destination_attraction_sample[np.newaxis,:],axis=0)
                # Increment acceptance
                self.log_destination_attraction_acceptance += log_dest_attract_acc
            
            # Run table sampling
            for k in tqdm(
                    range(K),
                    disable=self.sim_mcmc.sim.config.settings['mcmc']['contingency_table']['disable_tqdm'],
                    leave=(not self.od_mcmc.logger.disabled)
                ):
                # Take step
                table_sample, table_accepted = self.od_mcmc.table_gibbs_step(
                    table_sample,
                    log_intensity
                )
                self.tables = np.append(self.tables,table_sample[np.newaxis,:],axis=0)
                # Increment acceptance
                self.table_acceptance += table_accepted

            # Compute table likelihood for updated table
            negative_log_table_likelihood = self.sim_mcmc.negative_table_log_likelihood(
                    log_intensity,
                    table_sample
            )

            # Print metadata
            if ((int(self.print_percentage*N) > 0) and (i % int(self.print_percentage*N) == 0) or i == (N-1)):
                self.update_metadata(
                    i+1,
                    batch_counter,
                    print_flag=True,
                    update_flag=False
                )
            
            # Export batch and reset
            if ((i+1) >= sample_sizes[batch_counter]):
                # Append to result array
                self.results = [{"samples":{
                                    f"table_batch_{batch_counter}":np.asarray(self.tables,dtype='int32'),
                                    f"sign_batch_{batch_counter}":np.asarray(self.signs,dtype='int8'),
                                    f"theta_batch_{batch_counter}":np.asarray(self.thetas,dtype='float32'),
                                    f"log_destination_attraction_batch_{batch_counter}":np.asarray(self.log_destination_attractions,dtype='float32')
                                }}]
                
                self.update_metadata(
                    i+1,
                    batch_counter,
                    print_flag=False,
                    update_flag=True
                )
                
                # Write samples and metadata
                # self.write(metadata=True)

                # Reset tables and columns sums to release memory
                self.reset(metadata=False)

                # Increment batch counter
                batch_counter += 1

        # Unfix random seed
        set_seed(None)

        self.logger.info(f"Experimental results have been compiled.")

        # Append to result array
        self.results.append({"samples":{"log_destination_attraction":self.log_destination_attractions,
                                        "theta":self.thetas,
                                        "sign":self.signs,
                                        "table":self.tables}})

class Table_MCMC(Experiment):
    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)

        # Setup table
        ct = instantiate_ct(table=None,config=self.config)
        # Update table distribution
        self.config.settings['inputs']['contingency_table']['distribution_name'] = ct.distribution_name
        # Build spatial interaction model
        sim = instantiate_sim(self.config)
        
        # Update config with table dimension
        # self.config['table_dim'] = 'x'.join(map(str,ct.dims))
        # self.config['table_total'] = int(ct.margins[tuplize(range(ct.ndims()))])
        # Initialise intensities at ground truths
        if (ct is not None) and (ct.table is not None):
            self.logger.info("Using table as ground truth intensity")
            # Use true table to construct intensities
            with np.errstate(invalid='ignore',divide='ignore'):
                self.true_log_intensities = np.log(
                                                ct.table,
                                                dtype='float32'
                                            )
            
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

        # Contingency Table mcmc
        self.od_mcmc = ContingencyTableMarkovChainMonteCarlo(
            ct,
            table_mb=None,
            log_to_console=kwargs.get('log_to_console',True)
        )
        # Set table steps to 1
        self.od_mcmc.table_steps = 1
        # Delete duplicate of contingency table and spatial interaction model
        safe_delete(self.ct)
        safe_delete(self.sim)
        # Run garbage collector to release memory
        gc.collect()

        self.output_names = ['table']

    def initialise_parameters(self):
        if str_in_list('load_experiment',self.config['inputs']) and \
            len(self.config['inputs']['load_experiment']) > 0:
            # Read last batch of experiments 
            outputs = Outputs(
                self.config
            )
            # Parameter initialisations
            parameter_inits = {}
            # Total samples for joint posterior
            N = self.sim_mcmc.sim.config.settings['mcmc']['N']
            # Find last batch of samples and load it
            filenames = sorted(glob(path.join(outputs.outputs_path,f'samples/table*.npy')))
            if len(filenames) > 0:
                # Get last batch
                filepath = filenames[-1]
                # Extrach batch number 
                parameter_inits['batch_counter'] = int(filepath.split(f"batch_")[1].split(f"_samples.npy")[0])
                # Load samples
                samples = read_npy(filepath)
                # Initialise parameter
                parameter_inits["tables"] = samples.astype('float32')
                # Extract number of iterations
                parameter_inits['N0'] = len(samples)
            else:
                parameter_inits = {
                    "batch_counter":0,
                    "N0":1
                }
        else:
            self.od_mcmc.sample_unconstrained_margins()
            parameter_inits = {
                "batch_counter":0,
                "N0":1,
                "tables":np.asarray([self.od_mcmc.initialise_table(np.exp(self.true_log_intensities))],dtype='int32')
            }
        # Update metadata initially
        self.update_metadata(
            0,
            parameter_inits.get('batch_counter',0),
            print_flag=False,
            update_flag=True
        )
        return parameter_inits

    def run(self) -> None:

        # Time run
        self.start_time = time.time()

        # Fix random seed
        set_seed(self.seed)

        # Initialise parameters
        parameter_inits = self.initialise_parameters()
        
        # Initialise table
        self.tables = parameter_inits['tables'].astype('int32')
        table_sample = self.tables[-1]
        self.table_acceptance = 1

        # Store number of samples
        # Total samples for joint posterior
        N = self.od_mcmc.ct.config.settings['mcmc']['N']

        # Define sample batch sizes
        sample_sizes = self.define_sample_batch_sizes()
        # Initialise batch counter
        batch_counter = parameter_inits['batch_counter']

        # Loop through each sample batch
        for i in tqdm(range(parameter_inits['N0'],N),disable=self.config['disable_tqdm']):
                
            # Take step
            table_sample, table_accepted = self.od_mcmc.table_gibbs_step(
                table_sample, 
                self.true_log_intensities
            )
            # Append new tables samples to list
            self.tables = np.append(self.tables,table_sample[np.newaxis,:],axis=0)
            # Increment acceptance
            self.table_acceptance += table_accepted
                    
            # Print metadata
            if ((int(self.print_percentage*N) > 0) and (i % int(self.print_percentage*N) == 0) or i == (N-1)):
                self.update_metadata(
                    i+1,
                    batch_counter,
                    print_flag=True,
                    update_flag=False
                )
            
            # Export batch and reset
            if ((i+1) >= sample_sizes[batch_counter]):
                # Append to result array
                self.results = [{"samples":{
                                    f"table_batch_{batch_counter}":np.asarray(self.tables,dtype='int32')
                                }}]
                
                self.update_metadata(
                    i+1,
                    batch_counter,
                    print_flag=False,
                    update_flag=True
                )
                
                # Write samples and metadata
                # self.write(metadata=True)

                # Reset tables and columns sums to release memory
                self.reset(metadata=False)

                # Increment batch counter
                batch_counter += 1

        # Unfix random seed
        set_seed(None)

        self.logger.info(f"Experimental results have been compiled.")

        # Append to result array
        self.results.append({"samples":{"table":self.tables}})


class TableSummariesMCMCConvergence(Experiment):
    def __init__(self, config:Config, **kwargs):
        # Initalise superclass
        super().__init__(config,**kwargs)

        # Setup table
        ct = instantiate_ct(table=None,config=self.config)
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
            log_to_console=kwargs.get('log_to_console',True)
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
                    table_mb=self.samplers['0'].table_mb,
                    log_to_console=True
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

    
    def run(self) -> None:

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
        for i in tqdm(range(1,N),disable=self.config['disable_tqdm'],leave=False):
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
            instance = kwargs.get('instance','')
        )
        # Set up the neural net
        self.logger.note("Initializing the neural net ...")
        neural_network = NeuralNet(
            input_size=self.inputs.data.destination_attraction_ts.shape[1],
            output_size=len(config['inputs']['to_learn']),
            **config['neural_network']['hyperparameters'],
        ).to(self.device)

        # Pass inputs to device
        self.inputs.pass_to_device()

        # Instantiate Spatial Interaction Model
        self.logger.note("Initializing the spatial interaction model ...")

        sim = instantiate_sim(
            sim_type= config['spatial_interaction_model']['sim_type'],
            config = config,
            true_parameters = config['spatial_interaction_model']['parameters'],
            device = self.device,
            instance = kwargs.get('instance',''),
            **vars(self.inputs.data)
        )
        # Get and remove config
        config = pop_variable(sim,'config')

        # Build Harris Wilson model
        self.logger.note("Initializing the Harris Wilson physics model ...")
        harris_wilson_model = HarrisWilson(
            sim = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            device = self.device,
            instance = kwargs.get('instance','')
        )
        # Get and remove config
        config = pop_variable(harris_wilson_model,'config')

        # Instantiate harris and wilson neural network model
        self.logger.note("Initializing the Harris Wilson Neural Network model ...")
        self.harris_wilson_nn = HarrisWilson_NN(
            rng = rng,
            config = config,
            neural_net = neural_network,
            loss_function = config['neural_network'].pop('loss_function'),
            physics_model = harris_wilson_model,
            to_learn = config['inputs']['to_learn'],
            write_every = config['outputs']['write_every'],
            write_start = config['outputs']['write_start'],
            instance = kwargs.get('instance','')
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
            log_to_file=True,
            log_to_console=True,
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
        
    def run(self) -> None:

        self.logger.note(f"Running Neural Network training of Harris Wilson model.")

        # Time run
        start_time = time.time()

        # Initialise data structures
        self.initialise_data_structures()
        
        # Train the neural net
        num_epochs = self.config['training']['N']

        # For each epoch
        for e in range(num_epochs):

            # Track the epoch training time
            start_time = time.time()
            
            # Train neural net
            theta_sample, log_destination_attraction_sample = self.harris_wilson_nn.epoch(
                training_data=self.inputs.data.destination_attraction_ts, 
                experiment=self,
                **self.config['training']
            )
            
            # Add axis to every sample to ensure compatibility 
            # with the functions used below
            theta_sample = torch.unsqueeze(theta_sample,0)
            log_destination_attraction_sample = torch.unsqueeze(log_destination_attraction_sample,0)

            # intensity = np.exp(log_intensity.cpu().detach().numpy())
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
            self.logger.note("Simulation run finished.")
            # Write log file
            self.outputs.write_log(self.logger)
        else:
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
            instance = kwargs.get('instance','')
        )
        # Set up the neural net
        self.logger.note("Initializing the neural net ...")
        neural_network = NeuralNet(
            input_size=self.inputs.data.destination_attraction_ts.shape[1],
            output_size=len(config['inputs']['to_learn']),
            **config['neural_network']['hyperparameters'],
        ).to(self.device)

        # Pass inputs to device
        self.inputs.pass_to_device()

        # Instantiate Spatial Interaction Model
        self.logger.note("Initializing the spatial interaction model ...")

        sim = instantiate_sim(
            sim_type= config['spatial_interaction_model']['sim_type'],
            config = config,
            true_parameters = config['spatial_interaction_model']['parameters'],
            device = self.device,
            instance = kwargs.get('instance',''),
            **vars(self.inputs.data)
        )
        # Get and remove config
        config = pop_variable(sim,'config')

        # Build Harris Wilson model
        self.logger.note("Initializing the Harris Wilson physics model ...")
        harris_wilson_model = HarrisWilson(
            sim = sim,
            config = config,
            dt = config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.inputs.true_parameters,
            device = self.device,
            instance = kwargs.get('instance','')
        )
        # Get and remove config
        config = pop_variable(harris_wilson_model,'config')

        # Instantiate harris and wilson neural network model
        self.logger.note("Initializing the Harris Wilson Neural Network model ...")
        self.harris_wilson_nn = HarrisWilson_NN(
            rng = rng,
            config = config,
            neural_net = neural_network,
            loss_function = config['neural_network'].pop('loss_function'),
            physics_model = harris_wilson_model,
            to_learn = config['inputs']['to_learn'],
            write_every = config['outputs']['write_every'],
            write_start = config['outputs']['write_start'],
            instance = kwargs.get('instance','')
        )
        # Get config
        config = getattr(self.harris_wilson_nn,'config') if hasattr(self.harris_wilson_nn,'config') else None

        # Build contingency table
        ct = instantiate_ct(
            table = None,
            config = config
        )

        # Build contingency table MCMC
        self.ct_mcmc = ContingencyTableMarkovChainMonteCarlo(
            ct = ct,
            rng = rng,
            log_to_console = False,
            instance = kwargs.get('instance','')
        )

        # Update config
        if config is not None:
            self.config = config

        # Create outputs
        self.outputs = Outputs(
            self.config,
            module=__name__+kwargs.get('instance',''),
            sweep_params=kwargs.get('sweep_params',{}),
            experiment_id=self.sweep_experiment_id,
            log_to_file=True,
            log_to_console=True,
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

        self.output_names = ['log_destination_attraction','theta','loss', 'table']
        self.theta_names = config['inputs']['to_learn']
        
    def run(self) -> None:

        self.logger.note(f"Running Neural Network training of Harris Wilson model.")

        # Time run
        start_time = time.time()

        # Initialise data structures
        self.initialise_data_structures()
        
        # Train the neural net
        num_epochs = self.config['training']['N']

        # For each epoch
        for e in range(num_epochs):

            # Track the epoch training time
            start_time = time.time()
            
            # Track the training loss
            loss_sample = torch.tensor(0.0, requires_grad=True)

            # Count the number of batch items processed
            n_processed_steps = 0

            # Process the training set elementwise, updating the loss after batch_size steps
            for t, training_data in enumerate(self.inputs.data.destination_attraction_ts):
                
                # Perform neural net training
                loss_sample, theta_sample, log_destination_attraction_sample = self.harris_wilson_nn.epoch_time_step(
                    loss = loss_sample,
                    n_processed_steps = n_processed_steps,
                    experiment = self,
                    t = t,
                    dt = self.config['harris_wilson_model'].get('dt',0.001),
                    nn_data = training_data,
                    data_size = len(training_data),
                    **self.config['training']   
                )
            
                # Add axis to every sample to ensure compatibility 
                # with the functions used below
                theta_sample = torch.unsqueeze(theta_sample,0)
                log_destination_attraction_sample = log_destination_attraction_sample.unsqueeze(0).unsqueeze(0)

                # Compute log intensity
                log_intensity = self.harris_wilson_nn.physics_model.sim.log_intensity(
                    log_destination_attraction = log_destination_attraction_sample,
                    grand_total = self.ct_mcmc.ct.margins[tuplize(range(self.ct_mcmc.ct.ndims()))],
                    **dict(zip(self.harris_wilson_nn.parameters_to_learn,theta_sample.split(1,dim=1)))
                ).squeeze()

                # Sample table
                if e == 0:
                    table_sample = self.ct_mcmc.initialise_table(
                        intensity = torch.exp(log_intensity)
                    )
                else:
                    table_sample,_ = self.ct_mcmc.table_gibbs_step(
                        table_prev = table_sample,
                        log_intensity = log_intensity
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
            self.logger.note("Simulation run finished.")
            # Write log file
            self.outputs.write_log(self.logger)
        else:
            self.logger.note("Simulation run finished.")

class ExperimentSweep():

    def __init__(self,config:Config,sweep_key_paths:list,**kwargs):

        # Setup logger
        self.logger = setup_logger(
            __name__,
            level = config.level,
            log_to_file = True,
            log_to_console = True,
        )
        
        self.logger.info(f"Performing parameter sweep")

        # Get config
        self.config = config

        # Store one datetime
        # for all sweeps
        self.config.settings['datetime'] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        # Store all sweeped parameter names
        self.config.settings['sweeped_params_paths'] = sweep_key_paths

        # Load schema
        self.config.load_schema()

        # Initialise experiments 
        # (each one corresponds to one parameter sweep)
        self.experiments = []

        # Store number of workers
        self.n_workers = self.config.settings['inputs'].get("n_workers",1)

        # Parse sweep configurations
        self.sweep_params = self.parse_sweep_params(sweep_key_paths)

        # Temporarily disable sample output writing
        deep_update(self.config.settings,'export_samples',False)
        self.outputs = Outputs(
            self.config,
            sweep_params=kwargs.get('sweep_params',{})
        )
        # Prepare writing to file
        self.outputs.open_output_file(kwargs.get('sweep_params',{}))

        # Enable it again
        deep_updates(self.config.settings,{'export_samples':True})

        # Write metadata
        if self.config.get('export_metadata',True):
            self.outputs.write_metadata(
                dir_path='',
                filename=f"config"
            )

        self.logger.info(f"Experiment: {self.outputs.experiment_id}")
    
    def __repr__(self) -> str:
        return "ParameterSweep("+(self.experiment.__repr__())+")"

    def __str__(self) -> str:
        return f"""
            Sweep key paths: {self.sweep_key_paths}
        """

    def parse_sweep_params(self,params:list=None):
        sweep_params = {}
        for key_path in params:
            # Get sweep configuration
            sweep_input,_ = self.config.path_get(
                settings=self.config.settings,
                path=(key_path+["sweep","range"])
            )
            # Parse values
            sweep_vals = self.config.parse_data(sweep_input,(key_path+["sweep","range"]))

            sweep_params[">".join(key_path)] = {
                "var":key_path[-1],
                "path": key_path,
                "values": sweep_vals
            }
        return sweep_params

    def run(self):
        # Compute all combinations of sweep parameters
        sweep_configurations = list(itertools.product(*[val['values'] for val in self.sweep_params.values()]))
        # Pring parameter space size
        param_sizes_str = " x ".join([f"{v['var']} ({len(v['values'])})" for k,v in self.sweep_params.items()])
        total_size_str = np.product([len(v['values']) for v in self.sweep_params.values()])
        self.logger.info(f"Parameter space size: {param_sizes_str}. Total = {total_size_str}.")
        self.logger.info(f"Preparing configs...")
        
        # For each configuration update experiment config 
        # and instantiate new experiment
        self.prepare_experiments_sequential(sweep_configurations)

        # Decide whether to run sweeps in parallel or not
        if self.n_workers > 1:
            self.run_parallel()
        else:
            self.run_sequential()
    
    def prepare_experiments_sequential(self,sweep_configurations):
        for j,sval in tqdm(enumerate(sweep_configurations),total=len(sweep_configurations)):
            # Create new config
            new_config = deepcopy(self.config)
            # Deactivate sweep             
            new_config.settings["sweep_mode"] = False
            # Update config
            for i,key in enumerate(self.sweep_params.keys()):
                new_config.path_set(
                    new_config,
                    sval[i],
                    self.sweep_params[key]['path']
                )
                # new_val = get_value_from_path(new_config.settings,self.sweep_params[key]['path'])
                # print(key,new_val,type(new_val),get_value_from_path(new_config.settings,['sweep_mode']))
            # Create new experiment
            new_experiment = instantiate_experiment(
                experiment_type=new_config.settings['experiments'][0]['type'],
                config=new_config,
                sweep_params={val['var']:sval[i] for i,val in enumerate(self.sweep_params.values())},
                log_to_file=True,
                log_to_console=False,
                instance=str(j),
                experiment_id=self.outputs.experiment_id
            )
            # Append to experiments
            self.experiments.append(new_experiment)
        
    def prepare_experiments_parallel(self,sweep_configurations):
        return

    def run_sequential(self):
        self.logger.info("Running Parameter Sweep in sequence...")
        for exp in tqdm(self.experiments,total=len(self.experiments)):
            exp.run()
    
    def run_parallel(self):
        # Run experiments in parallel
        p = Pool(self.n_workers)
        result = [p.apply_async(exp.run()) for exp in tqdm(self.experiments)]
        p.close()
        p.join()
    
    def run_process(self,process):
        process.run()
        return True