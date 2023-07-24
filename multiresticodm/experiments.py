import gc
import sys
import time
import logging

from os import path
from tqdm import tqdm
from glob import glob
from typing import Dict
from copy import deepcopy
from datetime import datetime
from argparse import Namespace
from scipy.optimize import minimize
from joblib import Parallel, delayed

from multiresticodm.utils import *
from multiresticodm.config import Config
from multiresticodm.inputs import Inputs
from multiresticodm.outputs import Outputs
from multiresticodm.global_variables import *
from multiresticodm.markov_basis import MarkovBasis
from multiresticodm.contingency_table import ContingencyTable,instantiate_ct
from multiresticodm.math_utils import apply_norm,running_average_multivariate
from multiresticodm.contingency_table_mcmc import ContingencyTableMarkovChainMonteCarlo
from multiresticodm.harris_wilson_model import HarrisWilson
from multiresticodm.harris_wilson_model_neural_net import NeuralNet, HarrisWilson_NN
from multiresticodm.spatial_interaction_model import SpatialInteraction,instantiate_sim
from multiresticodm.spatial_interaction_model_mcmc import instantiate_spatial_interaction_mcmc

# Suppress scientific notation
np.set_printoptions(suppress=True)


class ExperimentHandler(object):

    def __init__(self, config:Config):
        # Import logger
        self.logger = logging.getLogger(__name__)
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)
        
        # Get contingency table
        self.config = config
        # Instatiate list of experiments
        self.experiments = []

        # Setup experiments
        self.setup_experiments()

    def map_experiment_type_to_class(self,experiment_type:str):
        if hasattr(sys.modules[__name__], experiment_type):
            return getattr(sys.modules[__name__], experiment_type)
        else:
            raise Exception(f'Experiment class {experiment_type} not found')


    def setup_experiments(self):

        self.experiments = {}

        # Only run experiments specified in command line
        for experiment_id in self.config.settings['experiments']['run_experiments']:
            # Check that such experiment already exists in the config file
            if experiment_id in self.config.settings['experiments'].keys():
                # Instatiate new experiment
                experiment = self.map_experiment_type_to_class(self.config.settings['experiments'][experiment_id]['type'])
                # Construct sub-config with only data relevant for experiment
                subconfig = Config(settings=deepcopy(self.config.settings['experiments'][experiment_id]))
                # Update id, seed and logging detail
                subconfig['experiment_id'] = experiment_id
                if self.config.settings['inputs'].get('dataset',None) is not None:
                    subconfig['experiment_data'] = path.basename(path.normpath(self.config.settings['inputs']['dataset']))
                else:
                    raise Exception(f'No dataset found for experiment id {experiment_id}')
                for k in self.config.settings.keys():
                    if k != 'experiments':
                        subconfig[k] = self.config.settings[k]
                # print(json.dumps(subconfig,indent=2))
                # Build it
                new_experiment = experiment(
                    subconfig,
                )

                # Append it to list of experiments
                self.experiments[experiment_id] = new_experiment

    def run_and_write_experiments_sequentially(self):
        # Run all experiments sequential
        for _,experiment in self.experiments.items():
            # Run experiment
            experiment.run()
            # Write outputs to file
            experiment.write(metadata=True)
            # Reset
            try:
                experiment.reset()
            except:
                pass

class Experiment(object):
    def __init__(self, subconfig:Config, disable_logger:bool=False, **kwargs):
        # Create logger
        self.logger = logging.getLogger(__name__)
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)
        # Store subconfig
        self.subconfig = subconfig
        if len(self.subconfig['inputs'].get('load_experiment',[])) > 0:
            # Get path
            filepath = self.subconfig['inputs'].get('load_experiment','')
            # Load metadata
            settings = read_json(path.join(filepath,path.basename(filepath)+'_metadata.json'))
            # Deep update config settings based on metadata
            settings_flattened = deep_flatten(settings,parent_key='',sep='')
            # Remove load experiment 
            del settings_flattened['load_experiment']
            deep_updates(self.subconfig,settings_flattened,overwrite=True)
            # Merge settings to config
            self.subconfig = Config(settings={**self.subconfig, **settings_flattened})
        
        # Update config with current timestamp ( but do not overwrite)
        datetime_results = list(deep_get(key='datetime',value=self.subconfig))
        if len(datetime_results) > 0:
            deep_update(self.subconfig, key='datetime', val=datetime.now().strftime("%d_%m_%Y_%H_%M_%S"), overwrite=False)
        else:
            self.subconfig['datetime'] = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        # Store progress
        if not str_in_list('store_progress',self.subconfig.keys()):
            self.subconfig['store_progress'] = 1
        # Initialise empty results
        self.results = []

        # Disable loggers if necessary
        if disable_logger:
            self.logger.disabled = True

        # Update current config
        # self.subconfig = self.sim.config.update_recursively(self.subconfig,updated_config,overwrite=True)
        # print(self.subconfig)
        # Decide how often to print statemtents
        self.print_statements = self.subconfig.get('print_statements',True)
        self.store_progress = self.subconfig.get('store_progress',0.05)
        self.print_percentage = min(0.05,self.store_progress)*int(self.print_statements)

        # Update seed if specified
        self.seed = None
        if "seed" in self.subconfig.keys():
            self.seed = int(self.subconfig["seed"])
            self.logger.warning(f'   Updated seed to {self.seed}')
        # Get experiment data
        self.logger.info(f"   Experiment {self.subconfig['experiment_id']} of type {self.subconfig['type']} has been set up.")

        # Get device name
        self.device = self.subconfig['inputs']['device']

    def run(self) -> None:
        pass

    def write(self,metadata:bool=False) -> None:
        # Initalise output handler
        outputs = Outputs(self,self.subconfig)

        if (metadata) and self.subconfig['export_metadata'] and hasattr(self,'subconfig'):
            # Write metadata
            outputs.write_metadata()

        if hasattr(self,'results') and len(self.results) > 0:
            if self.subconfig['export_samples'] and hasattr(self,'results'):
                # Write samples
                for res in self.results:
                    outputs.write_samples(list(res['samples'].keys()))
            self.logger.info(f'   Outputs have been written.')
        else:
            if self.subconfig.get('store_progress',1.0) >= 1.0:
                self.logger.warn(f"   Cannot write results: No {self.subconfig['type']} experiment (tabular or plotting) results found for experiment {self.subconfig['experiment_id']}")

    def reset(self,metadata:bool=False) -> None:
        self.logger.info(f'   Resetting experimental results to release memory.')
        
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
            safe_delete(self.subconfig)
            self.subconfig = Config(settings={})

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
    
    def initialise(self):
        # Get batch sizes
        batch_sizes = self.define_sample_batch_sizes()
        self.subconfig['n_batches'] = len(batch_sizes)

        if str_in_list('load_experiment',self.subconfig['inputs']) and len(self.subconfig['inputs']['load_experiment']) > 0:
            # Read last batch of experiments
            outputs = Outputs(self,self.subconfig)
            # Parameter initialisations
            parameter_inits = dict(zip(self.sample_names,[0]*len(self.sample_names)))
            parameter_acceptances = dict(zip(self.sample_names,[0]*len(self.sample_names)))
            # Total samples for table,theta,x posteriors, respectively
            K = deep_call(self,'.od_mcmc.table_steps',1)
            M = deep_call(self,'.sim_mcmc.theta_steps',1)
            L = deep_call(self,'.sim_mcmc.log_destination_attraction_steps',1)
            assert K == 1
            assert M == 1
            assert L == 1
            for sample_name in self.sample_names:
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
                    parameter_acceptances[sample_name] = self.subconfig.get((f"{sample_name}_acceptance"),0)*total_samples
                else:
                    raise Exception(f'Failed tried loading experiment {self.experiment_id}')
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
                    self.logger.warning('   Unconstrained margins could not be sampled.')
                
                try:
                    parameter_inits['table'] = deep_call(self,'.od_mcmc.initialise_table()',None)
                except:
                    parameter_inits['table'] = None
                    self.logger.warning('   Table could not be initialised.')
            
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
            self.logger.warning('   Experiment cannot be resumed.')

        # Update metadata initially
        self.update_metadata(
            0,
            parameter_inits.get('batch_counter',0),
            print_flag=False,
            update_flag=True
        )
        return parameter_inits,parameter_acceptances

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
        self.subconfig['batch_counter'] = batch_counter

        if batch_counter == 0 or N <= 1:
            if hasattr(self,'sim_mcmc'):
                # Compute total sum of squares for r2 computation
                self.w_data = np.exp(self.sim_mcmc.sim.log_destination_attraction)
                w_data_centred = self.w_data - np.mean(self.w_data)
                self.ss_tot = np.dot(w_data_centred, w_data_centred)
                
        if update_flag:
            if hasattr(self,'od_mcmc'):
                # Get dims from table
                self.subconfig['table_dim'] = 'x'.join(map(str,deep_call(self,'.od_mcmc.ct.dims',defaults=None)))
                # Get total from table
                self.subconfig['table_total'] = int(deep_call(
                    input=self,
                    expressions='.od_mcmc.ct.margins[args1]',
                    defaults=-1,
                    args1=tuplize(range(deep_call(self,'.od_mcmc.ct.ndims()',0)))
                ))
                # Get markov basis length from table mcmc
                if isinstance(deep_call(self,'.od_mcmc.table_mb',None),MarkovBasis):
                    self.subconfig['markov_basis_len'] = int(len(self.od_mcmc.table_mb))
            elif hasattr(self,'sim_mcmc'):
                # Compute total sum of squares for r2 computation
                self.w_data = np.exp(self.sim_mcmc.sim.log_destination_attraction)
                w_data_centred = self.w_data - np.mean(self.w_data)
                self.ss_tot = np.dot(w_data_centred, w_data_centred)
                # Get dims from sim
                self.subconfig['table_dim'] = 'x'.join(map(str,deep_call(self,'.sim_mcmc.sim.dims',defaults=None)))
                # Get sim auxiliary params
                self.subconfig['auxiliary_parameters'] = deep_call(
                    input=self,
                    expressions=[f'.sim_mcmc.sim.{param}' for param in ['delta','gamma','kappa','epsilon']],
                    defaults=[0.0,1.0,1.0,1.0]
                )
                self.subconfig['noise_regime'] = deep_call(self,'.sim_mcmc.sim.noise_regime','undefined')
                # Store ground truth parameters
                if self.sim_mcmc.sim.ground_truth_known:
                    self.subconfig['true_parameters'] = [str(self.sim_mcmc.sim.alpha_true),str(self.sim_mcmc.sim.beta_true)]
            elif hasattr(self,'ct'):
                # Get dims from table
                self.subconfig['table_dim'] = 'x'.join(map(str,deep_call(self,'.ct.dims',defaults=None)))
                # Get total from table
                self.subconfig['table_total'] = int(deep_call(
                    input=self,
                    expressions='.ct.margins[args1]',
                    defaults=-1,
                    args1=tuplize(range(deep_call(self,'.ct.ndims()',0)))
                ))
            elif hasattr(self,'sim'):
                # Get dims from sim
                self.subconfig['table_dim'] = 'x'.join(map(str,deep_call(self,'.sim.dims',defaults=None)))
                # Get sim auxiliary params
                self.subconfig['auxiliary_parameters'] = deep_call(
                    input=self,
                    expressions=[f'.sim.{param}' for param in ['delta','gamma','kappa','epsilon']],
                    defaults=[0.0,1.0,1.0,1.0]
                )
                self.subconfig['noise_regime'] = deep_call(self,'.sim.noise_regime','undefined')
                # Store ground truth parameters
                if self.sim.ground_truth_known:
                    self.subconfig['true_parameters'] = [str(self.sim.alpha_true),str(self.sim.beta_true)]
            else:
                self.subconfig['table_dim'] = [None,None]
                self.subconfig['table_total'] = None
                self.subconfig['noise_regime'] = None


        if hasattr(self,'signs') and hasattr(self,'thetas'):
            mu_theta = np.dot(self.signs.T,self.thetas)/np.sum(self.signs)
            std_theta = np.dot(self.signs.T,np.power(self.thetas,2))/np.sum(self.signs) - np.power(mu_theta,2)
            if update_flag:
                self.subconfig['theta_mu'] = mu_theta.tolist()
                self.subconfig['theta_sd'] = std_theta.tolist()
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
                self.subconfig['log_destination_attraction_r2'] = r2
        
        if update_flag:
            self.subconfig['execution_time'] = (time.time() - self.start_time)
            if hasattr(self,'theta_acceptance') and (M*N) > 0:
                self.subconfig['theta_acceptance'] = int(100*getattr(self,'theta_acceptance',0)/(M*N))
            if hasattr(self,'total_signs') and (M*N) > 0:
                self.subconfig['positives_percentage'] = int(100*getattr(self,'total_signs',0)/(M*N))
            if hasattr(self,'log_destination_attraction_acceptance') and (L*N) > 0:
                self.subconfig['log_destination_attraction_acceptance'] = int(100*getattr(self,'log_destination_attraction_acceptance',0)/(L*N))
            if hasattr(self,'table_acceptance') and (K*N) > 0:
                self.subconfig['table_acceptance'] = int(100*(getattr(self,'table_acceptance',0)/(K*N)))
            if hasattr(self,'total_signs') and (K*N) > 0:
                self.subconfig['total_signs'] = int(100*(getattr(self,'total_signs',0)/(K*N)))

        if print_flag:
            print('Iteration',N,'batch',f"{batch_counter+1}/{self.subconfig.get('n_batches',1)}")
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

    def __init__(self, subconfig:Config, disable_logger:bool=False):
        # Initalise superclass
        super().__init__(subconfig,disable_logger)
        # Build spatial interaction model
        self.sim = instantiate_sim(self.config)

        self.grid_size = subconfig['grid_size']
        self.amin,self.amax = subconfig['a_range']
        self.bmin,self.bmax = subconfig['b_range']

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

        if self.subconfig['print_statements']:
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
        self.subconfig['noise_regime'] = self.sim.noise_regime
        self.subconfig['fitted_alpha'] = alpha_values[idx[1]]
        self.subconfig['fitted_beta'] = beta_values[idx[0]]
        self.subconfig['fitted_scaled_beta'] = beta_values[idx[0]]*self.sim.bmax
        self.subconfig['R^2'] = float(r2_values[idx])
        self.subconfig['predicted_w'] = max_w_prediction.tolist()

        # Append to result array
        self.results = [{"samples":{"r2":r2_values}}]

class RSquaredAnalysisGridSearch(Experiment):

    def __init__(self, subconfig:Config, disable_logger:bool=False):
        # Initalise superclass
        super().__init__(subconfig,disable_logger)
        # Build spatial interaction model
        self.sim = instantiate_sim(self.config)
        
        self.grid_size = subconfig['grid_size']
        self.amin,self.amax = subconfig['a_range']
        self.bmin,self.bmax = subconfig['b_range']


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
                            
                
                if self.subconfig['print_statements']:
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
        self.subconfig['fitted_alpha'] = argmax_theta[0]
        self.subconfig['fitted_scaled_beta'] = argmax_theta[1]/(argmax_beta_scale)
        self.subconfig['fitted_beta_scaling_factor'] = argmax_beta_scale
        self.subconfig['fitted_beta'] = argmax_theta[1]
        self.subconfig['fitted_kappa'] = kappa_from_delta(argmax_theta[2])
        self.subconfig['fitted_delta'] = argmax_theta[2]
        self.subconfig['kappa_min'] = kappa_from_delta(delta_min)#kappa_min
        self.subconfig['kappa_max'] = kappa_from_delta(delta_max)#kappa_max
        self.subconfig['delta_min'] = delta_min
        self.subconfig['delta_max'] = delta_max
        self.subconfig['beta_scale_min'] = beta_scale_min
        self.subconfig['beta_scale_max'] = beta_scale_max
        self.subconfig['R^2'] = float(max_r2)
        self.subconfig['noise_regime'] = self.sim.noise_regime
        self.subconfig['predicted_w'] = argmax_w_prediction.tolist()

        # Append to result array
        self.results = [{"samples":{"r2":r2_values}}]


class LogTargetAnalysis(Experiment):

    def __init__(self, subconfig:Config, disable_logger:bool=False):
        # Initalise superclass
        super().__init__(subconfig,disable_logger)
        # Build spatial interaction model
        sim = instantiate_sim(self.config)

        self.grid_size = subconfig['grid_size']
        self.amin,self.amax = subconfig['a_range']
        self.bmin,self.bmax = subconfig['b_range']

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

        if self.subconfig['print_statements']:
            print("Fitted alpha, beta and scaled beta values:")
            print(XX[argmax_index],YY[argmax_index]*self.amax/(self.bmax), YY[argmax_index])
            print("Log target:")
            print(log_targets[argmax_index])

        # Compute estimated flows
        theta[0] = XX[argmax_index]
        theta[1] = YY[argmax_index]

        # Save fitted values to parameters
        self.subconfig['fitted_alpha'] = XX[argmax_index]
        self.subconfig['fitted_scaled_beta'] = YY[argmax_index]*self.amax/(self.bmax)
        self.subconfig['fitted_beta'] = YY[argmax_index]
        self.subconfig['kappa'] = self.sim.kappa
        self.subconfig['log_target'] = log_targets[argmax_index]
        self.subconfig['noise_regime'] = self.sim.noise_regime

        # Append to result array
        self.results = [{"samples":{"log_target":log_targets}}]

        
class LogTargetAnalysisGridSearch(Experiment):

    def __init__(self, subconfig:Config, disable_logger:bool=False):
        # Initalise superclass
        super().__init__(subconfig,disable_logger)
        
        # Build spatial interaction model
        sim = instantiate_sim(self.config)

        self.grid_size = subconfig['grid_size']
        self.amin,self.amax = subconfig['a_range']
        self.bmin,self.bmax = subconfig['b_range']

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


                if self.subconfig['print_statements']:
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
        self.subconfig['fitted_alpha'] = argmax_theta[0]
        self.subconfig['fitted_scaled_beta'] = argmax_theta[1]*1/(argmax_beta_scale)
        self.subconfig['fitted_beta'] = argmax_theta[1]
        self.subconfig['fitted_kappa'] = argmax_theta[4]
        self.subconfig['fitted_delta'] = argmax_theta[2]
        self.subconfig['kappa_min'] = self.kappa_min
        self.subconfig['kappa_max'] = self.kappa_max
        self.subconfig['delta_min'] = self.delta_min
        self.subconfig['delta_max'] = self.delta_max
        self.subconfig['beta_scale_min'] = self.beta_scale_min
        self.subconfig['beta_scale_max'] = self.beta_scale_max
        self.subconfig['log_target'] = max_target
        self.subconfig['noise_regime'] = self.sim.noise_regime

        # Append to result array
        self.results = [{"samples":{"log_target":log_targets}}]



class SIMLatentMCMC(Experiment):
    def __init__(self, subconfig:Config, disable_logger:bool=False):
        # Initalise superclass
        super().__init__(subconfig,disable_logger)

        # Build spatial interaction model
        sim = instantiate_sim(self.config)

        # Spatial interaction model MCMC
        self.sim_mcmc = instantiate_spatial_interaction_mcmc(sim,disable_logger)

        # Delete duplicate of spatial interaction model
        safe_delete(self.sim)
        
        # Run garbage collector
        gc.collect()

        self.sample_names = ['log_destination_attraction','theta','sign']
        
    def run(self) -> None:

        self.logger.info(f'   Running MCMC inference of {self.sim_mcmc.sim.noise_regime} noise SpatialInteraction.')

        # Time run
        self.start_time = time.time()

        # Fix random seed
        set_seed(self.seed)

        # Initialise parameters
        parameter_inits,parameter_acceptances = self.initialise()
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

        for i in tqdm(range(parameter_inits['N0'],N),disable=self.subconfig['disable_tqdm']):
        
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
                self.write(metadata=True)

                # Reset tables and columns sums to release memory
                self.reset(metadata=False)

                # Increment batch counter
                batch_counter += 1
        
        # Unfix random seed
        set_seed(None)

        self.logger.info(f'   Experimental results have been compiled.')

        # Append to result array
        self.results.append({"samples":{"log_destination_attraction":self.log_destination_attractions,
                                        "theta":self.thetas,
                                        "sign":self.signs}})


class JointTableSIMLatentMCMC(Experiment):

    def __init__(self, subconfig:Config, disable_logger:bool=False):
        # Initalise superclass
        super().__init__(subconfig,disable_logger)
        
        # Setup table
        ct = instantiate_ct(table=None,config=self.config)
        # Update table distribution
        self.config.settings['inputs']['contingency_table']['distribution_name'] = ct.distribution_name
        # Build spatial interaction model
        sim = instantiate_sim(self.config)

        # Spatial interaction model MCMC
        self.sim_mcmc = instantiate_spatial_interaction_mcmc(
            sim,
            disable_logger
        )
        # Contingency Table mcmc
        self.od_mcmc = ContingencyTableMarkovChainMonteCarlo(
            ct,
            table_mb=None,
            disable_logger=disable_logger
        )

        # Delete duplicate of contingency table and spatial interaction model
        safe_delete(self.ct)
        safe_delete(self.sim)
        # Run garbage collector
        gc.collect()

        self.sample_names = ['table','log_destination_attraction','theta','sign']

        # print_json(self.subconfig)
        print(self.sim_mcmc)
        print(self.od_mcmc)

    def run(self) -> None:

        self.logger.info(f'   Running joint MCMC inference of contingency tables and {self.sim_mcmc.sim.noise_regime} noise SpatialInteraction.')

        # Time run
        self.start_time = time.time()

        # Fix random seed
        set_seed(self.seed)

        # Initialise parameters
        parameter_inits,parameter_acceptances = self.initialise()
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

        for i in tqdm(range(parameter_inits['N0'],N),disable=self.subconfig['disable_tqdm']):
            
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
                self.write(metadata=True)

                # Reset tables and columns sums to release memory
                self.reset(metadata=False)

                # Increment batch counter
                batch_counter += 1

        # Unfix random seed
        set_seed(None)

        self.logger.info(f'   Experimental results have been compiled.')

        # Append to result array
        self.results.append({"samples":{"log_destination_attraction":self.log_destination_attractions,
                                        "theta":self.thetas,
                                        "sign":self.signs,
                                        "table":self.tables}})

class TableMCMC(Experiment):
    def __init__(self, subconfig:Config, disable_logger:bool=False):
        # Initalise superclass
        super().__init__(subconfig,disable_logger)

        # Setup table
        ct = instantiate_ct(table=None,config=self.config)
        # Update table distribution
        self.config.settings['inputs']['contingency_table']['distribution_name'] = ct.distribution_name
        # Build spatial interaction model
        sim = instantiate_sim(self.config)
        
        # Update config with table dimension
        # self.subconfig['table_dim'] = 'x'.join(map(str,ct.dims))
        # self.subconfig['table_total'] = int(ct.margins[tuplize(range(ct.ndims()))])
        # Initialise intensities at ground truths
        if (ct is not None) and (ct.table is not None):
            self.logger.info('   Using table as ground truth intensity')
            # Use true table to construct intensities
            with np.errstate(invalid='ignore',divide='ignore'):
                self.true_log_intensities = np.log(
                                                ct.table,
                                                dtype='float32'
                                            )
            
        elif (sim is not None) and (sim.ground_truth_known):
            self.logger.info('   Using SIM model as ground truth intensity')
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
            disable_logger=disable_logger
        )
        # Set table steps to 1
        self.od_mcmc.table_steps = 1
        # Delete duplicate of contingency table and spatial interaction model
        safe_delete(self.ct)
        safe_delete(self.sim)
        # Run garbage collector to release memory
        gc.collect()

        self.sample_names = ['table']

    def initialise(self):
        if str_in_list('load_experiment',self.subconfig['inputs']) and \
            len(self.subconfig['inputs']['load_experiment']) > 0:
            # Read last batch of experiments 
            outputs = Outputs(self,self.subconfig)
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
        parameter_inits = self.initialise()
        
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
        for i in tqdm(range(parameter_inits['N0'],N),disable=self.subconfig['disable_tqdm']):
                
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
                self.write(metadata=True)

                # Reset tables and columns sums to release memory
                self.reset(metadata=False)

                # Increment batch counter
                batch_counter += 1

        # Unfix random seed
        set_seed(None)

        self.logger.info(f'   Experimental results have been compiled.')

        # Append to result array
        self.results.append({"samples":{"table":self.tables}})


class TableSummariesMCMCConvergence(Experiment):
    def __init__(self, subconfig:Config, disable_logger:bool=False):
        # Initalise superclass
        super().__init__(subconfig,disable_logger)

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
            self.logger.error('   Cell constraints found in config.')
            raise Exception('TableSummariesMCMCConvergence cannot handle cell constraints due to the different margins generated')
        # Initialise intensities at ground truths
        if (ct is not None) and (ct.table is not None):
            self.logger.info('   Using table as ground truth intensity')
            # Use true table to construct intensities
            # with np.errstate(invalid='ignore',divide='ignore'):
            self.true_log_intensities = np.log(
                                            ct.table,
                                            dtype=np.float64
                                        )
            # np.log(ct.table,out=np.ones(np.shape(ct.table),dtype='float32')*(-1e10),where=(ct.table!=0),dtype='float32')
        elif (sim is not None) and (sim.ground_truth_known):
            self.logger.info('   Using SIM model as ground truth intensity')
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
        self.K = self.subconfig['K']
        # Create K different margins (data) for each MCMC sampler
        self.samplers = {}

        # Get table probabilities from which 
        # margin probs will be elicited
        self.samplers['0'] = ContingencyTableMarkovChainMonteCarlo(
            ct,
            table_mb=None,
            disable_logger=disable_logger
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
                    disable_logger=False
                )
                # Initialise fixed margins
                self.samplers[str(k)].sample_constrained_margins(np.exp(self.true_log_intensities))
        
        # Delete duplicate of contingency table and spatial interaction model
        safe_delete(self.ct)
        safe_delete(self.sim)
        # Run garbage collector to release memory
        gc.collect()

        self.sample_names = ['table']

    def initialise(self):
            
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
        # sys.exit()
        return tables0

    
    def run(self) -> None:

        # Time run
        self.start_time = time.time()

        # Fix random seed
        set_seed(self.seed)

        # Initialise table samples
        self.tables = self.initialise()

        # Initiliase means by MCMC iteration
        table_mean = np.mean(np.array(self.tables,dtype='float32'),axis=0)
        
        # Initialise error norms
        table_norm = apply_norm(
            tab=table_mean[np.newaxis,:],
            tab0=np.exp(self.true_log_intensities,dtype='float32'),
            name=self.subconfig['norm'],
            **self.subconfig
        )
        
        
        # Store number of samples
        # Total samples for joint posterior
        N = self.samplers['0'].ct.config.settings['mcmc']['N']

        print('Running MCMC')
        for i in tqdm(range(1,N),disable=self.subconfig['disable_tqdm'],leave=False):
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
                    name=self.subconfig['norm'],
                    **self.subconfig
                )), 
                axis=0
            )
            # print('after')
            # print(table_mean)
            # print('error')
            # print(np.sum(table_norm[-1]))
            # print('\n')

            if (self.subconfig['print_statements']) and (i in [int(p*N) for p in np.arange(0,1,0.1)]):
                # print('table mean')
            #     print(table_mean)
            #     print('ground truth')
            #     print(np.exp(self.true_log_intensities,dtype='float32'))
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

        self.logger.info(f'   Experimental results have been compiled.')

        self.results = [{
            "samples":{
                "tableerror":table_norm,
            }
        }]

        return self.results[-1]

class SIM_NN(Experiment):
    def __init__(self, subconfig:Config, disable_logger:bool=False):
        
        # Initalise superclass
        super().__init__(subconfig,disable_logger)

        # Fix random seed
        rng = set_seed(self.seed)
        # Store 
        subconfig.settings['inputs']['rng'] = self.seed

        # Prepare inputs
        self.inputs = Inputs(
            subconfig,
            model = 'neural_net',
            synthetic_data = False
        )

        # Set up the neural net
        self.logger.info("   Initializing the neural net ...")
        neural_network = NeuralNet(
            input_size=self.inputs.destination_attraction_ts.shape[1],
            output_size=len(subconfig['neural_net']['to_learn']),
            **subconfig['neural_net']['hyperparameters'],
        ).to(self.device)

        # Pass inputs to device
        self.inputs.pass_to_device()

        # Instantiate Spatial Interaction Model
        sim = instantiate_sim(
            config=subconfig,
            origin_demand=self.inputs.origin_demand,
            log_destination_attraction=np.log(self.inputs.destination_attraction_ts[:,-1].flatten()),
            cost_matrix=self.inputs.cost_matrix,
            true_parameters=subconfig['spatial_interaction_model']['parameters'],
            device = self.device
        )

        # Build Harris Wilson model
        harris_wilson_model = HarrisWilson(
            sim=sim,
            true_parameters = self.inputs.true_parameters,
            dt = subconfig['harris_wilson_model'].get('dt',0.001),
            device = self.device
        )

        # Instantiate model
        self.harris_wilson_nn = HarrisWilson_NN(
            rng=rng,
            # h5group=h5group,
            neural_net=neural_network,
            loss_function=subconfig['neural_net'].pop('loss_function'),
            physics_model=harris_wilson_model,
            to_learn=(subconfig['spatial_interaction_model']['sim_to_learn']+subconfig['harris_wilson_model']['hw_to_learn']),
            write_every=subconfig['outputs']['write_every'],
            write_start=subconfig['outputs']['write_start']
        )

        self.logger.info(f"   Initialized Harris Wilson Neural Net.")
        
        # Run garbage collector
        gc.collect()

        self.sample_names = subconfig['neural_net']['to_learn']
        
    def run(self) -> None:

        self.logger.info(f'   Running Neural Network training of Harris Wilson model.')

        # Time run
        self.start_time = time.time()
        
        # Train the neural net
        num_epochs = self.subconfig['neural_net']['N']

        for e in range(num_epochs):


            self.harris_wilson_nn.epoch(
                training_data=self.inputs.destination_attraction_ts, 
                **self.subconfig['neural_net']
            )

            self.logger.progress(f"   Completed epoch {e+1} / {num_epochs}.")

        self.logger.info("   Simulation run finished.")
        # h5file.close()
        
        # Append to result array
        # self.results.append({"samples":{"destination_attraction_ts":self.log_destination_attractions,
                                        # "theta":self.thetas}})