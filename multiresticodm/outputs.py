import os
import re
import gc
import sys
import logging
import traceback
import h5py as h5
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import geopandas as gpd
import concurrent.futures as concurrency

from tqdm import tqdm
from torch import int32
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Union,List,Tuple
from itertools import product,chain

import multiresticodm.probability_utils as ProbabilityUtils

from multiresticodm.utils import *
from multiresticodm.math_utils import *
from multiresticodm.config import Config
from multiresticodm.inputs import Inputs
from multiresticodm.global_variables import *
from multiresticodm.contingency_table import instantiate_ct
from multiresticodm.spatial_interaction_model import *

OUTPUTS_MODULE = sys.modules[__name__]

class Outputs(object):

    def __init__(self,
                 config:Config, 
                 settings:dict={},
                 data_names:list=['ground_truth_table'],
                 sweep_params:dict={},
                 **kwargs):
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__,
            console_level = level,
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update config level
        self.logger.setLevels(
            console_level = level
        )
        # Store output names
        self.data_names = data_names

        # Store settings
        self.settings = settings

        # Sample names must be a subset of all data names
        try:
            assert set(self.data_names).issubset(set(DATA_TYPES.keys()))
        except Exception:
            self.logger.error('Some sample names provided are not recognized')
            self.logger.error(','.join(self.data_names))
            self.logger.debug(traceback.format_exc())
            raise Exception('Cannot load outputs.')

        # Store coordinate slice
        self.coordinate_slice = kwargs.get('coordinate_slice',[])

        # Create semi-private xarray data 
        self.data = DataCollection(
            logger = self.logger
        )
        # Enable garbage collector
        gc.enable()
        
        # Store config
        if isinstance(config,Config):
            # Store config
            self.config = config
            
            # Get intensity model class
            self.intensity_model_class = [k for k in self.config.keys() if k in INTENSITY_MODELS and isinstance(self.config[k],dict)][0]

            # Update experiment id
            self.experiment_id = self.update_experiment_directory_id(kwargs.get('experiment_id',None))

            # Define output experiment path to directory
            self.outputs_path = os.path.join(
                    self.config['outputs']['out_directory'],
                    self.config['inputs']['dataset'],
                    self.config['outputs'].get('out_group',''),
                    self.experiment_id
            ) if kwargs.get('base_dir') is None else kwargs['base_dir']

        elif isinstance(config,str):
            # Store experiment id
            self.experiment_id = os.path.basename(os.path.normpath(config.split('samples/')[0]))
            
            # Load metadata
            assert os.path.exists(config)
            self.config = Config(
                path=os.path.join(config,"config.json"),
                logger = self.logger
            )

            # Get intensity model class
            self.intensity_model_class = [k for k in self.config.keys() if k in INTENSITY_MODELS and isinstance(self.config[k],dict)][0]
            
            # Define config experiment path to directory
            self.outputs_path = config if kwargs.get('base_dir') is None else kwargs['base_dir']
        
        else:
            raise Exception(f'Config {config} of type {type(config)} not recognised.')
        
        # Get name of intensity model
        self.intensity_model_name = self.config.settings[self.intensity_model_class]['name']

        # Store sample data requirements
        self.output_names = list(
            flatten([
                SAMPLE_DATA_REQUIREMENTS[sam] if sam != 'intensity' 
                else SAMPLE_DATA_REQUIREMENTS[sam][self.intensity_model_name]
                for sam in set(self.data_names).intersection(set(list(OUTPUT_TYPES.keys())))]
            )
        )
        self.input_names = [
            sam for sam in set(self.data_names).intersection(set(list(INPUT_TYPES.keys())))
        ]
        
        # Name output sample directory according 
        # to sweep params (if they are provided)
        self.sweep_id = ''
        if len(sweep_params) > 0 and isinstance(sweep_params,dict):
            # Create sweep id by grouping coupled sweep vars together
            # and isolated sweep vars separately
            sweep_id = []
            for v in set(list(self.config.target_names_by_sweep_var.values())).difference(set(['dataset'])):
                # Map sigma to noise regime
                if str(v) == 'sigma':
                    value = sigma_to_noise_regime(sweep_params[v])
                # Else use passed sweep value
                else:
                    value = sweep_params[v]
                # Add to key-value pair to unique sweep id
                sweep_id.append(f"{str(v)}_{stringify(value)}")
            # Join all grouped sweep vars into one sweep id 
            # which will be used to create an output folder
            if len(sweep_id) > 0:
                self.sweep_id = os.path.join(*sorted(sweep_id,key=lambda x: x.split('_')[0]))
            else:
                self.sweep_id = ''

    def samples(self):
        if hasattr(self,'data') and isinstance(self.data,DataCollection):
            return list(vars(self.data).keys())
        else:
            return []
        
    def has_sample(self,sample_name:str) -> bool:
        return sample_name in self.samples()
    
    def check_data_availability(self,sample_name:str,input_names:list=[],output_names:list=[]):
        available = True
        for input in input_names:
            try:
                assert hasattr(self.inputs.data,input)
            except:
                available = False
                self.logger.error(f"Sample {sample_name} requires input {input} \
                                  which does not exist in {','.join(self.inputs.data.vars())}")
        for output in output_names:
            try:
                assert hasattr(self.data,output)
            except:
                available = False
                self.logger.error(f"Sample {sample_name} requires output {output} which does not exist in {','.join(vars(self.data))}")
        return available

    def slice_sample_iterations(self,samples,settings:dict={}):
        # Get burnin parameter
        burnin = min(settings.get('burnin',1),samples.shape[0])

        # Get thinning parameter
        thinning = list(deep_get(key='thinning',value=settings))
        thinning = thinning[0] if len(thinning) > 0 else 1
        
        # Get iterations
        iters = np.arange(start=1,stop=samples.shape[0]+1,step=1,dtype='int32')

        # Apply burnin and thinning
        samples = samples[burnin:None:thinning]
        iters = iters[burnin:None:thinning]

        # Get total number of samples
        N = settings.get('N',samples.shape[0])
        if N is None:
            N = samples.shape[0]
        else:
            N = min(N,samples.shape[0])
        
        # Apply stop
        samples = samples[:N]
        iters = iters[:N]

        return samples,iters
    
    def load_experiment_data(self,inputs=None,settings:dict={},input_slice:dict={},slice_samples:bool=True):

        # Update config based on slice of NON-coordinate-like sweeped params
        # affecting only the inputs of the model
        if input_slice:
            self.config.path_set(
                settings = self.config.settings,
                value = input_slice['value'], 
                path = input_slice['path']
            )

        if inputs is None:
            # Import all input data
            self.inputs = Inputs(
                config = self.config,
                synthetic_data = False,
                logger = self.logger,
            )
        else:
            self.inputs = inputs
            # Convert all inputs to tensors
            self.inputs.pass_to_device()

        # Load output h5 file to dictionary
        output_dict_data = self.load_h5_data(
            settings = settings,
            slice_samples = slice_samples
        )
        return output_dict_data
    
    def load_geometry(self,geometry_filename,default_crs:str='epsg:27700'):
        # Load geometry from file
        geometry = gpd.read_file(geometry_filename)
        geometry = geometry.set_crs(default_crs,allow_override=True)
        
        return geometry


    def load_h5_data(self,settings:dict={},slice_samples:bool=True):
        self.logger.note('Loading h5 data into xarrays...')

        # Get all h5 files
        h5files = list(Path(os.path.join(self.outputs_path,'samples',f"{self.sweep_id}")).rglob("*.h5"))
        # Sort them by seed
        h5files = sorted(h5files, key = lambda x: int(str(x).split('seed_')[1].split('/',1)[0]) if 'seed' in str(x) else str(x))
        # Read h5 data
        local_coords,global_coords,data_vars = self.read_h5_files(h5files,settings=settings,slice_samples=slice_samples)
        # Convert set to list
        local_coords = {k:np.array(
                            list(v),
                            dtype=TORCH_TO_NUMPY_DTYPE[COORDINATES_DTYPES[k]]
                        ) for k,v in local_coords.items()}
        global_coords = {k:np.array(
                            list(v),
                            dtype=TORCH_TO_NUMPY_DTYPE[COORDINATES_DTYPES[k]]
                        ) for k,v in global_coords.items()}
        
        self.logger.progress('Populating data dictionary')
        
        # Create an xarray dataset for each sample
        xr_dict = {}
        for sample_name,sample_data in data_vars.items():
            
            coordinates = {}
            # Ignore first two dimensions
            # First dimension is the sweep dimension
            # Second dimension is the number of iterations per sweep
            if len(np.shape(sample_data)) > 2:
                # Get data dims
                dims = np.shape(sample_data)[2:]
                # For each dim create coordinate
                for i,d in enumerate(dims):
                    obj,func = XARRAY_SCHEMA[sample_name]['funcs'][i]
                    # Create coordinate ranges based on schema
                    coordinates[XARRAY_SCHEMA[sample_name]['coords'][i]] = deep_call(
                        globals()[obj],
                        func,
                        None,
                        start=1,
                        stop=d+1,
                        step=1
                    ).astype(XARRAY_SCHEMA[sample_name]['args_dtype'][i])
            
            # Update coordinates to include schema and sweep coordinates
            # Keep only coordinates that are 1) core
            # 2) isolated sweeps 
            # or 3) the targets of coupled sweeps
            coordinates = {
                **{k:v for k,v in local_coords.items() \
                   if k != sample_name and k in self.config.sweep_target_names},
                **global_coords,
                **coordinates
            }

            # Populate dictionary 
            xr_dict[sample_name] = {
                "data": sample_data.reshape(tuple([len(val) for val in coordinates.values()])),
                "coordinates": {k:list(flatten(v)) for k,v in coordinates.items()},
                "attrs": {
                    "experiment_id": self.experiment_id
                }
            }

        return xr_dict
    
    def read_h5_file(self,filename,settings:dict={},**kwargs):
        local_coords = {}
        global_coords = {}
        data_vars = {}
        try:
            with h5.File(filename) as h5data:
                self.logger.debug('Collect group-level attributes as coordinates')
                # Collect group-level attributes as coordinates
                # Group coordinates are file-dependent
                group_id = next(iter(h5data))
                if 'sweep_params' in list(h5data[group_id].attrs.keys()) and \
                    'sweep_values' in list(h5data[group_id].attrs.keys()):
                    # Loop through each sweep parameters and add it as a coordinate
                    for (k,v) in zip(h5data[group_id].attrs['sweep_params'],
                                h5data[group_id].attrs['sweep_values']):
                        local_coords[k] = {parse(v,np.float32(-1.0))}
                self.logger.debug('Store dataset')
                # Store dataset
                for sample_name,sample_data in h5data[group_id].items():
                    # If this data is not required, skip storing it
                    if sample_name not in self.output_names:
                        continue
                    # Apply burning, thinning and trimming
                    self.logger.debug(f'Data {sample_name} {np.shape(sample_data)}')
                    if kwargs.get('slice_samples',True):
                        # self.logger.debug(f'Before slicing {sample_name}: {np.shape(sample_data)}')
                        sample_data,iters = self.slice_sample_iterations(sample_data,settings=settings)
                        # self.logger.debug(f'After slicing {sample_name}: {np.shape(sample_data)}')
                    else:
                        # Get iterations
                        iters = np.arange(start=1,stop=sample_data.shape[0]+1,step=1,dtype='int32')
                    global_coords['iter'] = iters
                    # Append
                    self.logger.debug(f'Appending {sample_name}')
                    data_vars[sample_name] = np.array([sample_data[:]])
                self.logger.debug(f'Done with file')
        except BlockingIOError:
            self.logger.debug(f"Skipping in-use file: {filename}")
            return {str(filename):{"local_coords":{},"global_coords":{},"data_vars":{}}}
        except Exception:
            self.logger.debug(traceback.format_exc())
            raise Exception(f'Cannot read file {filename}')
        return {str(filename):{"local_coords":local_coords,"global_coords":global_coords,"data_vars":data_vars}}
    
    def read_h5_files(self,h5files:list,settings:dict={},**kwargs):
        # Get each file and add it to the new dataset
        h5data, local_coords, file_global_coords, data_vars = {},{},{},{}
        # Do it sequentially
        for filename in h5files:

            # Read h5 file
            h5data = self.read_h5_file(filename,settings=settings,**kwargs)
            file_global_coords = h5data[str(filename)]['global_coords']
            file_local_coords = h5data[str(filename)]['local_coords']
            file_data_vars = h5data[str(filename)]['data_vars']
            
            # Read coords
            if len(local_coords) > 0:
                for k,v in file_local_coords.items():
                    local_coords[k].update(v)
            else:
                local_coords = deepcopy(file_local_coords)
            
            # Read data
            for sample_name,sample_data in file_data_vars.items():
                if sample_name in list(data_vars.keys()) and len(data_vars) > 0:
                    data_vars[sample_name] = np.append(
                        data_vars[sample_name],
                        sample_data,
                        axis=0
                    )
                else:
                    data_vars[sample_name] = deepcopy(sample_data)
        
        return local_coords,file_global_coords,data_vars
    
        
    def update_experiment_directory_id(self,sweep_experiment_id:str=None):

        noise_level = list(deep_get(key='noise_regime',value=self.config.settings))
        if len(noise_level) <= 0:
            if 'sigma' in self.config.settings['inputs']['to_learn']:
                noise_level = 'learned'
            else:
                sigma = list(deep_get(key='sigma',value=self.config.settings))
                if len(sigma) == 1:
                    if isinstance(sigma[0],dict) and 'sweep' in list(sigma[0].keys()):
                        noise_level = 'sweeped'
                    else:
                        noise_level = sigma_to_noise_regime(sigma=sigma[0])
        else:
            noise_level = noise_level[0]
        noise_level = noise_level.capitalize()
        
        if not 'experiment_title' in list(self.config['outputs'].keys()):
            self.config['outputs']['experiment_title'] = ""
        elif isinstance(self.config['outputs']['experiment_title'],Iterable):
            self.config['outputs']['experiment_title'] = ""

        if sweep_experiment_id is None:
            if self.config['experiment_type'].lower() in ['tablesummariesmcmcconvergence','table_mcmc_convergence']:
                return self.config['experiment_type']+'_K'+\
                        str(self.config['K'])+'_'+\
                        self.config['mcmc']['contingency_table']['proposal']+'_'+\
                        self.config['outputs']['experiment_title']+'_'+\
                        self.config['datetime']
            elif self.config['experiment_type'].lower() == 'table_mcmc':
                return self.config['experiment_type']+'_'+\
                        self.config['mcmc']['contingency_table']['proposal']+'_'+\
                        self.config['outputs']['experiment_title']+'_'+\
                        self.config['datetime']
            else:
                return self.config['experiment_type']+'_'+\
                        noise_level+'Noise_'+\
                        self.config['outputs']['experiment_title']+'_'+\
                        self.config['datetime']
        else:
            # Return parameter sweep's experiment id
            # This avoids creating new output directories 
            # for sweeped sigma regimes
            return sweep_experiment_id

    def create_output_subdirectories(self,sweep_id:str='') -> None:
        export_samples = list(deep_get(key='export_samples',value=self.config.settings))
        export_metadata = list(deep_get(key='export_metadata',value=self.config.settings))
        export_samples = export_samples[0] if len(export_samples) > 0 else True
        export_metadata = export_metadata[0] if len(export_metadata) > 0 else True
        if export_samples or export_metadata:
            # Create output directories
            makedir(os.path.join(self.outputs_path,'samples'))
            if len(sweep_id) > 0 and isinstance(sweep_id,str):
                makedir(os.path.join(self.outputs_path,'samples',sweep_id))
            makedir(os.path.join(self.outputs_path,'figures'))

    def write_log(self):
        if isinstance(self.logger,DualLogger):
            for i,hand in enumerate(self.logger.file.handlers):
                if isinstance(hand,logging.FileHandler):
                    # Do not write to temporary filename
                    if not hand.filename.startswith("logs/temp_"):
                        # Close handler
                        self.logger.file.handlers[i].flush()
                        self.logger.file.handlers[i].close()
        elif isinstance(self.logger,logging.Logger):
            for i,hand in enumerate(self.logger.handlers):
                if isinstance(hand,logging.FileHandler):
                    # Do not write to temporary filename
                    if not hand.filename.startswith("logs/temp_"):
                        # Close handler
                        self.logger.handlers[i].flush()
                        self.logger.handlers[i].close()
        else:
            raise Exception(f'Cannot write outputs of invalid type logger {type(self.logger)}')

    def write_metadata(self,dir_path:str,filename:str) -> None:
        # Define filepath
        filepath = os.path.join(self.outputs_path,dir_path,f"{filename.split('.')[0]}.json")
        if (os.path.exists(filepath) and self.config['experiments'][0]['overwrite']) or (not os.path.exists(filepath)):
            if isinstance(self.config,Config):
                write_json(self.config.settings,filepath,indent=2)
            elif isinstance(self.config,dict):
                write_json(self.config,filepath,indent=2)
            else:
                raise Exception(f'Cannot write metadata of invalid type {type(self.config)}')

    def print_metadata(self) -> None:
        print_json(self.config,indent=2)

    def open_output_file(self,sweep_params:dict={}):
        # Create output directories if necessary
        self.create_output_subdirectories(sweep_id=self.sweep_id)
        if hasattr(self,'config') and hasattr(self.config,'settings'):
            export_samples = list(deep_get(key='export_samples',value=self.config.settings))
            export_metadata = list(deep_get(key='export_metadata',value=self.config.settings))
            export_samples = export_samples[0] if len(export_samples) > 0 else True
            export_metadata = export_metadata[0] if len(export_metadata) > 0 else True
            # Write to file
            if export_samples:
                self.logger.note(f"Creating output file at:\n        {self.outputs_path}")
                try:
                    self.h5file = h5.File(os.path.join(self.outputs_path,'samples',f"{self.sweep_id}","data.h5"), mode="w")
                except:
                    print(os.path.join(self.outputs_path,'samples',f"{self.sweep_id}","data.h5"))
                    print(os.path.exists(os.path.join(self.outputs_path,'samples',f"{self.sweep_id}","data.h5")))
                    raise Exception('FAILED')
                self.h5group = self.h5file.create_group(self.experiment_id)
                # Store sweep configurations as attributes 
                self.h5group.attrs.create("sweep_params",list(sweep_params.keys()))
                self.h5group.attrs.create("sweep_values",['' if val is None else str(val) for val in sweep_params.values()])
                # Update log filename
                if isinstance(self.logger,DualLogger):
                    for i,hand in enumerate(self.logger.file.handlers):
                        if isinstance(hand,logging.FileHandler):
                            # Make directory
                            makedir(os.path.join(self.outputs_path,'samples',self.sweep_id))
                            # Define filename
                            self.logger.file.handlers[i].filename = os.path.join(
                                self.outputs_path,
                                'samples',
                                self.sweep_id,
                                f"outputs.log"
                            )
                elif isinstance(self.logger,logging.Logger):
                    for i,hand in enumerate(self.logger.handlers):
                        if isinstance(hand,logging.FileHandler):
                            # Make directory
                            makedir(os.path.join(self.outputs_path,'samples',self.sweep_id))
                            # Define filename
                            self.logger.handlers[i].filename = os.path.join(
                                self.outputs_path,
                                'samples',
                                self.sweep_id,
                                f"outputs.log"
                            )

    def write_data_collection(self, sample_name, overwrite:bool=False):
        # Make output directory
        output_directory = os.path.join(self.outputs_path,'sample_collections')
        makedir(output_directory)
        # Write output to file
        filepath = os.path.join(
            output_directory,
            f"collection_data.nc"
        )
        self.logger.info(f'Writing output collection to {filepath}')
        for i,datum in enumerate(getattr(
            self.data,
            sample_name
        )):
            # Writing or appending mode
            if ((not os.path.exists(filepath) or not os.path.isfile(filepath)) or overwrite) and i == 0:
                mode = 'w'
            else:
                mode = 'a'
            write_netcdf(
                datum,
                filepath,
                mode = mode,
                group = f"/{sample_name}/{i}"
            )

    def read_data_collection(self, sample_name:str=None):
        # Outputs filepath
        output_filepath = os.path.join(self.outputs_path,'sample_collections',"collection_data.nc")
        
        if not os.path.isfile(output_filepath) or not os.path.exists(output_filepath):
            return self.output_names
        else:
            samples_not_loaded = []
            # Get all sample names and collection ids
            sample_collections = {}
            # Read data array
            if sample_name is not None:
                with nc.Dataset(output_filepath, 'r') as nc_file:
                    sample_collections[sample_name] = list(nc_file.groups[sample_name].groups.keys())
            else:
                with nc.Dataset(output_filepath, 'r') as nc_file:
                    for sam_name, sample_collection in nc_file.groups.items():
                        sample_collections[sam_name] = list(sample_collection.groups.keys())
            
            data_arrs = []
            for sam_name,sample_collection in sample_collections.items():
                for collection_id in sample_collection:
                    try:
                        # Read data array
                        data_array = read_netcdf_array(
                            filepath = output_filepath,
                            group=f'/{sam_name}/{collection_id}'
                        )
                        # Convert to torch
                        data_arrs.append(data_array)
                    except:
                        samples_not_loaded.append(sample_name)
                        self.logger.warning(f"Could not load /{sam_name}/{collection_id} from {self.outputs_path} sample collections")
            
            self.data = DataCollection(
                *data_arrs,
                logger = self.logger
            )
            return samples_not_loaded

    def create_filename(self,sample=None):
        # Decide on filename
        if (sample is None) or (not isinstance(sample,str)):
            filename = f"{','.join(self.settings['sample'])}"
        else:
            filename = f"{sample}"
        if 'statistic' in list(self.settings.keys()):
            arr = []
            for metric,statistic in self.settings['statistic'].items():
                arr.append(str(metric.lower()) + ','.join([str(stat) for stat in list(flatten(statistic[0]))]))
            filename += f"{'_'.join(arr)}"
        if 'table_dim' in list(self.config.keys()):
            filename += f"_{self.config['table_dim']}"
        if 'table_total' in list(self.config.keys()):
            filename += f"_{self.config['table_total']}"
        if 'type' in list(self.config.keys()) and len(self.config['type']) > 0:
            filename += f"_{self.config['type']}"
        if 'experiment_title' in list(self.settings.keys()) and len(self.settings['experiment_title']) > 0:
            filename += f"_{self.settings['experiment_title']}"
        if 'viz_type' in list(self.settings.keys()):
            filename += f"_{self.settings['viz_type']}"
        if 'burnin' in list(self.settings.keys()):
            filename += f"_burnin{self.settings['burnin']}" 
        if 'thinning' in list(self.settings.keys()):
            filename += f"_thinning{self.settings['thinning']}"
        if 'N' in list(self.settings.keys()):
            filename += f"_N{self.settings['N']}"
        # filename += f"_N{self.config['mcmc']['N']}"
        return filename

    def get_sample(self,sample_name:str):

        if sample_name == 'intensity':
            # Get sim model 
            self.logger.debug('getting sim model')
            sim_model = globals()[self.config.settings['spatial_interaction_model']['name']+'SIM']
            # Check that required data is available
            self.logger.debug('checking sim data availability')
            self.check_data_availability(
                sample_name=sample_name,
                input_names=sim_model.REQUIRED_INPUTS,
                output_names=sim_model.REQUIRED_OUTPUTS,
            )
            # Compute intensities for all samples
            table_total = self.settings.get('table_total') if self.settings.get('table_total',-1.0) > 0 else 1.0
            # Instantiate ct
            sim = instantiate_sim(
                config = self.config,
                logger=self.logger,
                **{input:self.get_sample(input) for input in sim_model.REQUIRED_INPUTS}
            )
            # Compute log intensity
            samples = sim.log_intensity(
                grand_total = torch.tensor(table_total,dtype=int32),
                torch = False,
                **{output:self.get_sample(output) for output in sim_model.REQUIRED_OUTPUTS}
            )
            # Create new dataset
            samples = samples.rename('intensity')
            # Exponentiate
            samples = np.exp(samples)

        elif sample_name.endswith("__error"):
            # Load all samples
            samples = self.get_sample(sample_name.replace("__error",""))
            # Make sure you have ground truth
            try:
                assert self.ground_truth_table is not None
            except:
                self.logger.error('Ground truth table missing. Sample error cannot be computed.')
                raise
        
        elif sample_name == 'ground_truth_table':
            # Get config and sim
            dummy_config = Config(settings=self.config)
            ct = instantiate_ct(
                config=dummy_config,
                log_to_console=False,
                logger=self.logger
            )
            samples = xr.DataArray(
                data=torch.tensor(ct.table).int(),
                name='ground_truth_table',
                dims=['origin','destination'],
                coords=dict(
                    origin=np.arange(1,ct.dims[0]+1,1,dtype='int16'),
                    destination=np.arange(1,ct.dims[1]+1,1,dtype='int16')
                )
            )
        
        elif sample_name in list(INPUT_TYPES.keys()):
            # Get sim model 
            sim_model = globals()[self.config.settings['spatial_interaction_model']['name']+'SIM']
            self.check_data_availability(
                sample_name=sample_name,
                input_names=sim_model.REQUIRED_INPUTS
            )
            # Get samples and cast them to appropriate type
            if torch.is_tensor(getattr(self.inputs.data,sample_name)):
                samples = torch.clone(
                    getattr(self.inputs.data,sample_name).to(
                        INPUT_TYPES[sample_name]
                    )
                )
            else:
                samples = torch.tensor(
                    getattr(self.inputs.data,sample_name), 
                    dtype=INPUT_TYPES[sample_name]
                )

        else:
            if not hasattr(self.data,sample_name):
                raise Exception(f"{sample_name} not found in data.")
            elif self.data.sizes(dim = sample_name) > 1:
                raise Exception(f"Cannot process {sample_name} Data Collection of size {self.data.sizes(dim = sample_name)} > 1.")
            else:
                # Get xarray
                samples = getattr(self.data,sample_name)
            
            # Find iteration dimensions
            iter_dims = [x for x in samples.dims if x in ['iter','seed']]
            
            # Find sweep dimensions that are not core coordinates
            sweep_dims = [d for d in samples.dims if d not in (list(CORE_COORDINATES_DTYPES.keys()))]

            if len(sweep_dims) > 0:
                # Stack all non-core coordinates into new coordinate
                samples = samples.stack(id=tuplize(iter_dims),sweep=tuplize(sweep_dims))
            else:
                samples = samples.stack(id=tuplize(iter_dims))

            # If parameter is beta, scale it by bmax
            if sample_name == 'beta' and self.intensity_model_class == 'spatial_interaction_model':
                samples *= self.config.settings[self.intensity_model_class]['parameters']['bmax']

        self.logger.progress(f"Loaded {sample_name} sample")
        return samples
    
    def compute_statistic(self,data,sample_name,statistic,**kwargs):
        # print('compute_statistic',sample_name,type(data),statistic)
        if statistic is None or statistic.lower() == '' or 'sample' in statistic.lower() or len(kwargs.get('dim',[])) == 0:
            return data
        
        elif statistic.lower() == 'signedmean' and \
            sample_name in list(OUTPUT_TYPES.keys()):
            if sample_name in list(INTENSITY_TYPES.keys())+['intensity'] and hasattr(self.data,'sign'):
                signs = self.get_sample('sign')
                print(signs)
                # Compute moments
                return ( np.einsum('nk,n...->k...',signs,data) / np.sum(np.ravel(signs)))
            else:
                return self.compute_statistic(data,sample_name,'mean',dim=kwargs['dim'])

        elif (statistic.lower() == 'signedvariance' or statistic.lower() == 'signedvar') and \
            sample_name in list(OUTPUT_TYPES.keys()):

            if sample_name in list(INTENSITY_TYPES.keys())+['intensity']:
                # Compute mean
                samples_mean = self.compute_statistic(data,sample_name,'signedmean',**kwargs)
                # Compute squared mean
                signs = self.get_sample('sign')
                samples_squared_mean = np.einsum('nk,n...->k...',signs,np.pow(data,2)) / torch.sum(torch.ravel(signs))
                # Compute intensity variance
                return (samples_squared_mean - torch.pow(samples_mean,2))
            else:
                return deep_call(
                    data,
                    f".var(dim)",
                    data,
                    dim=kwargs['dim']
                )
                # return self.compute_statistic(data,sample_name,'var',**kwargs)
        
        elif statistic.lower() == 'error' and \
            sample_name in [param for param in list(OUTPUT_TYPES.keys()) if 'error' not in param]:
            # Apply error norm
            return apply_norm(
                tab=data,
                tab0=self.ground_truth_table,
                name=self.settings['norm'],
                **self.settings
            )

        elif any([op in statistic.lower() for op in OPERATORS]):
            return operate(
                data,
                statistic,
                **kwargs
            )

        elif hasattr(data,statistic):
            return deep_call(
                data,
                f".{statistic}(dim)",
                data,
                dim=kwargs['dim']
            )
        
        else:
            return deep_call(
                np,
                f".{statistic}(data)",
                data,
                data=data,
            )
    
    def apply_sample_statistics(self,samples,sample_name,statistic_dims:Union[List,Tuple]=[],**kwargs):
        
        if isinstance(statistic_dims,Tuple):
            statistic_dims = [statistic_dims]
        sample_statistic = samples
        
        # For every collection of statistic-axes
        for stat,dims in statistic_dims:

            # Convert dims to list
            if dims is None:
                dims = [None]

            sample_statistic = self.compute_statistic(
                                    data=sample_statistic,
                                    sample_name=sample_name,
                                    statistic=stat,
                                    dim=dims,
                                    **kwargs
                                )

        return sample_statistic
    
class DataCollection(object):

    def __init__(self,
                 *data,
                 **kwargs):
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__,
            console_level = level,
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update config level
        self.logger.setLevels(
            console_level = level
        )

        # All items must be of type xarray data array
        assert all([isinstance(datum,xr.DataArray) for datum in data])

        for datum in data:
            # Update sample data collection
            self.update_sample(datum)
        
        # Combine coords for each list element of the Data Collection
        for sample_name in vars(self).keys():
            if sample_name in DATA_TYPES:
                for i,datum in enumerate(getattr(
                    self,
                    sample_name
                )):
                    getattr(
                        self,
                        sample_name
                    )[i] = xr.combine_by_coords(datum)
        
    def data_vars(self):
        return {k:v for k,v in vars(self).items() if k in DATA_TYPES}
    
    def update_sample(self, new_data, group_by:list=[]):

        # Get sample name
        sample_name = new_data.attrs['name']

        # Core dimensions for sample must be shared
        sample_shared_dims = XARRAY_SCHEMA[sample_name]['new_shape']
        # Grouped by sweep params that will be shared
        sample_shared_dims = set(sample_shared_dims).union(set(group_by))
        # All input-related sweep params that will be shared
        sample_shared_dims = set(sample_shared_dims).union(set(list(INPUT_SCHEMA.keys())))

        # Flag for whether update has completed
        complete = False
        
        if not sample_name in vars(self):
            setattr(
                self,
                sample_name,
                [[new_data]]
            )
        else:
            # Compute intersection of shared dims provided
            # and dims existing in sample
            existing_dims = list(getattr(self,sample_name)[0][0].dims)
            sample_shared_dims = sample_shared_dims.intersection(set(existing_dims))
            # Create a slice of dimensions that should be shared
            shared_dims_slice = dict(zip(list(sample_shared_dims),[slice(None)]*len(sample_shared_dims)))

            for i,datum in enumerate(getattr(self,sample_name)):
                # Check if old and new data arrays share exactly the same coordinates along specified dimensions
                # Then update the old data array
                coordinates_matched = all([
                    set(datum[0].coords.get(k).values) == set(new_data.coords.get(k).values)
                    for k in shared_dims_slice
                ])
                if coordinates_matched:
                    getattr(
                        self,
                        sample_name
                    )[i].append(new_data)
                    complete = True
                    # No need for more searching
                    # Otherwise new outputs will be appended at the end
                    break
            # Just add this new data array to the collection
            if not complete:
                getattr(
                    self,
                    sample_name
                ).append([new_data])

    def del_sample(self,sample_name):
        delattr(
            self,
            sample_name
        )

    def __getitem__(self,index):
        if index >= len(self):
            raise KeyError(f"Index {index} out of bounds for length {len(self)}.")
        else:
            new_data = deepcopy(self)
            for sample_name, sample_data in vars(self).items():
                if sample_name in DATA_TYPES:
                    setattr(
                        new_data,
                        sample_name,
                        sample_data[index]
                    )

            return new_data

    def __setitem__(self, index, new_data):
        if index >= len(self):
            raise KeyError(f"Index {index} out of bounds for length {len(self)}.")
        else:
            for sample_name in vars(new_data).keys():
                if sample_name in DATA_TYPES and sample_name in vars(self):
                    getattr(
                        self,
                        sample_name
                    )[index] = getattr(
                        new_data,
                        sample_name
                    )

    def __delitem__(self,index):
        if index >= len(self):
            raise KeyError(f"Index {index} out of bounds for length {len(self)}.")
        else:
            # Delete index element of Data Collection
            del self.data[index]

    def __repr__(self):
        return "\n\n".join([
            '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'+str(sample_name)+'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' + \
            (' \n'.join([str(dict(elem.sizes)) for elem in sample_data])
            if isinstance(sample_data,list) \
            else str(dict(sample_data.sizes))) + \
            '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
            for sample_name,sample_data in vars(self).items() \
            if sample_name in DATA_TYPES
        ])

    def sizes(self,dim:str=None):
        if dim is None:
            return {
                sample_name : (
                    len(sample_data)
                    if isinstance(sample_data,list)
                    else 1
                )
                for sample_name,sample_data in vars(self).items() 
                if sample_name in DATA_TYPES
            }
        else:
            elem = getattr(
                    self,
                    dim
                )
            return len(elem) if isinstance(elem,list) else 1


    def __len__(self):
        try:
            assert len(set([size for size in self.sizes().values()])) == 1
        except:
            raise Exception(f"Irregular DataCollection with sizes {str(self.sizes())}")
        
        return list(self.sizes().values())[0]


class OutputSummary(object):

    def __init__(self,
                 settings:dict={}, 
                 **kwargs):
        
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__,
            console_level = level,
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevels(
            console_level = level
        )

        # Store settings
        self.settings = settings
        # Store device
        self.device = self.settings.get('device','cpu')
        # Enable garbage collector
        gc.enable()
        # Update settings to parse statistics argument
        unpacked_stats = unpack_statistics(self.settings)
        if unpacked_stats is not None:
            deep_updates(
                self.settings,
                unpacked_stats
            )
        # Update settings to parse slice by argument
        self.settings['slice_by'] = parse_slice_by(
            self.settings.get('slice_by',[])
        )
        # Find output folders in collection
        self.output_folders = self.find_matching_output_folders(self)

    @classmethod
    def find_matching_output_folders(cls,__self__):
        if 'directories' in list(__self__.settings.keys()) and len(__self__.settings['directories']) > 0:
            output_dirs = []
            for _dir in list(__self__.settings['directories']):
                for dataset_name in __self__.settings['dataset_name']:
                    path_dir = os.path.join(
                        __self__.settings['out_directory'],
                        dataset_name,
                        __self__.settings['outputs'].get('out_group',''),
                        _dir
                    )
                    if os.path.exists(path_dir):
                        output_dirs.append(path_dir)
        else:
            # Search metadata based on search parameters
            # Get output directory and group
            output_directory = __self__.settings['out_directory']
            output_group = __self__.settings.get('out_group','')
            # Get experiment title
            experiment_titles = __self__.settings.get('experiment_title',[''])
            # Get dataset name
            dataset_names = __self__.settings.get('dataset_name',['.*'])
            # Get type of experiment
            experiment_types = __self__.settings.get('experiment_type',[''])

            # Get date
            if len(__self__.settings.get('dates',[])) <= 0:
                dates = ['']
            else:
                dates = list(__self__.settings.get('dates',['']))
            
            # Grab all output directories
            folder_patterns = []
            for data_name in dataset_names:
                for exp_type in experiment_types:
                    for exp_title in experiment_titles:
                        for dt in dates:
                            folder_patterns.append(
                                os.path.join(
                                    data_name,
                                    (f"{(exp_type+'.*') if len(exp_type) > 0 else ''}"+
                                    f"{('_'+exp_title+'.*') if len(exp_title) > 0 else ''}"+
                                    f"{(dt+'*') if len(dt) > 0 else ''}")
                                )
                            )
            # Combine them all into one pattern
            folder_patterns_re = "(" + ")|(".join(folder_patterns) + ")"
            # Get all output directories matching dataset name(s)
            output_dirs = flatten([
                get_all_subdirectories(
                    os.path.join(
                        output_directory,
                        dataset,
                        output_group
                    ) for dataset in dataset_names
                )
            ])
            # Sort them by string
            output_dirs = sorted(list(output_dirs))
            # Get all output dirs that match the pattern
            output_dirs = [output_folder for output_folder in output_dirs if re.search(folder_patterns_re,output_folder)]
            # Exclude those that are specified
            if len(__self__.settings.get('exclude',[])) > 0:
                output_dirs = [
                    output_folder for output_folder in output_dirs if __self__.settings['exclude'] not in output_folder
                ]
            # Sort by datetime
            date_pattern = re.compile(r"\d{1,2}\_\d{1,2}\_\d{2,4}\_\d{1,2}\_\d{1,2}\_\d{1,2}")
            output_dirs = sorted(
                output_dirs,
                key=(
                    lambda dt: datetime.strptime(
                        date_pattern.search(dt).group(0), 
                        "%d_%m_%Y_%H_%M_%S"
                    )
                )
            )
            # If no directories found terminate
            if len(output_dirs) == 0 :
                __self__.logger.error(f'No directories found in {os.path.join(output_directory,"*")}')
                raise Exception('Cannot read outputs.')
            else:
                __self__.logger.info(f"{len(output_dirs)} output folders found.")
        return output_dirs

    def collect_experiments_metadata(self):
        
        experiment_metadata = {}
        for i,output_folder in enumerate(self.output_folders):
            
            self.logger.info(f"Scanning folder {i+1}/{len(self.output_folders)}")

            # Collect outputs from folder
            outputs = self.get_folder_outputs(output_folder)

            # Loop through each member of the data collection
            for j in range(len(outputs.data)):
                # Collect metric metadata
                metric_data = self.get_experiment_metadata(j,outputs)
                
                if output_folder in experiment_metadata:
                    experiment_metadata[output_folder]= np.append(
                        experiment_metadata[output_folder],
                        metric_data,
                        axis = 0
                    )
                else:
                    experiment_metadata[output_folder] = experiment_metadata
        
        return experiment_metadata
    
    def get_experiment_metadata(self,index:int,outputs:Outputs):

        self.logger.info(f"Getting data element {index+1}/{len(outputs.data)}")
        # Create outputs copy
        new_outputs = deepcopy(outputs)
        # Update value of new outputs to the specific element of Data Collection
        new_outputs.data = outputs.data[index]

        # Apply these metrics to the data 
        metric_data = self.apply_metrics(
            experiment_id = outputs.experiment_id,
            outputs = new_outputs
        )
        
        # Extract useful data from config
        useful_metadata = {}
        for key in self.settings['metadata_keys']:
            # Replace iter with N
            path, found = self.config.path_find(
                key = key if key != 'iter' else 'N',
                settings = new_outputs.config.settings,
                key_path = [],
                found = False
            )
            if not found:
                self.logger.error(f"{key if key != 'iter' else 'N'} not found in experiment metadata.")
                continue
            # Extract directly from config
            has_sweep = self.config.has_sweep(path)
            if found and not has_sweep:
                useful_metadata[key],_ = self.config.path_get(
                    key_path = path
                )
            # Extract from data collection element
            elif self.config.has_sweep(path):
                # Grab first dataarray
                data_arr = list(new_outputs.data.data_vars().values())[0]
                useful_metadata[key] = data_arr.sizes[key]
        
        # Add useful metadata to metric data
        for m in range(len(metric_data)):
            metric_data[m]['folder'] = os.path.join(self.base_dir)
            for k,v in useful_metadata.items():
                metric_data[m][k] = v

        # Return metrix data
        return metric_data

    def get_sweep_outputs(
        self,
        sweep_configuration,
        experiment_id:str,
        sweep_dims:list,
        inputs:Inputs=None,
        group_by:list=[],
        input_slice:dict={},
        coordinate_slice:dict={}
    ):
        # Get specific sweep config 
        new_config,sweep = self.config.prepare_experiment_config(
            sweep_params = self.sweep_params,
            sweep_configuration = sweep_configuration
        )
        # Get outputs and unpack its statistics
        # This is the case where there are SOME input slices provided
        outputs = Outputs(
            config = new_config,
            settings = self.settings,
            data_names = self.settings['sample'],
            coordinate_slice = coordinate_slice,
            sweep_params = sweep,
            base_dir = self.base_dir,
            console_handling_level = self.settings['logging_mode'],
            logger = self.logger
        )
        
        # Get dictionary output data to be passed into xarray
        xr_dict_data = outputs.load_experiment_data(
            inputs=inputs,
            settings=self.settings,
            input_slice=input_slice,
            slice_samples=True
        )

        data_arr,slice_dict = {},{}
        for sample_name,xr_data in xr_dict_data.items():
            # Get sample dimensions
            sample_dims = XARRAY_SCHEMA[sample_name]['new_shape']
            # Get sample xr_data
            data = xr_data.pop('data')
            # Coordinates of output dataset
            coords = xr_data.pop('coordinates')
            # Create slice dictionary
            slice_dict = {
                k: [stringify_index(parse(elem)) for elem in coords[k]]
                for k in coords.keys()
            }
            data_arr[sample_name] = xr.DataArray(
                data = torch.tensor(
                    data,
                    dtype=DATA_TYPES[sample_name],
                    device=self.settings.get('device','cpu')
                ),
                coords = slice_dict,
                dims = (sweep_dims+sample_dims),
                attrs = dict(
                    name = sample_name,
                    experiment_id = experiment_id,
                    **{
                        k:sweep[k] for k in (list(CORE_COORDINATES_DTYPES.keys())+list(group_by))
                        if k in sweep and k != 'seed'
                    }
                )
            )

        return data_arr

    def get_folder_outputs(self,output_folder):
            
        # Read metadata config
        self.config = Config(
            path = os.path.join(output_folder,"config.json"),
            logger = self.logger
        )
        self.config.find_sweep_key_paths()

        # Parse sweep configurations
        self.sweep_params = self.config.parse_sweep_params()
        
        # Slice sweep params
        input_slice,coordinate_slice = self.create_sweep_slice()

        # Get all sweep configurations
        sweep_configurations, \
        param_sizes_str, \
        total_size_str = self.config.prepare_sweep_configurations(self.sweep_params)

        # Get output folder
        self.base_dir = output_folder.split(
            'samples/'
        )[0]
        output_folder_succinct = self.base_dir.split(
            self.config['inputs']['dataset']
        )[-1]
        self.logger.info("----------------------------------------------------------------------------------")
        self.logger.info(f'{output_folder_succinct}')
        self.logger.info(f"Parameter space size: {param_sizes_str}")
        self.logger.info(f"Total = {total_size_str}.")
        self.logger.info("----------------------------------------------------------------------------------")

        # Get sweep slice parameter names
        sweep_slice_param_names = list(input_slice.keys())+list(coordinate_slice.keys())
        
        # If there is no overlap between slice and available sweep params continue
        # as long as some slice params have been provided
        if set(sweep_slice_param_names) and \
            not set(sweep_slice_param_names).intersection(set(self.config.sweep_param_names)):
            self.logger.error(f"Slice parameters {sweep_slice_param_names} do not overlap with sweep parameters {self.config.sweep_param_names}.")
            raise Exception('Cannot read outputs.')
        
        # If inputs are not sweeped pre-load them
        
        # Import all input data
        inputs = Inputs(
            config = self.config,
            synthetic_data = False,
            logger = self.logger,
        )
        
        # If sweep is over input data
        if self.config.sweep_param_names and (
            (set(self.config.sweep_param_names).intersection(set(list(INPUT_SCHEMA.keys())))) or \
            (set(self.config.sweep_param_names).intersection(set(list(PARAMETER_DEFAULTS.keys()))))
        ):
            # Load inputs for every single output
            passed_inputs = None
        else:
            passed_inputs = deepcopy(inputs)

        # Reload global outputs
        outputs = Outputs(
            config = self.config,
            settings = self.settings,
            data_names = self.settings['sample'],
            base_dir = self.base_dir,
            console_handling_level = self.settings['logging_mode'],
            logger = self.logger
        )
        # Attach inputs to outputs object
        outputs.inputs = inputs
        safe_delete(inputs)

        # Try to load ground truth table
        self.settings['table_total'] = int(self.settings.get('table_total',1))
        # Read ground truth table
        if hasattr(inputs.data,'ground_truth_table'):
            # Gather dims from ground truth table
            # which is guaranteed to NOT be sweepable
            origin,destination = np.shape(outputs.inputs.data.ground_truth_table)
            outputs.inputs.data.ground_truth_table = xr.DataArray(
                data=torch.tensor(outputs.inputs.data.ground_truth_table,dtype=int32,device=self.device),
                name='ground_truth_table',
                dims=['origin','destination'],
                coords=dict(
                    origin=np.arange(1,origin+1,1,dtype='int16'),
                    destination=np.arange(1,destination+1,1,dtype='int16')
                )
            )
            # Try to update metadata on table samples
            self.settings['table_total'] = int(outputs.inputs.data.ground_truth_table.sum(dim=['origin','destination']).values)
        else:
            raise Exception('Inputs are missing ground truth table.')
        
        # If outputs are forced to be reloaded reload them all
        if self.settings.get('force_reload',False):
            overwrite = True
            samples_not_loaded = outputs.output_names
        else:
            # Attempt to load all samples
            # Keep track of samples not loaded
            overwrite = False
            samples_not_loaded = outputs.read_data_collection()

        # Load all necessary samples that were not loaded
        if len(samples_not_loaded) > 0:

            # Gather sweep dimension names
            sweep_dims = list(self.sweep_params['isolated'].keys())
            sweep_dims += list(self.sweep_params['coupled'].keys())

            # Gather h5 data from multiple files
            # and store them in xarray-type dictionaries
            output_datasets = []
            # Do it concurrently
            stop = None
            if self.settings.get('n_workers',1) > 1:
                # Initialise progress bar
                progress = tqdm(
                    total=len(sweep_configurations[:stop]),
                    desc='Collecting h5 data',
                    leave=False,
                    miniters=1,
                    position=0
                )
                with concurrency.ProcessPoolExecutor(self.settings.get('n_workers',1)*2) as executor:
                    futures = []
                    # Start the processes and ignore the results
                    for sweep_configuration in sweep_configurations[:stop]:
                        try:
                            future = executor.submit(
                                self.get_sweep_outputs,
                                sweep_configuration = sweep_configuration,
                                experiment_id = outputs.experiment_id,
                                sweep_dims = sweep_dims,
                                inputs = passed_inputs,
                                group_by = self.settings.get('group_by',[]),
                                input_slice = input_slice,
                                coordinate_slice = coordinate_slice 
                            )
                            # Update progress
                            futures.append(future)
                        except:
                            print(traceback.format_exc())
                            raise Exception('Getting sweep outputs failed.')

                    # Wait for all processes to finish
                    for future in concurrency.as_completed(futures):
                        try:
                            result = future.result()
                        except:
                            raise Exception(f"Future {future} failed")
                        progress.update(n=1)
                        output_datasets.append(result)
                    
                    # Delete futures and executor
                    safe_delete(futures)
                    executor.shutdown(True)
                    safe_delete(executor)
                # Delete progress bar
                progress.close()
                safe_delete(progress)

            # Do it sequentially
            else:
                for sweep_configuration in tqdm(
                    sweep_configurations[:stop],
                    desc='Collecting metadata',
                    leave=False,
                    position=0
                ):
                    # Get metric data for sweep dataset
                    output_datasets.append(
                        self.get_sweep_outputs(
                            sweep_configuration = sweep_configuration,
                            experiment_id = outputs.experiment_id,
                            sweep_dims = sweep_dims,
                            group_by = self.settings.get('group_by',[]),
                            inputs = passed_inputs,
                            input_slice = input_slice,
                            coordinate_slice = coordinate_slice 
                        )
                    )

            # Create xarray dataset
            try:
                self.logger.info(f"Attempting to load {', '.join(sorted(samples_not_loaded))}.")
                
                for sample_name in sorted(samples_not_loaded):
                    
                    self.logger.info(f"Combining {sample_name} homogeneous DataArray(s)")
                    # Homogeneous data arrays are the ones that have common coordinates
                    # along all core dimensions and group_by dimensions

                    for dataset in output_datasets:
                            
                        # Slice according to coordinate slice
                        if len(coordinate_slice) > 0:
                            outputs.data.update_sample(
                                dataset.pop(sample_name).sel(
                                    **{
                                        k:(v['values'] \
                                            if len(v['values']) > 1 \
                                            else v['values'][0]) \
                                        for k,v in coordinate_slice.items()
                                    }
                                ),
                                group_by = self.settings.get('group_by',[])
                            )
                        else:
                            outputs.data.update_sample(
                                dataset.pop(sample_name),
                                group_by = self.settings.get('group_by',[])
                            )
                    
                    # Combine coords for each list element of the Data Collection
                    self.logger.info(f'Combining {sample_name} by coords')
                    for i,datum in enumerate(getattr(
                        outputs.data,
                        sample_name
                    )):
                        getattr(
                            outputs.data,
                            sample_name
                        )[i] = xr.combine_by_coords(datum)
                    
                    # Write sample data collection to file
                    outputs.write_data_collection(
                        sample_name = sample_name,
                        overwrite = overwrite
                    )
                    # Do not overwite file
                    overwrite = False
            except:
                print(traceback.format_exc())
                self.logger.error('Failed creating xarray DataArray')
            

        return outputs

    def write_metadata_summaries(self,experiment_metadata:dict):
        if len(experiment_metadata.keys()) > 0:
            # Create dataframe
            experiment_metadata_df = pd.DataFrame(list(chain(*experiment_metadata.values())))
            experiment_metadata_df = experiment_metadata_df.set_index('folder')

            # Sort by values specified
            if len(self.settings['sort_by']) > 0 and \
                all([sb in experiment_metadata_df.columns.values for sb in self.settings['sort_by']]):
                experiment_metadata_df = experiment_metadata_df.sort_values(
                    by=list(self.settings['sort_by']),
                    ascending=self.settings['ascending']
                )

            # Find dataset directory name
            dataset = find_dataset_directory(self.settings['dataset_name'])

            # Make output directory
            output_directory = os.path.join(
                self.settings['out_directory'],
                dataset,
                self.settings['outputs'].get('out_group',''),
                'summaries'
            )
            makedir(output_directory)

            date_strings = '__'.join(self.settings['dates'])
            if 'directories' in list(self.settings.keys()) and len(self.settings['directories']) > 0:
                if 'filename_ending' in list(self.settings.keys()):
                    filepath = os.path.join(
                        output_directory,
                        f"{self.settings['filename_ending']}.csv"
                    )
                else:
                    filepath = os.path.join(
                        output_directory,
                        f"{'_'.join(os.path.basename(folder) for folder in set(list(experiment_metadata.keys())))}_"+\
                        f"{self.settings['filename_ending']}.csv"
                    )
            else:
                filepath = os.path.join(
                    output_directory,
                    f"{'_'.join(self.settings['experiment_type'])}_"+\
                    f"{'_'.join(self.settings['experiment_title'])+'_' if len(self.settings['experiment_title']) > 0 else ''}"+\
                    f"{date_strings if len(date_strings) < 4 else 'multiple_dates'}_"+\
                    f"burnin{self.settings['burnin']}_"+\
                    f"thinning{self.settings['thinning']}_"+\
                    f"N{self.settings.get('N',None)}_"+\
                    f"{self.settings['filename_ending']}.csv"
                )
            # Write experiment summaries to file
            self.logger.info(f"Writing summaries to {filepath}")
            write_csv(experiment_metadata_df,filepath,index=True)
            print('\n')
    
    
    def apply_metrics(self,experiment_id,outputs):
        # Get outputs and unpack its statistics
        self.logger.progress('Applying metrics...')
        
        metric_data = []
        for sample_name in self.settings['sample']:
            
            # Get sample
            self.logger.progress(f'Getting sample {sample_name}...')
            try:
                samples = outputs.get_sample(sample_name)
                # Unstack id multi-dimensional index
                samples = samples.unstack('id')
            except Exception:
                self.logger.error(f'Experiment {os.path.basename(experiment_id)} does not have sample {sample_name}')
                continue
            self.logger.progress(f"samples {np.shape(samples)}, {samples.dtype}")

            sample_data = {}
            for metric,statistics in self.settings['statistic'].items():
                # Issue warning if three set of statistics are provided
                # even though we need 2; one for the samples and one for the metric
                if len(statistics) > 2:
                    self.logger.warning(f"{len(statistics)} provided. Any more than 2 statistics will be ignored.")
                
                # Unpack sample and metric statistics
                sample_statistics_axes = statistics[0]
                metric_statistics_axes = statistics[1]
                
                self.logger.progress(f"{metric.lower()} {sample_statistics_axes} {metric_statistics_axes}")
                
                # Compute statistic before applying metric
                try:
                    samples_summarised = outputs.apply_sample_statistics(
                        samples = samples,
                        sample_name = sample_name,
                        statistic_dims = sample_statistics_axes
                    )
                except Exception:
                    self.logger.debug(traceback.format_exc())
                    self.logger.error(f"samples {np.shape(samples)}, {samples.dtype}")
                    self.logger.error(f"Applying statistic {' over axes '.join([str(s) for s in sample_statistics_axes])} \
                                    for sample {sample_name} for metric {metric.lower()} of experiment {experiment_id} failed")
                    print('\n')
                    continue
                self.logger.progress(f"samples_summarised {np.shape(samples_summarised)}, {samples_summarised.dtype}")

                # Get all attributes and their values
                attribute_keys = METRICS[metric.lower()]['loop_over']

                # Get all combinations of metric attribute values
                attribute_values = list(product(*[self.settings.get(attr,['none']) for attr in attribute_keys]))
                
                # Get copy of settings
                settings_copy = deepcopy(self.settings)
                    
                # Loop over all possible combinations of attribute values                        
                for attr_id, value_tuple in enumerate(attribute_values):
                    # Create metric attribute dictionary
                    attribute_settings = dict(zip(attribute_keys,value_tuple))
                    # Remove invalid ('none') attribute values
                    attribute_settings = {k:v for k,v in attribute_settings.items() if v != 'none'}

                    # Create an attributes string
                    attribute_settings_string = ','.join([f"{k}_{v}" for k,v in attribute_settings.items()])
                    for key,val in attribute_settings.items():
                        # Update settings values of attributes
                        settings_copy[key] = val

                    # Update kwargs specific for each metric
                    try:
                        metric_kwargs = self.update_metric_arguments(
                            metric.lower(),
                            outputs,
                            settings_copy
                        )
                    except Exception:
                        self.logger.debug(traceback.format_exc())
                        self.logger.error(f"metric {np.shape(metric)}, {metric.dtype}")
                        self.logger.error(f'Arguments for metric {metric} cannot be updated')
                        print('\n')
                        continue
                    try:
                        if metric.lower() != 'none' and metric.lower() != '' and metric.lower() in METRICS:
                            samples_metric = globals()[metric.lower()](
                                tab=samples_summarised,
                                **metric_kwargs
                            )
                            # Rename metric xr data array
                            samples_metric = samples_metric.rename(metric.lower())
                        else:
                            samples_metric = deepcopy(samples_summarised)
                    except Exception:
                        self.logger.debug(traceback.format_exc())
                        self.logger.error(f"samples_summarised {np.shape(samples_summarised)}, {samples_summarised.dtype}")
                        self.logger.error(f"tab0 {np.shape(metric_kwargs['tab0'])}, {metric_kwargs['tab0'].dtype}")
                        self.logger.error(f'Applying metric {metric.lower()} for {attribute_settings_string} \
                                            over sample {sample_name} \
                                            for experiment {experiment_id} failed')
                        print('\n')
                        continue

                    # print(sample_name,metric,samples_metric)
                    self.logger.progress(f"Samples metric is {np.shape(samples_metric)}")
                    # Apply statistics after metric
                    try:
                        self.settings['axis'] = METRICS[metric.lower()]['apply_axis']
                        metric_summarised = outputs.apply_sample_statistics(
                            samples = samples_metric,
                            sample_name = metric.lower(),
                            statistic_dims = metric_statistics_axes
                        )
                    except Exception:
                        self.logger.debug(traceback.format_exc())
                        self.logger.error(f"samples_metric {np.shape(samples_metric)}, {samples_metric.dtype}")
                        self.logger.error(f"Applying statistic(s) {'>'.join([str(m) for m in metric_statistics_axes])} \
                                            over metric {metric.lower()} and sample {sample_name} for experiment {experiment_id} failed")
                        print('\n')
                        continue
                    self.logger.debug(f"Summarised metric is {np.shape(metric_summarised)}")
                    
                    # Squeeze output
                    metric_summarised = np.squeeze(metric_summarised)
                    self.logger.progress(f"Summarised metric is squeezed to {np.shape(metric_summarised)}")
                    # Get metric data in pandas dataframe
                    metric_summarised = metric_summarised.to_dataframe().reset_index(drop=True)

                    # This loops over remaining sweep configurations
                    for _,sweep in metric_summarised.iterrows():
                        # Get sweep id
                        sweep_id = '& '.join([str(k)+'_'+str(v) for k,v in sweep.to_dict().items() if k != metric])
                        # Gather all key-value pairs from every row (corresponding to a single sweep setting)
                        # this is corresponds to variable 'row'
                        # Add every sweep configuration to this metric data
                        if sweep_id not in sample_data:
                            sample_data[sweep_id] = [{
                                **{
                                   "sample_name" : sample_name,
                                   f"{metric}_sample_statistic" : '|'.join(
                                       [stringify_statistic(_stat_dim) 
                                        for _stat_dim in sample_statistics_axes]
                                    ),
                                    f"{metric}_metric_statistic" : '|'.join(
                                        [stringify_statistic(_stat_dim) 
                                         for _stat_dim in metric_statistics_axes]
                                    )
                                },
                                **sweep.to_dict(),
                                **attribute_settings
                            }]*len(attribute_values)
                            
                        else:
                            # Skip first element to avoid overwritting 
                            # metric settings
                            sample_data[sweep_id][attr_id].update({
                                f"{metric}_sample_statistic" : '|'.join(
                                       [stringify_statistic(_stat_dim) 
                                        for _stat_dim in sample_statistics_axes]
                                ),
                                f"{metric}_metric_statistic" : '|'.join(
                                    [stringify_statistic(_stat_dim) 
                                        for _stat_dim in metric_statistics_axes]
                                ),
                                metric:sweep[metric],
                                **attribute_settings
                            })
                
                # Add sample data to metric data
                for sweep_metric_list in sample_data.values():
                    metric_data += list(sweep_metric_list)
            
        return metric_data
    
    def update_metric_arguments(self,metric,outputs,settings):        
        # Initialise metric arguments
        metric_arguments = {}
        settings_copy = deepcopy(settings)
        if metric.lower() == 'shannon_entropy':
            dummy_config = Config(settings=outputs.config)
            ct = instantiate_ct(
                config=dummy_config,
                log_to_console=False,
                logger=self.logger
            ) 
            try:
                settings_copy['distribution_name'] = f"log_{ct.distribution_name}_pmf_normalised"
                assert hasattr(ProbabilityUtils,settings_copy['distribution_name'])
            except Exception as e:
                settings_copy['distribution_name'] = ''
                self.logger.debug(traceback.format_exc())
                self.logger.error(f"No distribution matching key {ct.distribution_name}")
                print('\n')
                # raise Exception(f'Arguments for metric {metric} cannot be updated')
            # Pass log intensities as argument
            metric_arguments['tab0'] = np.log(outputs.get_sample('intensity'),dtype='float32')
        else:
            # Pass ground truth table as argument
            metric_arguments['tab0'] = outputs.inputs.data.ground_truth_table
        
        # Pass standard metric arguments
        metric_arguments.update(settings_copy)

        return metric_arguments
    

    def create_sweep_slice(self):
        if len(self.config.isolated_sweep_paths) > 0 or len(self.config.coupled_sweep_paths) > 0:

            coordinate_slice = {}
            input_slice = {}
            # Loop through key-value pairs used
            # to subset the output samples
            for name,sliced_values in self.settings['slice_by'].items():
                # Loop through experiment's isolated sweeped parameters
                for target_name,target_path in self.config.isolated_sweep_paths.items():
                    # If there is a match between the two
                    if name == target_name:
                        # If is a coordinate add to the coordinate slice
                        # This slices the xarray created from the outputs samples
                        if self.config.is_sweepable(name):
                            if name in list(coordinate_slice.keys()):
                                coordinate_slice[name]['values'] = np.append(
                                    coordinate_slice[name]['values'],
                                    sliced_values
                                )
                            else:
                                coordinate_slice[name] = {
                                    "paths": target_path,
                                    "values": sliced_values
                                }
                        # If is NOT a coordinate add to the input slice
                        # This slices the input data and requires reinstantiating outputs
                        else:
                            input_slice[name] = {
                                "values":sliced_values,
                                "key_path":target_path
                            }
                # Loop through experiment's coupled sweeped parameters
                for target_name,target_paths in self.config.coupled_sweep_paths.items():
                    # If any of the coupled sweeps contain the target name
                    if name in list(target_paths.keys()):
                        for coupled_name,target_path in target_paths.items():
                            # If is a coordinate add to the coordinate slice
                            # This slices the xarray created from the outputs samples
                            if self.config.is_sweepable(coupled_name):
                                if name in list(coordinate_slice.keys()):
                                    coordinate_slice[coupled_name]['values'] = np.append(
                                        coordinate_slice[coupled_name]['values'],
                                        sliced_values
                                    )
                                else:
                                    coordinate_slice[coupled_name] = {
                                        "paths": target_path,
                                        "values": sliced_values
                                    }
                            # If is NOT a coordinate add to the input slice
                            # This slices the input data and requires reinstantiating outputs
                            else:
                                input_slice[name] = {
                                    "values":sliced_values,
                                    "key_path":target_path
                                }
            if len(input_slice) == 0:
                input_slice = {}
        else:
            coordinate_slice = {}
            input_slice = {}
        return input_slice,coordinate_slice
