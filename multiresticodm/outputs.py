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

from tqdm import tqdm
from torch import int32
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Union,List,Tuple
from itertools import product,chain

from multiresticodm.utils.misc_utils import *
from multiresticodm.utils.exceptions import *
from multiresticodm.utils.math_utils import *
from multiresticodm.config import Config
from multiresticodm.inputs import Inputs
from multiresticodm.fixed.global_variables import *
from multiresticodm.spatial_interaction_model import *
from multiresticodm.utils.multiprocessor import BoundedQueueProcessPoolExecutor

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
        self.data_names = None \
            if (data_names is None) or (len(data_names) <= 0) \
            else list(data_names)
        # Store settings
        self.settings = settings

        # Sample names must be a subset of all data names
        if self.data_names is not None:
            try:
                assert set(self.data_names).issubset(set(DATA_SCHEMA.keys()))
            except Exception:
                self.logger.error('Some sample names provided are not recognized')
                self.logger.error(','.join(self.data_names))
                self.logger.debug(traceback.format_exc())
                raise InvalidDataNames('Cannot load outputs.')

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
            ) if kwargs.get('base_dir',None) is None else kwargs['base_dir']
            

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
            raise InvalidConfigType(f'Config {config} of type {type(config)} not recognised.')
        
        # Get name of intensity model
        self.intensity_model_name = self.config.settings[self.intensity_model_class]['name']
        # If these are sweeped store their range otherwse
        self.intensity_model_name = self.intensity_model_name['sweep']['range'] \
            if isinstance(self.intensity_model_name,dict) \
            else [self.intensity_model_name]
        
        # Store sample data requirements
        self.output_names = []
        # Setup universe of output names
        avail_output_names = set(self.data_names).intersection(set(list(OUTPUT_SCHEMA.keys()))) \
            if self.data_names is not None \
            else set(list(OUTPUT_SCHEMA.keys()))

        for sam in avail_output_names:
            if sam == 'intensity':

                # Add all intensity-related data names
                for model_name in self.intensity_model_name:
                    self.output_names.extend(SAMPLE_DATA_REQUIREMENTS[sam][model_name])
                # Add signs if required
                if 'sign' in EXPERIMENT_OUTPUT_NAMES[self.config['experiment_type']]:
                    self.output_names.append('sign')
                # Take only unique values
                self.output_names = list(set(self.output_names))
                
            elif sam in LOSS_DATA_REQUIREMENTS \
                and 'neural_network' in self.config.settings \
                and 'loss' in EXPERIMENT_OUTPUT_NAMES[self.config['experiment_type']]:
                # Grab all loss names
                loss_names = list(flatten(self.config['neural_network']['loss']['loss_name']['sweep']['range'])) \
                    if isinstance(self.config['neural_network']['loss']['loss_name'],dict) \
                    else [self.config['neural_network']['loss']['loss_name']]
                loss_names = set(list(flatten(loss_names))+['total_loss'])
                # Add them to output names
                if sam in loss_names:
                    self.output_names.append(sam)
            elif sam in EXPERIMENT_OUTPUT_NAMES[self.config['experiment_type']]:
                self.output_names.append(sam)
        
        # Keep only unique values
        self.output_names = list(flatten(self.output_names))
        self.output_names = list(set(self.output_names))
        # Get input names
        self.input_names = [
            sam for sam in set(self.data_names).intersection(set(list(INPUT_SCHEMA.keys())))
        ] if self.data_names is not None else list(INPUT_SCHEMA.keys())
        
        # Name output sample directory according 
        # to sweep params (if they are provided)
        self.sweep_id = self.config.get_sweep_id(sweep_params = sweep_params)

        # Create parameter slice
        self.update_parameter_slice()
        
        if self.coordinate_slice:
            self.logger.info("//////////////////////////////////////////////////////////////////////////////////")
            self.logger.info("Slicing coordinates by:")
            for dimkey,dimval in self.coordinate_slice.items():
                self.logger.info(f"{dimkey}: {', '.join([str(dv) for dv in dimval])}")
            self.logger.info("//////////////////////////////////////////////////////////////////////////////////")

    def get(self,index:int):
        self_copy = deepcopy(self)
        self_copy.data = self_copy.data[index]
        return self_copy
    
    def data_vars(self):
        return {k:v for k,v in self.data._vars_().items() if k in DATA_SCHEMA}

    def trim_sweep_configurations(self,sweep_configurations:list=[],sweep_params:dict={}):
        # Loop through each sweep configuration
        for sweep_conf in sweep_configurations:
            # Extract sweep params for this configuration
            _,sweep = self.config.prepare_experiment_config(
                sweep_params,
                sweep_conf
            )
            # Get sweep id
            sweep_id = self.config.get_sweep_id(sweep_params = sweep)
            # Check if 'metadata.json' or 'config.json' and 'data.h5' and 'outputs.log'
            # exist in output directory
            file_exists = {}
            for file in ['metadata.json','data.h5','outputs.log']:
                # Create filepath
                filepath = os.path.join(
                    self.outputs_path,
                    'samples',
                    sweep_id,
                    f"outputs.log"
                )
                # Check if path exists and filepath corresponds to a file
                file_exists[file] = os.path.exists(filepath) and os.path.isfile(filepath)
            # Check if necessary data exists 
            data_exists = file_exists['metadata.json'] and \
                file_exists['data.h5'] and \
                file_exists['outputs.log']
            # If necessary data does not exist
            # Add sweep configurations that need to be run
            if not data_exists:
                yield sweep_conf
                

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
                                  which does not exist in {','.join(list(self.inputs.data_vars().keys()))}")
        for output in output_names:
            try:
                assert hasattr(self.data,output)
            except:
                available = False
                self.logger.error(f"Sample {sample_name} requires output {output} \
                                  which does not exist in {','.join(list(self.data_vars().keys()))}")
        return available

    def slice_samples_by_index(self,samples):

        for dim_names, slice_setts in self.settings['burnin_thinning_trimming'].items():
            
            # Gather dim names
            dim_names = dim_names.split('+')
            # Get the intersection between available dims and specified dims
            dim_names = list(set(samples.dims).intersection(dim_names))

            # if samples do not have these dimensions, carry on
            if len(dim_names) <= 0:
                continue

            # Stack all dims together
            samples = samples.stack(temp_dim=dim_names)

            # Get total number of iterations
            total_samples = samples.sizes['temp_dim']

            # Get burnin parameter
            burnin = min(slice_setts.get('burnin',0),total_samples)

            # Get thinning parameter
            thinning = min(slice_setts.get('thinning',1),total_samples)

            # Get iterations
            iters = np.arange(start=burnin,stop=total_samples,step=thinning,dtype='int32')
            
            # Get number of samples to keep
            trimming = min(slice_setts.get('trimming',None),len(iters))

            # Trim iterations
            iters = iters[:trimming]
            
            # Apply burnin, thinning and trimming to samples
            samples = samples.isel(temp_dim = iters)

            # Unstack temp dim
            samples = samples.unstack('temp_dim')

        return samples
    
    
    def load_geometry(self,geometry_filename,default_crs:str='epsg:27700'):
        # Load geometry from file
        geometry = gpd.read_file(geometry_filename)
        geometry = geometry.set_crs(default_crs,allow_override=True)
        
        return geometry


    def load_h5_data(self):
        self.logger.note('Loading h5 data into xarrays...')
        # Get h5 file
        h5file = os.path.join(self.outputs_path,'samples',f"{self.sweep_id}","data.h5")
        try:
            assert os.path.exists(h5file) and os.path.isfile(h5file)
        except:
            raise MissingFiles(f"H5 file {h5file} not found.")
        # Read h5 data
        local_coords,global_coords,data_vars = self.read_h5_file(h5file)
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
                if DATA_SCHEMA[sample_name]["is_iterable"]:                    
                    dims = np.shape(sample_data)[2:]
                else:
                    dims = np.shape(sample_data)[1:]
                
                # For each dim create coordinate
                for i,d in enumerate(dims):
                    obj,func = DATA_SCHEMA[sample_name]['funcs'][i]
                    # Create coordinate ranges based on schema
                    coordinates[DATA_SCHEMA[sample_name]["dims"][i]] = deep_call(
                        globals()[obj],
                        func,
                        None,
                        start=1,
                        stop=d+1,
                        step=1
                    ).astype(DATA_SCHEMA[sample_name]['args_dtype'][i])

            # Keep only necessary global coordinates
            sample_global_dims = ['iter'] if DATA_SCHEMA[sample_name]['is_iterable'] else []
            sample_global_dims += DATA_SCHEMA[sample_name].get("dims",[])

            # Update coordinates to include schema and sweep coordinates
            # Keep only coordinates that are 1) core
            # 2) isolated sweeps 
            # or 3) the targets of coupled sweeps
            coordinates = {
                **{k:v for k,v in local_coords.items() \
                   if k != sample_name},
                **{k:v for k,v in global_coords.items() \
                   if k in sample_global_dims},
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
    
    def read_h5_file(self,filename,**kwargs):
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
                        local_coords[k] = {parse(v,None)}

                self.logger.debug('Store dataset')
                # Store dataset
                for sample_name,sample_data in h5data[group_id].items():
                    # If this data is not required, skip storing it
                    if sample_name not in self.output_names:
                        continue
                    
                    start_idx = 0
                    if OUTPUT_SCHEMA[sample_name]["is_iterable"]:
                        if 'iter' in global_coords:
                            global_coords.update({
                                "iter" : np.arange(
                                    start = 1,
                                    stop = sample_data.shape[start_idx]+1,
                                    step = 1,
                                    dtype = 'int32'
                                )
                            })
                        else:
                            global_coords['iter'] = np.arange(
                                start = 1,
                                stop = sample_data.shape[start_idx]+1,
                                step = 1,
                                dtype = 'int32'
                            )
                        start_idx = 1
                    for i,dim in enumerate(OUTPUT_SCHEMA[sample_name].get("dims",[])):
                        if dim in global_coords:
                            global_coords.update({
                                dim : np.arange(
                                    start=1,
                                    stop=sample_data.shape[i+start_idx]+1,
                                    step=1,
                                    dtype='int32'
                                )
                            })
                        else:
                            global_coords[dim] = np.arange(
                                start=1,
                                stop=sample_data.shape[i+start_idx]+1,
                                step=1,
                                dtype='int32'
                            )
                    # print(global_coords)
                    # Append
                    self.logger.debug(f'Appending {sample_name}')
                    data_vars[sample_name] = np.array([sample_data[:]])

        except BlockingIOError:
            self.logger.debug(f"Skipping in-use file: {filename}")
            return {},{},{}
        except Exception:
            self.logger.debug(traceback.format_exc())
            raise CorruptedFileRead(f'Cannot read file {filename}')
        return local_coords,global_coords,data_vars
        
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

        title = self.config['outputs']['title']
        title = title if isinstance(title,str) else None
        
        proposal = self.config['mcmc']['contingency_table']['proposal'] \
            if 'mcmc' in self.config.settings and 'contingency_table' in self.config.settings['mcmc'] \
            else None
        proposal = proposal if isinstance(proposal,str) else None

        if sweep_experiment_id is None:
            if self.config['experiment_type'].lower() in ['tablesummariesmcmcconvergence','table_mcmc_convergence']:
                return self.config['experiment_type']+'_K'+\
                        str(self.config['K'])+'_'+\
                        ((proposal+'_') if proposal is not None else '') +\
                        ((title+'_') if title is not None else '') +\
                        self.config['datetime']
            elif self.config['experiment_type'].lower() == 'table_mcmc':
                return self.config['experiment_type']+'_'+\
                        ((proposal+'_') if proposal is not None else '') +\
                        ((title+'_') if title is not None else '') +\
                        self.config['datetime']
            else:
                return self.config['experiment_type']+'_'+\
                        ((noise_level+'Noise_') if noise_level is not None else '') +\
                        ((title+'_') if title is not None else '') +\
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
            raise InvalidLoggerType(f'Cannot write outputs of invalid type logger {type(self.logger)}')

    def write_metadata(self,dir_path:str,filename:str) -> None:
        # Define filepath
        filepath = os.path.join(self.outputs_path,dir_path,f"{filename.split('.')[0]}.json")
        if (os.path.exists(filepath) and self.config['experiments'][0]['overwrite']) or (not os.path.exists(filepath)):
            if isinstance(self.config,Config):
                write_json(self.config.settings,filepath,indent=2)
            elif isinstance(self.config,dict):
                write_json(self.config,filepath,indent=2)
            else:
                raise InvalidMetadataType(f'Cannot write metadata of invalid type {type(self.config)}')

    def print_metadata(self) -> None:
        print_json(self.config,indent=2)

    def open_output_file(self,sweep_params:dict={}):
        # Create output directories if necessary
        self.create_output_subdirectories(sweep_id=self.sweep_id)
        if hasattr(self,'config') and hasattr(self.config,'settings'):
            export_samples = list(deep_get(key='export_samples',value=self.config.settings))
            export_metadata = list(deep_get(key='export_metadata',value=self.config.settings))
            # Keep first entry of these values
            export_samples = export_samples[0] if len(export_samples) > 0 else True
            export_metadata = export_metadata[0] if len(export_metadata) > 0 else True
            
            # Write to file
            if export_samples:
                self.logger.note(f"Creating output file at:\n        {self.outputs_path}")
                try:
                    self.h5file = h5.File(
                        os.path.join(
                            self.outputs_path,
                            'samples',
                            f"{self.sweep_id}",
                            "data.h5"
                        ), 
                        mode='w'
                    )
                except Exception as exc:
                    self.logger.debug(traceback.format_exc())
                    raise MissingFiles(
                        message = f"H5 file {os.path.join(self.outputs_path,'samples',self.sweep_id,'data.h5')} not found"
                    )
                
                # Store experiment id
                self.h5group = self.h5file.create_group(self.experiment_id)

                # Store sweep configurations as attributes
                self.h5group.attrs.create("sweep_params",list(sweep_params.keys()))
                self.h5group.attrs.create("sweep_values",['none' if val is None else str(val) for val in sweep_params.values()])
                
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

    def write_data_collection(self, sample_names:list = None):
        # Make output directory
        output_directory = os.path.join(self.outputs_path,'sample_collections')
        makedir(output_directory)
        # Write output to file
        filepath = os.path.join(
            output_directory,
            f"data_collection.nc"
        )
        self.logger.info(f'Writing output collection to {filepath}')
        # Get specific sample names
        sample_names = sample_names if sample_names is not None else list(self.data_vars().keys())
        sample_names = set(sample_names).intersection(set(list(self.data_vars().keys())))
        
        # Writing or appending mode
        if (not os.path.exists(filepath) and not os.path.isfile(filepath)):
            mode = 'w'
        else:
            mode = 'a'

        for sam_name in sample_names:
            for i,datum in enumerate(getattr(self.data,sam_name)):
                write_xr_data(
                    datum,
                    filepath,
                    group = f"/{sam_name}/{i}"
                )

    def read_data_collection(self, group_by:list):
        # Outputs filepath
        output_filepath = os.path.join(self.outputs_path,'sample_collections',"data_collection.nc")

        if not os.path.isfile(output_filepath) or \
            not os.path.exists(output_filepath) or \
            self.settings.get('force_reload',False):

            # Remove existing file
            if os.path.exists(output_filepath):
                os.remove(output_filepath)

            return self.output_names
        else:
            # Start with the premise that all available samples should be loaded
            samples_not_loaded = deepcopy(self.output_names)
            # Get all sample names and collection ids (all of that constitutes the group ids)
            sample_group_ids = []
            with nc.Dataset(output_filepath, mode='r') as nc_file:
                sample_group_ids = read_xr_group_ids(nc_file,list_format=False)
            # If this throws an exception it means that some 
            # elements corresponding to some sample names 
            # are missing from the data collection
            try:
                sample_ids = {
                    s: ','.join(ids) \
                    for s,ids in sample_group_ids.items()
                }
                assert len(set([sid for sid in sample_ids.values()])) == 1
            except Exception as exc:
                self.logger.debug(exc)
                # Force reload all data
                return self.output_names

            data_arrs = []
            for sam_name,sample_collection in sample_group_ids.items():
                sample_loaded = True
                for collection_id in sample_collection:
                    # Read data array
                    data_array = read_xr_data(
                        filepath = output_filepath,
                        group = f'/{sam_name}/{collection_id}'
                    )
                    data_arrs.append(data_array)
                # remove loaded sample from consideration
                # since it has been succesfully loaded
                if sample_loaded and sam_name in samples_not_loaded:
                    samples_not_loaded.remove(sam_name)

            self.data = DataCollection(
                *data_arrs,
                group_by = group_by,
                logger = self.logger
            )
            return samples_not_loaded

    def create_filename(self,sample=None):
        # Decide on filename
        if (sample is None) or (not isinstance(sample,str)):
            filename = f"{','.join(self.settings['sample'])}"
        else:
            filename = f"{sample}"
        if 'statistic' in self.settings:
            arr = []
            for metric,statistic in self.settings['statistic'].items():
                arr.append(str(metric.lower()) + ','.join([str(stat) for stat in list(flatten(statistic[0]))]))
            filename += f"{'_'.join(arr)}"
        if 'table_dim' in self.config:
            filename += f"_{self.config['table_dim']}"
        if 'table_total' in self.config:
            filename += f"_{self.config['table_total']}"
        if 'type' in self.config and len(self.config['type']) > 0:
            filename += f"_{self.config['type']}"
        if 'title' in self.settings and len(self.settings['title']) > 0:
            filename += f"_{self.settings['title']}"
        if 'viz_type' in self.settings:
            filename += f"_{self.settings['viz_type']}"
        if 'burnin_thinning_trimming' in self.settings:
            filename += '_'+'_'.join([
                        f"({var},burnin{setts['burnin']},thinning{setts['thinning']},trimming{setts['trimming']})"
                        for var,setts in self.settings['burnin_thinning_trimming'].items()
                    ])
        return filename

    def get_sample(self,sample_name:str):

        if sample_name == 'intensity':
            # Get sim model 
            self.logger.debug('getting sim model')
            # Read from config
            intensity_name = self.config.settings[self.intensity_model_class]['name']
            # Otherwise read from coords of first dataarray
            first_da = self.get_sample(self.output_names[0])
            intensity_name = intensity_name if isinstance(intensity_name,str) else first_da.coords['name'].item()

            # Get intensity model
            IntensityModelClass = globals()[intensity_name+'SIM']
            # Check that required data is available
            self.logger.debug('checking sim data availability')
            self.check_data_availability(
                sample_name = sample_name,
                input_names = IntensityModelClass.REQUIRED_INPUTS,
                output_names = IntensityModelClass.REQUIRED_OUTPUTS,
            )
            # Compute intensities for all samples
            table_total = self.settings.get('table_total') if self.settings.get('table_total',-1.0) > 0 else 1.0

            # Instantiate ct
            IntensityModel = IntensityModelClass(
                config = self.config,
                logger = self.logger,
                **{input:self.get_sample(input) for input in IntensityModelClass.REQUIRED_INPUTS}
            )

            # Compute log intensity
            samples = IntensityModel.log_intensity(
                grand_total = torch.tensor(table_total,dtype=int32),
                torch = False,
                **{output:self.get_sample(output) for output in IntensityModelClass.REQUIRED_OUTPUTS}
            )

            # Create new dataset
            samples = samples.rename('intensity')
            # Exponentiate
            samples = np.exp(samples)

        
        elif sample_name in INPUT_SCHEMA:

            # Cast to xr DataArray
            self.inputs.cast_to_xarray()

            # Read from config
            intensity_name = self.config.settings[self.intensity_model_class]['name']
            # Otherwise read from coords of first dataarray
            first_da = self.get_sample(self.output_names[0])
            intensity_name = intensity_name if isinstance(intensity_name,str) else first_da.coords['name'].item()

            # Get intensity model
            IntensityModelClass = globals()[intensity_name+'SIM']

            self.check_data_availability(
                sample_name = sample_name,
                input_names = IntensityModelClass.REQUIRED_INPUTS
            )
            # Get samples and cast them to appropriate type
            if torch.is_tensor(getattr(self.inputs.data,sample_name)):
                samples = torch.clone(
                    getattr(self.inputs.data,sample_name).to(
                        NUMPY_TO_TORCH_DTYPE[INPUT_SCHEMA[sample_name]['dtype']]
                    )
                )
            else:
                samples = torch.tensor(
                    getattr(self.inputs.data,sample_name), 
                    dtype=NUMPY_TO_TORCH_DTYPE[INPUT_SCHEMA[sample_name]['dtype']]
                )

        else:
            if not hasattr(self.data,sample_name):
                raise MissingData(
                    missing_data_name = sample_name,
                    data_names = ', '.join(list(self.data_vars().keys())),
                    location = 'Outputs'
                )
            elif self.data.sizes(dim = sample_name) > 1:
                raise IrregularDataCollectionSize(f"Cannot process {sample_name} Data Collection of size {self.data.sizes(dim = sample_name)} > 1.")
            else:
                # Get xarray
                samples = getattr(self.data,sample_name)

            # Slice according to coordinate slice
            if self.coordinate_slice:
                slice_dict = {
                    k:v
                    for k,v in self.coordinate_slice.items()
                    if k in list(samples.dims)
                }
                self.logger.debug(f'Before coordinate slicing {sample_name}: {dict(samples.sizes)}')
                try:
                    samples = samples.sel( **slice_dict )
                except Exception as exc:
                    self.logger.debug(f"{samples.coords.values}")
                    raise CoordinateSliceMismatch(f"""{sample_name} coordinate slice
                        {', '.join([str(k)+': '+str(v)
                        for k,v in self.coordinate_slice.items() 
                        if k in list(samples.dims)])}
                        failed due to coordinate value mismatch.
                        {str(exc)}
                        {dict({k:samples.coords[k].values.tolist() for k in self.coordinate_slice.keys()})}
                    """)
                self.logger.debug(f'After coordinate slicing {sample_name}: {dict(samples.sizes)}')
            
            # Apply burning, thinning and trimming
            self.logger.debug(f'Before index slicing {sample_name}: {dict(samples.sizes)}')
            # Keep track of previous number of iterations
            prev_iter = deepcopy({
                k:samples.sizes[k] \
                for ktuple in self.settings['burnin_thinning_trimming'].keys() \
                for k in ktuple.split('+')
                if k in samples.dims
            })
            samples = self.slice_samples_by_index(samples)
            self.logger.debug(f'After index  slicing {sample_name}: {dict(samples.sizes)}')
            # If no data remains after slicing raise exception
            if any([samples.sizes[k] <= 0 for k in samples.dims]):
                raise EmptyData(
                    data_names = sample_name,
                    message = f"slicing {list(prev_iter.keys())} with shape {prev_iter} using {self.settings['burnin_thinning_trimming']}"
                )
            
            # Find iteration dimensions
            iter_dims = [x for x in samples.dims if x in ['iter','seed']]


            # Find sweep dimensions that are not core coordinates
            sweep_dims = [d for d in samples.dims if d not in (list(CORE_COORDINATES_DTYPES.keys()))]
            
            # print(iter_dims,sweep_dims)

            # Stack variables and reorder data
            if len(sweep_dims) > 0 and len(iter_dims) > 0:
                # Stack all non-core coordinates into new coordinate
                samples = samples.stack(
                    id = tuplize(iter_dims),
                    sweep = tuplize(sweep_dims)
                )
                # Reorder coordinate names
                samples = samples.transpose(
                    'id',*OUTPUT_SCHEMA[sample_name].get("dims",[]),'sweep'
                )
            elif len(iter_dims) > 0:
                samples = samples.stack(
                    id = tuplize(iter_dims)
                )
                # Reorder coordinate names
                samples = samples.transpose(
                    'id',*OUTPUT_SCHEMA[sample_name].get("dims",[])
                )
            elif len(sweep_dims) > 0:
                samples = samples.stack(
                    sweep = tuplize(sweep_dims)
                )
                # Reorder coordinate names
                samples = samples.transpose(
                    *OUTPUT_SCHEMA[sample_name].get("dims",[]),'sweep'
                )
            else:
                # Reorder coordinate names
                samples = samples.transpose(
                    *OUTPUT_SCHEMA[sample_name].get("dims",[])
                )
            

            # If parameter is beta, scale it by bmax
            if sample_name == 'beta' and self.intensity_model_class == 'spatial_interaction_model':
                return samples * self.config.settings[self.intensity_model_class]['parameters']['bmax']

        self.logger.progress(f"Loaded {sample_name} sample")
        return samples
    
    def compute_statistic(self,data,sample_name,statistic,**kwargs):
        self.logger.debug(f"""compute_statistic {sample_name},{type(data)},{statistic}""")
        if statistic is None or statistic.lower() == '' or 'sample' in statistic.lower() or len(kwargs.get('dim',[])) == 0:
            return data
        
        elif statistic.lower() == 'signedmean' and \
            sample_name in list(OUTPUT_SCHEMA.keys()):
            if sample_name in list(INTENSITY_SCHEMA.keys()) and \
                'sign' in EXPERIMENT_OUTPUT_NAMES[self.config.settings['experiment_type']]:
                # Get sign samples
                signs = self.get_sample('sign')
                # Unstack dimensions
                signs = unstack_dims(signs,['id'])
                # Compute moments
                numerator = data.dot(signs,dims=['iter'])
                denominator = signs.sum('iter')
                numerator,denominator = xr.align(numerator,denominator, join='exact')
                return numerator/denominator
            else:
                return self.compute_statistic(data,sample_name,'mean',dim=kwargs['dim'])

        elif (statistic.lower() == 'signedvariance' or statistic.lower() == 'signedvar') and \
            sample_name in list(OUTPUT_SCHEMA.keys()):

            if sample_name in list(INTENSITY_SCHEMA.keys()) and \
                'sign' in EXPERIMENT_OUTPUT_NAMES[self.config.settings['experiment_type']]:
                # Compute mean
                samples_mean = self.compute_statistic(data,sample_name,'signedmean',**kwargs)
                # Get sign samples
                signs = self.get_sample('sign')
                # Unstack dimensions
                signs = unstack_dims(signs,['id'])
                # Compute moments
                numerator = (data**2).dot(signs,dims=['iter'])
                denominator = signs.sum('iter')
                numerator,denominator = xr.align(numerator,denominator, join='exact')
                return (numerator/denominator - samples_mean**2)
            else:
                return deep_call(
                    data,
                    f".var(dim)",
                    data,
                    dim=kwargs['dim']
                )
                # return self.compute_statistic(data,sample_name,'var',**kwargs)
        
        elif statistic.lower() == 'error' and \
            sample_name in [param for param in list(OUTPUT_SCHEMA.keys()) if 'error' not in param]:
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
            if dims is not None:
                dims = [d for d in dims if d in list(sample_statistic.dims)]
            else:
                dims = [None] 

            sample_statistic = self.compute_statistic(
                                    data=sample_statistic,
                                    sample_name=sample_name,
                                    statistic=stat,
                                    dim=dims,
                                    **kwargs
                                )

        return sample_statistic
    
    def update_parameter_slice(self):
        if len(self.config.isolated_sweep_paths) > 0 or len(self.config.coupled_sweep_paths) > 0:

            coordinate_slice = {}
            # Loop through key-value pairs used
            # to subset the output samples
            for coord_slice in self.settings.get('coordinate_slice',{}):
                # Unpack coordinate slice
                dim_name,dim_values = coord_slice
                dim_values = list(map(sigma_to_noise_regime,dim_values)) if dim_name == 'sigma' else dim_values
                # Loop through experiment's isolated sweeped parameters
                for target_name,target_path in self.config.isolated_sweep_paths.items():
                    # If there is a match between the two
                    if dim_name == target_name:
                        # If is a coordinate add to the coordinate slice
                        # This slices the xarray created from the outputs samples
                        if self.config.is_sweepable(dim_name):
                            if dim_name in list(coordinate_slice.keys()):
                                coordinate_slice[dim_name]['values'] = np.append(
                                    coordinate_slice[dim_name]['values'],
                                    [
                                        stringify_coordinate(parse(elem),None) \
                                        for elem in dim_values
                                    ]
                                )
                            else:
                                coordinate_slice[dim_name] = [
                                    stringify_coordinate(parse(elem),None) \
                                    for elem in dim_values
                                ]
                # Loop through experiment's coupled sweeped parameters
                for target_name,target_paths in self.config.coupled_sweep_paths.items():
                    # If any of the coupled sweeps contain the target name
                    if dim_name in target_paths:
                        # Get all sliced dim values
                        sliced_dim_values,found = self.config.path_get(
                            key_path = target_paths[dim_name]+['sweep','range'],
                            settings = self.config.settings
                        )
                        if not found: continue

                        sliced_dim_values = list(map(sigma_to_noise_regime,sliced_dim_values)) \
                        if dim_name == 'sigma' \
                        else list(map(parse, sliced_dim_values))
                        
                        # Get indices of dim values so that 
                        # coupled dim values can be sliced accordingly
                        try:
                            dim_value_indices = [sliced_dim_values.index(val) for val in dim_values]
                        except:
                            raise MissingData(
                                missing_data_name = dim_values,
                                data_names = sliced_dim_values,
                                location = 'Output Coordinate Slice'
                            )
                        for coupled_name,target_path in target_paths.items():
                            # Get all coupled dim values
                            all_coupled_values,found = self.config.path_get(
                                key_path = target_path+['sweep','range'],
                                settings = self.config.settings
                            )
                            if not found: continue

                            # Get coupled dim values using dim value indices
                            coupled_dim_values = [
                                all_coupled_values[ind] \
                                for ind in dim_value_indices
                            ]

                            # If is a coordinate add to the coordinate slice
                            # This slices the xarray created from the outputs samples
                            if self.config.is_sweepable(coupled_name):
                                if coupled_name in coordinate_slice:
                                    coordinate_slice[coupled_name] = np.append(
                                        coordinate_slice[coupled_name],
                                        [parse(str(elem),'None') \
                                         for elem in coupled_dim_values]
                                    )
                                else:
                                    coordinate_slice[coupled_name] = [
                                        parse(str(elem),'None') \
                                        for elem in coupled_dim_values
                                    ]
                            # Flatten list of values
                            coordinate_slice[coupled_name] = list(flatten(coordinate_slice[coupled_name]))
        else:
            coordinate_slice = {}
        
        # Store as global var
        self.coordinate_slice = coordinate_slice
    
    def get_data_collection_group_by(self,sweep_dims:list,sweep_params:dict):
        group_by = []
        combined_dims = []
        # Get all non-core sweep dims
        non_core_sweep_dims = [k for k in sweep_dims if k not in CORE_COORDINATES_DTYPES]
        for gb in list(self.settings.get('group_by',[]))+non_core_sweep_dims:
            # If this is an isolated sweep parameter
            # add it to the group by
            if gb in sweep_params['isolated']:
                group_by.append(gb)
            # If it is a coupled sweep parameter
            # add the coupled parameters too
            if gb in sweep_params['coupled']:
                group_by.append(gb)
                # If this parameter is the target name
                # add its coupled parameters to the group by
                for coupled_param in sweep_params['coupled'].get(gb,[]):
                    # Make sure there are no duplicate group by params
                    if coupled_param['var'] not in group_by:
                        group_by.append(coupled_param['var'])
            # If this parameter is the coupled parameter
            # of a target name add the target name and 
            # the rest of the coupled parameters to the group by
            target_name = self.config.target_names_by_sweep_var.get(gb,'none')
            if target_name != 'none':
                for coupled_param in sweep_params['coupled'].get(target_name,[]):
                    # Add target name
                    if target_name not in group_by:
                        group_by.append(target_name)
                    # Add rest of coupled params
                    if coupled_param['var'] not in group_by:
                        group_by.append(coupled_param['var'])
        
        for dim in sweep_params['isolated'].keys():
            if dim not in group_by:
                combined_dims.append(dim)
        for vals in sweep_params['coupled'].values():
            for coupled_dims in vals :
                if coupled_dims['var'] not in group_by:
                    combined_dims.append(coupled_dims['var'])
        return group_by,combined_dims
    
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

        # Update sample data collection
        for datum in data:
            self.group_sample(
                datum,
                group_by = kwargs.get('group_by',[])
            )
        
        # Combine coords for each list element of the Data Collection
        for sample_name in vars(self).keys():
            if sample_name in DATA_SCHEMA:
                for i,datum in enumerate(getattr(
                    self,
                    sample_name
                )):
                    getattr(
                        self,
                        sample_name
                    )[i] = xr.combine_by_coords(datum,combine_attrs='drop_conflicts')
    
    def group_sample(self, new_data, group_by:list=[]):
        # Get sample name
        sample_name = new_data.attrs['arr_name']

        # Core dimensions for sample must be shared
        sample_shared_dims = ['iter'] if DATA_SCHEMA[sample_name]["is_iterable"] else []
        sample_shared_dims += DATA_SCHEMA[sample_name].get("dims",[])
        # Grouped by sweep params that will be shared
        sample_shared_dims = set(sample_shared_dims).union(set(group_by))
        # All input-related sweep params that will be shared
        sample_shared_dims = set(sample_shared_dims).union(set(list(INPUT_SCHEMA.keys())))

        # Flag for whether update has completed
        complete = False

        if sample_name not in list(self._vars_().keys()):
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

        return self
    
    def group_samples_sequentially(self,output_datasets,group_by:list=[]):
        for datasets in tqdm(
            output_datasets,
            leave = False,
            miniters = 1,
            position = 0,
            desc = 'Binning samples by group id sequentially'
        ):
            self.group_sample(
                new_data = datasets,
                group_by = group_by
            )

    def combine_by_coords_sequentially(self,sample_name:str,combined_dims:list):
        dataset_list = getattr(
            self,
            sample_name
        )
        for i,datasets in tqdm(
            enumerate(dataset_list),
            total = len(dataset_list),
            leave = False,
            position = 0,
            miniters = 1,
            desc = 'Combining DataArray(s) by coordinates sequentially'
        ):
            getattr(
                self,
                sample_name
            )[i] = self.combine_by_coords(
                i = i,
                combined_dims = combined_dims,
                datasets = datasets
            )[1]

    def combine_by_coords_concurrently(self,sample_data,indx:int,combined_dims:list,n_workers:int=1):
        # Every result corresponds to a unique pair of 
        # sweep id and sample name
        # Initialise progress bar
        progress = tqdm(
            total = len(sample_data),
            leave = False,
            position = 0,
            miniters = 1,
            desc = 'Combining DataArray(s) by coordinates concurrently'
        )
        combined_coords = []
        # Create callback function for this index
        # The index is required because the following
        # function is created within a loop
        def make_callback(idx):
            def _callback_(fut):
                progress.update()
                combined_coords.append(fut.result())
            return _callback_
        my_callback = make_callback(indx)

        with BoundedQueueProcessPoolExecutor(max_waiting_tasks = n_workers) as executor:
            # Gather all group and group elements that need to be combined
            for i,datasets in enumerate(sample_data):
                try:
                    future = executor.submit(
                        self.combine_by_coords,
                        i = i,
                        combined_dims = combined_dims,
                        datasets = datasets
                    )
                    future.add_done_callback(my_callback)
                except:
                    print(traceback.format_exc())
                    raise MultiprocessorFailed(
                        'Getting sweep outputs failed.',
                        name = 'combine_by_coords_concurrently'
                    )

        # Delete executor and progress bar
        progress.close()
        safe_delete(progress)
        executor.shutdown(wait=True)
        safe_delete(executor)
        return combined_coords
            

    def combine_by_coords(self,i:int,combined_dims:list,datasets:list):
        # Sort datasets by seed
        datasets = sorted(
            datasets, 
            key = lambda x: tuple([x.coords[var].item() for var in combined_dims if var not in ['N','iter']])
        )
        result = None
        for j,datum in enumerate(datasets):
            if j == 0:
                result = datum
            else:
                result = xr.combine_by_coords(
                    [result,datum], 
                    combine_attrs = 'drop_conflicts'
                )
        return (i,result)

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
            for sample_name, sample_data in self._vars_().items():
                if sample_name in DATA_SCHEMA:
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
                if sample_name in DATA_SCHEMA and sample_name in self._vars_():
                    getattr(
                        self,
                        sample_name
                    )[index] = getattr(
                        new_data,
                        sample_name
                    )
    
    # def __delitem__(self,index):
    #     if index >= len(self):
    #         raise KeyError(f"Index {index} out of bounds for length {len(self)}.")
    #     else:
    #         # Delete index element of Data Collection
    #         del self[index]

    def _vars_(self):
        return {k:v for k,v in vars(self).items() if k in DATA_SCHEMA}
    
    def __repr__(self):
        return "\n\n".join([
            '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'+str(sample_name)+'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' + \
            (' \n'.join([str(dict(elem.sizes)) for elem in sample_data])
            if isinstance(sample_data,list) \
            else str(dict(sample_data.sizes))) + \
            '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
            for sample_name,sample_data in self._vars_().items()
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
                if sample_name in DATA_SCHEMA
            }
        else:
            elem = getattr(
                    self,
                    dim
                )
            return len(elem) if isinstance(elem,list) else 1


    def __len__(self):
        length = 0
        try:
            assert len(set([size for size in self.sizes().values()])) <= 1
            length = len(set([size for size in self.sizes().values()]))
        except:
            raise IrregularDataCollectionSize(f"Irregular DataCollection with sizes {str(self.sizes())}")
        
        if length <= 0:
            return 0
        else:
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
        # Find output folders in collection
        self.output_folders = self.find_matching_output_folders(self)

    @classmethod
    def find_matching_output_folders(cls,__self__):
        if 'directories' in list(__self__.settings.keys()) and len(__self__.settings['directories']) > 0:
            output_dirs = []
            output_directory = __self__.settings['out_directory']
            for dataset_name in __self__.settings['dataset_name']:
                for directory in list(__self__.settings['directories']):
                    path_dir = os.path.join(
                        output_directory,
                        dataset_name,
                        directory
                    )
                    if os.path.exists(path_dir):
                        output_dirs.append(path_dir)
        else:
            # Search metadata based on search parameters
            # Get output directory and group
            output_directory = __self__.settings['out_directory']
            output_group = __self__.settings.get('out_group','')
            # Get experiment title
            experiment_titles = __self__.settings.get('title',[''])
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
                                    (f"{(exp_type+'.*') if len(exp_type) > 0 else ''}") +\
                                    (f"{('_'+exp_title+'.*') if len(exp_title) > 0 else ''}") +\
                                    (f"{(dt+'*') if len(dt) > 0 else ''}")
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
                    )
                ) for dataset in dataset_names
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
            raise MissingFiles('Cannot read outputs.')
        else:
            __self__.logger.info(f"{len(output_dirs)} output folders found.")
        return output_dirs
    
    def collect_experiments_metadata(self):
        
        experiment_metadata = {}
        for indx,output_folder in enumerate(self.output_folders):
            
            self.logger.info(f"Scanning folder {indx+1}/{len(self.output_folders)}")

            # Collect outputs from folder
            outputs = self.get_folder_outputs(indx,output_folder)

            # Loop through each member of the data collection
            if self.settings.get('n_workers',1) > 1:
                metric_data_collection = self.get_experiment_metadata_concurrently(outputs)
            else:
                metric_data_collection = self.get_experiment_metadata_sequentially(outputs)
            
            # Convert generator to list
            metric_data_collection = list(metric_data_collection)

            if len(metric_data_collection) <= 0:
                self.logger.error(f"Empty output dataset in {output_folder}")
                raise MissingMetadata("Failed collecting experiments metadata")
            
            # Convert metric data collection to list
            metric_data_collection = list(metric_data_collection)
            for metric_data in metric_data_collection:
                if len(metric_data) > 0:
                    if output_folder in experiment_metadata:
                        experiment_metadata[output_folder]= np.append(
                            experiment_metadata[output_folder],
                            metric_data,
                            axis = 0
                        )
                    else:
                        experiment_metadata[output_folder] = metric_data
        return experiment_metadata
    
    def get_experiment_metadata_sequentially(self,outputs):
        # Every result corresponds to a unique pair of 
        # sweep id and sample name
        # Loop through each member of the data collection
        for j in tqdm(
            range(len(outputs.data)),
            total=len(outputs.data),
            desc='Collecting experiment metadata sequentially',
            leave=False,
            miniters=1,
            position=0
        ):
            # Collect metric metadata
            yield self.get_experiment_metadata(outputs.get(j))

    def get_experiment_metadata_concurrently(self,outputs):
        # Every result corresponds to a unique pair of 
        # sweep id and sample name
        # Initialise progress bar
        progress = tqdm(
            total=len(outputs.data),
            desc='Collecting experiment metadata concurrently',
            leave=False,
            miniters=1,
            position=0
        )
        results = []
        def my_callback(fut):
            progress.update()
            results.append(fut.result())

        with BoundedQueueProcessPoolExecutor(max_waiting_tasks = self.settings.get('n_workers',1)) as executor:
            # Start the processes and ignore the results
            for j in range(len(outputs.data)):
                try:
                    future = executor.submit(
                        self.get_experiment_metadata,
                        outputs = outputs.get(j),
                    )
                    future.add_done_callback(my_callback)
                except:
                    print(traceback.format_exc())
                    raise MultiprocessorFailed(
                        'Getting sweep outputs failed.',
                        name = 'get_experiment_metadata_concurrently'
                    )

        # Delete executor and progress bar
        progress.close()
        safe_delete(progress)
        executor.shutdown(wait=True)
        safe_delete(executor)
        return results

    def get_experiment_metadata(self,outputs:Outputs):
        
        # Extract useful data from config
        useful_metadata = {}
        for key in self.settings['metadata_keys']:
            # Replace iter with N
            key_paths = outputs.config.path_find(
                key = key if key != 'iter' else 'N',
                settings = outputs.config.settings,
                current_key_path = [],
                all_key_paths = []
            )
            if len(key_paths) <= 0:
                self.logger.error(f"{key if key != 'iter' else 'N'} not found in experiment metadata.")
                continue
            # Extract directly from config
            has_sweep = outputs.config.has_sweep(key_paths[0])
            if len(key_paths) > 0 and not has_sweep:
                useful_metadata[key],_ = outputs.config.path_get(
                    key_path = key_paths[0]
                )
            # NOTE: If metadata key is sweeped
            # it will be read directly from 
            # the sweep dimensions of the xarray data

        # Apply these metrics to the data 
        metric_data = self.apply_metrics(
            experiment_id = outputs.experiment_id,
            outputs = outputs
        )

        # Add useful metadata to metric data
        for m in range(len(metric_data)):
            metric_data[m]['folder'] = os.path.join(self.base_dir)
            for k,v in useful_metadata.items():
                metric_data[m][k] = v

        # Return metric data
        return metric_data

    def get_sweep_outputs_sequentially(
        self,
        sweep_configurations:list,
        outputs: Outputs,
        inputs: Inputs,
        group_by: list = [],
        **kwargs
    ):
        output_datasets = []
        for sweep_configuration in tqdm(
            sweep_configurations[::kwargs.get('step',1)][:kwargs.get('stop',None)],
            desc='Collecting h5 data sequentially',
            leave=False,
            position=0
        ):
            # Get metric data for sweep dataset
            output_datasets.append(
                self.get_sweep_outputs(
                    base_config = outputs.config,
                    sweep_configuration = sweep_configuration,
                    experiment_id = outputs.experiment_id,
                    inputs = inputs,
                    group_by = group_by
                )
            )
        return output_datasets
    
    def get_sweep_outputs_concurrently(
            self,
            sweep_configurations:list,
            outputs: Outputs,
            inputs: Inputs,
            group_by: list = [],
            **kwargs
        ):
        # Gather h5 data from multiple files
        # and store them in xarray-type dictionaries
        output_datasets = []
    
        # Initialise progress bar
        progress = tqdm(
            total=len(sweep_configurations[::kwargs.get('step',1)][:kwargs.get('stop',None)]),
            desc='Collecting h5 data in parallel',
            leave=False,
            miniters=1,
            position=0
        )
        def my_callback(fut):
            progress.update()
            try:
                res = fut.result()
                if len(res) > 0:
                    output_datasets.append(res)
            except (MissingFiles,CorruptedFileRead):
                pass
            except Exception as exc:
                raise exc

        with BoundedQueueProcessPoolExecutor(max_waiting_tasks = self.settings.get('n_workers',1)) as executor:
            # Start the processes and ignore the results
            for sweep_configuration in sweep_configurations[::kwargs.get('step',1)][:kwargs.get('stop',None)]:
                try:
                    future = executor.submit(
                        self.get_sweep_outputs,
                        base_config = outputs.config,
                        sweep_configuration = sweep_configuration,
                        experiment_id = outputs.experiment_id,
                        inputs = inputs,
                        group_by = group_by
                    )
                    future.add_done_callback(my_callback)
                except:
                    print(traceback.format_exc())
                    raise MultiprocessorFailed(
                        'Getting sweep outputs failed.',
                        name = 'get_sweep_outputs_concurrently'
                    )

            # Delete executor and progress bar
            progress.close()
            safe_delete(progress)
            executor.shutdown(wait=True)
            safe_delete(executor)

        return output_datasets

    def get_sweep_outputs(
        self,
        base_config:Config,
        sweep_configuration,
        experiment_id:str,
        inputs:Inputs=None,
        group_by:list=[]
    ):
        # Get specific sweep config 
        new_config,sweep = base_config.prepare_experiment_config(
            sweep_params = self.sweep_params,
            sweep_configuration = sweep_configuration
        )
        # Get outputs and unpack its statistics
        # This is the case where there are SOME input slices provided
        outputs = Outputs(
            config = new_config,
            settings = self.settings,
            data_names = self.settings['sample'],
            base_dir = self.base_dir,
            sweep_params = sweep,
            console_handling_level = self.settings['logging_mode'],
            logger = self.logger
        )
        # Load inputs
        if inputs is None:
            # Import all input data
            outputs.inputs = Inputs(
                config = new_config,
                synthetic_data = False,
                logger = self.logger
            )
            # Cast to xr DataArray
            outputs.inputs.cast_to_xarray()
        else:
            outputs.inputs = inputs

        # Get dictionary output data to be passed into xarray
        xr_dict_data = outputs.load_h5_data()

        data_arr,slice_dict = {},{}
        for sample_name,xr_data in xr_dict_data.items():
            # Get sample xr_data
            data = xr_data.pop('data')
            # Coordinates of output dataset
            coords = xr_data.pop('coordinates')
            # Create slice dictionary
            slice_dict = {
                k: [stringify_coordinate(parse(elem)) for elem in coords[k]]
                for k in coords.keys()
            }
            data_arr[sample_name] = xr.DataArray(
                data = data,
                coords = slice_dict,
                # dims = (sweep_dims+sample_dims),
                attrs = dict(
                    arr_name = sample_name,
                    experiment_id = experiment_id,
                    sweep_id = outputs.sweep_id,
                    **{
                        k:stringify_coordinate(parse(sweep[k])) for k in (list(CORE_COORDINATES_DTYPES.keys())+list(group_by))
                        if k in sweep and k != 'seed' and k not in slice_dict
                    }
                )
            )
        return data_arr

    def get_folder_outputs(self,indx:str,output_folder):
            
        # Read metadata config
        config = Config(
            path = os.path.join(output_folder,"config.json"),
            logger = self.logger
        )
        config.find_sweep_key_paths()

        # Parse sweep configurations
        self.sweep_params = config.parse_sweep_params()

        # Get all sweep configurations
        sweep_configurations, \
        param_sizes_str, \
        total_size_str = config.prepare_sweep_configurations(self.sweep_params)
        # Get output folder
        self.base_dir = output_folder.split(
            'samples/'
        )[0]
        output_folder_succinct = self.base_dir.split(
            config['inputs']['dataset']
        )[-1]
        self.logger.info("----------------------------------------------------------------------------------")
        self.logger.info(f'{output_folder_succinct}')
        self.logger.info(f"Parameter space size: {param_sizes_str}")
        self.logger.info(f"Total = {total_size_str}.")
        self.logger.info("----------------------------------------------------------------------------------")
        
        # If sweep is over input data
        input_sweep_param_names = set(config.sweep_param_names).intersection(
            set(list(INPUT_SCHEMA.keys()))
        )
        input_sweep_param_names = set(input_sweep_param_names).difference(set(['to_learn']))

        # input_sweep_param_names = input_sweep_param_names.union(
        #     set(config.sweep_param_names).intersection(
        #         set(list(PARAMETER_DEFAULTS.keys()))
        #     )
        # )
        if len(input_sweep_param_names) > 0:
            # Load inputs for every single output
            passed_inputs = None
        else:
            # Import all input data
            passed_inputs = deepcopy(Inputs(
                config = config,
                synthetic_data = False,
                logger = self.logger,
            ))

        # Reload global outputs
        outputs = Outputs(
            config = config,
            settings = self.settings,
            data_names = self.settings['sample'],
            base_dir = self.base_dir,
            console_handling_level = self.settings['logging_mode'],
            logger = self.logger
        )
        
        # Attach inputs to outputs object
        outputs.inputs = passed_inputs
        safe_delete(passed_inputs)

        # Gather sweep dimension names
        sweep_dims = list(self.sweep_params['isolated'].keys())
        sweep_dims += list(self.sweep_params['coupled'].keys())

        # Additionally group data collection by these attributes
        group_by,combined_dims = outputs.get_data_collection_group_by(
            sweep_dims = sweep_dims,
            sweep_params = self.sweep_params
        )

        # Attempt to load all samples
        # Keep track of samples not loaded
        samples_not_loaded = outputs.read_data_collection(
            group_by = group_by
        )

        # Load all necessary samples that were not loaded
        if len(samples_not_loaded) > 0:
            self.logger.info(f"Collecting samples {', '.join(samples_not_loaded)}.")

            # Do it concurrently
            if self.settings.get('n_workers',1) > 1:
                output_datasets = self.get_sweep_outputs_concurrently(
                    sweep_configurations = sweep_configurations,
                    outputs = outputs,
                    inputs = passed_inputs,
                    group_by = group_by
                )

            # Do it sequentially
            else:
                output_datasets = self.get_sweep_outputs_sequentially(
                    sweep_configurations = sweep_configurations,
                    outputs = outputs,
                    inputs = passed_inputs,
                    group_by = group_by
                )
            # Create xarray dataset
            try:
                self.logger.note(f"Attempting to create xarray(s) for {', '.join(sorted(samples_not_loaded))}.")
                
                for sample_name in sorted(samples_not_loaded):
                    
                    # Homogeneous data arrays are the ones that have common coordinates
                    # along all core dimensions and group_by dimensions
                    outputs.data.group_samples_sequentially(
                        output_datasets = [ds.pop(sample_name,None) for ds in output_datasets],
                        group_by = group_by
                    )
                    
                    # Combine coords for each list element of the Data Collection
                    parallel = False#self.settings.get('n_workers',1) > 1
                    # if parallel:
                    #     combined_coords = outputs.data.combine_by_coords_concurrently(
                    #         sample_data = getattr(outputs.data,sample_name),
                    #         indx = indx,
                    #         combined_dims = combined_dims,
                    #         n_workers = self.settings.get('n_workers',1)
                    #     )
                    #     # Add results to self
                    #     for cc in combined_coords:
                    #         getattr(
                    #             outputs.data,
                    #             sample_name
                    #         )[cc[0]] = cc[1]
                    # else:
                    outputs.data.combine_by_coords_sequentially(
                        sample_name = sample_name,
                        combined_dims = combined_dims
                    )
                
                # Write sample data collection to file
                outputs.write_data_collection(
                    sample_names = samples_not_loaded
                )
            except Exception as exc:
                self.logger.error(traceback.format_exc())
                sys.exit()
            

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
                self.settings['directories'][0] if len(self.settings['directories']) == 1 else '',
                'summaries'
            )
            makedir(output_directory)

            if len(self.settings.get('directories',[])) > 0:
                filepath = os.path.join(
                    output_directory,
                    f"{'_'+self.settings['filename_ending'] if len(self.settings['filename_ending']) else 'summaries'}" +\
                    '_'.join([
                        f"({var},burnin{setts['burnin']},thinning{setts['thinning']},trimming{setts['trimming']})"
                        for var,setts in self.settings['burnin_thinning_trimming'].items()
                    ]) +\
                    ".csv"
                )
            else:
                # Gather all dates provided
                date_strings = '__'.join(self.settings['dates'])

                filepath = os.path.join(
                    output_directory,
                    f"{'_'.join(list(self.settings['experiment_type']))}"+\
                    f"{'_'+'_'.join(self.settings['title']) if len(self.settings['title']) > 0 else ''}"+\
                    f"{'_multiple_dates' if len(date_strings) > 4 else '_'+date_strings if len(date_strings) > 0 else ''}"+\
                    '_'+'_'.join([
                        f"({var},burnin{setts['burnin']},thinning{setts['thinning']},trimming{setts['trimming']})"
                        for var,setts in self.settings['burnin_thinning_trimming'].items()
                    ])+\
                    f"{'_'+self.settings['filename_ending'] if len(self.settings['filename_ending']) else 'summaries'}.csv"
                )
            # Write experiment summaries to file
            self.logger.info(f"Writing summaries to {filepath}")
            write_csv(experiment_metadata_df,filepath,index=True)
            print('\n')
    
    
    def apply_metrics(self,experiment_id:str,outputs:Outputs):
        # Get outputs and unpack its statistics
        self.logger.progress('Applying metrics...')
        metric_data = []

        for sample_name in self.settings['sample']:
            
            # Get sample
            self.logger.progress(f'Getting sample {sample_name}...')
            try:
                samples = outputs.get_sample(sample_name)
                # Unstack id multi-dimensional index
                samples = samples.unstack('id') if 'id' in samples.dims else samples
            except CoordinateSliceMismatch as exc:
                self.logger.debug(exc)
                continue
            except MissingData as exc:
                self.logger.debug(exc)
                continue
            except Exception as exc:
                print(traceback.format_exc())
                self.logger.error(exc)
                sys.exit()
            self.logger.progress(f"samples {dict(samples.sizes)}, {samples.dtype}")
            
            # Unstack sweep dims
            # samples = samples.unstack(dim='sweep')
            sample_data = {}
            for metric,statistics in self.settings['statistic'].items():
                # Issue warning if three set of statistics are provided
                # even though we need 2; one for the samples and one for the metric
                if len(statistics) > 2:
                    self.logger.warning(f"{len(statistics)} provided. Any more than 2 statistics will be ignored.")
                
                # Unpack sample and metric statistics
                sample_statistics_axes = statistics[0]
                metric_statistics_axes = statistics[1]
                
                # Rename metric if required
                if metric == '' or metric == 'none':
                    metric = deepcopy(sample_name)
                
                self.logger.progress(f"{metric.lower()} {sample_statistics_axes} {metric_statistics_axes}")
                
                # Compute statistic before applying metric
                try:
                    samples_summarised = outputs.apply_sample_statistics(
                        samples = samples,
                        sample_name = sample_name,
                        statistic_dims = sample_statistics_axes
                    )
                    samples_summarised.rename(sample_name)
                except Exception:
                    print(traceback.format_exc())
                    self.logger.debug(traceback.format_exc())
                    self.logger.error(f"samples {dict(samples.sizes)}, {samples.dtype}")
                    self.logger.error(f"Applying statistic(s) \
                    {' over axes '.join([str(s) for s in sample_statistics_axes])} \
                    for sample {sample_name} \
                    for metric {metric.lower()} of experiment {experiment_id} failed")
                    sys.exit()
                    # continue
                self.logger.progress(f"samples_summarised {dict(samples.sizes)}, {samples_summarised.dtype}")

                # Get all attributes and their values
                attribute_keys = METRICS.get(metric.lower(),{}).get('loop_over',[])

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
                    
                    # Get ground truth data to compute metric
                    try:
                        if metric.lower() not in ['none',''] and metric.lower() in METRICS:
                            ground_truth_data = deep_call(
                                outputs,
                                METRICS[metric.lower()]['ground_truth'],
                                None
                            )
                            assert ground_truth_data is not None
                        else:
                            ground_truth_data = None
                    except:
                        raise MissingData(
                            missing_data_name = metric,
                            data_names = ', '.join(list(outputs.inputs.data_vars().keys())),
                            location = 'Inputs'
                        )

                    try:
                        if metric.lower() not in ['none',''] and metric.lower() in METRICS:
                            samples_metric = globals()[metric.lower()](
                                prediction = samples_summarised,
                                ground_truth = ground_truth_data,
                                **attribute_settings
                            )
                            # Rename metric xr data array
                            samples_metric = samples_metric.rename(metric.lower())
                        else:
                            samples_metric = deepcopy(samples_summarised).rename(metric.lower())
                    except Exception:
                        print(traceback.format_exc())
                        self.logger.debug(traceback.format_exc())
                        self.logger.debug(f"samples_summarised {dict(samples_summarised.sizes)}, {samples_summarised.dtype}")
                        self.logger.error(f"ground_truth {dict(ground_truth_data.sizes)}, {ground_truth_data.dtype}")
                        self.logger.error(f'Applying metric {metric.lower()} for {attribute_settings_string} \
                        over sample {sample_name} \
                        for experiment {experiment_id} failed')
                        print('\n')
                        sys.exit()

                    self.logger.progress(f"Samples metric is {dict(samples_metric.sizes)}")
                    # Unstack sweep dimensions
                    samples_metric = samples_metric.unstack(dim='sweep') if 'sweep' in samples_metric.dims else samples_metric
                    # Apply statistics after metric
                    try:
                        if metric.lower() not in ['none',''] and metric.lower() in METRICS:
                            self.settings['axis'] = METRICS[metric.lower()]['apply_axis']
                            metric_summarised = outputs.apply_sample_statistics(
                                samples = samples_metric,
                                sample_name = metric.lower(),
                                statistic_dims = metric_statistics_axes
                            )
                        else:
                            metric_summarised = deepcopy(samples_metric).rename(metric.lower())
                    except Exception:
                        self.logger.debug(traceback.format_exc())
                        self.logger.error(f"samples_metric {dict(samples_metric.sizes)}, {samples_metric.dtype}")
                        self.logger.error(f"Applying statistic(s) {'>'.join([str(m) for m in metric_statistics_axes])} \
                                            over metric {metric.lower()} and sample {sample_name} for experiment {experiment_id} failed")
                        print('\n')
                        continue
                    self.logger.debug(f"Summarised metric is {dict(metric_summarised.sizes)}")
                    
                    # Squeeze output
                    metric_summarised = np.squeeze(metric_summarised)

                    self.logger.progress(f"Summarised metric is squeezed to {dict(metric_summarised.sizes)}")
                    # Get metric data in pandas dataframe
                    try:
                        metric_summarised = metric_summarised.to_dataframe().reset_index(drop=True)
                    except:
                        # Create row out of coordinates
                        row = {k:[np.asarray(v.data)] for k,v in metric_summarised.coords.items()}
                        # add metric
                        row[metric_summarised.name] = [np.array(metric_summarised.to_pandas())]
                        # Convert to dataframe
                        metric_summarised = pd.DataFrame(row)

                    # This loops over remaining sweep configurations
                    for _,sweep in metric_summarised.iterrows():
                        # remove sweep key
                        sweep = sweep.to_dict()
                        sweep = {k:get_value(sweep,k) for k in sweep.keys()}
                        # Get sweep id
                        sweep_id = ' & '.join([str(k)+'_'+str(v) for k,v in sweep.items() if k not in [metric,'sweep']])
                        # Gather all key-value pairs from every row (corresponding to a single sweep setting)
                        # this is corresponds to variable 'row'
                        # Add every sweep configuration to this metric data
                        if sweep_id not in sample_data:
                            metric_value = sweep.pop(metric,None)
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
                                    ),
                                    metric:parse(metric_value)
                                },
                                **sweep,
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
                                metric:parse(sweep[metric]),
                                **attribute_settings
                            })
                
            # Add sample data to metric data
            for sweep_metric_list in sample_data.values():
                metric_data += list(sweep_metric_list)
            safe_delete(sample_data)
        
        return metric_data