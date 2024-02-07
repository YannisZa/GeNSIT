import os
import re
import gc
import sys
import time
import torch
import shutil
import logging
import traceback
import h5py as h5
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import geopandas as gpd

from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from itertools import product
from typing import Union,List,Tuple
from itertools import product,chain

from multiresticodm.config import Config
from multiresticodm.inputs import Inputs
from multiresticodm.utils.misc_utils import *
from multiresticodm.utils.exceptions import *
from multiresticodm.static.global_variables import *
from multiresticodm.spatial_interaction_model import *
from multiresticodm.utils import misc_utils as MiscUtils
from multiresticodm.utils import math_utils as MathUtils
from multiresticodm.contingency_table import instantiate_ct
from multiresticodm.harris_wilson_model import HarrisWilson
from multiresticodm.utils import probability_utils as ProbabilityUtils
from multiresticodm.utils.multiprocessor import BoundedQueueProcessPoolExecutor
from multiresticodm.harris_wilson_model_mcmc import instantiate_harris_wilson_mcmc
from multiresticodm.harris_wilson_model_neural_net import NeuralNet, HarrisWilson_NN
from multiresticodm.contingency_table_mcmc import ContingencyTableMarkovChainMonteCarlo

OUTPUTS_MODULE = sys.modules[__name__]

class Outputs(object):
    def __init__(self,
                 config:Config, 
                 settings:dict={},
                 data_names:list = None,
                 sweep:dict={},
                 inputs:Inputs = None,
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
        self.data_names = settings.get('sample',[]) \
            if (data_names is None) or (len(data_names) <= 0) \
            else list(data_names)
        
        # Store settings
        self.settings = settings

        # Store device
        self.device = self.settings.get('device','cpu')

        # Store inputs
        self.inputs = inputs

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
            data = [],
            logger = self.logger
        )
        # Enable garbage collector
        gc.enable()

        self.sweep_id = ''

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
            
            # Remove potentially added filenames
            config = config.replace('metadata.json','')
            config = config.replace('config.json','')
            # Store experiment id and update sweep id if possible
            if 'samples' in config:
                self.experiment_id = os.path.basename(os.path.normpath(config.split('samples/')[0]))
                # Attempt to find sweep id inside path
                path_suffix = config.split('samples/')[-1]
                # If match is found, extract the string before the suffix
                if path_suffix:
                    self.sweep_id = path_suffix
            else:
                self.experiment_id = os.path.basename(os.path.normpath(config.split('/',-1)[0]))
            
            
            # Load metadata
            config_filepath = ''
            if os.path.exists(os.path.join(config,"config.json")):
                config_filepath = os.path.join(config,"config.json")
            elif os.path.exists(os.path.join(config,"metadata.json")):
                config_filepath = os.path.join(config,"metadata.json")

            try:
                assert config_filepath != ''
            except:
                raise MissingFiles(
                    message = f'Missing config.json or metadata.json in {config}'
                )

            # Load config
            self.config = Config(
                path = config_filepath,
                logger = self.logger
            )

            # Get sweep-related data
            self.config.get_sweep_data()

            # Get intensity model class
            self.intensity_model_class = [k for k in self.config.keys() if k in INTENSITY_MODELS and isinstance(self.config[k],dict)][0]
            
            # Define config experiment path to directory
            self.outputs_path = config.split('samples/')[0] if kwargs.get('base_dir') is None else kwargs['base_dir']
        
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
        if self.sweep_id == '':
            self.sweep_id = self.config.get_sweep_id(sweep = sweep)

        if kwargs.get('slice',True):
        
            # Create coordinate slice conditions
            self.create_slicing_conditions()
            if self.coordinate_slice or self.settings.get('burnin_thinning_trimming',[]):
                self.logger.info("//////////////////////////////////////////////////////////////////////////////////")
                self.logger.info("Slicing coordinates:")
                for coord_slice_expression in self.coordinate_slice:
                    self.logger.info(f"{coord_slice_expression.replace('da.','')}")
                for coord_slice in self.settings.get('burnin_thinning_trimming',[]):
                    for dimkey,dimval in coord_slice.items():
                        self.logger.info(f"{dimkey}: {', '.join([str(key)+' = '+str(val) for key,val in dimval.items()])}")
                self.logger.info("//////////////////////////////////////////////////////////////////////////////////")

    def get(self,index:int):
        self_copy = deepcopy(self)
        self_copy.data = self_copy.data[index]
        # Update config
        first_dataset = list(self_copy.data_vars().values())[0]
        # Find sweep dimensions that are not core coordinates
        sweep = dict(zip(
            first_dataset.get_index('sweep').names,
            [unstringify(d) for d in first_dataset.coords['sweep'].values.tolist()[0]]
        ))
        # NOTE: We are not using config's native 'prepare_experiment_config' function
        # because some of the sweep dimensions might have be grouped when loading the outputs
        # e.g. seed is often grouped and does not appear in the sweep coordinates of the output data array
        # Reset config-global quantities
        self_copy.config.reset()
        # Update config
        self_copy.config.update(sweep)
        # Update sweep mode flag
        self_copy.config.find_sweep_key_paths()
        try:
            # Either there are no sweep params
            # or if there are then these are must be group_by dims
            assert not self_copy.config.sweep_mode() or \
                not (set(self_copy.config.sweep_param_names).difference(self.settings.get('group_by',[])))
        except:
            raise InvalidMetadataType(
                message = f"""
                    No sweeps should be contained in Outputs' config. 
                    {self_copy.config.sweep_param_names} found with {self.settings.get('group_by',[])} group by params specified.
                """
            )

        return self_copy
    

    def strip_data(self,keep_inputs:list=[],keep_outputs:list=[],keep_collection_ids:list=[]):
        # Remove all but keep_inputs from input data
        if len(keep_inputs) <= 0:
            safe_delete(self.inputs)
        else:
            if self.inputs is not None:
                removed_inputs = set(list(self.inputs.data_vars().keys())).difference(set(keep_inputs))
                for removed_inpt in removed_inputs:
                    delattr(self.inputs.data,removed_inpt)
        
        # Remove all but keep_outputs from output data
        if len(keep_outputs) <= 0 and len(keep_collection_ids) <= 0:
            safe_delete(self.data)
            self.data = DataCollection(
                data = [],
                logger = self.logger
            )
        
        elif len(keep_outputs) <= 0 and len(keep_collection_ids) > 0:
            for sample_name in self.data_vars().keys():
                removed_collection_ids = set(range(len(self.data_vars()[sample_name]))).difference(set(keep_collection_ids))
                for i in sorted(removed_collection_ids,reverse = True):
                    del getattr(self.data,sample_name)[i]
        
        elif len(keep_outputs) > 0 and len(keep_collection_ids) <= 0:
            removed_outputs = set(list(self.data_vars().keys())).difference(set(keep_outputs))
            for removed_outpt in removed_outputs:
                delattr(self.data,removed_outpt)
        
        elif len(keep_outputs) > 0 and len(keep_collection_ids) > 0:
            removed_outputs = set(list(self.data_vars().keys())).difference(set(keep_outputs))
            for removed_outpt in removed_outputs:
                removed_collection_ids = set(range(len(self.data_vars()[removed_outpt]))).difference(set(keep_collection_ids))
                for i in sorted(removed_collection_ids,reverse = True):
                    del getattr(self.data,removed_outpt)[i]
        
        time.sleep(3)
        gc.collect()
        time.sleep(3)
    
    def group_by(self,dim:str):
        # Get all data vars by each group
        data_by_group = {}
        for sample_name,sample_data in self.data_vars().items(): 
            # Stack sweep and iteration dims
            self.stack_sweep_and_iter_dims(self)
            try:
                assert dim in sample_data.dims
            except:
                raise InvalidDataNames(
                    f"Grouping {sample_name} by {dim} which is not included in {sample_data.dims}"
                )
            for group_id,group_data in sample_data.groupby(dim):
                print(str(group_id))
                # data_by_group[str(group_id)].setdefault(sample_name, group_data)
                if str(group_id) in data_by_group:
                    data_by_group[str(group_id)][sample_name] = group_data
                else:
                    data_by_group[str(group_id)] = {sample_name: group_data}
            print('\n')
        return data_by_group

    def slice_coordinates(self):
        # Slice according to coordinate value slice
        if self.coordinate_slice or self.settings.get('burnin_thinning_trimming',[]):
            progress = tqdm(
                total = self.data.size(),
                leave = False,
                position = 0,
                miniters = 1,
                desc = 'Slicing coordinates sequentially'
            )
            # Based on first sample name slice the rest of sample names
            sample_name = list(self.data_vars().keys())[0]
            samples = self.data_vars()[sample_name]
            # Keep track of removed collection ids
            removed_collection_ids = set()
            for i in range(len(samples)):
                # Apply coordinate value slice
                try:
                    samples[i],successful_value_slices = self.slice_coordinates_by_value(
                        da = samples[i],
                        sample_name = sample_name,
                        i = i
                    )
                except Exception as exc:
                    # traceback.print_exc()
                    # If coordinate slice failed remove group from data collection
                    removed_collection_ids.add(i)
                    self.logger.debug(exc)
                    # Update progress
                    progress.update(1)
                    continue

                # Apply burning, thinning and trimming
                try:
                    samples[i],successful_index_slices = self.slice_coordinates_by_index(
                        samples = samples[i],
                        sample_name = sample_name
                    )
                    self.logger.progress(f"After index slicing {sample_name}[{i}]: {({k:v for k,v in dict(samples[i].sizes).items() if v > 1})}")
                except Exception as exc:
                    # If index slice failed do NOT remove group from data collection
                    # Instead just keep the data as it was before index slicing
                    traceback.print_exc()
                    raise exc
                
                # Make sure you keep the samples for this collection id
                getattr(self.data,sample_name)[i] = samples[i]
                # Slice the rest of sample data
                for sam_name, current_samples in self.data_vars().items():
                    # Do not reslice the data you just sliced!
                    if sam_name != sample_name:
                        # Slice sam_name's data
                        current_samples[i],_ = self.slice_coordinates_by_value(
                            da = current_samples[i],
                            sample_name = sam_name,
                            i = i
                        )
                        current_samples[i],_ = self.slice_coordinates_by_index(
                            samples = current_samples[i],
                            sample_name = sam_name
                        )
                        # Update data with sliced data
                        getattr(self.data,sam_name)[i] = current_samples[i]
                
                # Update progress
                progress.update(1)
                
            # Remove collection ids that are not matching coordinate slice
            kept_collection_ids = set(list(range(self.data.size()))).difference(removed_collection_ids)
            self.logger.info(f"""
                {len(kept_collection_ids)} collection ids kept out of {self.data.size()}.
                Kept ids: {list(sorted(kept_collection_ids))}
            """)
            for cid in sorted(list(removed_collection_ids), reverse = True):
                for sam_name in self.data_vars().keys():
                    del getattr(self.data,sam_name)[cid]
                    gc.collect()
            
            # Print successful slices
            for cslice in successful_value_slices:
                self.logger.success(f"Slicing using coordinate slice {cslice} succeded")
            for dim_names,islice in successful_index_slices.items():
                # Announce successful coordinate index slices
                self.logger.success(f"Slicing {dim_names} {islice['slice_settings']} succeded {islice['new_shape']}")
            # Sleep for 3 secs so that gc cleans memory
            time.sleep(3)

            progress.close()

    def data_vars(self):
        return {k:v for k,v in self.data._vars_().items() if k in DATA_SCHEMA}            

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

    def slice_coordinates_by_value(self,da,sample_name:str,i:int):
        # Get latest sample collection element
        # NOTE: you have to name this dataset 'da'
        # so that slice expressions can be evaluated in the next step
        self.logger.progress(f"Before coordinate slicing {sample_name}[{i}]: {({k:v for k,v in dict(da.sizes).items() if v > 1})}")
        # Monitor successful coordinate slices
        successful_slices = set()
        # print(sample_name,i,'/',len(samples))
        for coord_slice in self.coordinate_slice:
            try:
                # Slice based on these conditions
                # Reassign da to sliced data
                da = da.where( 
                    eval(
                        coord_slice,
                        {"da":da}
                    ),
                    drop = True
                )
            except Exception as exc:
                self.logger.debug(f"Slicing using {sample_name}[{i}] {coord_slice} failed with {exc}")
                continue
            # Make sure dataset is not empty
            if da.size <= 0:
                raise EmptyData(
                    message = f"Slicing using {sample_name}[{i}] {coord_slice} yielded zero size dataset. Removing collection id {i}.",
                    data_names = sample_name
                )
            # Keep track of slices that succeded
            if str(coord_slice) not in successful_slices:
                successful_slices.add(str(coord_slice))

        self.logger.progress(f"After coordinate slicing {sample_name}[{i}]: {({k:v for k,v in dict(da.sizes).items() if v > 1})}")
        
        return da,successful_slices
    
    def slice_coordinates_by_index(self,samples,sample_name:str):

        # Keep track of previous number of iterations
        prev_iter = deepcopy({
            k:samples.sizes[k] \
            for index_slice in self.settings['burnin_thinning_trimming'] \
            for ktuple in index_slice.keys()
            for k in ktuple.split('+')
            if k in samples.dims
        })
        
        # Keep track of sliced dimensions/variables
        sliced_dims = {}
        for index_slice in self.settings['burnin_thinning_trimming']:
            
            # Extract variable name(s)
            dim_names = list(index_slice.keys())[0]
            # Extract index slice settings
            slice_setts = list(index_slice.values())[0]
            
            # Gather dim names
            dim_names = dim_names.split('+')
            # Get the intersection between available dims and specified dims
            dim_names = list(set(samples.dims).intersection(dim_names))

            # if samples do not have these dimensions, carry on
            if len(dim_names) <= 0:
                continue

            # Stack all dims together
            samples = samples.stack(temp_dim = dim_names)

            # Get total number of iterations
            total_samples = samples.sizes['temp_dim']

            # Get burnin parameter
            burnin = slice_setts.get('burnin',0)

            # Get thinning parameter
            thinning = slice_setts.get('thinning',1)

            # Get iterations
            iters = np.arange(start = burnin,stop = total_samples,step = thinning,dtype='int32')
            
            # Get number of samples to keep
            trimming = slice_setts.get('trimming',None)

            # Trim iterations
            iters = iters[:trimming]
            
            # Make sure you do not slice the same variable twice!
            try:
                assert all([dim not in sliced_dims for dim in sliced_dims])
            except:
                self.logger.debug(f"Slicing {dim_names} has already been applied.")
                # Unstack temp dim
                samples = samples.unstack('temp_dim')
                continue

            # Apply burnin, thinning and trimming to samples
            try:
                # Slice based on index
                sliced_samples = samples.isel(temp_dim = iters)
            except:
                self.logger.debug(f"Slicing {dim_names} {slice_setts} failed")
                # Unstack temp dim
                samples = samples.unstack('temp_dim')
                continue
            
            # Unstack temp dim
            sliced_samples = sliced_samples.unstack('temp_dim')

            # If no samples remain after slicing - ignore last applied slice
            if sliced_samples.size <= 0:
                self.logger.debug(f"Slicing {dim_names} {slice_setts} yield zero size data")
                # Unstack temp dim
                samples = samples.unstack('temp_dim')
                continue
            else:
                # Success - samples were sliced
                for d in dim_names: 
                    if d not in sliced_dims:
                        sliced_dims[d] = {"slice_settings":slice_setts,"new_shape":({k:v for k,v in dict(samples.sizes).items() if v > 1})}
                # Update current samples to be the sliced samples
                samples = sliced_samples

        # If no data remains after slicing raise exception
        if any([samples.sizes[k] <= 0 for k in samples.dims]):
            raise EmptyData(
                data_names = sample_name,
                message = f"Slicing {list(prev_iter.keys())} with shape {prev_iter} using {self.settings['burnin_thinning_trimming']}"
            )
        
        return samples,sliced_dims
    
    
    def load_geometry(self,geometry_filename,default_crs:str='epsg:27700'):
        # Load geometry from file
        geometry = gpd.read_file(geometry_filename)
        geometry = geometry.set_crs(default_crs,allow_override = True)
        
        return geometry


    def load_h5_data(self,sample_names:list):
        self.logger.note('Loading h5 data into xarrays...')
        
        # Get h5 file
        h5file = os.path.join(self.outputs_path,'samples',f"{self.sweep_id}","data.h5")

        try:
            assert os.path.exists(h5file) and os.path.isfile(h5file)
        except:
            raise MissingFiles(f"H5 file {h5file} not found.")
        
        # Read h5 data
        local_coords,global_coords,data_vars = self.read_h5_file(
            filename = h5file,
            sample_names = sample_names
        )

        # Convert set to list
        local_coords = {k:np.array(
                            list(v),
                            dtype = TORCH_TO_NUMPY_DTYPE[COORDINATES_DTYPES[k]]
                        ) for k,v in local_coords.items()}
        global_coords = {k:np.array(
                            list(v),
                            dtype = TORCH_TO_NUMPY_DTYPE[COORDINATES_DTYPES[k]]
                        ) for k,v in global_coords.items()}
        self.logger.progress('Populating data dictionary')
        # Create an xarray dataset for each sample
        xr_dict = {}
        for sample_name,sample_data in data_vars.items():
            
            # Keep only necessary global coordinates
            sample_global_dims = ['iter'] if DATA_SCHEMA[sample_name]['is_iterable'] else []
            sample_global_dims += DATA_SCHEMA[sample_name].get("dims",[])

            # Update coordinates to include schema and sweep coordinates
            # Keep only coordinates that are 1) core
            # 2) isolated sweeps 
            # or 3) the targets of coupled sweeps
            coordinates = {
                **{k:v for k,v in local_coords.items() if k != sample_name},
                **{k:global_coords[k] for k in sample_global_dims},
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
    
    def read_h5_file(self,filename:str,sample_names:list,**kwargs):
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
                    if sample_name not in sample_names:
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
                                    start = 1,
                                    stop = sample_data.shape[i+start_idx]+1,
                                    step = 1,
                                    dtype='int32'
                                )
                            })
                        else:
                            global_coords[dim] = np.arange(
                                start = 1,
                                stop = sample_data.shape[i+start_idx]+1,
                                step = 1,
                                dtype='int32'
                            )
                    # Append
                    self.logger.debug(f'Appending {sample_name}')
                    data_vars[sample_name] = np.array(
                        sample_data[:],
                        dtype = DATA_SCHEMA[sample_name].get('dtype','float32')
                    )

        except BlockingIOError:
            self.logger.debug(f"Skipping in-use file: {filename}")
            return {},{},{}
        except Exception:
            self.logger.debug(traceback.format_exc())
            raise CorruptedFileRead(f'Cannot read file {filename}')
        return local_coords,global_coords,data_vars

    def load(self,indx:int = 0):
        
        # Additionally group data collection by these attributes
        group_by,combined_dims = self.config.get_group_id(
            group_by = self.settings.get('group_by',[])
        )

        if len(self.config.sweep_configurations) > 0:
            # Attempt to load all samples
            # Keep track of samples not loaded
            samples_not_loaded = self.read_data_collection(
                group_by = group_by
            )

            # Load all necessary samples that were not loaded
            if len(samples_not_loaded) > 0:
                self.logger.info(f"Collecting samples {', '.join(sorted(samples_not_loaded))}.")

                # Do it concurrently
                if self.settings.get('n_workers',1) > 1:
                    output_datasets = self.get_sweep_outputs_concurrently(
                        sample_names = samples_not_loaded,
                        group_by = group_by
                    )

                # Do it sequentially
                else:
                    output_datasets = self.get_sweep_outputs_sequentially(
                        sample_names = samples_not_loaded,
                        group_by = group_by
                    )
                
                # Create xarray dataset
                try:
                    self.logger.info(f"Creating xarray(s) for {', '.join(sorted(samples_not_loaded))}.")
                    
                    for sample_name in sorted(samples_not_loaded):
                        
                        # Homogeneous data arrays are the ones that have common coordinates
                        # along all core dimensions and group_by dimensions
                        self.data.group_samples_sequentially(
                            output_datasets = [x for x in [ds.pop(sample_name,None) for ds in output_datasets] if x is not None],
                            sample_name = sample_name,
                            group_by = group_by
                        )

                        # Combine coords for each list element of the Data Collection
                        parallel = False #self.settings.get('n_workers',1) > 1
                        if parallel:
                            combined_coords = self.data.combine_by_coords_concurrently(
                                indx = indx,
                                sample_name = sample_name,
                                combined_dims = combined_dims
                            )
                            # Add results to self
                            for cc in combined_coords:
                                getattr(
                                    self.data,
                                    sample_name
                                )[cc[0]] = cc[1]
                        else:
                            self.data.combine_by_coords_sequentially(
                                sample_name = sample_name,
                                combined_dims = combined_dims
                            )
                        
                        # Write sample data collection to file
                        dirpath = self.write_data_collection(
                            sample_names = [sample_name]
                        )
                    
                    self.logger.info(f'Wrote output collection to {dirpath}')
                except Exception as exc:
                    self.logger.error(traceback.format_exc())
                    raise exc

        else:
            # Load data array for this given sweep
            data_array = self.load_single(
                sample_names = self.output_names,
                group_by = group_by,
                sweep = None,
            )
            for sample_name,sample_data in data_array.items():
                setattr(
                    self.data,
                    sample_name,
                    [sample_data]
                )
        
        # Slice according to coordinate and index slice
        self.slice_coordinates()

        # If output dataset is empty raise Error
        if self.data.size() <= 0:
            raise EmptyData(
                message = 'Outputs data is empty after slicing by coordinates and/or indices',
                data_names = 'all'
            )
        
        # Stack sweep and iter dimensions
        self.stack_sweep_and_iter_dims(self)


    def load_single(self,sample_names:list = None, group_by:list = None, sweep:dict = None):
        # Load inputs
        if self.inputs is None:
            # Import all input data
            self.inputs = Inputs(
                config = self.config,
                synthetic_data = False,
                logger = self.logger
            )
        
        # Cast to xr DataArray
        self.inputs.cast_to_xarray()

        # Get dictionary output data to be passed into xarray
        xr_dict_data = self.load_h5_data(sample_names)

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
            # Decide on extra set of attributes
            if not sweep:
                attrs = {}
            else:
                attrs = {
                    k:stringify_coordinate(parse(sweep[k])) for k in (list(CORE_COORDINATES_DTYPES.keys())+list(group_by))
                    if k in sweep and k != 'seed' and k not in slice_dict
                }
            data_arr[sample_name] = xr.DataArray(
                data = data,
                coords = slice_dict,
                attrs = dict(
                    arr_name = sample_name,
                    experiment_id = self.experiment_id,
                    sweep_id = self.sweep_id,
                    **attrs
                )
            ).astype(DATA_SCHEMA[sample_name]["dtype"])
        return data_arr

        
    def update_experiment_directory_id(self,sweep_experiment_id:str = None):

        noise_level = list(deep_get(key='noise_regime',value = self.config.settings))
        if len(noise_level) <= 0:
            if 'sigma' in self.config.settings['inputs']['to_learn']:
                noise_level = 'learned'
            else:
                sigma = list(deep_get(key='sigma',value = self.config.settings))
                if len(sigma) == 1:
                    if isinstance(sigma[0],dict) and 'sweep' in list(sigma[0].keys()):
                        noise_level = 'sweeped'
                    else:
                        noise_level = sigma_to_noise_regime(sigma = sigma[0])
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
        export_samples = list(deep_get(key='export_samples',value = self.config.settings))
        export_metadata = list(deep_get(key='export_metadata',value = self.config.settings))
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
                write_json(self.config.settings,filepath,indent = 2)
            elif isinstance(self.config,dict):
                write_json(self.config,filepath,indent = 2)
            else:
                raise InvalidMetadataType(f'Cannot write metadata of invalid type {type(self.config)}')

    def print_metadata(self) -> None:
        print_json(self.config,indent = 2)

    def open_output_file(self,sweep:dict={}):
        # Create output directories if necessary
        self.create_output_subdirectories(sweep_id = self.sweep_id)
        if hasattr(self,'config') and hasattr(self.config,'settings'):
            export_samples = list(deep_get(key='export_samples',value = self.config.settings))
            export_metadata = list(deep_get(key='export_metadata',value = self.config.settings))
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
                self.h5group.attrs.create("sweep_params",list(sweep.keys()))
                self.h5group.attrs.create("sweep_values",['none' if val is None else str(val) for val in sweep.values()])
                
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
        print(output_directory)
        
        # Get specific sample names
        sample_names = sample_names if sample_names is not None else list(self.data_vars().keys())
        sample_names = set(sample_names).intersection(set(list(self.data_vars().keys())))

        # Create sample_name - collection_id pairs to export in parallel
        group_ids = []
        for sam_name in sample_names:
            for collection_id in range(len(getattr(self.data,sam_name))):
                group_ids.append([sam_name,collection_id])

        # Write data arrays each one of which 
        # corresponds to a different group
        # if self.settings.get('n_workers',1) > 1:
        #     self.write_xr_data_concurrently(
        #         group_ids = group_ids,
        #         sample_names = sample_names,
        #         dirpath = output_directory
        #     )
        # else:
        self.write_xr_data_sequentially(
            group_ids = group_ids,
            sample_names = sample_names,
            dirpath = output_directory
        )
        return output_directory
    
    def write_xr_data_sequentially(self,group_ids,dirpath:str,sample_names:list):
        for grid in tqdm(
            group_ids, 
            leave = False,
            miniters = 1,
            position = 0,
            desc = f"Writing {','.join(sample_names)} group data sequentially"
        ):
            sam_name, collection_id = grid[0], grid[1]
            write_xr_data(
                getattr(self.data,sam_name)[collection_id],
                dirpath,
                group = grid
            )
    
    def write_xr_data_concurrently(self,group_ids,dirpath:str,sample_names:list):
        # Initialise progress bar
        progress = tqdm(
            total = len(group_ids),
            desc = f"Writing {','.join(sample_names)} group data",
            leave = False,
            miniters = 1,
            position = 0
        )
        def my_callback(fut):
            progress.update()
            try:
                fut.result()
            except Exception as exc:
                raise ValueError("write_xr_data_concurrently failed") from exc

        with BoundedQueueProcessPoolExecutor(max_waiting_tasks = 2*self.settings.get('n_workers',1)) as executor:
            # Start the processes and ignore the results
            for grid in group_ids:
                try:
                    sam_name, collection_id = grid[0], grid[1]
                    future = executor.submit(
                        write_xr_data,
                        getattr(self.data,sam_name)[collection_id],
                        dirpath,
                        group = grid
                    )
                    future.add_done_callback(my_callback)
                except:
                    traceback.print_exc()
                    raise MultiprocessorFailed(
                        keys = 'write_xr_data_concurrently',
                        message = f"Writing {','.join(sample_names)} group data failed"
                    )

            # Delete executor and progress bar
            progress.close()
            safe_delete(progress)
            executor.shutdown(wait = True)
            safe_delete(executor)


    def read_data_collection(self, group_by:list):
    
        # Outputs filepath
        dirpath = os.path.join(self.outputs_path,'sample_collections')

        if not os.path.isdir(dirpath) or \
            not os.path.exists(dirpath) or \
            self.settings.get('force_reload',False):
            self.logger.warning(f'Removing {dirpath}')

            # Remove existing file
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)

            return self.output_names
        else:
            # Start with the premise that all available samples should be loaded
            samples_not_loaded = deepcopy(sorted(self.output_names))
            # Get all sample names and collection ids (all of that constitutes the group ids)
            samples_to_load_group_ids = {}
            for group_id in list(os.walk(dirpath))[0][-1]:
                samples_to_load_group_ids.setdefault(
                    group_id.split('>')[0],
                    [x.replace('.nc','') for x in group_id.split('>')[1:]]
                ).extend([x.replace('.nc','') for x in group_id.split('>')[1:]])
            # Find unique group ids by sample name and sort them
            samples_to_load_group_ids = {k:sorted(list(set(v)),key = lambda x: eval(x)) for k,v in samples_to_load_group_ids.items()}
            
            # Raise error if no data collection elements found
            if len(samples_to_load_group_ids) <= 0:
                return self.output_names
            
            # Update samples not loaded
            samples_to_load = sorted([s for s in samples_not_loaded if s in samples_to_load_group_ids])
            samples_not_loaded = sorted([s for s in samples_not_loaded if s not in samples_to_load_group_ids])
            
            # If no more samples need to be loaded 
            # check each sample's data collection elements
            if len(samples_not_loaded) == 0:
                # If this throws an exception it means that some 
                # elements corresponding to some sample names 
                # are missing from the data collection
                sample_ids = {
                    s: ','.join(sorted(ids)) \
                    for s,ids in samples_to_load_group_ids.items()
                }
                try:
                    assert len(set([sid for sid in sample_ids.values()])) == 1
                except Exception:
                    self.logger.warning(f"Some elements might be missing from data collection")
                    for key,val in sample_ids.items():
                        self.logger.debug(f"{key}: {val}")
                    # Force reload all data
                    # return self.output_names
            
            # Create list of all group identifiers
            all_groups = [
                (sam_name,gid) \
                for sam_name in sorted(samples_to_load) \
                for gid in samples_to_load_group_ids[sam_name]
            ]

            self.logger.info(f"Reading samples {', '.join(sorted(samples_to_load))}.")
            
            data_arrs = []
            # Gather all group and group elements that need to be combined
            if False:#self.settings.get('n_workers',1) > 1:
                data_arrs,sample_names_loaded = self.read_xr_data_concurrently(
                    all_groups = all_groups,
                    samples_to_load = sorted(samples_to_load),
                    dirpath = dirpath
                )
            else:
                data_arrs,sample_names_loaded = self.read_xr_data_sequentially(
                    all_groups = all_groups,
                    samples_to_load = sorted(samples_to_load),
                    dirpath = dirpath
                )
            
            # remove loaded samples from consideration
            # since they has been succesfully loaded
            for sample_name in sample_names_loaded:
                if sample_name in samples_not_loaded:
                    samples_not_loaded.remove(sample_name)
                    
            self.logger.info(f"Creating Data Collection for each group.")
            # Pass all samples into a data collection object
            self.data = DataCollection(
                data = data_arrs,
                group_by = group_by,
                logger = self.logger
            )
            return samples_not_loaded
        
    def read_xr_data_sequentially(self,all_groups,samples_to_load:list,dirpath:str):
        data_arrs = []
        sample_names_loaded = set([])
        for group in tqdm(
            all_groups,
            leave = False,
            miniters = 1,
            position = 0,
            desc = f"Reading group data"
        ):
            try:
                sample_dict = read_xr_data(
                    dirpath = dirpath,
                    sample_gid = group
                )
                # Extract sample name and data
                sample_name = list(sample_dict.keys())[0]
                sample_data = list(sample_dict.values())[0]
                # append array to data arrays
                if sample_data is not None:
                    data_arrs.append(sample_data.astype(DATA_SCHEMA[sample_name]["dtype"]))
                # add sample name to set of sample names loaded
                sample_names_loaded.add(sample_name)
            except:
                traceback.print_exc()
                raise MultiprocessorFailed(
                    keys = 'read_xarray_group',
                    message = f"Reading {','.join(samples_to_load)} group"
                )
        
        return data_arrs,sample_names_loaded
    
    def read_xr_data_concurrently(self,all_groups,samples_not_loaded:list,dirpath:str):
        # Gather h5 data from multiple files
        # and store them in xarrays
        data_arrs = []

        # Initialise progress bar
        progress = tqdm(
            total = len(all_groups),
            desc = f"Reading {','.join(samples_not_loaded)} group concurrently",
            leave = False,
            miniters = 1,
            position = 0
        )
        def my_callback(fut):
            progress.update()
            try:
                res = fut.result()
                if len(res) > 0:
                    # Extract sample name and data
                    sample_name = list(res.keys())[0]
                    sample_data = list(res.values())[0]
                    # append array to data arrays
                    if sample_data is not None:
                        data_arrs.append(sample_data)
                    # remove loaded sample from consideration
                    # since it has been succesfully loaded
                    if sample_name in samples_not_loaded and sample_data is not None:
                        samples_not_loaded.remove(sample_name)
            except (MissingFiles,CorruptedFileRead):
                pass
            except Exception as exc:
                raise ValueError(f"Reading {','.join(samples_not_loaded)} group concurrently failed") from exc

        with BoundedQueueProcessPoolExecutor(max_waiting_tasks = 2*self.settings.get('n_workers',1)) as executor:
            # Start the processes and ignore the results
            for group in all_groups:
                try:
                    future = executor.submit(
                        read_xr_data,
                        dirpath = dirpath,
                        sample_gid = group
                    )
                    future.add_done_callback(my_callback)
                except:
                    traceback.print_exc()
                    raise MultiprocessorFailed(
                        keys = 'read_xr_data_concurrently'
                    )

            # Delete executor and progress bar
            progress.close()
            safe_delete(progress)
            executor.shutdown(wait = True)
            safe_delete(executor)

        return data_arrs,samples_not_loaded
        
    def get_sweep_outputs_sequentially(
        self,
        sample_names:list = [],
        group_by:list = []
    ):  
        output_datasets = []
        for sweep_configuration in tqdm(
            self.config.sweep_configurations,
            desc='Collecting h5 data sequentially',
            leave = False,
            position = 0
        ):
            # Get metric data for sweep dataset
            res = self.get_sweep_outputs(
                base_config = self.config,
                sweep_configuration = sweep_configuration,
                sample_names = sample_names,
                group_by = group_by
            )
            if len(res) > 0:
                output_datasets.append(res)
        return output_datasets
    
    def get_sweep_outputs_concurrently(
            self,
            sample_names:list = [],
            group_by:list = []
        ):
        # Gather h5 data from multiple files
        # and store them in xarray-type dictionaries
        output_datasets = []

        # Initialise progress bar
        progress = tqdm(
            total = len(self.config.sweep_configurations),
            desc='Collecting h5 data in parallel',
            leave = False,
            miniters = 1,
            position = 0
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
                raise ValueError("Getting sweep outputs failed") from exc

        with BoundedQueueProcessPoolExecutor(max_waiting_tasks = 2*self.settings.get('n_workers',1)) as executor:
            # Start the processes and ignore the results
            for sweep_configuration in self.config.sweep_configurations:
                try:
                    future = executor.submit(
                        self.get_sweep_outputs,
                        base_config = self.config,
                        sweep_configuration = sweep_configuration,
                        sample_names = sample_names,
                        group_by = group_by
                    )
                    future.add_done_callback(my_callback)
                except:
                    traceback.print_exc()
                    raise MultiprocessorFailed(
                        keys = 'get_sweep_outputs_concurrently'
                    )

            # Delete executor and progress bar
            progress.close()
            safe_delete(progress)
            executor.shutdown(wait = True)
            safe_delete(executor)

        return output_datasets

    def get_sweep_outputs(
        self,
        base_config:Config,
        sweep_configuration:list,
        sample_names:list,
        group_by:list=[]
    ):
        # Get specific sweep config 
        new_config,sweep = deepcopy(base_config).prepare_experiment_config(
            sweep_configuration = sweep_configuration
        )
        # Get outputs and unpack its statistics
        # This is the case where there are SOME input slices provided
        outputs = Outputs(
            config = new_config,
            settings = self.settings,
            data_names = self.settings['sample'],
            base_dir = self.outputs_path,
            sweep = sweep,
            console_handling_level = self.settings['logging_mode'],
            slice = False,
            logger = self.logger
        )
        # Load inputs
        if self.inputs is None:
            # Import all input data
            outputs.inputs = Inputs(
                config = new_config,
                synthetic_data = False,
                logger = self.logger
            )
        data_array = outputs.load_single(
            sample_names = sample_names, 
            group_by = group_by,
            sweep = sweep
        )
        return data_array
            

    def get_sample(self,sample_name:str):
        
        # NOTE: This function is applied to data with only one sweep!

        # Instantiate inputs if required
        if getattr(self,'inputs',None) is None:
            self.inputs = Inputs(
                config = self.config,
                synthetic_data = False,
                logger = self.logger
            )

        if sample_name == 'intensity':
            # Get sim model 
            self.logger.debug('getting sim model')
            
            # Read from config
            intensity_name = self.config.settings[self.intensity_model_class]['name']

            # Get intensity model
            IntensityModelClass = globals()[intensity_name+'SIM']
            
            # Check that required data is available
            self.logger.debug('checking sim data availability')
            self.check_data_availability(
                sample_name = sample_name,
                input_names = IntensityModelClass.REQUIRED_INPUTS,
                output_names = IntensityModelClass.REQUIRED_OUTPUTS,
            )
            # Instantiate ct
            IntensityModel = IntensityModelClass(
                config = self.config,
                logger = self.logger,
                **{input:self.get_sample(input) for input in IntensityModelClass.REQUIRED_INPUTS}
            )
            # Grand total
            grand_total = self.inputs.data.grand_total \
                if torch.is_tensor(self.inputs.data.grand_total) \
                else torch.tensor(
                    self.inputs.data.grand_total,
                    device = self.config['inputs']['device'],
                    dtype = torch.int32
                )
            # Compute log intensity
            samples = IntensityModel.log_intensity(
                torch = False,
                grand_total = grand_total,
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
            elif isinstance(getattr(self.inputs.data,sample_name),(xr.DataArray,xr.Dataset)):
                samples = getattr(self.inputs.data,sample_name)
            else:
                samples = torch.tensor(
                    getattr(self.inputs.data,sample_name),
                    dtype = NUMPY_TO_TORCH_DTYPE[INPUT_SCHEMA[sample_name]['dtype']],
                    device = self.device
                )


        else:
            if not hasattr(self.data,sample_name):
                raise MissingData(
                    missing_data_name = sample_name,
                    data_names = ', '.join(list(self.data_vars().keys())),
                    location = 'Outputs'
                )
            elif self.data.sizes(dim = sample_name) > 1:
                raise IrregularDataCollectionSize(
                    message = f"""
                        Cannot process {sample_name} Data Collection of size > 1.
                    """,
                    sizes = {sample_name: self.data.sizes(dim = sample_name)}
                )
            else:
                # Get xarray
                samples = getattr(self.data,sample_name)

            # If parameter is beta, scale it by bmax
            if sample_name == 'beta' and self.intensity_model_class == 'spatial_interaction_model':
                return samples * self.config.settings[self.intensity_model_class]['parameters']['bmax']

        self.logger.progress(f"Loaded {sample_name} sample")
        return samples

    
    def create_slicing_conditions(self):
        if len(self.config.isolated_sweep_paths) > 0 or len(self.config.coupled_sweep_paths) > 0:

            coordinate_slice = []
            # Loop through key-operator-value tuples used
            # to subset the output samples
            for coord_slice in self.settings.get('coordinate_slice',[]):
                # Get all data names appearing in slice expression
                dim_names = [dim for dim in re.findall(r'\bda\.(\w+)\b',coord_slice) if dim not in ['isin']]
                
                # If all included data names are sweepable dimensions
                if all([self.config.is_sweepable(dim) for dim in dim_names]):
                    # Store this coordinate slice expression
                    coordinate_slice.append(coord_slice)
        else:
            coordinate_slice = []
        
        # Store as global var
        self.coordinate_slice = coordinate_slice
    
    def create_filename(self,sample = None):
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
                        for coord_slice in self.settings['burnin_thinning_trimming']
                        for var,setts in coord_slice.items()
                    ])
        return filename

    @classmethod
    def stack_sweep_and_iter_dims(cls,__self__):

        for sample_name,samples in __self__.data_vars().items(): 

            if isinstance(samples,xr.DataArray):
                samples = [samples]

            for i in range(len(samples)):

                # Get sample data
                sample_data = samples[i]
                
                # Find iteration dimensions
                iter_dims = [x for x in sample_data.dims if x in ['iter','seed']]

                # Find sweep dimensions that are not core coordinates
                sweep_dims = [d for d in sample_data.dims if d not in (list(CORE_COORDINATES_DTYPES.keys()))]
                
                # print(iter_dims,sweep_dims)

                # Stack variables and reorder data
                if len(sweep_dims) > 0 and len(iter_dims) > 0:
                    # Stack all non-core coordinates into new coordinate
                    sample_data = sample_data.stack(
                        id = tuplize(iter_dims),
                        sweep = tuplize(sweep_dims)
                    )
                    # Reorder coordinate names
                    samples[i] = sample_data.transpose(
                        'id',*OUTPUT_SCHEMA[sample_name].get("dims",[]),'sweep'
                    )
                elif len(iter_dims) > 0:
                    sample_data = sample_data.stack(
                        id = tuplize(iter_dims)
                    )
                    # Reorder coordinate names
                    samples[i] = sample_data.transpose(
                        'id',*OUTPUT_SCHEMA[sample_name].get("dims",[])
                    )
                elif len(sweep_dims) > 0:
                    sample_data = sample_data.stack(
                        sweep = tuplize(sweep_dims)
                    )
                    # Reorder coordinate names
                    samples[i] = sample_data.transpose(
                        *OUTPUT_SCHEMA[sample_name].get("dims",[]),'sweep'
                    )
                else:
                    # Reorder coordinate names
                    samples[i] = sample_data.transpose(
                        *OUTPUT_SCHEMA[sample_name].get("dims",[])
                    )
                
                # Update data
                getattr(
                    __self__.data,
                    sample_name
                )[i] = samples[i]

    @classmethod
    def check_object_availability(cls,__self__,reqs:list,object_name:str,**kwargs):
        for req in reqs:
            if getattr(__self__,req,None) is None:
                _ = __self__.instantiate_object_from_expression(
                    __self__,
                    f'{object_name}.{req}.',
                    object_name,
                    **kwargs
                )
    
    @classmethod
    def instantiate_object_from_expression(cls,__self__,expression:str,object_name:str='self',**kwargs):
        for obj in re.findall(rf'\b{object_name}\.[^.]*\b', expression):
            obj_call = obj.split(f"{object_name}.")[-1]

            if obj_call in ['config','inputs']:
                try:
                    assert getattr(__self__,obj_call,None) is not None
                except:
                    raise DataException(f"config and/or inputs not found")
            
            elif obj_call == 'ct':
                __self__.check_object_availability(
                    __self__,
                    ['config','inputs'],
                    object_name,
                    **kwargs
                )
                inputs_copy = deepcopy(__self__.inputs)
                inputs_copy.cast_from_xarray()
                __self__.ct = instantiate_ct(
                    config = __self__.config,
                    **inputs_copy.data_vars(),
                    **kwargs
                )
            
            elif obj_call == 'ct_mcmc':
                __self__.check_object_availability(
                    __self__,
                    ['ct'],
                    object_name,
                    **kwargs
                )
                __self__.ct_mcmc = ContingencyTableMarkovChainMonteCarlo(
                    ct = __self__.ct,
                    **kwargs
                )
                safe_delete(__self__.ct)
            
            elif obj_call == 'sim':
                __self__.check_object_availability(
                    __self__,
                    ['config','inputs'],
                    object_name,
                    **kwargs
                )
                inputs_copy = deepcopy(__self__.inputs)
                inputs_copy.cast_from_xarray()
                __self__.intensity_model = instantiate_sim(
                    name = __self__.config['spatial_interaction_model']['name'],
                    config = __self__.config,
                    true_parameters = __self__.config['spatial_interaction_model']['parameters'],
                    **inputs_copy.data_vars(),
                    **kwargs
                )
            
            elif obj_call == 'physics_model':
                __self__.check_object_availability(
                    __self__,
                    ['sim','config'],
                    object_name,
                    **kwargs
                )
                __self__.physics_model = HarrisWilson(
                    config = __self__.config,
                    intensity_model = __self__.intensity_model,
                    dt = __self__.config['harris_wilson_model'].get('dt',0.001),
                    true_parameters = __self__.inputs.true_parameters,
                    **kwargs
                )
                safe_delete(__self__.intensity_model)

            elif obj_call == 'neural_network':
                __self__.check_object_availability(
                    __self__,
                    ['config','inputs'],
                    object_name,
                    **kwargs
                )
                __self__.neural_network = NeuralNet(
                    input_size = __self__.inputs.data.dims['destination'],
                    output_size = len(__self__.config['inputs']['to_learn']),
                    **__self__.config['neural_network']['hyperparameters'],
                    **kwargs
                ).to(__self__.device)

            elif obj_call == 'learning_model':
                __self__.check_object_availability(
                    __self__,
                    ['config','neural_network','physics_model'],
                    object_name,
                    **kwargs
                )
                __self__.learning_model = HarrisWilson_NN(
                    config = __self__.config,
                    neural_net = __self__.neural_network,
                    loss = __self__.config['neural_network'].pop('loss'),
                    physics_model = __self__.physics_model,
                    write_every = 1,
                    write_start = 0
                    **kwargs
                )
                safe_delete(__self__.intensity_model)
                safe_delete(__self__.physics_model)
                safe_delete(__self__.neural_network)
    
class DataCollection(object):


    def __init__(self,
                 data = [],
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

        if len(data) > 0:
            if isinstance(data, list):
                # All items must be of type xarray data array
                assert all([isinstance(datum,xr.DataArray) for datum in data])

                # Update sample data collection
                for datum in tqdm(data, desc = 'Grouping Data Collection samples sequentially'):
                    self.group_sample(
                        datum,
                        group_by = kwargs.get('group_by',[])
                    )
                # Combine coords for each list element of the Data Collection
                for sample_name in vars(self).keys():
                    if sample_name in DATA_SCHEMA:
                        for i,group_datum in tqdm(
                            enumerate(getattr(
                                self,
                                sample_name
                            )),
                            total = len(getattr(self,sample_name)),
                            desc = 'Combining Data Collection group elements'
                        ):
                            # Combine by coords iff there are more than one elements in the group
                            if len(datum) > 1:
                                getattr(
                                    self,
                                    sample_name
                                )[i] = xr.combine_by_coords(
                                    group_datum,
                                    combine_attrs='drop_conflicts'
                                )
                            else:
                                getattr(
                                    self,
                                    sample_name
                                )[i] = group_datum[0]
            elif isinstance(data, dict):
                # All items must be of type xarray data array
                assert all([isinstance(datum,xr.DataArray) for datum in data.values()])
                
                for sample_name,sample_data in data.items():
                    if sample_name in DATA_SCHEMA:
                        setattr(
                            self,
                            sample_name,
                            [sample_data]
                        )
    
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
    
    def group_samples_sequentially(self,output_datasets,sample_name:str,group_by:list=[]):
        for datasets in tqdm(
            output_datasets,
            leave = False,
            miniters = 1,
            position = 0,
            desc = f'Grouping {sample_name} samples sequentially'
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
            desc = f'Combining {sample_name} DataArray(s) by coordinates sequentially'
        ):
            getattr(
                self,
                sample_name
            )[i] = self.combine_by_coords(
                i = i,
                combined_dims = combined_dims,
                datasets = datasets
            )[1]

    def combine_by_coords_concurrently(self,indx:int,sample_name:str,combined_dims:list):
        dataset_list = getattr(
            self,
            sample_name
        )
        # Every result corresponds to a unique pair of 
        # sweep id and sample name
        # Initialise progress bar
        progress = tqdm(
            total = len(dataset_list),
            leave = False,
            position = 0,
            miniters = 1,
            desc = f'Combining {sample_name} DataArray(s) by coordinates concurrently'
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

        with BoundedQueueProcessPoolExecutor(max_waiting_tasks = self.settings.get('n_workers',1)) as executor:
            # Gather all group and group elements that need to be combined
            for i,datasets in enumerate(dataset_list):
                try:
                    future = executor.submit(
                        self.combine_by_coords,
                        i = i,
                        combined_dims = combined_dims,
                        datasets = datasets
                    )
                    future.add_done_callback(my_callback)
                except:
                    # traceback.print_exc()
                    raise MultiprocessorFailed(
                        keys = 'combine_by_coords_concurrently'
                    )

        # Delete executor and progress bar
        progress.close()
        safe_delete(progress)
        executor.shutdown(wait = True)
        safe_delete(executor)
        return combined_coords
            

    def combine_by_coords(self,i:int,combined_dims:list,datasets:list):
        # Sort datasets by seed
        datasets = sorted(
            datasets, 
            key = lambda x: tuple([
                x.coords[var].item() \
                    for var in combined_dims \
                    if var not in ['N','iter']
            ])
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

    def __getitem__(self,index:int,sample_names:list = None):
        new_data = deepcopy(self)
        # Get intersection of provided and available sample names
        sample_names = sample_names if sample_names is not None else list(self._vars_().keys())
        sample_names = set(sample_names).intersection(set(list(self._vars_().keys())))
        for sample_name, sample_data in self._vars_().items():
            if int(index) >= len(getattr(self,sample_name)):
                raise KeyError(f"Sample {sample_name} index {index} out of bounds for length {len(getattr(self,sample_name))}.")
            setattr(
                new_data,
                sample_name,
                sample_data[index]
            )

        return new_data

    def __setitem__(self, index:int, new_data):
        for sample_name in vars(new_data).keys():
            # Get intersection of provided and available sample names
            sample_names = sample_names if sample_names is not None else list(self._vars_().keys())
            sample_names = set(sample_names).intersection(set(list(self._vars_().keys())))
            if sample_name in DATA_SCHEMA and sample_name in self._vars_():
                if int(index) >= len(getattr(self,sample_name)):
                    raise KeyError(f"Sample {sample_name} index {index} out of bounds for length {len(getattr(self,sample_name))}.")
                getattr(
                    self,
                    sample_name
                )[index] = getattr(
                    new_data,
                    sample_name
                )
    
    def __delitem__(self,index:int,sample_names:list = None):
        # Delete index element of Data Collection
        sample_names = set(sample_names).intersection(set(list(self._vars_().keys())))
        for sample_name in sample_names:
            if int(index) >= len(getattr(self,sample_name)):
                raise KeyError(f"Sample {sample_name} index {index} out of bounds for length {len(getattr(self,sample_name))}.")
            del getattr(
                self,
                sample_name
            )[index]

    def _vars_(self):
        return {k:v for k,v in vars(self).items() if k in DATA_SCHEMA}
    
    def __repr__(self):
        if all([
            datum.size <= 0
            for sample_data_colls in self._vars_().values()
            for datum in sample_data_colls
        ]):
            return 'Empty Dataset'
        
        else:
            return "\n\n".join([
                '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'+str(sample_name)+'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' + \
                (' \n'.join([str(dict(elem.sizes)) for elem in sample_data])
                if isinstance(sample_data,list) \
                else str(dict(sample_data.sizes))) + \
                '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
                for sample_name,sample_data in self._vars_().items()
            ])
    
    
    def sizes(self,dim:str = None):
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

    def size(self):
        return sum(list(self.sizes().values()))
    
    def __len__(self):
        length = 0
        try:
            assert len(set([size for size in self.sizes().values()])) <= 1
            length = len(set([size for size in self.sizes().values()]))
        except:
            raise IrregularDataCollectionSize(sizes= self.sizes())
        
        if length <= 0:
            return 0
        else:
            return list(self.sizes().values())[0]



class OutputSummary(object):

    def __init__(
        self,
        settings:dict={},
        **kwargs
    ):
        
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
    
    def collect_metadata(self):
        
        experiment_metadata = {}
        for indx,output_folder in enumerate(self.output_folders):
            
            # Get metadata collection for this 
            metadata_collection,_ = self.collect_folder_metadata(indx,output_folder)
            
            for metadata in metadata_collection:
                if len(metadata) > 0:
                    if output_folder in experiment_metadata:
                        experiment_metadata[output_folder]= np.append(
                            experiment_metadata[output_folder],
                            metadata,
                            axis = 0
                        )
                    else:
                        experiment_metadata[output_folder] = metadata
        return experiment_metadata
    
    def collect_folder_metadata(self, indx:int, output_folder:str):
        self.logger.info(f"\n\n\n Scanning folder {indx+1}/{len(self.output_folders)}")
        self.logger.info(output_folder)
            
        # Collect outputs from folder
        outputs = self.get_folder_outputs(indx,output_folder)

        # Loop through each member of the data collection
        if self.settings.get('n_workers',1) > 1:
            metadata_collection = self.get_experiment_metadata_concurrently(outputs)
        else:
            metadata_collection = self.get_experiment_metadata_sequentially(outputs)
        
        # Convert generator to list
        metadata_collection = list(metadata_collection)

        if len(metadata_collection) <= 0:
            self.logger.error(f"Empty output dataset in {output_folder}")
            raise MissingMetadata("Failed collecting experiments metadata")
        
        # Strip outputs of all unnecessary data
        outputs.strip_data(keep_inputs=['dims'])

        # Collect garbage
        gc.collect()

        # Convert metric data collection to list
        return list(metadata_collection),outputs

    def get_experiment_metadata_sequentially(self,outputs):
        # Every result corresponds to a unique pair of 
        # sweep id and sample name
        # Loop through each member of the data collection
        for j in tqdm(
            range(len(outputs.data)),
            total = len(outputs.data),
            desc='Collecting experiment metadata sequentially',
            leave = False,
            miniters = 1,
            position = 0
        ):
            # Collect metric metadata
            yield self.get_experiment_metadata(outputs.get(j))

    def get_experiment_metadata_concurrently(self,outputs):
        # Every result corresponds to a unique pair of 
        # sweep id and sample name
        # Initialise progress bar
        progress = tqdm(
            total = len(outputs.data),
            desc='Collecting experiment metadata concurrently',
            leave = False,
            miniters = 1,
            position = 0
        )
        results = []
        def my_callback(fut):
            progress.update()
            try:
                res = fut.result()
                results.append(res)
            except Exception as exc:
                raise ValueError("Getting experiment metadata failed") from exc

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
                    # traceback.print_exc()
                    raise MultiprocessorFailed(
                        keys = 'get_experiment_metadata_concurrently'
                    )

        # Delete executor and progress bar
        progress.close()
        safe_delete(progress)
        executor.shutdown(wait = True)
        safe_delete(executor)
        return results

    def get_experiment_metadata(self,outputs:Outputs):

        # Read inputs if they are sweeped
        if outputs.inputs is None:
            outputs.inputs = Inputs(
                config = outputs.config,
                synthetic_data = False,
                logger = self.logger
            )
        # Cast inputs to xr DataArray
        outputs.inputs.cast_to_xarray()

        # Apply these operations to the data 
        expression_data = self.evaluate_expressions(outputs = outputs)

        # Delete outputs
        safe_delete(outputs)

        # Convert data to df 
        if len(expression_data) > 0:
            expression_data_df = pd.DataFrame(expression_data)
        
        # Make sure either metric or expression evaluation data have been computed
        try:
            assert len(expression_data) > 0
        except:
            raise MissingData(
                missing_data_name = "expression_data_df", 
                data_names = "empty",
                location = "Outputs(get_experiment_metadata)"
            )
        
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
            # the metadata are included in the sweep columns 
            # of the expression_data_df
                
        # Add useful metadata to metric and operation data
        expression_data_df = expression_data_df.assign(
            folder = os.path.join(outputs.outputs_path),
            **useful_metadata
        )
        
        # Return all data as list of dictionaries
        return expression_data_df.to_dict('records')

    def get_folder_outputs(self,indx:str,output_folder:str):
            
        # Read metadata config
        config = Config(
            path = os.path.join(output_folder,"config.json"),
            logger = self.logger
        )
        # Get sweep-related data
        config.get_sweep_data()
        
        # If sweep is over input data
        input_sweep_param_names = set(config.sweep_param_names).intersection(
            set(list(INPUT_SCHEMA.keys()))
        )
        input_sweep_param_names = set(input_sweep_param_names).difference(set(['to_learn']))
        input_sweep_param_names = input_sweep_param_names.union(
            set(config.sweep_param_names).intersection(
                set(list(PARAMETER_DEFAULTS.keys()))
            )
        )
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

        # Instantiate global outputs
        outputs = Outputs(
            config = config,
            settings = self.settings,
            data_names = self.settings['sample'],
            inputs = passed_inputs,
            console_handling_level = self.settings['logging_mode'],
            logger = self.logger
        )
        
        # Load all output data
        outputs.load(indx = indx)

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
                    by = list(self.settings['sort_by']),
                    ascending = self.settings['ascending']
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

            index_slice_str = '_'.join([
                f"({var},burnin{setts['burnin']},thinning{setts['thinning']},trimming{setts['trimming']})"
                for coord_slice in self.settings['burnin_thinning_trimming']
                for var,setts in coord_slice.items()
            ])
            if len(self.settings.get('directories',[])) > 0:
                str_list = [
                    (index_slice_str if index_slice_str else ''),
                    f"{self.settings['filename_ending'] if len(self.settings['filename_ending']) else 'summaries'}.csv"
                ]
                filepath = os.path.join(
                    output_directory,
                    '_'.join([s for s in str_list if s])
                )
            else:
                # Gather all dates provided
                date_strings = '__'.join(self.settings['dates'])
                str_list = [
                    "{'_'.join(list(self.settings['experiment_type']))}",
                    f"{'_'.join(self.settings['title']) if len(self.settings['title']) > 0 else ''}",
                    f"{'multiple_dates' if len(date_strings) > 4 else date_strings if len(date_strings) > 0 else ''}",
                    (index_slice_str if index_slice_str else ''),
                    f"{self.settings['filename_ending'] if len(self.settings['filename_ending']) else 'summaries'}.csv"
                ]

                filepath = os.path.join(
                    output_directory,
                    '_'.join([s for s in str_list if s])
                )
            # Write experiment summaries to file
            self.logger.info(f"Writing summaries to {filepath}")
            write_csv(experiment_metadata_df,filepath,index = True)
            print('\n')
    
        
    def evaluate_expressions(self,outputs:Outputs):
        # Get outputs and unpack its statistics
        if len(self.settings['evaluate']) == 0:
            return []

        self.logger.progress(f"Evaluating expressions for {outputs.experiment_id}")

        # Create a copy of global outputs
        sweep_outputs = deepcopy(outputs)
        
        # Get first sample
        sample_data = list(outputs.data_vars().values())[0]
        
        # Update config using sweep configuration
        sweep_outputs.config.update({
            "sweep_mode":False,
            **dict(zip(
                [x for x in sample_data.get_index('sweep').names if x != 'sweep'],
                list(map(unstringify,sample_data.get_index('sweep')[0]))
            ))
        })

        # Gather all arguments for evaluating every expression later on
        keyword_args = {}
        # Get all keys that correspond to raw data
        raw_data_keys = [k for k,_ in self.settings['evaluation_kwargs'] if k in list(DATA_SCHEMA.keys())]
        # First, gather all input/outputs
        for key in raw_data_keys:
            # Get sample
            self.logger.progress(f'Getting sample {key}...')
            try:
                samples = sweep_outputs.get_sample(key)
                keyword_args[key] = samples
            except CoordinateSliceMismatch as exc:
                self.logger.debug(exc)
                continue
            except MissingData as exc:
                self.logger.debug(exc)
                continue
            except Exception as exc:
                self.logger.error(f"Getting sample {key} failed")
                self.logger.debug(traceback.format_exc())
                self.logger.error(exc)
                sys.exit()
            self.logger.progress(f"{key} {dict(samples.sizes)}, {samples.dtype}")
        
        keyword_expressions = [(k,v) for k,v in self.settings['evaluation_kwargs'] if k not in raw_data_keys]
        self.logger.progress([k for k,_ in keyword_expressions])

        # Second, gather all data derivative arguments
        for key,expression in keyword_expressions:
            self.logger.progress(f"trying {key} {expression}")
            # You might need to instantiate some objects first
            if 'outputs.' in expression:
                # Instantiate necessary objects
                try:
                    sweep_outputs.instantiate_object_from_expression(
                        sweep_outputs,
                        expression,
                        object_name = 'outputs',
                        instance = 0,
                        level = 'EMPTY'
                    )
                except Exception as exc:
                    self.logger.debug(traceback.format_exc())
                    raise InstantiationException(f"{exc} by instantiating {expression}")
            # Evaluate keyword argument only if no such argument 
            # has already been evaluated
            if keyword_args.get(key,None) is None:
                try:
                    keyword_eval = eval(
                        expression,
                        {
                            **sweep_outputs.inputs.data_vars(),
                            **sweep_outputs.inputs.data.dims,
                            **sweep_outputs.data_vars(),
                            "outputs":sweep_outputs,
                            **{str(k):eval(str(k)) for k in self.settings['evaluation_library']}
                        },
                        {
                            **keyword_args
                        }
                    )
                except Exception as exc:
                    # traceback.print_exc()
                    self.logger.warning(f"{key} with keyword expression {expression} failed: {exc}")
                    self.logger.debug(traceback.format_exc())
                    self.logger.debug(f"""Available data: {
                    list(sweep_outputs.inputs.data_vars().keys()) +
                    list(sweep_outputs.inputs.data.dims.keys()) +
                    ['outputs'] + list(keyword_args.keys())
                    }""")
                    self.logger.debug(traceback.format_exc())
                    continue
                
                # Store evaluation of keyword argument
                if keyword_eval is not None:
                    keyword_args[key] = keyword_eval
                
                self.logger.progress(f"Keyword {key} expression {expression} succeded.")

            # print_json(keyword_args,newline = True)
        
        # Delete temporary objects
        for attr in ['ct','intensity_model','ct_mcmc','physics_model','learning_model']:
            if getattr(sweep_outputs,attr,None) is not None:
                safe_delete(getattr(sweep_outputs,attr))
        # Garbage collect
        gc.collect()

            
        evaluation_data = {}
        evaluation_kwargs = {}
        for operation_name, expression in self.settings['evaluate']:
            self.logger.progress(f"trying {operation_name} {expression}")
            # Evaluate expression only if no such expression
            # has already been evaluated
            if operation_name not in evaluation_kwargs:
                try:
                    evaluation = eval(
                        expression,
                        {
                            **outputs.inputs.data_vars(),
                            **outputs.inputs.data.dims,
                            **{str(k):eval(str(k)) for k in self.settings['evaluation_library']}
                        },
                        {
                            **keyword_args,
                            **evaluation_kwargs
                        }
                    )
                    # Update list of evaluated expressions
                    evaluation_kwargs[operation_name] = evaluation
                except Exception as exc:
                    traceback.print_exc()
                    self.logger.warning(f"{operation_name} with operation expression {expression} failed: {exc}")
                    self.logger.debug(traceback.format_exc())
                    continue
            
            if isinstance(evaluation,(xr.DataArray,xr.Dataset)):
                if 'sweep' in evaluation.dims:
                    self.logger.note(f"sweep: {evaluation['sweep'].values.tolist()}")

            self.logger.success(f"Evaluation {operation_name} using {expression} succeded {np.shape(evaluation) if not isinstance(evaluation,xr.DataArray) else dict(evaluation.sizes)}")
            print('\n')
            if isinstance(evaluation,(xr.DataArray,xr.Dataset)):
                if 'sweep' in evaluation.dims:
                    # Rename xr data array
                    evaluation = evaluation.rename(operation_name.lower())
                    # This loops over sweep configurations
                    sweep_keys = list(evaluation.get_index('sweep').names)
                    for sweep_values,eval_data in evaluation.groupby('sweep'):
                        self.logger.progress(f"Gathering {operation_name} data for { dict(zip(sweep_keys,sweep_values))}")
                        # Gather all key-value pairs from every row 
                        # (corresponding to a single sweep setting)
                        sweep = dict(zip(sweep_keys,sweep_values))
                        sweep = {k:get_value(sweep,k) for k in sweep.keys()}
                        # Get sweep id in string form
                        sweep_id = ' & '.join([str(k)+'_'+str(v) for k,v in sweep.items() if k not in [operation_name,'sweep']])
                        # Get scalar if only one item is provided otherwise get list
                        data = eval_data.values.ravel()
                        try:
                            data = data.item()
                        except:
                            data = data.tolist()
                        # Add every sweep configuration to this evaluation data
                        if sweep_id not in evaluation_data:
                            evaluation_data[sweep_id] = {
                                **{
                                    f"{operation_name}_expression":expression,
                                    operation_name:data
                                },
                                **sweep
                            }
                            
                        else:
                            evaluation_data[sweep_id].update({
                                operation_name:data
                            })
                else:
                    # Get scalar if only one item is provided otherwise get list
                    data = evaluation.values.ravel()
                    try:
                        data = data.item()
                    except:
                        data = data.tolist()
                    # Rename xr data array
                    evaluation = evaluation.rename(operation_name.lower())
                    # Add data to every existing sweep
                    for sweep_id in evaluation_data.keys():
                        evaluation_data[sweep_id].update({
                            operation_name:data
                        })
            else:
                # Get data list
                if isinstance(evaluation,np.generic):
                    data = evaluation.tolist()
                elif isinstance(evaluation,Iterable) and not isinstance(evaluation, str):
                    data = evaluation
                else:
                    data = [evaluation]
                # Add data to every existing sweep
                for sweep_id in evaluation_data.keys():
                    if operation_name in evaluation_data[sweep_id]:
                        evaluation_data[sweep_id][operation_name].append(data)
                    else:
                        evaluation_data[sweep_id].update({operation_name:data})

        return list(evaluation_data.values())
        