# import memray

from memory_profiler import profile

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
import geopandas as gpd
import dask.array as da
import scipy.sparse as sp
import concurrent.futures as concurrency

from tqdm import tqdm
from torch import int32
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from operator import itemgetter
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

class OutputSummary(object):

    def __init__(self, settings, **kwargs):
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__,
            console_level = level,
        ) if kwargs.get('logger',None) is None else kwargs['logger']

        # Get command line settings
        self.settings = settings
        # Store device
        self.device = self.settings.get('device','cpu')
        # Instatiate list of experiments
        self.experiment_metadata = {}
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
        # Setup experiments
        self.collect_experiment_metadata()
     
    @classmethod
    def find_matching_output_folders(cls,__self__):
        if 'directories' in list(__self__.settings.keys()) and len(__self__.settings['directories']) > 0:
            output_dirs = []
            for _dir in list(__self__.settings['directories']):
                for dataset_name in __self__.settings['dataset_name']:
                    path_dir = os.path.join(
                        __self__.settings['out_directory'],
                        dataset_name,
                        _dir
                    )
                    if os.path.exists(path_dir):
                        output_dirs.append(path_dir)
        else:
            # Search metadata based on search parameters
            # Get output directory
            output_directory = __self__.settings['out_directory']
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
            output_dirs = flatten([get_all_subdirectories(output_directory+dataset) for dataset in dataset_names])
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
        return output_dirs
    
    @profile
    def collect_experiment_metadata(self):
        # Find matching directories
        output_dirs = self.find_matching_output_folders(self)
        for i,output_folder in enumerate(output_dirs):
            
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
            self.logger.info(f"Scanning folder {i+1}/{len(output_dirs)}")
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
                continue
            
            # If inputs are not sweeped pre-load them
            
            # Import all input data
            inputs = Inputs(
                config = self.config,
                synthetic_data = False,
                logger = self.logger,
            )

            # Try to load ground truth table
            self.settings['table_total'] = self.settings.get('table_total',1)
            # Read ground truth table
            if hasattr(inputs.data,'ground_truth_table'):
                # Gather dims from ground truth table
                # which is guaranteed to NOT be sweepable
                origin,destination = np.shape(inputs.data.ground_truth_table)
                self.ground_truth_table = xr.DataArray(
                    data=torch.tensor(inputs.data.ground_truth_table,dtype=int32,device=self.device),
                    name='ground_truth_table',
                    dims=['origin','destination'],
                    coords=dict(
                        origin=np.arange(1,origin+1,1,dtype='int16'),
                        destination=np.arange(1,destination+1,1,dtype='int16')
                    )
                )
                # Delete inputs ground truth table 
                # and keep only the current ground truth table xarray
                safe_delete(inputs.data.ground_truth_table)
                # Try to update metadata on table samples
                self.settings['table_total'] = self.ground_truth_table.sum(dim=['origin','destination']).values
            else:
                raise Exception('Inputs are missing ground truth table.')
            
            # If sweep is over input data
            if self.config.sweep_param_names and (
                (set(self.config.sweep_param_names).intersection(set(list(INPUT_SCHEMA.keys())))) or \
                (set(self.config.sweep_param_names).intersection(set(list(PARAMETER_DEFAULTS.keys()))))
            ):
                # Load inputs for every single output
                passed_inputs = None
            else:
                passed_inputs = deepcopy(inputs)

            # Gather results
            output_datasets = []
            # Do it concurrently
            stop = None
            if self.settings.get('n_workers',1) > 1:

                # Initialise progress bar
                with tqdm(
                    sweep_configurations[:stop],
                    desc='Collecting outputs',
                    leave=False,
                    position=0
                ) as pbar:
                    with concurrency.ProcessPoolExecutor(self.settings.get('n_workers',1)*2) as executor:
                        # Start the processes and ignore the results
                        futures = [executor.submit(
                            self.get_sweep_outputs,
                            sweep_configuration = sweep_configuration,
                            inputs = passed_inputs,
                            coordinate_slice = coordinate_slice
                        ) for sweep_configuration in sweep_configurations[:stop]]


                        # Wait for all processes to finish
                        for index,fut in enumerate(concurrency.as_completed(futures)):
                            try:
                                None
                                output_datasets.append(fut.result())
                            except:
                                self.logger.error(traceback.format_exc())
                                raise Exception(f"Future {index} failed")
                            pbar.update(1)
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
                            sweep_configuration,
                            inputs = passed_inputs,
                            coordinate_slice = coordinate_slice
                        )
                    )
                    # sys.exit()

            # Reload global outputs
            outputs = Outputs(
                config = self.config,
                settings = self.settings,
                dataset = self.settings['sample'],
                base_dir = self.base_dir,
                console_handling_level = self.settings['logging_mode'],
                logger = self.logger
            )

            # Get found output names
            first_output = deepcopy(output_datasets[0])
            output_names = deepcopy(list(first_output.keys()))
            # Gather sweep dimension names
            sweep_dims = list(self.sweep_params['isolated'].keys())
            sweep_dims += list(self.sweep_params['coupled'].keys())

            # Create xarray dataset
            try:
                self.logger.info(f"{len(output_names)} outputs found: {','.join(output_names)}.")
                for sample_name in output_names[:-2]:
                    sample_dims = XARRAY_SCHEMA[sample_name]['new_shape']+sweep_dims
                    with tqdm(
                        total = len(output_datasets[:stop]),
                        desc=f'Creating xarray dataset for {sample_name}',
                        position=0,
                        leave=False
                    ) as pbar:
                        setattr(outputs.data,sample_name,[])
                        indx = 0
                        while indx < len(sweep_configurations[:stop]):
                            # Get sample xr_data
                            xr_data = output_datasets[indx].pop(sample_name)
                            # Coordinates of output dataset
                            data_coords = xr_data.pop('coordinates')
                            # Create slice dictionary
                            slice_dict = {
                                k: [stringify_index(parse(elem)) for elem in data_coords[k]]
                                for k in data_coords.keys()
                                if k in sample_dims
                            }
                            # Store data
                            getattr(
                                outputs.data,
                                sample_name
                            ).append(
                                xr.DataArray(
                                    da.from_array(xr_data.pop('data'),chunks=1),
                                    name = sample_name,
                                    dims = list(slice_dict.keys()),
                                    coords = slice_dict,
                                    attrs = dict(
                                        experiment_id = outputs.experiment_id,
                                        name = sample_name,
                                    )
                                )
                            )
                                
                            # Update progress bar
                            pbar.update(1)
                            indx += 1
                    
                    # Combine dataarrays by coordinates
                    self.logger.info(f"Combinining {sample_name} DataArrays by coordinates")
                    setattr(
                        outputs.data,
                        sample_name,
                        xr.combine_by_coords(
                            getattr(
                                outputs.data,
                                sample_name
                            )
                        )
                    )
                    # Chunk dataarray  by seed
                    setattr(
                        outputs.data,
                        sample_name,
                        getattr(
                            outputs.data,
                            sample_name
                        ).chunk(dict(seed=1))
                    )

                # print(getattr(
                #         outputs.data,
                #         sample_name
                #     ))
                # sys.exit()
                # for sample_name in vars(outputs.data).keys():
                #     # Slice according to coordinate slice
                #     if len(coordinate_slice) > 0:
                #         setattr(
                #             outputs.data,
                #             sample_name, 
                #             getattr(outputs.data,sample_name).sel(
                #                 **{
                #                     k:(v['values'] \
                #                         if len(v['values']) > 1 \
                #                         else v['values'][0]) \
                #                     for k,v in coordinate_slice.items()
                #                 }
                #             )
                #         )

            except:
                print(traceback.format_exc())
            return
            # Apply these metrics to the data 
            metric_data = self.apply_metrics(
                experiment_id = output_folder,
                outputs = outputs
            )
            for j in range(len(metric_data)):
                metric_data[j]['folder'] = os.path.join(self.base_dir)
                for k,v in folder_metadata.items():
                    metric_data[j][k] = v                 
            # Store useful metadata
            if not output_folder in list(self.experiment_metadata.keys()):
                self.experiment_metadata[output_folder] = metric_data
            else:
                self.experiment_metadata[output_folder] = np.append(
                    self.experiment_metadata[output_folder],
                    metric_data,
                    axis=0
                )            
    
                    
    def get_sweep_outputs(
        self,
        sweep_configuration,
        inputs:Inputs=None,
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
            dataset = self.settings['sample'],
            coordinate_slice = coordinate_slice,
            sweep_params = sweep,
            base_dir = self.base_dir,
            console_handling_level = self.settings['logging_mode'],
            logger = self.logger
        )
        # Get dictionary output data to be passed into xarray
        xr_dict_data = outputs.load(inputs=inputs)

        return xr_dict_data


    def write_metadata_summaries(self):
        if len(self.experiment_metadata.keys()) > 0:
            # Create dataframe
            experiment_metadata_df = pd.DataFrame(list(chain(*self.experiment_metadata.values())))
            experiment_metadata_df = experiment_metadata_df.set_index('folder')

            # Sort by values specified
            if len(self.settings['sort_by']) > 0 and all([sb in experiment_metadata_df.columns.values for sb in self.settings['sort_by']]):
                experiment_metadata_df = experiment_metadata_df.sort_values(by=list(self.settings['sort_by']),ascending=self.settings['ascending'])

            # Find dataset directory name
            dataset = find_dataset_directory(self.settings['dataset_name'])

            # Make output directory
            output_directory = os.path.join(
                self.settings['out_directory'],
                dataset,
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
                        f"{'_'.join(os.path.basename(folder) for folder in set(list(self.experiment_metadata.keys())))}_"+\
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
             # Get samples
            self.logger.progress(f'Getting sample {sample_name}...')
            try:
                samples = outputs.get_sample(sample_name)
            except Exception as e:
                self.logger.error(f'Experiment {os.path.basename(experiment_id)} does not have sample {sample_name}')
                continue
            self.logger.debug(f"samples {np.shape(samples)}, {samples.dtype}")
            for statistics in outputs.settings['statistic']:
                # Unpack sample and metric statistics
                sample_statistics_axes = statistics[0]
                metric_statistics_axes = statistics[1]
                # Compute statistic before applying metric
                # samples_summarised = None
                try:
                    samples_summarised = outputs.apply_sample_statistics(samples,sample_name,sample_statistics_axes)
                except Exception as e:
                    self.logger.debug(traceback.format_exc())
                    self.logger.error(f"samples {np.shape(samples)}, {samples.dtype}")
                    self.logger.error(f"Applying statistic {' over axes '.join([str(s) for s in sample_statistics_axes])} \
                                    for sample {sample_name} of experiment {experiment_id} failed")
                    print('\n')
                    continue
                self.logger.debug(f"samples_summarised {np.shape(samples_summarised)}, {samples_summarised.dtype}")

                for metric in self.settings['metric']:
                    # Get all attributes and their values
                    attribute_keys = METRICS[metric]['loop_over']
                    if len(attribute_keys) > 0:
                        attribute_values = product(*[self.settings[attr] for attr in attribute_keys])
                        # Get copy of settings
                        settings_copy = deepcopy(self.settings)
                        other_attrs = list(set(attribute_keys).difference(set(attribute_keys)))                
                        # Delete all other attributes in metrics
                        if len(other_attrs) > 0:
                            settings_copy = deep_delete(settings_copy,other_attrs)
                        # Loop over all possible combinations of attribute values                        
                        for value_tuple in attribute_values:
                            attribute_settings = dict(zip(attribute_keys,value_tuple))
                            attribute_settings_string = ','.join([f"{k}_{v}" for k,v in attribute_settings.items()])
                            for key,val in attribute_settings.items():
                                # Update settings values of attributes
                                settings_copy[key] = val

                            # Create string 
                            # Update kwargs specific for each metric
                            try:
                                metric_kwargs = self.update_metric_arguments(
                                    metric,
                                    outputs,
                                    settings_copy
                                )
                            except Exception as e:
                                self.logger.debug(traceback.format_exc())
                                self.logger.error(f"metric {np.shape(metric)}, {metric.dtype}")
                                self.logger.error(f'Arguments for metric {metric} cannot be updated')
                                print('\n')
                            
                            try:
                                if metric != 'none':
                                    samples_metric = globals()[metric](
                                        tab=samples_summarised,
                                        **metric_kwargs
                                    )
                                else:
                                    samples_metric = deepcopy(samples_summarised)
                                # Reshape samples metric
                                samples_metric = samples_metric.reshape(metric_shape)
                            except Exception as e:
                                self.logger.debug(traceback.format_exc())
                                self.logger.error(f"samples_summarised {np.shape(samples_summarised)}, {samples_summarised.dtype}, {samples_summarised.device}")
                                self.logger.error(f"tab0 {np.shape(metric_kwargs['tab0'])}, {metric_kwargs['tab0'].dtype}, {metric_kwargs['tab0'].device}")
                                self.logger.error(f'Applying metric {metric} for {attribute_settings_string} \
                                                  over sample {sample_name} \
                                                  for experiment {experiment_id} failed')
                                print('\n')
                                continue
                            # print(sample_name,metric,samples_metric)
                            self.logger.debug(f"Samples metric is {np.shape(samples_metric)}")
                            # Apply statistics after metric
                            try:
                                self.settings['axis'] = METRICS[metric]['apply_axis']
                                metric_summarised = outputs.apply_sample_statistics(
                                                        samples=samples_metric,
                                                        sample_name=metric,
                                                        statistic_axes=metric_statistics_axes
                                )
                            except Exception as e:
                                self.logger.debug(traceback.format_exc())
                                self.logger.error(f"samples_metric {np.shape(samples_metric)}, {samples_metric.dtype}")
                                self.logger.error(f"Applying statistic(s) {'>'.join([str(m) for m in metric_statistics_axes])} \
                                                    over metric {metric} and sample {sample_name} for experiment {experiment_id} failed")
                                print('\n')
                                continue
                            self.logger.debug(f"Summarised metric is {np.shape(metric_summarised)}")
                            # Squeeze output
                            metric_summarised = np.squeeze(metric_summarised)
                            # Add to data records
                            metric_data_keys = [
                                "sample_statistic",
                                "sample_name",
                                "metric",
                                *attribute_settings.keys(),
                                "metric_statistic",
                                "value"
                            ]
                            metric_data_vals = [
                                f"{'|'.join([stringify_statistic(_stat_ax) for _stat_ax in sample_statistics_axes])}>",
                                f"{sample_name}",
                                f"{metric}",
                                *attribute_settings.values(),
                                f"{'|'.join([stringify_statistic(_stat_ax) for _stat_ax in metric_statistics_axes])}>",
                                f"{metric_summarised}"
                            ]
                            metric_data.append(dict(zip(metric_data_keys,metric_data_vals)))
                            self.logger.debug(f"Summarised metric is updated to {np.shape(metric_summarised)}")
                    else:                        
                        # Update kwargs specific for each metric
                        metric_kwargs = self.update_metric_arguments(
                            metric,
                            outputs,
                            self.settings
                        )
                        # try:
                        if metric != 'none':
                            samples_metric = samples_summarised.groupby('sweep').apply(
                                lambda group: globals()[metric](
                                    tab=group,
                                    **metric_kwargs
                                )
                            )
                            # Rename metric xr data array
                            samples_metric = samples_metric.rename(metric)
                        else:
                            samples_metric = deepcopy(samples_summarised)
                        # except Exception as e:
                        #     self.logger.debug(traceback.format_exc())
                        #     self.logger.error(f"samples_summarised {np.shape(samples_summarised)}, {samples_summarised.dtype}")
                        #     self.logger.error(f"tab0 {metric_kwargs['tab0'].shape}, {metric_kwargs['tab0'].dtype}")
                        #     self.logger.error(f'Applying metric {metric} over sample {sample_name} for experiment {experiment_id} failed')
                        #     print('\n')
                        #     continue
                        
                        # Apply statistics after metric
                        # try:
                        metric_summarised = outputs.apply_sample_statistics(samples_metric,metric,metric_statistics_axes)
                        # except Exception as e:
                        #     self.logger.debug(traceback.format_exc())
                        #     self.logger.error(f"Shape of metric is {np.shape(samples_metric)}")
                        #     self.logger.error(f"Applying statistic {'|'.join([stringify_statistic(_stat_ax) for _stat_ax in metric_statistics_axes])}>" + \
                        #                       f"over metric {metric} and sample {sample_name} for experiment {experiment_id} failed")
                        #     print('\n')
                        #     continue
                        self.logger.debug(f"Summarised metric is {np.shape(metric_summarised)}")
                        # Get metric data in pandas dataframe
                        metric_summarised = metric_summarised.to_dataframe().reset_index(drop=True)

                        # Add to data records
                        metric_data_keys = [
                            "sample_statistic",
                            "sample_name",
                            "metric",
                            "metric_statistic"
                        ]
                        metric_data_vals = [
                            f"{'|'.join([stringify_statistic(_stat_dim) for _stat_dim in sample_statistics_axes])}",
                            f"{sample_name}",
                            f"{metric}",
                            f"{'|'.join([stringify_statistic(_stat_dim) for _stat_dim in metric_statistics_axes])}",
                        ]
                        for _,row in metric_summarised.iterrows():
                            # Gather all key-value pairs from every row (corresponding to a single sweep setting)
                            item = {(k if k != metric else 'value'):str(v) for k,v in row.to_dict().items()}
                            # Add them to list of metric data
                            metric_data.append(
                                dict(
                                    zip(
                                        metric_data_keys+list(item.keys()),
                                        metric_data_vals+list(item.values())
                                    )
                                )
                            )

                        self.logger.debug(f"Summarised metric is updated to {np.shape(metric_summarised)}")
    
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
            metric_arguments['tab0'] = outputs.ground_truth_table
        
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
                        if SWEEPABLE_PARAMS[name]['is_coord']:
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
                            if SWEEPABLE_PARAMS[coupled_name]['is_coord']:
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

class Outputs(object):

    def __init__(self,
                 config:Config, 
                 settings:dict={}, 
                 dataset:list=['ground_truth_table'],
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
        self.data_names = dataset
        # Store sample data requirements
        self.output_names = list(
            flatten([
                SAMPLE_DATA_REQUIREMENTS[sam] 
                for sam in set(self.data_names).intersection(set(list(OUTPUT_TYPES.keys())))]
            )
        )
        self.input_names = [
            sam for sam in set(self.data_names).intersection(set(list(INPUT_TYPES.keys())))
        ]

        # Sample names must be a subset of all data names
        try:
            assert set(self.data_names).issubset(set(DATA_TYPES.keys()))
        except Exception:
            self.logger.error('Some sample names provided are not recognized')
            self.logger.error(','.join(self.data_names))
            self.logger.debug(traceback.format_exc())
            raise Exception('Cannot load outputs.')

        # Store settings
        self.settings = settings
        # Store device
        self.device = self.settings.get('device','cpu')
        # Store coordinate slice
        self.coordinate_slice = kwargs.get('coordinate_slice',[])
        # Create semi-private xarray data 
        self.data = Dataset()
        # Enable garbage collector
        gc.enable()
        
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

    def load(self,inputs=None,input_slice:dict={},slice_samples:bool=True):

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
            slice_samples = slice_samples
        )
        return output_dict_data

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
                    # sys.exit()
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

    def slice_sample_iterations(self,samples):
        # Get burnin parameter
        burnin = min(self.settings.get('burnin',1),samples.shape[0])

        # Get thinning parameter
        thinning = list(deep_get(key='thinning',value=self.settings))
        thinning = thinning[0] if len(thinning) > 0 else 1
        
        # Get iterations
        iters = np.arange(start=1,stop=samples.shape[0]+1,step=1,dtype='int32')

        # Apply burnin and thinning
        samples = samples[burnin:None:thinning]
        iters = iters[burnin:None:thinning]

        # Get total number of samples
        N = self.settings.get('N',samples.shape[0])
        if N is None:
            N = samples.shape[0]
        else:
            N = min(N,samples.shape[0])
        
        # Apply stop
        samples = samples[:N]
        iters = iters[:N]

        return samples,iters

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
                        sample_data,iters = self.slice_sample_iterations(sample_data)
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

    def read_h5_data(self,h5files:list,**kwargs):
        # Get each file and add it to the new dataset
        h5data, local_coords, file_global_coords, data_vars = {},{},{},{}

        # Do it sequentially
        for filename in h5files:

            # Read h5 file
            h5data = self.read_h5_file(filename,**kwargs)
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

    def load_h5_data(self,slice_samples:bool=True):
        self.logger.note('Loading h5 data into xarrays...')

        # Get all h5 files
        h5files = list(Path(os.path.join(self.outputs_path,'samples',f"{self.sweep_id}")).rglob("*.h5"))
        # Sort them by seed
        h5files = sorted(h5files, key = lambda x: int(str(x).split('seed_')[1].split('/',1)[0]) if 'seed' in str(x) else str(x))
        # Read h5 data
        local_coords,global_coords,data_vars = self.read_h5_data(h5files,slice_samples=slice_samples)
        # Convert set to list
        local_coords = {k:np.array(
                            list(v),
                            dtype=TORCH_TO_NUMPY_DTYPE[COORDINATES_DTYPES[k]]
                        ) for k,v in local_coords.items()}
        global_coords = {k:np.array(
                            list(v),
                            dtype=TORCH_TO_NUMPY_DTYPE[COORDINATES_DTYPES[k]]
                        ) for k,v in global_coords.items()}
        
        # self.logger.progress('Creating xarray dataset(s)')
        
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
            self.logger.debug('instantiate sim')
            # Instantiate ct
            sim = instantiate_sim(
                config = self.config,
                logger=self.logger,
                **{input:self.get_sample(input) for input in sim_model.REQUIRED_INPUTS}
            )
            self.logger.debug('groupby apply intensity function')
            # CAUTION: Sweep coordinates must be unidimensional here!!!
            samples = sim.log_intensity(
                grand_total = torch.tensor(table_total,dtype=int32),
                torch = False,
                **{output:self.get_sample(output) for output in sim_model.REQUIRED_OUTPUTS}
            )
            # Create new dataset
            samples = samples.rename('intensity')
            # Exponentiate
            self.logger.debug('groupby exponentiate intensity')
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
            samples = torch.clone(
                getattr(self.inputs.data,sample_name).to(
                    INPUT_TYPES[sample_name]
                )
            )

        else:
            if not hasattr(self.data,sample_name):
                raise Exception(f"{sample_name} not found in output data [{','.join(list(vars(self.data).keys()))}]")
            
            # Get xarray
            samples = getattr(self.data,sample_name)
            
            # Find iteration coordinates
            iter_coords = [x for x in samples.dims if x in ['iter','seed']]
            
            # Find sweep coordinates that are not iteration-related
            sweep_coords = [d for d in samples.dims if d not in (list(CORE_COORDINATES_DTYPES.keys()))]

            if len(sweep_coords) > 0:
                # Stack all non-core coordinates into new coordinate
                samples = samples.stack(id=tuplize(iter_coords),sweep=tuplize(sweep_coords))
            else:
                samples = samples.stack(id=tuplize(iter_coords))

            # If parameter is beta, scale it by bmax
            if sample_name == 'beta' and self.intensity_model_class == 'spatial_interaction_model':
                samples *= self.config.settings[self.intensity_model_class]['parameters']['bmax']
        
        return samples

    def load_geometry(self,geometry_filename,default_crs:str='epsg:27700'):
        # Load geometry from file
        geometry = gpd.read_file(geometry_filename)
        geometry = geometry.set_crs(default_crs,allow_override=True)
        
        return geometry


    def create_filename(self,sample=None):
        # Decide on filename
        if (sample is None) or (not isinstance(sample,str)):
            filename = f"{','.join(self.settings['sample'])}"
        else:
            filename = f"{sample}"
        if 'statistic' in list(self.settings.keys()):
            filename += f"_{','.join([str(stat) for stat in list(flatten(self.settings['statistic'][0]))])}"
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

    def compute_sample_statistics(self,data,sample_name,statistic,**kwargs):
        # print('compute_sample_statistics',sample_name,type(data),statistic,axis)
        if statistic is None or statistic.lower() == '' or 'sample' in statistic.lower() or len(kwargs.get('dim',[])) == 0:
            return data
        
        elif not sample_name in list(OUTPUT_TYPES.keys()):
            return deep_call(
                data,
                f".{statistic}(dim)",
                data,
                dim=kwargs['dim']
            )
        
        elif statistic.lower() == 'signedmean' and \
            sample_name in list(OUTPUT_TYPES.keys()):
            if sample_name in list(INTENSITY_TYPES.keys())+['intensity'] and hasattr(self.data,'sign'):
                signs = self.get_sample('sign')
                print(signs)
                # Compute moments
                return ( torch.einsum('nk,n...->k...',signs.float(),data.float()) / torch.sum(torch.ravel(signs.float())))
            else:
                return self.compute_sample_statistics(data,sample_name,'mean',dim=kwargs['dim'])
       
        elif (statistic.lower() == 'signedvariance' or statistic.lower() == 'signedvar') and \
            sample_name in list(OUTPUT_TYPES.keys()):

            if sample_name in list(INTENSITY_TYPES.keys())+['intensity']:
                # Compute mean
                samples_mean = self.compute_sample_statistics(data,sample_name,'signedmean',**kwargs)
                # Compute squared mean
                signs = self.get_sample('sign')
                samples_squared_mean = np.einsum('nk,n...->k...',signs,torch.pow(data.float(),2)) / torch.sum(torch.ravel(signs.float()))
                # Compute intensity variance
                return (samples_squared_mean.float() - torch.pow(samples_mean.float(),2))
            else:
                return deep_call(
                    data,
                    f".var(dim)",
                    data,
                    dim=kwargs['dim']
                )
                # return self.compute_sample_statistics(data,sample_name,'var',**kwargs)
        
        elif statistic.lower() == 'error' and \
            sample_name in [param for param in list(OUTPUT_TYPES.keys()) if 'error' not in param]:
            # Apply error norm
            return apply_norm(
                tab=data,
                tab0=self.ground_truth_table,
                name=self.settings['norm'],
                **self.settings
            )
       
        else:
            return deep_call(
                data,
                f".{statistic}(dim)",
                data,
                dim=kwargs['dim']
            )
            # convert_string_to_torch_function(statistic)(data)
    
    def apply_sample_statistics(self,samples,sample_name,statistic_dims:Union[List,Tuple]=[]):
        # print('apply_sample_statistics',sample_name,statistic_axes)
        
        if isinstance(statistic_dims,Tuple):
            statistic_dims = [statistic_dims]
        sample_statistic = samples
        
        # For every collection of statistic-axes
        for stats,dims in statistic_dims:
            # print('stats',type(stats),stats)
            # print('dims',type(dims),dims)
            # Extract statistics and axes tuples applied to specific sample
            if isinstance(stats,str) and '|' in stats and len(stats):
                stats_list = [s for s in stats.split('|') if len(s) > 0]
            else:
                stats_list = [stats]
            if isinstance(dims,str) and '|' in dims:
                dims_list = [a if len(a) > 0 else None for a in dims.split('|')]
            else:
                dims_list = [dims]
            
            # If no stats applied, move on
            if len(stats_list) == 0:
                continue

            # print('stats_list',type(stats_list),stats_list)
            # print('dims_list',type(dims_list),dims_list)
            # Sequentially apply all the statistics along the corresponding axes tuple
            for i in range(len(stats_list)):
                stat,dim = stats_list[i],dims_list[i]
                
                # Skip computation if no statistic is provided
                if isinstance(stat,str) and len(stat) == 0:
                    continue
                # Convert axes to tuple of integers
                if isinstance(dim,(str)):
                    dim = list(map(str,dim.split('_')))
                elif hasattr(dim,'__len__'):
                    dim = list(map(str,dim))
                sample_statistic = self.compute_sample_statistics(
                                        data=sample_statistic,
                                        sample_name=sample_name,
                                        statistic=stat,
                                        dim=dim
                                    )
                # print(sample_statistic.shape)

        return sample_statistic

