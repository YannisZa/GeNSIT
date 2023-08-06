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

from glob import glob
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from itertools import product,chain
from argparse import Namespace
from datetime import datetime
from typing import Union,List,Tuple
from multiresticodm.inputs import Inputs
# from numba_progress import ProgressBar

import multiresticodm.probability_utils as ProbabilityUtils

from multiresticodm.utils import *
from multiresticodm.math_utils import *
from multiresticodm.config import Config
from multiresticodm.global_variables import *
from multiresticodm.contingency_table import instantiate_ct
from multiresticodm.spatial_interaction_model import *

OUTPUTS_MODULE = sys.modules[__name__]

class OutputSummary(object):

    def __init__(self, settings):
        # Setup logger
        self.logger = setup_logger(
            __name__,
            level=settings.get('logging_mode','info').upper(),
            log_to_file=True,
            log_to_console=True
        )
        # Get command line settings
        self.settings = settings
        # Instatiate list of experiments
        self.experiment_metadata = {}
        # Enable garbage collector
        gc.enable()
        # Update settings to parse statistics argument
        deep_updates(
            self.settings,
            unpack_statistics(self.settings)
        )
        # Setup experiments
        self.collect_experiment_metadata()
        # Compile metadata
        self.write_metadata_summaries()
     
    @classmethod
    def find_matching_output_folders(cls,__self__):
        if str_in_list('directories',__self__.settings.keys()) and len(__self__.settings['directories']) > 0:
            output_dirs = list(__self__.settings['directories'])

            for _dir in output_dirs:
                path_dir = Path(_dir)
                __self__.settings['dataset_name'] = [os.path.basename(path_dir.parents[0])]
                __self__.settings['output_directory'] = path_dir.parents[1].absolute()
                
        else:
            # Search metadata based on search parameters
            # Get output directory
            output_directory = __self__.settings['output_directory']
            # Get experiment title
            experiment_titles = __self__.settings['experiment_title']
            # Get dataset name
            dataset_names = __self__.settings['dataset_name']
            # Get type of experiment
            experiment_types = __self__.settings['experiment_type']

            # Get date
            if len(__self__.settings['dates']) <= 0:
                dates = ['']
            else:
                dates = list(__self__.settings.get('dates',['']))
                # dates = []
                # Read dates
                # for dt in __self__.settings['dates']:
                    # dates.append(dt)
                    # # Try different formats
                    # for format in DATE_FORMATS:
                    #     try:
                    #         dates.append(dt.strftime(format))
                    #         break
                    #     except:
                    #         pass
            # Grab all output directories
            folder_patterns = []
            for data_name in dataset_names:
                for exp_type in experiment_types:
                    for exp_title in experiment_titles:
                        for dt in dates:
                            folder_patterns.append(
                                os.path.join(
                                    output_directory,
                                    data_name,
                                    (f"{(exp_type+'.*') if len(exp_type) > 0 else exp_type}"+
                                    f"{('_'+exp_title+'.*') if len(exp_title) > 0 else ''}"+
                                    f"{(dt+'*') if len(dt) > 0 else ''}")
                                )
                            )
            # Combine them all into one pattern
            folder_patterns_re = "(" + ")|(".join(folder_patterns) + ")"
            # Get all output directories matching dataset name
            output_dirs = sorted(get_directories_and_immediate_subdirectories(output_directory))
            # Get all output dirs that match the pattern
            output_dirs = [output_folder for output_folder in output_dirs if re.match(folder_patterns_re,output_folder)]
            # Exclude those that are specified
            if len(__self__.settings['exclude']) > 0:
                output_dirs = [
                    output_folder for output_folder in output_dirs if __self__.settings['exclude'] not in output_folder
                ]
            # Sort by datetime
            output_dirs = sorted(output_dirs,key=(lambda dt: datetime.strptime(dt[-19:], "%d_%m_%Y_%H_%M_%S")))
            # If no directories found terminate
            if len(output_dirs) == 0 :
                __self__.logger.error(f'No directories found in {os.path.join(output_directory,"*")}')
                raise Exception('Cannot read outputs.')
        return output_dirs

    def collect_experiment_metadata(self):
        # Find matching directories
        output_dirs = self.find_matching_output_folders(self)
        for i,output_folder in tqdm(enumerate(output_dirs),total=len(output_dirs)):
            
            self.logger.info(f'Collecting metadata from {output_folder}')
            
            # Get name of folder
            folder_name = Path(output_folder).stem
            
            # Read metadata config
            experiment_metadata = read_json(os.path.join(output_folder,"config.json"))

            # Extract useful data
            useful_metadata = {}
            for key in self.settings['metadata_keys']:
                try:
                    # Get first instance of key
                    useful_metadata[key] = list(deep_get(key=key,value=experiment_metadata))[0]
                except:
                    self.logger.error(f"{key} not found in experiment metadata.")

            if 'sweeped_params_paths' in list(experiment_metadata.keys()) and len(experiment_metadata['sweeped_params_paths']) > 0:
                coordinate_slice = {}
                input_slice = []
                # Loop through key-value pairs used
                # to subset the output samples
                for param_tup in self.settings['slice_by']:
                    name,value = param_tup
                    # Loop through experiment's sweeped parameters
                    for key_path in experiment_metadata['sweeped_params_paths']:
                        # If there is a much between the two
                        if name == key_path[-1]:
                            # If is a coordinate add to the coordinate slice
                            # This slices the xarray created from the outputs samples
                            if SWEEPABLE_PARAMS[name]['is_coord']:
                                if name in list(coordinate_slice.keys()):
                                    coordinate_slice[name]['value'] = np.append(coordinate_slice[name]['value'],[value])
                                else:
                                    coordinate_slice[name] = {"path": key_path, "value": [value]}
                            # If is NOT a coordinate add to the input slice
                            # This slices the input data and requires reinstantiating outputs
                            else:
                                input_slice.append({"value":value,"key_path":key_path})
                if len(input_slice) == 0:
                    input_slice = [None]
            else:
                coordinate_slice = {}
                input_slice = [None]
            
            for path_value in input_slice:
                
                # Get outputs and unpack its statistics
                # This is the case where there are NO input slices provided
                if path_value is None:
                    outputs = Outputs(
                        config=output_folder,
                        settings=self.settings,
                        output_names=(list(self.settings['sample'])+['ground_truth_table']),
                        coordinate_slice=coordinate_slice
                    )
                # This is the case where there are SOME input slices provided
                else:
                    outputs = Outputs(
                        config=output_folder,
                        settings=self.settings,
                        output_names=(list(self.settings['sample'])+['ground_truth_table']),
                        coordinate_slice=coordinate_slice,
                        input_slice=path_value
                    )

                # Apply these metrics to the data 
                metric_data = self.apply_metrics(
                    experiment_id=output_folder,
                    outputs=outputs
                )
                for j in range(len(metric_data)):
                    metric_data[j]['folder'] = folder_name
                    for k,v in useful_metadata.items():
                        metric_data[j][k] = v
                
                # Store useful metadata
                if not str_in_list(output_folder,self.experiment_metadata.keys()):
                    self.experiment_metadata[output_folder] = metric_data
                else:
                    self.experiment_metadata[output_folder] = np.append(
                        self.experiment_metadata[output_folder],
                        metric_data,
                        axis=0
                    )

                print('\n')

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
                self.settings['output_directory'],
                dataset,
                'summaries'
            )
            makedir(output_directory)

            # Get filepath experiment filepath
            # date_strings = []
            # for dt in self.settings['dates']:
                # Try different formats
                # for format in DATE_FORMATS:
                #     try:
                #         date_strings.append(dt.strftime(format))
                #         break
                #     except:
                #         pass
            date_strings = '__'.join(self.settings['dates'])
            if str_in_list('directories',self.settings.keys()) and len(self.settings['directories']) > 0:
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
        
        metric_data = []

        for sample_name in self.settings['sample']:
            # Read samples
            try:
                assert hasattr(outputs.data,sample_name)
                samples = getattr(outputs.data,sample_name)
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
                # try:
                samples_summarised = outputs.apply_sample_statistics(samples,sample_name,sample_statistics_axes)
                # except Exception as e:
                #     self.logger.debug(traceback.format_exc())
                #     self.logger.error(f"samples {np.shape(samples)}, {samples.dtype}")
                #     self.logger.error(f"Applying statistic {' over axes '.join([str(s) for s in sample_statistics_axes])} \
                #                     for sample {sample_name} of experiment {experiment_id} failed")
                #     print('\n')
                #     continue
                if samples_summarised.dim() !=  samples.dim():
                    samples_summarised = samples_summarised.unsqueeze(dim=0)
                self.logger.debug(f"samples_summarised {np.shape(samples_summarised)}, {samples_summarised.dtype}")
                
                # Get shape of samples
                N = np.shape(samples_summarised)[0]
                dims = np.shape(samples_summarised)[1:]

                for metric in self.settings['metric']:
                    metric_shape = convert_string_to_numpy_shape(METRICS[metric]['shape'],N=N,dims=dims)
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
                                        tab0=metric_kwargs['tab0'],
                                        kwargs=metric_kwargs['kwargs']
                                    )
                                else:
                                    samples_metric = deepcopy(samples_summarised)
                                # Reshape samples metric
                                samples_metric = samples_metric.reshape(metric_shape)
                            except Exception as e:
                                self.logger.debug(traceback.format_exc())
                                self.logger.error(f"samples_summarised {np.shape(samples_summarised)}, {samples_summarised.dtype}")
                                self.logger.error(f"tab0 {np.shape(metric_kwargs['tab0'])}, {metric_kwargs['tab0'].dtype}")
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
                            # Make sure metric summarised is not multidimensional
                            if metric_summarised.dim() > 1:
                                self.logger.warning(f"{'|'.join([stringify_statistic(_stat_ax) for _stat_ax in sample_statistics_axes])}>"+
                                                    f"{sample_name}>"+
                                                    f"{metric}>"+
                                                    f"{attribute_settings_string}>"+
                                                    f"{'|'.join([stringify_statistic(_stat_ax) for _stat_ax in metric_statistics_axes])} is multidimensional and will not be written to file")
                                self.logger.warning(f"Shape of summarised metric is {np.shape(metric_summarised)}")
                                continue
                            else:
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
                        try:
                            if metric != 'none':
                                samples_metric = globals()[metric](
                                    tab=samples_summarised,
                                    tab0=metric_kwargs['tab0'],
                                    kwargs=metric_kwargs['kwargs']
                                )
                            else:
                                samples_metric = deepcopy(samples_summarised)
                        except Exception as e:
                            self.logger.debug(traceback.format_exc())
                            self.logger.error(f"samples_summarised {np.shape(samples_summarised)}, {samples_summarised.dtype}")
                            self.logger.error(f"tab0 {metric_kwargs['tab0'].shape}, {metric_kwargs['tab0'].dtype}")
                            self.logger.error(f'Applying metric {metric} over sample {sample_name} for experiment {experiment_id} failed')
                            print('\n')
                            continue
                        # Reshape samples metric
                        try:
                            samples_metric = samples_metric.reshape(metric_shape)
                        except Exception as e:
                            self.logger.debug(traceback.format_exc())
                            self.logger.error(f"samples_summarised {np.shape(samples_summarised)}, {samples_summarised.dtype}")
                            self.logger.error(f"tab0 {np.shape(metric_kwargs['tab0'])}, {metric_kwargs['tab0'].dtype}")
                            self.logger.error(f'Applying metric {metric} over sample {sample_name} for experiment {experiment_id} failed')
                            print('\n')
                            continue
                        self.logger.debug(f"Samples metric is {np.shape(samples_metric)}")
                        
                        # Apply statistics after metric
                        try:
                            metric_summarised = outputs.apply_sample_statistics(samples_metric,metric,metric_statistics_axes)
                        except Exception as e:
                            self.logger.debug(traceback.format_exc())
                            self.logger.error(f"Shape of metric is {np.shape(samples_metric)}")
                            self.logger.error(f"Applying statistic {'|'.join([stringify_statistic(_stat_ax) for _stat_ax in metric_statistics_axes])}>" + \
                                              f"over metric {metric} and sample {sample_name} for experiment {experiment_id} failed")
                            print('\n')
                            continue
                        self.logger.debug(f"Summarised metric is {np.shape(metric_summarised)}")
                        # Squeeze output
                        metric_summarised = torch.squeeze(metric_summarised)
                        if metric_summarised.dim() > 1:
                            self.logger.debug(traceback.format_exc())
                            # Make sure metric summarised is not multidimensional
                            self.logger.warning(f"{'|'.join([stringify_statistic(_stat_ax) for _stat_ax in sample_statistics_axes])}>"+
                                                f"{sample_name}>"+
                                                f"{metric}>"+
                                                f"{'|'.join([stringify_statistic(_stat_ax) for _stat_ax in metric_statistics_axes])} is multidimensional \
                                                and will not be written to file")
                            self.logger.warning(f"Shape of summarised metric is {np.shape(metric_summarised)}")
                            print('\n')
                            continue
                        else:
                            # Add to data records
                            metric_data_keys = [
                                "sample_statistic",
                                "sample_name",
                                "metric",
                                "metric_statistic",
                                "value"
                            ]
                            metric_data_vals = [
                                f"{'|'.join([stringify_statistic(_stat_ax) for _stat_ax in sample_statistics_axes])}",
                                f"{sample_name}",
                                f"{metric}",
                                f"{'|'.join([stringify_statistic(_stat_ax) for _stat_ax in metric_statistics_axes])}>",
                                f"{metric_summarised}"
                            ]                        
                            metric_data.append(dict(zip(metric_data_keys,metric_data_vals)))

                        self.logger.debug(f"Summarised metric is updated to {np.shape(metric_summarised)}")
    
        return metric_data
    
    def update_metric_arguments(self,metric,outputs,settings):        
        # Initialise metric arguments
        metric_arguments = {}
        settings_copy = deepcopy(settings)
        if metric.lower() == 'shannon_entropy':
            dummy_config = Namespace(**{'settings':outputs.config})
            ct = instantiate_ct(table=None,config=dummy_config,log_to_console=False) 
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
            metric_arguments['tab0'] = np.log(outputs.load_samples('intensity',slice_samples=True),dtype='float32')
        else:
            # Pass ground truth table as argument
            metric_arguments['tab0'] = outputs.ground_truth_table
        
        # Pass standard metric arguments
        metric_arguments['kwargs'] = settings_copy

        return metric_arguments

class Outputs(object):

    def __init__(self,
                 config, 
                 module:str=__name__,
                 settings:dict={}, 
                 output_names:list=['ground_truth_table'], 
                 slice_samples:bool=True,
                 coordinate_slice:dict={},
                 input_slice:dict={},
                 **kwargs):
        # Setup logger
        self.logger = setup_logger(
            module,
            level=config.level if hasattr(config,'level') else kwargs.get('level','INFO'),
            log_to_file=kwargs.get('log_to_file',True),
            log_to_console=kwargs.get('log_to_console',True),
        )

        # Sample names must be a subset of all data names
        try:
            assert set(output_names).issubset(set(DATA_TYPES.keys()))
        except Exception as e:
            self.logger.error('Some sample names provided are not recognized')
            self.logger.error(','.join(output_names))
            self.logger.debug(traceback.format_exc())
            raise Exception('Cannot load outputs.')

        # Store settings
        self.settings = settings
        # Store coordinate slice
        self.coordinate_slice = coordinate_slice
        # Create semi-private xarray data 
        self._data = Dataset()
        # Create public xarray data 
        self.data = Dataset()
        # Enable garbage collector
        gc.enable()
        if isinstance(config,str):
            # Store experiment id
            self.experiment_id = os.path.basename(os.path.normpath(config))
            
            # Load metadata
            assert os.path.exists(config)
            self.config = Config(
                path=os.path.join(config,"config.json"),
                level='info'
            )

            # Update config based on slice of coordinate-like sweeped params
            # affecting only the outputs of the model
            if self.coordinate_slice:
                for param in self.coordinate_slice.keys():
                    self.config.path_set(
                        settings = self.config.settings,
                        value = self.coordinate_slice[param]['value'], 
                        path = self.coordinate_slice[param]['path']
                    )
            # Update config based on slice of NON-coordinate-like sweeped params
            # affecting only the inputs of the model
            if input_slice:
                self.config.path_set(
                    settings = self.config.settings,
                    value = input_slice['value'], 
                    path = input_slice['path']
                )
            
            # Get intensity model class
            self.intensity_model_class = [k for k in self.config.keys() if k in INTENSITY_MODELS and isinstance(self.config[k],dict)][0]
            
            # Define config experiment path to directory
            self.outputs_path = config
            assert str_in_list('input_path',list(self.config['inputs'].keys()))
            self.inputs_path = self.config['inputs'].get('input_path',None)

            # Import all input data
            self.inputs = Inputs(
                config = self.config,
                synthetic_data = False,
            )
            # Convert all inputs to tensors
            self.inputs.pass_to_device()

            # Try to load ground truth table
            self.ground_truth_table = None
            self.settings['table_total'] = self.settings.get('table_total',1)
            if 'ground_truth_table' in output_names:
                # Try reading it from settings path
                try:
                    self.ground_truth_table = np.loadtxt(
                        os.path.join(
                            self.config['inputs']['dataset'],
                            self.settings['table']
                        )
                    )
                    # Convert to tensor
                    self.ground_truth_table = torch.tensor(self.ground_truth_table,dtype=torch.int32)
                except:
                    # Try reading it from inputs
                    try:
                        self.ground_truth_table = self.inputs.table
                    except:
                        pass
            
            # Try to get table total (number of agents)
            if self.ground_truth_table is not None:
                # Remove it from sample names
                output_names.remove('ground_truth_table')
                # Reshape it
                self.ground_truth_table = self.ground_truth_table.reshape((1,*self.ground_truth_table.shape))
                # Extract metadata
                self.settings['table_total'] = self.ground_truth_table.ravel().sum()
                self.settings['dims'] = list(np.shape(self.ground_truth_table))
                self.logger.info(f'Ground truth table loaded')

            # Load output h5 file to xarrays
            self.load_h5_data(config,coordinate_slice=self.coordinate_slice)
            
            # Try to load all output data
            for sample_name in output_names:
                # try:
                setattr(
                    self.data,
                    sample_name, 
                    self.get_sample(
                        sample_name,
                        slice_samples = slice_samples
                    )
                )

                if sample_name == 'table' and self.settings['table_total'] != 1:
                    self.settings['table_total'] = list(torch.sum(torch.ravel(self.data.table)))[0]
                self.logger.info(f'Sample {sample_name} loaded with shape {list(getattr(self.data,sample_name).shape)}')
                # except:
                #     self.logger.debug(traceback.format_exc())
                #     self.logger.warning(f'Sample {sample_name} could not be loaded')
                    # sys.exit()

            if self.settings['table_total'] == 0:
                self.logger.warning('Ground truth missing')
            
            
        elif isinstance(config,Config):
            # Store config
            self.config = config
            
            # Remove unnecessary data
            # for attr in ['inputs','harris_wilson_nn','sim_mcmc','sim','ct']:
            #     if hasattr(self.experiment,attr):
            #         safe_delete(getattr(self.experiment,attr))
            # gc.collect()
            
            # Get intensity model class
            self.intensity_model_class = [k for k in self.config.keys() if k in INTENSITY_MODELS and isinstance(self.config[k],dict)][0]
            
            # Update experiment id
            self.experiment_id = self.update_experiment_directory_id(kwargs.get('experiment_id',None))
            
            # Define output experiment path to directory
            self.outputs_path = os.path.join(
                    self.config['outputs']['directory'],
                    self.config['experiment_data'],
                    self.experiment_id
            )
    
            # Name output sample directory according 
            # to sweep params (if they are provided)
            sweep_params = kwargs.get('sweep_params',{})
            self.sweep_id = ''
            if len(sweep_params) > 0 and isinstance(sweep_params,dict):
                self.sweep_id = os.path.join(*[str(k)+"_"+str(v) for k,v in sweep_params.items()])

            # Create output directories if necessary
            self.create_output_subdirectories(sweep_id=self.sweep_id)
        
        else:
            raise Exception(f'Config {config} of type {type(config)} not recognised.')

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
        
        if not str_in_list('experiment_title',self.config['outputs'].keys()):
            self.config['outputs']['experiment_title'] = ""

        if sweep_experiment_id is None:
            if str_in_list(self.config['experiment_type'].lower(),['tablesummariesmcmcconvergence','table_mcmc_convergence']):
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
            makedir(os.path.join(self.outputs_path,'sample_derivatives'))

    def write_log(self,logger):
        if isinstance(logger,logging.Logger):
            for i,hand in enumerate(logger.handlers):
                if isinstance(hand,logging.FileHandler):
                    # Do not write to temporary filename
                    if hand.filename != 'temp.log':
                        # Close handler
                        logger.handlers[i].flush()
                        logger.handlers[i].close()
        else:
            raise Exception(f'Cannot write outputs of invalid type logger {type(logger)}')
        

    def write_metadata(self,dir_path:str,filename:str) -> None:
        # settings_copy = deepcopy(self.config.settings)
        # settings_copy = deep_apply(settings_copy, type)
        # print(settings_copy)
        # Define filepath
        filepath = os.path.join(self.outputs_path,dir_path,f"{filename.split('.')[0]}.json")
        # print('writing metadata',filepath)
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
        if hasattr(self,'config') and hasattr(self.config,'settings'):
            export_samples = list(deep_get(key='export_samples',value=self.config.settings))
            export_metadata = list(deep_get(key='export_metadata',value=self.config.settings))
            export_samples = export_samples[0] if len(export_samples) > 0 else True
            export_metadata = export_metadata[0] if len(export_metadata) > 0 else True
            
            # Write to file
            if export_samples:
                self.logger.note(f"Creating output file at:\n        {self.outputs_path}")
                self.h5file = h5.File(os.path.join(self.outputs_path,'samples',f"{self.sweep_id}","data.h5"), mode="w")
                self.h5group = self.h5file.create_group(self.experiment_id)
                # Store sweep configurations as attributes 
                self.h5group.attrs.create("sweep_params",list(sweep_params.keys()))
                self.h5group.attrs.create("sweep_values",list(sweep_params.values()))
                # Update log filename
                if isinstance(self.logger,logging.Logger):
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

        # Apply burnin and thinning
        samples = samples.isel(iter=slice(burnin,None,thinning))

        # Get total number of samples
        N = self.settings.get('N',samples['iter'].shape[0])
        N = min(N,samples['iter'].shape[0])
        
        # Apply stop
        samples = samples.isel(iter=slice(None,N,None))

        return samples

    def load_h5_data(self,output_path,coordinate_slice:dict={}):
        self.logger.info('Loading h5 data into xarrays...')

        # Get all h5 files
        h5files = list(Path(os.path.join(output_path,'samples')).rglob("*.h5"))
        # Sort them by seed
        h5files = sorted(h5files, key = lambda x: int(str(x).split('seed_')[1].split('/',1)[0]) if 'seed' in str(x) else str(x))
        # Store data attributes for xarray
        coords,data_vars,data_variables = {},{},{}
        # print(len(h5files))
        # Get each file and add it to the new dataset
        for filename in h5files:
            
            with h5.File(filename) as h5data:

                # Collect group-level attributes as coordinates
                # Group coordinates are file-dependent
                if 'sweep_params' in list(h5data[self.experiment_id].attrs.keys()) and \
                    'sweep_values' in list(h5data[self.experiment_id].attrs.keys()):
                
                    # Loop through each sweep parameters and add it as a coordinate
                    for (k,v) in zip(h5data[self.experiment_id].attrs['sweep_params'],
                                h5data[self.experiment_id].attrs['sweep_values']):
                        if k in list(coords.keys()) and len(coords) > 0:
                            coords[k].add(v)
                        else:
                            coords[k] = {v}
                
                # Store dataset
                for sample_name,sample_data in h5data[self.experiment_id].items():
                    if sample_name in list(data_vars.keys()):
                        data_vars[sample_name] = np.append(
                            data_vars[sample_name],
                            np.array([sample_data[:]]),
                            axis=0
                        )
                    else:
                        data_vars[sample_name] = np.array([sample_data[:]])
        
        # Convert set to list
        coords = {k:np.array(list(v)) for k,v in coords.items()}

        # print({k:np.shape(v) for k,v in data_vars.items()})
        # Create an xarray dataset for each sample
        for sample_name,sample_data in data_vars.items():

            # Get data dims
            dims = np.shape(sample_data)[1:]
            coordinates = {}
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
            # print('coordinates',coordinates.keys())
            # print('coords',coords.keys())
            all_coordinates = {**{k:v for k,v in coordinates.items()},**{k:v for k,v in coords.items()}}
            
            # For each coordinate name
            # get data variable
            # print(sample_name)
            # print('coordinates')
            # print({k:np.shape(v) for k,v in all_coordinates.items()})
            # print('data')
            # print(np.shape(sample_data.reshape(*[len(val) for val in all_coordinates.values()])))
            # print(list(all_coordinates.keys()))
            # if len(coordinates.keys()) > 0:
            data_variables[sample_name] = (
                list(all_coordinates.keys()),
                sample_data.reshape(tuple([len(val) for val in all_coordinates.values()]))
            )
            # else:
                # data_variables[sample_name] = sample_data
            
            # Create xarray dataset
            xr_data = xr.Dataset(
                data_vars = {sample_name:data_variables[sample_name]},
                coords = all_coordinates,
                attrs = dict(
                    experiment_id = self.experiment_id
                ),
            )
            # Slice according to coordinate slice
            if len(coordinate_slice) > 0:
                xr_data = xr_data.isel(**coordinate_slice)
            
            # Store dataset
            setattr(self._data,sample_name,xr_data)
            
            
            

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
                assert hasattr(self._data,output)
            except:
                available = False
                self.logger.error(f"Sample {sample_name} requires output {output} which does not exist in {','.join(vars(self._data))}")
        return available

    def get_sample(self,sample_name:str,slice_samples:bool=True):

        if sample_name == 'intensity':
            # Get sim model 
            sim_model = globals()[self.config.settings['spatial_interaction_model']['sim_type']+'SIM']
            # Check that required data is available
            self.check_data_availability(
                sample_name=sample_name,
                input_names=sim_model.REQUIRED_INPUTS,
                output_names=sim_model.REQUIRED_OUTPUTS,
            )

            # Prepare input arguments 
            data = {}
            for input in sim_model.REQUIRED_INPUTS:
                data[input] = self.get_sample(input,slice_samples)

            # Compute intensities for all samples
            table_total = self.settings.get('table_total') if self.settings.get('table_total',-1.0) > 0 else 1.0

            # Instantiate ct
            sim = instantiate_sim(
                sim_type = next(deep_get(key='sim_type',value=self.config.settings), None),
            **data
            )
            
            data = {}
            # Prepare output arguments
            for output in sim_model.REQUIRED_OUTPUTS:
                data[output] = torch.tensor(
                    self.get_sample(output,slice_samples),
                    dtype=NUMPY_TO_TORCH_DTYPE[OUTPUT_TYPES[output]]
                )

            # Compute log intensity function
            samples = sim.log_intensity(
                grand_total=torch.tensor(table_total,dtype=torch.int32),
                **data
            )

            # Exponentiate
            samples = torch.exp(samples).to(dtype=torch.float32)

        elif sample_name.endswith("__error"):
            # Load all samples
            samples = self.get_sample(sample_name.replace("__error",""),slice_samples)
            # Make sure you have ground truth
            try:
                assert self.ground_truth_table is not None
            except:
                self.logger.error('Ground truth table missing. Sample error cannot be computed.')
                raise
        
        elif sample_name == 'ground_truth_table':
            # Get config and sim
            dummy_config = Namespace(**{'settings':self.config})
            ct = instantiate_ct(
                table=None,
                config=dummy_config,
                log_to_console=False
            )
            samples = torch.tensor(ct.table).int().reshape((1,*ct.dims))
        
        elif str_in_list(sample_name, INPUT_TYPES.keys()):
            # Get sim model 
            sim_model = globals()[self.config.settings['spatial_interaction_model']['sim_type']+'SIM']
            self.check_data_availability(
                sample_name=sample_name,
                input_names=sim_model.REQUIRED_INPUTS
            )
            # Get samples and cast them to appropriate type
            samples = torch.clone(
                getattr(self.inputs.data,sample_name).to(
                    dtype=NUMPY_TO_TORCH_DTYPE[INPUT_TYPES[sample_name]]
                )
            )

        else:
            if not hasattr(self._data,sample_name):
                raise Exception(f"{sample_name} not found in output data {','.join(vars(self._data).keys())}")
            
            # Get xarray
            xr_samples = getattr(self._data,sample_name)#[sample_name]
            # print(sample_name,samples.shape)
            print('first',xr_samples.data_vars)
            # Apply burning, thinning and trimming
            if slice_samples:
                xr_samples = self.slice_sample_iterations(xr_samples)
            # print(sample_name,samples.shape)
            # Remove non-core coordinates
            noncore_coords = list(set(xr_samples.dims) - set(XARRAY_SCHEMA[sample_name]['coords']))
            # Stack all non-core coordinates into new coordinate
            xr_samples = xr_samples.stack(new_iter=(noncore_coords+['iter']))
            # Get number of samples
            N = xr_samples['new_iter'].shape[0]
            # Convert xarray DataSet to DataArray
            xr_samples = xr_samples[sample_name]
            # Convert to numpy
            samples = xr_samples.values
            # Cast to specific data type
            samples = samples.astype(dtype=OUTPUT_TYPES[sample_name])
            # Reshape
            dims = {"N":N,"I":self.inputs.data.dims[0],"J":self.inputs.data.dims[1]}
            # print(sample_name,samples.shape)
            samples = samples.reshape(*[string_to_numeric(var) if var.isnumeric() else dims.get(var,None) for var in XARRAY_SCHEMA[sample_name]['new_shape']])
            # print(sample_name,samples.shape)
            # print('\n')

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
        if str_in_list('statistic',self.settings.keys()):
            filename += f"_{','.join([str(stat) for stat in list(flatten(self.settings['statistic'][0]))])}"
        if str_in_list('table_dim',self.config.keys()):
            filename += f"_{self.config['table_dim']}"
        if str_in_list('table_total',self.config.keys()):
            filename += f"_{self.config['table_total']}"
        if str_in_list('type',self.config.keys()) and len(self.config['type']) > 0:
            filename += f"_{self.config['type']}"
        if str_in_list('experiment_title',self.settings.keys()) and len(self.settings['experiment_title']) > 0:
            filename += f"_{self.settings['experiment_title']}"
        if str_in_list('viz_type',self.settings.keys()):
            filename += f"_{self.settings['viz_type']}"
        if str_in_list('burnin',self.settings.keys()):
            filename += f"_burnin{self.settings['burnin']}" 
        if str_in_list('thinning',self.settings.keys()):
            filename += f"_thinning{self.settings['thinning']}"
        if str_in_list('N',self.settings.keys()):
            filename += f"_N{self.settings['N']}"
        # filename += f"_N{self.config['mcmc']['N']}"
        return filename

    def compute_sample_statistics(self,data,sample_name,statistic,axis:int=0):
        # print('compute_sample_statistics',sample_name,statistic,axis)
        if statistic is None or statistic.lower() == '' or 'sample' in statistic.lower():
            return data
        
        elif not str_in_list(sample_name,OUTPUT_TYPES.keys()):
            return convert_string_to_torch_function(statistic)(data.float(),dim=axis).to(dtype=torch.float32)
        
        elif statistic.lower() == 'signedmean' and \
            str_in_list(sample_name,OUTPUT_TYPES.keys()): 
            if str_in_list(sample_name,INTENSITY_TYPES.keys()) \
                and hasattr(self.data,'sign'):
                signs = self.data.sign.unsqueeze(1)
                # Compute moments
                return ( torch.einsum('nk,n...->k...',signs.float(),data.float()) / torch.sum(torch.ravel(signs.float()))).to(dtype=torch.float32)
            else:
                return self.compute_sample_statistics(data,sample_name,'mean',axis)
       
        elif (statistic.lower() == 'signedvariance' or statistic.lower() == 'signedvar') and \
            str_in_list(sample_name,OUTPUT_TYPES.keys()):

            if str_in_list(sample_name,INTENSITY_TYPES.keys()) \
                and hasattr(self.data,'sign'):
                signs = self.data.sign.unsqueeze(1)
                # Compute intensity variance
                samples_mean = self.compute_sample_statistics(data,sample_name,'signedmean',axis)
                samples_squared_mean = np.einsum('nk,n...->k...',signs,torch.pow(data.float(),2)) / torch.sum(torch.ravel(signs.float()))
                return (samples_squared_mean.float() - torch.pow(samples_mean.float(),2)).to(dtype=torch.float32)
            else:
                return self.compute_sample_statistics(data,sample_name,'var',axis)
        
        elif statistic.lower() == 'error' and \
            str_in_list(sample_name,[param for param in OUTPUT_TYPES.keys() if 'error' not in param]):
            # Apply error norm
            return apply_norm(
                tab=data,
                tab0=self.ground_truth_table,
                name=self.settings['norm'],
                **self.settings
            )
       
        else:
            return convert_string_to_torch_function(statistic)(data.float(),dim=axis).to(dtype=torch.float32)

    def apply_sample_statistics(self,samples,sample_name,statistic_axes:Union[List,Tuple]=[]):
        # print('apply_sample_statistics',sample_name,statistic_axes)
        
        if isinstance(statistic_axes,Tuple):
            statistic_axes = [statistic_axes]
        sample_statistic = samples
        
        # For every collection of statistic-axes
        for stats,axes in statistic_axes:
            # print('stats',type(stats),stats)
            # print('axes',type(axes),axes)
            # Extract statistics and axes tuples applied to specific sample
            if isinstance(stats,str) and '|' in stats and len(stats):
                stats_list = [s for s in stats.split('|') if len(s) > 0]
            else:
                stats_list = [stats]
            if isinstance(axes,str) and '|' in axes:
                axes_list = [a if len(a) > 0 else None for a in axes.split('|')]
            else:
                axes_list = [axes]
            
            # If no stats applied, move on
            if len(stats_list) == 0:
                continue

            # print('stats_list',type(stats_list),stats_list)
            # print('axes_list',type(axes_list),axes_list)
            # Sequentially apply all the statistics along the corresponding axes tuple
            for i in range(len(stats_list)):
                stat,ax = stats_list[i],axes_list[i]
                
                # Skip computation if no statistic is provided
                if isinstance(stat,str) and len(stat) == 0:
                    continue

                # print('stat',type(stat),stat)
                # print('ax',type(ax),ax)
                # Convert axes to tuple of integers
                if isinstance(ax,(str)):
                    ax = list(map(int,ax.split('_')))
                elif hasattr(ax,'__len__'):
                    ax = list(map(int,ax))

                sample_statistic = self.compute_sample_statistics(
                                        data=sample_statistic,
                                        sample_name=sample_name,
                                        statistic=stat,
                                        axis=tuplize(ax)
                                    )
                # print(sample_statistic.shape)

        return sample_statistic