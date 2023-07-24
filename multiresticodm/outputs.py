import os
import re
import gc
import sys
import logging
import traceback
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
# from numba_progress import ProgressBar

import multiresticodm.probability_utils as ProbabilityUtils

from multiresticodm.utils import *
from multiresticodm.math_utils import *
from multiresticodm.global_variables import *
from multiresticodm.contingency_table import instantiate_ct
from multiresticodm.spatial_interaction_model import instantiate_sim

OUTPUTS_MODULE = sys.modules[__name__]

class OutputSummary(object):

    def __init__(self, settings):
        # Import logger
        self.logger = logging.getLogger(__name__)
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)
        # Get contingency table
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
                                    (f"exp.*{(exp_type+'.*') if len(exp_type) > 0 else exp_type}"+
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
            # Read metadata
            experiment_metadata = read_json(os.path.join(output_folder,f"{folder_name}_metadata.json"))

            # Extract useful data
            useful_metadata = {}
            for key in self.settings['metadata_keys']:
                try:
                    # Get first instance of key
                    useful_metadata[key] = list(deep_get(key=key,value=experiment_metadata))[0]
                except:
                    self.logger.error(f'No "{key}" found in experiment metadata)')
            # Get outputs and unpack its statistics
            outputs = Outputs(output_folder,self.settings,(list(self.settings['sample'])+['ground_truth_table']))

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
            samples = None
            try:
                samples = outputs.experiment.results.get(sample_name,None)
                assert samples is not None
            except Exception as e:
                self.logger.error(f'Experiment {experiment_id} does not have sample {sample_name}')
                continue
            self.logger.debug(f"samples {np.shape(samples)}, {samples.dtype}")
            for statistics in outputs.settings['statistic']:
                # Unpack sample and metric statistics
                sample_statistics_axes = statistics[0]
                metric_statistics_axes = statistics[1]
                # Compute statistic before applying metric
                samples_summarised = None
                try:
                    samples_summarised = outputs.apply_sample_statistics(samples,sample_name,sample_statistics_axes)
                except Exception as e:
                    self.logger.debug(traceback.format_exc())
                    self.logger.error(f"samples {np.shape(samples)}, {samples.dtype}")
                    self.logger.error(f"Applying statistic {' over axes '.join([str(s) for s in sample_statistics_axes])} \
                                    for sample {sample_name} of experiment {experiment_id} failed")
                    print('\n')
                    continue
                if not np.array_equal(samples_summarised.ndim,samples.ndim):
                    samples_summarised = samples_summarised[np.newaxis,:]
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
                            print(sample_name,metric,samples_metric)
                            self.logger.debug(f"Samples metric is {np.shape(samples_metric)}")
                            # Apply statistics after metric
                            try:
                                self.settings['axis'] = METRICS[metric]['apply_axis']
                                metric_summarised = outputs.apply_sample_statistics(
                                                        samples=samples_metric,
                                                        sample_name='metric',
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
                            if np.size(metric_summarised) > 1:
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
                            # safe_delete_and_clean([samples_metric,metric_summarised])
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
                            self.logger.error(f"tab0 {np.shape(metric_kwargs['tab0'])}, {metric_kwargs['tab0'].dtype}")
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
                            metric_summarised = outputs.apply_sample_statistics(samples_metric,'metric',metric_statistics_axes)
                        except Exception as e:
                            self.logger.debug(traceback.format_exc())
                            self.logger.error(f"Shape of metric is {np.shape(samples_metric)}")
                            self.logger.error(f"Applying statistic {'|'.join([stringify_statistic(_stat_ax) for _stat_ax in metric_statistics_axes])}>" + \
                                              f"over metric {metric} and sample {sample_name} for experiment {experiment_id} failed")
                            print('\n')
                            continue
                        self.logger.debug(f"Summarised metric is {np.shape(metric_summarised)}")
                        # Squeeze output
                        metric_summarised = np.squeeze(metric_summarised)
                        if np.size(metric_summarised) > 1:
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
                        # safe_delete_and_clean([samples_metric,metric_summarised])
            #     safe_delete_and_clean(samples_summarised)
            # safe_delete_and_clean(samples)
    
        return metric_data
    
    def update_metric_arguments(self,metric,outputs,settings):        
        # Initialise metric arguments
        metric_arguments = {}
        settings_copy = deepcopy(settings)
        if metric.lower() == 'shannon_entropy':
            dummy_config = Namespace(**{'settings':outputs.experiment.subconfig})
            ct = instantiate_ct(table=None,config=dummy_config,disable_logger=True) 
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
                 experiment, 
                 settings:dict={}, 
                 sample_names:list=['ground_truth_table'], 
                 slice_samples:bool=True,
                 disable_logger:bool=False):
        # Import logger
        self.logger = logging.getLogger(__name__)
        self.logger.disabled = disable_logger

        # Sample names must be a subset of all data names
        try:
            assert set(sample_names).issubset(set(DATA_TYPES.keys()))
        except Exception as e:
            self.logger.error('Some sample names provided are not recognized')
            self.logger.error(','.join(sample_names))
            self.logger.debug(traceback.format_exc())
            raise Exception('Cannot load outputs.')

        # Store settings
        self.settings = settings
        # Enable garbage collector
        gc.enable()
        if isinstance(experiment,str):
            # Load metadata
            assert os.path.exists(experiment)
            self.experiment_id = os.path.basename(os.path.normpath(experiment))
            metadata = read_json(os.path.join(experiment,self.experiment_id+"_metadata.json"))
            # Store experiment id
            self.experiment = Namespace(**{'subconfig':metadata,'results':{}})
            # Get intensity model class
            self.intensity_model_class = [input_name for input_name in self.experiment.subconfig.keys() if input_name != 'contingency_table' and isinstance(self.experiment.subconfig[input_name],dict)][0]
            # Define output experiment path to directory
            self.outputs_path = experiment
            assert str_in_list('input_path',list(metadata['inputs'].keys()))
            self.inputs_path = metadata['inputs'].get('input_path',None)
            # Try to load ground truth table
            self.ground_truth_table = None
            self.settings['table_total'] = self.settings.get('table_total',1)
            if 'ground_truth_table' in sample_names:
                try:
                    self.ground_truth_table = np.loadtxt(
                        os.path.join(
                            self.experiment.subconfig['inputs']['dataset'],
                            self.settings['table']
                        )
                    ).astype('int32')
                except:
                    try:
                        self.ground_truth_table = np.loadtxt(
                            os.path.join(
                                self.experiment.subconfig['inputs']['dataset'],
                                self.experiment.subconfig['inputs']['data_files']['table']
                            )
                        ).astype('int32')
                    except:
                        pass
            # Try to get table total (number of agents)
            if self.ground_truth_table is not None:
                # Remove it from sample names
                sample_names.remove('ground_truth_table')
                # Reshape it
                self.ground_truth_table = self.ground_truth_table.reshape((1,*self.ground_truth_table.shape))
                # Extract metadata
                self.settings['table_total'] = self.ground_truth_table.ravel().sum()
                self.settings['dims'] = list(np.shape(self.ground_truth_table))
                self.logger.info(f'Ground truth table loaded')

            # Try to load all results
            for sample_name in sample_names:
                try:
                    self.experiment.results[sample_name] = self.load_samples(
                        sample_name,
                        slice_samples=slice_samples
                    )
                    if sample_name == 'table' and self.settings['table_total'] != 1:
                        self.settings['table_total'] = self.experiment.results[sample_name][0].ravel().sum()
                    self.logger.info(f'Sample {sample_name} loaded with shape {np.shape(self.experiment.results[sample_name])}')
                except:
                    self.logger.debug(traceback.format_exc())
                    self.logger.warning(f'Sample {sample_name} could not be loaded')

            if self.settings['table_total'] == 0:
                self.logger.warning('Ground truth missing')

            
        else:
            # Store experiment
            self.experiment = experiment
            # Get intensity model class
            self.intensity_model_class = [input_name for input_name in self.experiment.subconfig.keys() if input_name != 'contingency_table' and isinstance(self.experiment.subconfig[input_name],dict)][0]
            # Define output experiment directory
            self.experiment_id = self.update_experiment_directory_id()
            # Define output experiment path to directory
            self.outputs_path = os.path.join(
                    self.experiment.subconfig['outputs']['directory'],
                    self.experiment.subconfig['experiment_data'],
                    self.experiment_id
            )
            # Try to load ground truth table
            try:
                self.ground_truth_table = self.experiment.ct.table
            except:
                pass
            # Create output directories
            self.create_output_subdirectories()
        
        # self.logger.info(f"Output directory is set to {self.outputs_path}")

    def update_experiment_directory_id(self):
        if hasattr(self.experiment,'sim'):
            noise_level = self.experiment.sim.noise_regime
        else:
            noise_level = next(deep_get('noise_regime',self.experiment.subconfig.settings))
            if noise_level is None: noise_level = 'unknown'
        noise_level = noise_level.capitalize()
        
        if not str_in_list('experiment_title',self.experiment.subconfig['outputs'].keys()):
            self.experiment.subconfig['outputs']['experiment_title'] = ""

        if str_in_list(self.experiment.subconfig['type'].lower(),['tablesummariesmcmcconvergence','tablemcmcconvergence']):
            return self.experiment.subconfig['experiment_id']+'_K'+\
                    str(self.experiment.subconfig['K'])+'_'+\
                    self.experiment.subconfig['mcmc']['contingency_table']['proposal']+'_'+\
                    self.experiment.subconfig['type']+'_'+\
                    self.experiment.subconfig['outputs']['experiment_title']+'_'+\
                    self.experiment.subconfig['datetime']
        elif self.experiment.subconfig['type'].lower() == 'tablemcmc':
            return self.experiment.subconfig['experiment_id']+'_'+\
                    self.experiment.subconfig['mcmc']['contingency_table']['proposal']+'_'+\
                    self.experiment.subconfig['type']+'_'+\
                    self.experiment.subconfig['outputs']['experiment_title']+'_'+\
                    self.experiment.subconfig['datetime']
        else:
            return self.experiment.subconfig['experiment_id']+'_'+\
                    self.experiment.subconfig['type']+'_'+\
                    noise_level+'Noise_'+\
                    self.experiment.subconfig['outputs']['experiment_title']+'_'+\
                    self.experiment.subconfig['datetime']

    def create_output_subdirectories(self) -> None:
    
        makedir(os.path.join(self.outputs_path,'samples'))
        makedir(os.path.join(self.outputs_path,'figures'))
        makedir(os.path.join(self.outputs_path,'sample_derivatives'))

    def write_metadata(self) -> None:
        # Define filepath
        filepath = os.path.join(self.outputs_path,f"{self.experiment_id}_metadata.json")
        if (os.path.exists(filepath) and self.experiment.subconfig['overwrite']) or (not os.path.exists(filepath)):
            write_json(self.experiment.subconfig.settings,filepath,indent=2)

    def print_metadata(self) -> None:
        print_json(self.experiment.subconfig,indent=2)

    def write_samples(self, vector_names:List[str]=None) -> None:

        try:
            assert len(self.experiment.results) > 0
        except:
            self.logger.error(f"No experimental tabular results found.")
            return

        # Get all vector names that have been stored in results and match the specified ones
        existing_vector_names = list(set(vector_names).intersection(set(self.experiment.results[0]['samples'].keys())))
        
        try:
            assert (vector_names is None) or (set(vector_names) <= set(self.experiment.results[0]['samples'].keys()))
        except:
            missing_vector_names = set(self.experiment.results[0]['samples'].keys()).difference(set(vector_names))
            self.logger.warning(f"Vectors {','.join(missing_vector_names)} not found in {','.join(self.experiment.results[0]['samples'].keys())}")
        
        # Define filepath
        dirpath = os.path.join(self.outputs_path,'samples')

        # Only export samples if vector names are not null
        if len(existing_vector_names) > 0:
            # Loop through experimental results and write samples
            for res in tqdm(self.experiment.results):

                for k in existing_vector_names:

                    # Add type of sample in filename
                    if 'id' in res.keys():
                        filename = f"{k}_samples_{res['id']}.npy"
                    else:
                        filename = f"{k}_samples.npy"

                    # Write function samples (list of dictionaries) in compress json format
                    if (os.path.exists(os.path.join(dirpath,filename)) and self.experiment.subconfig['overwrite']) or (not os.path.exists(os.path.join(dirpath,filename))):
                        write_npy(res['samples'][k],os.path.join(dirpath,filename))

    # def instantiate_plotting_function(self,pid:str):# -> Union[Outputs,None]:
    #     if hasattr(self, self.settings['type']):
    #         return getattr(self, self.settings['type'])(pid)
    #     else:
    #         raise Exception(f"Input class {self.settings['type']} not found")

    def slice_samples(self,samples):

        # Get burnin parameter
        burnin = min(self.settings.get('burnin',0),np.shape(samples)[0])
        # Get thinning parameter
        thinning = list(deep_get(key='thinning',value=self.settings))
        thinning = thinning[0] if len(thinning) > 0 else 1
        # Apply burnin and thinning
        samples = samples[burnin:None:thinning,...]
        # Get total number of samples
        N = self.settings.get('N',samples.shape[0])
        N = N if N is not None else samples.shape[0]
        N = min(N,samples.shape[0])
        # Trim total number of samples
        samples = samples[:N,...]

        return samples

    def load_samples(self,sample_name,slice_samples:bool=True):
        if sample_name == 'intensity':
            # Load all samples
            theta_samples = self.load_samples('theta',slice_samples)
            log_destination_attraction_samples = self.load_samples('log_destination_attraction',slice_samples)
            # Load inputs
            origin_demand = self.load_input_data(
                'origin_demand'
            ).astype('float32')
            cost_matrix = self.load_input_data(
                'cost_matrix'
            ).astype('float32')
            self.log_destination_attraction = self.load_input_data(
                'log_destination_attraction'
            ).astype('float32')
            # Get total number of samples
            N = theta_samples.shape[0]
            # Scale beta
            theta_samples[:,1] *= self.experiment.subconfig[self.intensity_model_class]['beta_max']

            # Compute intensities for all samples
            table_total = self.settings.get('table_total') if self.settings.get('table_total',-1.0) > 0 else 1.0

            # Instantiate ct
            sim = instantiate_sim({
                'sim_type':next(deep_get('sim_type',self.experiment.subconfig), None),
                'cost_matrix':cost_matrix,
                'origin_demand':origin_demand,
                'log_destination_attraction':self.log_destination_attraction
            })
            # with ProgressBar(total=N) as progress_bar:
            log_flow = sim.log_flow_matrix_vectorised(
                log_destination_attraction_samples,
                theta_samples,
                origin_demand,
                cost_matrix,
                table_total,
                None
            )
            samples = np.exp(log_flow,dtype='float32')
        elif sample_name.endswith("__error"):
            # Load all samples
            samples = self.load_samples(sample_name.replace("__error",""),slice_samples)
            # Make sure you have ground truth
            try:
                assert self.ground_truth_table is not None
            except:
                self.logger.error('Ground truth table missing. Sample error cannot be computed.')
                raise
        elif sample_name == 'ground_truth_table':
            # Get config and sim
            dummy_config = Namespace(**{'settings':self.experiment.subconfig})
            ct = instantiate_ct(table=None,config=dummy_config,disable_logger=True)
            samples = ct.table#[np.newaxis,:]
        elif str_in_list(sample_name, INPUT_TYPES.keys()):
            samples = self.load_input_data(
                sample_name
            ).astype(INPUT_TYPES[sample_name])
        else:
            # Load samples
            filenames = sorted(glob(os.path.join(self.outputs_path,f'samples/{sample_name}*.npy')))
            if len(filenames) <= 0:
                raise Exception(f"No {sample_name} files found in {os.path.join(self.outputs_path,'samples')}")
            samples = read_npy(filenames[0])
            for filename in filenames[1:]:
                # Read new batch
                sample_batch = read_npy(filename)
                # Append it to batches
                samples = np.append(samples,sample_batch,axis=0)
            
            # Apply burning, thinning and trimming
            if slice_samples:
                samples = self.slice_samples(samples)
            
        return samples.astype(DATA_TYPES[sample_name])

    def load_geometry(self,geometry_filename,default_crs:str='epsg:27700'):
        # Load geometry from file
        geometry = gpd.read_file(geometry_filename)
        geometry = geometry.set_crs(default_crs,allow_override=True)
        
        return geometry
    
    def load_input_data(self,input_name):
        # Define path to input files
        path_dir = Path(self.experiment.subconfig['inputs']['dataset'])
        filepath = os.path.join(
            self.inputs_path,
            os.path.basename(path_dir),
            self.experiment.subconfig['inputs']
                .get(self.intensity_model_class,'')['import']
                .get(input_name,'')
        )
        # Sort 
        input_files_matched = sorted(glob(filepath))
        if len(input_files_matched) <= 0:
            raise Exception(f"No {input_name} files found in {filepath}")
        else:
            # Return first file matching input name
            return read_file(input_files_matched[0])



    def create_filename(self,sample=None):
        # Decide on filename
        if (sample is None) or (not isinstance(sample,str)):
            filename = f"{','.join(self.settings['sample'])}"
        else:
            filename = f"{sample}"
        if str_in_list('statistic',self.settings.keys()):
            filename += f"_{','.join([str(stat) for stat in list(flatten(self.settings['statistic'][0]))])}"
        if str_in_list('table_dim',self.experiment.subconfig.keys()):
            filename += f"_{self.experiment.subconfig['table_dim']}"
        if str_in_list('table_total',self.experiment.subconfig.keys()):
            filename += f"_{self.experiment.subconfig['table_total']}"
        if str_in_list('type',self.experiment.subconfig.keys()) and len(self.experiment.subconfig['type']) > 0:
            filename += f"_{self.experiment.subconfig['type']}"
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
        # filename += f"_N{self.experiment.subconfig['mcmc']['N']}"
        return filename

    def compute_sample_statistics(self,data,sample_name,statistic,axis:int=0):
        if statistic is None or statistic.lower() == '' or 'sample' in statistic.lower():
            return data
        elif not str_in_list(sample_name,SAMPLE_TYPES.keys()):
            return convert_string_to_numpy_function(statistic)(data,axis=axis).astype('float32')#[np.newaxis,:]
        elif statistic.lower() == 'signedmean' and \
            str_in_list(sample_name,['table','intensity','theta','log_destination_attraction']):
            if sample_name == 'table':
                # Compute mean,var
                return np.mean(data,keepdims=True,axis=axis).astype('float32')#[np.newaxis,:]
            elif str_in_list(sample_name,['intensity','theta','log_destination_attraction']):
                sign_samples = self.load_samples('sign',slice_samples=True)
                sign_samples = sign_samples.reshape((np.shape(sign_samples)[0],1))
                # Compute moments
                return ( np.einsum('nk,n...->k...',sign_samples,data) / np.sum(sign_samples.ravel())).astype('float32')#[np.newaxis,:]
            else:
                raise Exception(f'Signed mean could not be computed for sample {sample_name}')
        elif (statistic.lower() == 'signedvariance' or statistic.lower() == 'signedvar') and \
            str_in_list(sample_name,['table','intensity','theta','log_destination_attraction']):
            if sample_name == 'table':
                # Compute table variance
                return (np.var(data,axis=axis)).astype('float32')#[np.newaxis,:]
            elif str_in_list(sample_name,['intensity','theta','log_destination_attraction']):
                sign_samples = self.load_samples('sign',slice_samples=True)
                sign_samples = sign_samples.reshape((np.shape(sign_samples)[0],1))
                # Compute intensity variance
                samples_mean = np.einsum('nk,n...->k...',sign_samples,data) / np.sum(sign_samples.ravel())
                samples_squared_mean = np.einsum('nk,n...->k...',sign_samples,data**2) / np.sum(sign_samples.ravel())
                return (samples_squared_mean - samples_mean**2).astype('float32')#[np.newaxis,:]
            else:
                raise Exception('Signed variance could not be computed')
        elif statistic.lower() == 'error' and \
            str_in_list(sample_name,['table','intensity','theta','log_destination_attraction']):
            # Apply error norm
            return apply_norm(
                tab=data,
                tab0=self.ground_truth_table,
                name=self.settings['norm'],
                **self.settings
            )
        else:
            return convert_string_to_numpy_function(statistic)(data,axis=axis).astype('float32')

    def apply_sample_statistics(self,samples,sample_name,statistic_axes:Union[List,Tuple]=[]):
        
        if isinstance(statistic_axes,Tuple):
            statistic_axes = [statistic_axes]
        sample_statistic = samples
        # print('statistic_axes',statistic_axes)
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
                if len(stat) == 0:
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

        # print('\n')
        return sample_statistic