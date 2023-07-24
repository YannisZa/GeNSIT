import os
os.environ['USE_PYGEOS'] = '0'
import gc
import sys
import logging
import traceback
import seaborn as sns
import geopandas as gpd
import sklearn.manifold
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.cm as cm
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from glob import glob
from pathlib import Path
from tqdm.auto import tqdm
from copy import deepcopy
from scipy import interpolate
from argparse import Namespace
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
from statsmodels.graphics.tsaplots import plot_acf

from multiresticodm.utils import *
from multiresticodm.global_variables import *
from multiresticodm.colormaps import *
from multiresticodm.outputs import Outputs,OutputSummary
from multiresticodm.contingency_table import instantiate_ct
from multiresticodm.spatial_interaction_model import instantiate_sim
from multiresticodm.probability_utils import log_odds_ratio_wrt_intensity
from multiresticodm.math_utils import running_average,apply_norm,positive_sigmoid,logsumexp,map_distance_name_to_function,coverage_probability,calculate_min_interval

latex_preamble = r'''
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
'''

mpl.rcParams['text.latex.preamble'] = latex_preamble


class Plot(object):

    def __init__(self,plot_ids:List[str],outputs_directories:List[str],settings:dict):
        self.logger = logging.getLogger(__name__)

        # Enable garbage collector
        gc.enable()
        # Store settings
        self.settings = settings
        # Find matching output directories
        self.outputs_directories = OutputSummary.find_matching_output_folders(self)

        self.logger.info(f"Loaded {len(self.outputs_directories)} experiments")
        # self.logger.info(f"{','.join([Path(out_dir).stem for out_dir in self.outputs_directories])}")
        # Run plots
        for plot_id in plot_ids:
            self.compile_plot(plot_id)

    def compile_plot(self,visualiser_name):
        if hasattr(self, PLOT_HASHMAP[visualiser_name]):
            return getattr(self, PLOT_HASHMAP[visualiser_name])()
        else:
            raise Exception(f'Experiment class {PLOT_HASHMAP[visualiser_name]} not found')
    
    def compile_table_records_in_geodataframe(self,table,geometry):
        # Extract ids from geometry
        origin_geometry_ids = geometry[geometry.geometry_type == self.settings['origin_geometry_type']].geometry_id.tolist()
        destination_geometry_ids = geometry[geometry.geometry_type == self.settings['destination_geometry_type']].geometry_id.tolist()
        # Create dataframe
        table_df = pd.DataFrame(table,index=origin_geometry_ids,columns=destination_geometry_ids)
        # Create pairs of flow records instead of 2D flows
        table_df = table_df.stack().reset_index()
        # Rename columns
        table_df.rename(columns={"level_0":"origin","level_1":"destination",0:"flow"},inplace=True)
        # Attach origin geometry
        table_df = table_df.merge(
                        geometry[['geometry_id','LONG','LAT','geometry','origin_demand']].set_index('geometry_id'),
                        left_on='origin',
                        right_index=True,
                        how='left'
        )
        # Rename geometry
        table_df.rename(columns={"LONG":"origin_long","LAT":"origin_lat","geometry":"origin_geometry"},inplace=True)
        # Attach destination geometry
        table_df = table_df.merge(
                        geometry[['geometry_id','LONG','LAT','geometry','destination_demand']].set_index('geometry_id'),
                        left_on='destination',
                        right_index=True,
                        how='left'
        )
        # Rename geometry
        table_df.rename(
                columns={
                    "LONG":"destination_long",
                    "LAT":"destination_lat",
                    "geometry":"destination_geometry"
                },inplace=True)

        # Convert to geopandas
        return gpd.GeoDataFrame(table_df,geometry='origin_geometry')

    def infer_ground_truth_intensity(self,ct,log_intensity):
        # Get table distribution name
        distribution_name = ct.distribution_name.lower()
        if distribution_name == 'poisson':
            return np.exp(log_intensity)
        
        # Get smallest in length constrained axes
        axis_constrained = min(ct.constraints['constrained_axes'], key=len)
        # print(distribution_name)
        if distribution_name == 'multinomial':
            # Compute table and intensity totals
            table_total = ct.margins[tuplize(range(ct.ndims()))]
            # Reshape intensity totals
            new_shape = np.ones((ct.ndims()+1),dtype='int8')
            new_shape[0] = log_intensity.shape[0]
            # intensity_totals = np.sum(intensity,axis=tuplize([i for i in range(1,ct.ndims()+1)])).reshape(new_shape)
            log_intensity_totals = np.array([logsumexp(log_intensity[i,...].ravel()) for i in range(log_intensity.shape[0])]).reshape(new_shape)
            # return table_total*intensity/intensity_totals
            return table_total * np.exp(log_intensity-log_intensity_totals)
        
        elif distribution_name == 'product_multinomial':
            # Compute unconstrained axis
            unconstrained_axis = ct.axes_complement(axis_constrained,same_length=True)[0]
            
            # Reshape table margins
            new_shape = np.array([1]*ct.ndims())
            new_shape[unconstrained_axis] = ct.dims[unconstrained_axis]
            # Compute table and intensity constrained margins
            table_constrained_margins = ct.margins[tuplize(axis_constrained)].reshape(new_shape)
            # Reshape intensity margins
            new_shape = np.array([log_intensity.shape[0]]+[1]*ct.ndims())
            new_shape[unconstrained_axis[0]+1] = ct.dims[unconstrained_axis[0]]
            # Compute log intensity margin totals
            log_intensity_constrained_margins = np.array([
                logsumexp(lam[index,:]) if unconstrained_axis == (0,) else logsumexp(lam[:,index])
                for lam in log_intensity 
                for index in range(ct.dims[unconstrained_axis])
            ]).reshape(new_shape)
            return table_constrained_margins*np.exp(log_intensity-log_intensity_constrained_margins)
        
        elif distribution_name == 'fishers_hypergeometric':
            new_shape = (log_intensity.shape[0],1,log_intensity.shape[2])
            # Compute odd ratios
            log_odd_ratios = np.array([log_odds_ratio_wrt_intensity(llam) for llam in log_intensity])
            # Compute odd ratios margins
            log_odd_ratios_margins = np.array([
                logsumexp(log_or[:,j])
                for log_or in log_odd_ratios
                for j in range(new_shape[2])
            ]).reshape(new_shape)
            # Reshape table colsums
            new_shape = np.array([1]*ct.ndims())
            new_shape[1] = ct.dims[1]
            # Table colsums
            table_constrained_margins = ct.margins[(0,)].reshape(new_shape)
            return table_constrained_margins*np.exp(log_odd_ratios - log_odd_ratios_margins)
        
        else:
            raise Exception(f"No ground truth matching distribution {distribution_name} found")

        
    def plot_error_norms(self,errors,statistic_name,statistic_symbol,sample_name,K:int=None):
        self.logger.info(f"Plotting {self.settings['norm']} error norms")
        # Define output filename
        if len(errors.keys()) > 1:
            # Get filename for comparison plot
            filename = f"{statistic_name}_{sample_name}_" + \
                        f"{self.settings['norm']}_norm_vs_iteration_comparison_" + \
                        f"burnin{self.settings['burnin']}_" + \
                        f"thinning{self.settings['thinning']}_" + \
                        f"by_{'_'.join(list(self.settings['label_by']))}"
            if K is not None:
                filename += f"_{K}"

            # Get path of first output directory
            output_directory = Path(self.outputs_directories[0])
            # Find dataset directory name
            dataset = find_dataset_directory(self.settings['dataset_name'])
            # Make directory for paper figures in parent directory
            new_output_directory = os.path.join(
                output_directory.parents[1].absolute(),
                'paper_figures',
                dataset
            )
            # Make new directory
            makedir(new_output_directory)
            # Define filepath
            filepath = os.path.join(
                new_output_directory,
                filename
            )
        else:
            subconfig = errors[list(errors.keys())[0]]['subconfig']
            outputs_path = errors[list(errors.keys())[0]]['outputs_path']
            filename = f"{subconfig['table_dim']}_{statistic_name}_{sample_name}_" + \
                    f"{self.settings['norm']}_norm_vs_iteration_" + \
                    f"N{subconfig['mcmc']['N']}_" + \
                    f"burnin{self.settings['burnin']}_" + \
                    f"thinning{self.settings['thinning']}_" + \
                    f"{subconfig['mcmc']['contingency_table']['proposal']}"
            filepath = os.path.join(outputs_path,'figures',filename)

        # Plot
        fig = plt.figure(figsize=self.settings['figure_size'])
        x_min = self.settings.get('burnin',0)

        for i,experiment_id in enumerate(list(errors.keys())):
            # Read outputs
            self.logger.debug(f"Experiment id {experiment_id}")

            # Read label(s) from settings
            errors[experiment_id]['label'], \
            label_by_key, \
            label_by_val = create_dynamic_data_label(__self__=self,data=errors[experiment_id]['subconfig'])

            # Plot error norm for current file
            x_max = self.settings.get('N',errors[experiment_id]['subconfig']['mcmc'].get('N',-1))
            x_range = list(range(
                        self.settings['burnin'],
                        errors[experiment_id]['subconfig']['mcmc'].get('N',1),
                        self.settings['thinning']
            ))
            x_range = x_range[0:int(x_max):1]
            # Add x range to error data
            errors[experiment_id]['x'] = x_range
            print('experiment_id',experiment_id,'x_range',np.min(x_range),np.max(x_range))

            try:
                assert len(x_range) == len(errors[experiment_id]['y'])
            except:
                print('x range',np.shape(x_range))
                print('y range',np.shape(errors[experiment_id]['y']))
                raise Exception('Mismatch in x and y range shapes')

            # Plot error norms for each value in x range
            if self.settings['marker_frequency'] is not None:
                plt.plot(
                    x_range,
                    errors[experiment_id]['y'], 
                    label=errors[experiment_id]['label'], 
                    marker='o',
                    markevery=int(self.settings['marker_frequency']), 
                    markersize=int(self.settings['marker_size'])
                )
            else:
                plt.plot(
                    x_range,
                    errors[experiment_id]['y'], 
                    label=errors[experiment_id]['label']
                )

        # Title
        if self.settings['figure_title'] is not None:
            plt.title(self.settings['figure_title'],fontsize=self.settings['title_label_size'])
        # X,Y labels
        if self.settings['x_label'] is None:
            plt.xlabel('Iteration',fontsize=self.settings['axis_font_size'])
        elif self.settings['x_label'].lower() != 'none' or self.settings['x_label'].lower() != '':
            plt.xlabel(self.settings['x_label'].replace("_"," "),fontsize=self.settings['axis_font_size'])
        if self.settings['y_label'] is None:
            norm_name = self.settings['norm'].replace("relative_","").capitalize()
            plt.ylabel(fr"${{{norm_name}}}$ norm of ${{{statistic_symbol}}}$",fontsize=self.settings['axis_font_size'])
        elif self.settings['y_label'].lower() != 'none' or self.settings['y_label'].lower() != '':
            plt.ylabel(self.settings['y_label'].replace("_"," "),fontsize=self.settings['axis_font_size'])
        # X,Y limits
        if x_min is not None and x_max is not None:
            plt.xlim(x_min,min(x_max,np.max(x_range)))
        else:
            plt.xlim(-10,np.max(x_range))
        # Legend
        if len(errors.keys()) > 0:
            leg = plt.legend()
            leg._ncol = 1

        y_min,y_max = self.settings['y_limit']
        if y_min is not None and y_max is not None:
            plt.ylim(y_min,min(y_max,np.max([max(errors[k]['y']) for k in errors.keys()])))
        else:
            plt.ylim(0.0,np.max([np.max(errors[k]['y']) for k in errors.keys()]))

        # X,Y label ticks
        plt.locator_params(axis='x', nbins=self.settings['x_tick_frequency'])
        # benchmark
        if self.settings['benchmark']:
            plt.axhline(y=0,color='red')
        # Tight layout
        plt.tight_layout()
        # Export figure
        write_figure_data(
            errors,
            Path(filepath).parent,
            groupby=[],
            key_type={'x':'int','y':'float'},
            **self.settings
        )
        write_figure(fig,filepath,**self.settings)
        
    
    def plot_lower_dimensional_embedding(self,filepath,embedded_data):
        
        self.logger.info(f"Plotting lower dimensional embedding of table data")

        if embedded_data.shape[1] == 3:

            # Figure size 
            fig,ax = plt.subplots(1,1,figsize=self.settings['figure_size'])
            # Title
            if self.settings['figure_title'] is not None:
                ax.set_title(self.settings['figure_title'],fontsize=self.settings['title_label_size'])
            
            # Plot embedded tabular data
            for sample_name,projected_sample in embedded_data.groupby("sample_name"):
                if sample_name == 'ground_truth':
                    ax.scatter(
                        projected_sample.x.values,
                        projected_sample.y.values,
                        label=sample_name,
                        zorder=1,
                        s=200,
                        linewidth=5
                    )
                else:
                    ax.scatter(
                        projected_sample.x.values,
                        projected_sample.y.values,
                        label=sample_name,
                        zorder=11,
                        s=5
                    )
            
            # ax.scatter(
            #     0.0000375,
            #     0.00003,
            #     label='ground truth',
            #     marker="+",
            #     s=200,
            #     linewidths=5,
            #     zorder=1
            # )

            # Plot axis labels
            if self.settings['x_label'] is None:
                ax.set_xlabel('Projected dimension 1',fontsize=self.settings['axis_font_size'])
            elif self.settings['x_label'].lower() != 'none' or self.settings['x_label'].lower() != '':
                ax.set_xlabel(self.settings['x_label'].replace("_"," "),fontsize=self.settings['axis_font_size'])
            if self.settings['y_label'] is None:
                ax.set_ylabel('Projected dimension 2',fontsize=self.settings['axis_font_size'])
            elif self.settings['y_label'].lower() != 'none' or self.settings['y_label'].lower() != '':
                ax.set_ylabel(self.settings['y_label'].replace("_"," "),fontsize=self.settings['axis_font_size']) 
            # Legend
            if str_in_list('legend_label_size',self.settings.keys()):
                leg = plt.legend(prop={'size': self.settings['legend_label_size']},loc = 'upper left')
            else:
                leg = plt.legend(loc = 'upper left')
            leg._ncol = 1
            # Ticks
            # plt.xticks(fontsize=self.settings['tick_font_size'])
            # plt.yticks(fontsize=self.settings['tick_font_size'])
            # Tight layout
            plt.tight_layout()
            # Write figure
            write_figure_data(
                embedded_data,
                Path(filepath).parent,
                key_type={'x':'float','y':'float'},
                groupby=['sample_name'],
                **self.settings
            )
            write_figure(fig,filepath,**self.settings)

            self.logger.info(f"Figure exported to {filepath}")

        else:
            raise Exception(f"Cannot handled embedded data of dimension {embedded_data.shape[1]}")

    def check_table_posterior_mean_convergence_fixed_intensity_input_validity(self):
        # if the experiment is not of the right type remove it
        outputs_directories = deepcopy(self.outputs_directories)
        for i,output_directory in enumerate(self.outputs_directories):
            self.logger.debug(f"Experiment id {output_directory}")
            # Load contingency table
            outputs = Outputs(
                output_directory,
                self.settings,
                disable_logger=True
            )
            if not valid_experiment_type(outputs.experiment_id,['tablesummariesmcmcconvergence','tablemcmcconvergence']):
                self.logger.info(f'Experiment {outputs.experiment_id} is not of type Table(Summaries)MCMCConvergence')
                # Remove experiment from output directories
                self.outputs_directories.remove(outputs_directories[i])
                # Remove relevant metadata
                del outputs.experiment.subconfig
                continue

            if i == 0:
                K = outputs.experiment.subconfig['K']
            else:
                # All experiment must be using the same ensemble size
                try: 
                    assert outputs.experiment.subconfig['K'] == K
                except:
                    # Remove experiment from output directories
                    self.outputs_directories.remove(outputs_directories[i])
                    # Remove relevant metadata
                    del outputs.experiment.subconfig
                    continue    

    def table_posterior_mean_convergence_fixed_intensity(self):
        
        self.logger.info('Producing plot for table_posterior_mean_convergence_fixed_intensity')

        # Check for validity of inputs
        self.check_table_posterior_mean_convergence_fixed_intensity_input_validity()
        if len(self.outputs_directories) > 0:
            # Read all table norms
            table_norms = {}
            # Find out when norm drops below epsilon
            for output_directory in self.outputs_directories:
                # Read outputs
                outputs = Outputs(
                    output_directory,
                    self.settings,
                    sample_names=['tableerror'],
                    disable_logger=True
                )
                # Apply statistic to error norm
                table_error_statistic = outputs.compute_sample_statistics(
                    outputs.experiment.results['tableerror'],
                    'tableerror',
                    list(flatten(self.settings['statistic']))[0],
                    axis = (1,2)
                )
                table_norms[outputs.experiment_id] = {
                        'y':table_error_statistic,
                        'subconfig':outputs.experiment.subconfig,
                        'outputs_path':outputs.outputs_path
                }
                # Find supremum of norm below epsilon
                convergence_index = np.argmax(table_error_statistic < (self.settings['epsilon_threshold']))
                # Get metadata
                N = list(deep_get(key='N',value=outputs.experiment.subconfig))[0]
                try:
                    thinning = list(deep_get(key='thinning',value=outputs.experiment.subconfig))[0]
                except:
                    thinning = 1
                # Extraction iteration number
                sample_sizes = range(0,N,thinning)
                convergence_iteration = sample_sizes[convergence_index]
                if table_error_statistic[convergence_index] < self.settings['epsilon_threshold']:
                    print(f"{self.settings['norm']} below {self.settings['epsilon_threshold']} achieved at iteration {convergence_iteration} for {outputs.experiment_id}")
                else:
                    print(f"{self.settings['norm']} below {self.settings['epsilon_threshold']} not achieved in {sample_sizes[-1]} iterations for {outputs.experiment_id}...")
                    print(f"Smallest {self.settings['norm']} was {np.min(table_error_statistic)}")

            # Decide on statistic symbol based on size of ensemble of datasets
            statistic_symbol = ''
            if outputs.experiment.subconfig['K'] == 1:
                statistic_symbol = "\mathbb{{E}}[\mathbf{T}|\mathcal{C}_{T},\Lambda]"
            else:
                statistic_symbol = "\mathbb{{E}}[\mathbb{{E}}[\mathbf{T}|\mathcal{C}_{T},\Lambda]]"
            # print(table_error_statistic)
            # Plot errors
            self.plot_error_norms(
                        table_norms,
                        'mean',
                        statistic_symbol,
                        'table',
                        outputs.experiment.subconfig['K']
            )
        else:
            self.logger.error('No valid experiments found')

    def check_table_posterior_mean_convergence_input_validity(self):
        # if the experiment is not of the right type remove it
        for output_directory in self.outputs_directories: 
            # Load outputs
            outputs = Outputs(
                output_directory,
                self.settings,
                disable_logger=True
            )
            if not valid_experiment_type(outputs.experiment_id,['jointtablesimlatent']):
                self.logger.info(f'Experiment {outputs.experiment_id} is not of type JointTableSIMLatent')
                # Remove experiment from output directories
                self.outputs_directories.remove(output_directory)

            # Load contingency table
            dummy_config = Namespace(**{'settings':outputs.experiment.subconfig})
            ct = instantiate_ct(table=None,config=dummy_config,disable_logger=True)
            # If no table is provided
            if ct.table is None:
                # Remove experiment from output directories
                self.outputs_directories.remove(output_directory)
                # Remove relevant metadata
                del outputs.experiment.subconfig
                continue

    def table_posterior_mean_convergence(self):
        
        self.logger.info('Producing plot for table_posterior_mean_convergence')

        # Check for validity of inputs
        self.check_table_posterior_mean_convergence_input_validity()
        if len(self.outputs_directories) > 0:
            error_norms = {}
            for output_directory in tqdm(self.outputs_directories):
                self.logger.debug(f"Experiment id {output_directory}")
                # Get experiment id
                outputs = Outputs(
                    output_directory,
                    self.settings,
                    sample_names=['intensity','table','sign'],
                    slice_samples=False,
                    disable_logger=True
                )
                
                # Instantiate contingency table
                dummy_config = Namespace(**{'settings':outputs.experiment.subconfig})
                ct = instantiate_ct(table=None,config=dummy_config,disable_logger=True)
                
                # Load samples and apply moving averages
                table_rolling_mean = running_average(outputs.experiment.results['table'])
                
                # Make sure that no infs are introduced
                if (not np.isfinite(table_rolling_mean).all()):
                    self.logger.error(output_directory)
                    raise Exception(f'Table rolling mean produced Infs/NaNs for distribution {ct.distribution_name}')
                
                # Elicit ground truth intensity
                intensity = self.infer_ground_truth_intensity(ct,np.log(outputs.experiment.results['intensity']))

                # Compute its mean
                intensity_rolling_mean = running_average(intensity,signs=outputs.experiment.results['sign'])

                # Slice samples for plotting
                table_rolling_mean = outputs.slice_samples(table_rolling_mean)
                intensity_rolling_mean = outputs.slice_samples(intensity_rolling_mean)
                
                # Make sure that no infs are introduced
                if (not np.isfinite(intensity_rolling_mean).all()):
                    self.logger.error(output_directory)
                    raise Exception(f'Intensity signed rolling mean produced Infs/NaNs for distribution {ct.distribution_name}')
                
                # Update error norm normalisation constant only in the unconstrained case
                # if ct.distribution_name.lower() == 'poisson':
                self.settings['normalisation_constant'] = ct.margins[tuplize(range(ct.ndims()))]

                # Compute error norm between the two averages
                # Table is used as a ground truth because
                # its grand totals are conserved in all but the constrained case
                # Therefore normalisation in relative norms is the same for all samples
                error_norm = apply_norm(
                    tab=intensity_rolling_mean,
                    tab0=table_rolling_mean,
                    name=self.settings['norm'],
                    **self.settings
                )

                # Compute statistic over error
                mean_error_norm = outputs.compute_sample_statistics(
                    error_norm,
                    'table',
                    'mean',
                    axis = (1,2)
                )
                
                # Add error norms to dict
                error_norms[outputs.experiment_id] = {
                        'y':mean_error_norm,
                        'subconfig':outputs.experiment.subconfig,
                        'outputs_path':outputs.outputs_path
                }
                

            # Plot table mean error norm
            self.plot_error_norms(
                error_norms,
                'mean',
                "\mathbb{{E}}[\mathbf{T}|\mathcal{C}_{T},\mathbf{Y}] - \mathbb{{E}}[\Lambda|\mathcal{C}_{T},\mathbf{Y}]",
                'table',
            )
            
        else:
            self.logger.error('No valid experiments found')

    def check_colsum_posterior_mean_convergence_fixed_intensity_input_validity(self):
        # if the experiment is not of the right type remove it
        for output_directory in tqdm(self.outputs_directories): 
            self.logger.debug(f"Experiment id {output_directory}")
            # Load contingency table
            outputs = Outputs(
                output_directory,
                self.settings,
                disable_logger=True
            )
            # If experiment is not of the right type
            if 'tablesummariesmcmcconvergence' not in outputs.experiment_id.lower():
                self.logger.info(f'Experiment {outputs.experiment_id} is not of type TableSummariesMCMCConvergence')
                # Remove experiment from output directories
                self.outputs_directories.remove(output_directory)
                # Remove relevant metadata
                del outputs.experiment.subconfig
                continue
            # If experiment does not use ensemble size of 1
            elif outputs.experiment.subconfig['K'] != 1:
                # Remove experiment from output directories
                self.outputs_directories.remove(output_directory)
                # Remove relevant metadata
                del outputs.experiment.subconfig
                continue   

    def colsum_posterior_mean_convergence_fixed_intensity(self):        
        self.logger.info('Producing plot for column sum posterior mean convergence')
       # Check for validity of inputs
        self.check_colsum_posterior_mean_convergence_fixed_intensity_input_validity()

        if len(self.outputs_directories) > 0:
            # Read all norms
            column_sum_norms = {}
            for output_directory in self.outputs_directories:
                self.logger.debug(f"Experiment id {output_directory}")
                # Get experiment id
                experiment_id = os.path.basename(os.path.normpath(output_directory))
                # Grab all files in output directory
                filenames = glob(os.path.join(output_directory,f'samples/table*_error*.npy'))
                for fname in filenames:
                    if f"{self.settings['norm']}_norm" in fname:
                        column_sum_norms[experiment_id] = read_npy(fname)
            # Decide on statistic symbol based on size of ensemble of datasets
            statistic_symbol = "\mathbb{{E}}[\mathbf{{n}}_{{\cdot,+}}|\mathbf{\lambda}]"
        
            # Plot errors
            self.plot_error_norms(
                        column_sum_norms,
                        'mean',
                        statistic_symbol,
                        'table',
            )
        else:
            self.logger.error('No valid experiments found')

    def table_distribution_low_dimensional_embedding(self):
        
        self.logger.info('Running table_distribution_low_dimensional_embedding')
        valid_experiment_types = ['tablemcmc','jointtablesimlatent','simlatent','neuralabm']
        
        # Add ground truth table to embedding
        outputs = Outputs(
            self.outputs_directories[0],
            self.settings,
            ['ground_truth_table'],
            slice_samples=False,
            disable_logger=True
        )
        # Store sample name, the actual sample (y), and its size (x)
        dims = np.shape(np.squeeze(outputs.ground_truth_table))
        # ground_truth_table
        embedded_data_ids = np.array(['ground_truth'],dtype=str)
        embedded_data_vals = np.array([outputs.ground_truth_table.ravel()],dtype='int32')
        
        # Decide on figure output dir
        if len(self.outputs_directories) > 1:
            output_dir = 'paper_figures'
        else:
            output_dir = os.path.join('figures',outputs.experiment_id)

        for output_directory in tqdm(sorted(self.outputs_directories)): 
            self.logger.info(f"Experiment id {output_directory}")
            # Load contingency table
            outputs = Outputs(
                output_directory,
                self.settings,
                list(self.settings.get('sample')),
                slice_samples=True,
                disable_logger=False
            )
            # Make sure the right experiments are provided
            try:
                assert np.any([exp_type in outputs.experiment.subconfig['type'].lower() for exp_type in valid_experiment_types])
            except:
                self.logger.warning(f"Skipping invalid experiment {outputs.experiment.subconfig['type'].lower()}")
                continue
            
            # Read label(s) from settings
            try:
                label = ''
                for k in list(self.settings['label_by']):
                    value = list(deep_get(key=k,value=outputs.experiment.subconfig))[0]
                    # If label not included in metadata ignore it
                    if value is not None:
                        label += '_'+str(value)
                        if k == 'noise_regime':
                            label += '_noise'
            except:
                label = outputs.experiment_id
            # Get only first n_samples as specified by settings
            if str_in_list('table',list(outputs.experiment.results.keys())):
                table_samples = outputs.experiment.results['table']
                table_samples = table_samples.reshape((table_samples.shape[0], np.prod(table_samples.shape[1:])))
                # Add to data
                embedded_data_ids = np.append(embedded_data_ids,np.repeat(("table"+label),table_samples.shape[0]))
                embedded_data_vals = np.append(embedded_data_vals,table_samples,axis=0)
                
            if str_in_list('intensity',list(outputs.experiment.results.keys())):
                intensity_samples = outputs.experiment.results['intensity']
                intensity_samples = intensity_samples.reshape((intensity_samples.shape[0], np.prod(intensity_samples.shape[1:])))
                # Get only the first n_samples after burnin
                embedded_data_ids = np.append(embedded_data_ids,np.repeat(("intensity"+label),intensity_samples.shape[0]))
                embedded_data_vals = np.append(embedded_data_vals,intensity_samples,axis=0)

        self.logger.info(f"Getting lower dimensional embedding of {embedded_data_vals.shape[0]} table samples using {self.settings['embedding_method']}, nearest neighbours = {self.settings['nearest_neighbours']} and distance metric = {self.settings['distance_metric']}")
        
        # Get data embedding
        if self.settings['embedding_method'] == 'isomap':
            # Instantiate isomap
            isomap = sklearn.manifold.Isomap(
                    n_components=2,
                    n_neighbors=self.settings['nearest_neighbours'],
                    max_iter=int(1e4),
                    path_method='auto',#'D',
                    eigen_solver='auto',
                    neighbors_algorithm='auto',#'ball_tree',
                    n_jobs=int(self.settings['n_workers']),
                    metric=map_distance_name_to_function(self.settings['distance_metric']),
                    metric_params={"dims":dims,"ord":self.settings.get('ord',None)}
            )
            # Get lower dimensional embedding
            embedded_data_vals = isomap.fit_transform(embedded_data_vals)
            print(f"Reconstruction error = {isomap.reconstruction_error()}")
        
        elif self.settings['embedding_method'] == 'tsne':
            # Instantiate tsne
            tsne = sklearn.manifold.TSNE(
                n_components=2,
                perplexity=float(self.settings['nearest_neighbours']),
                learning_rate='auto',
                n_iter=self.settings.get('K',int(1e4)),
                n_iter_without_progress=300,
                n_jobs=int(self.settings['n_workers']),
                metric=map_distance_name_to_function(self.settings['distance_metric']),
                metric_params={"dims":dims,"ord":self.settings.get('ord',None)}
            )
            # Get lower dimensional embedding
            embedded_data_vals = tsne.fit_transform(embedded_data_vals)
            print(f"KL Divergence = {tsne.kl_divergence_}")
        
        # Store projection data
        embedded_data_df = pd.DataFrame(
            data = np.array([
                embedded_data_ids,
                embedded_data_vals[:,0],
                embedded_data_vals[:,1]
            ]).T,
            columns = ['sample_name','x','y']
        )
        # Change types
        embedded_data_df = embedded_data_df.astype({'x': 'float32','y': 'float32'})
        
        # Get filename for comparison plot
        filename = f"2d_{self.settings['embedding_method']}_embedding_" + \
                    f"label_by_{'_'.join(list(self.settings['label_by']))}_" + \
                    f"burnin{self.settings['burnin']}_" + \
                    f"thinning{self.settings['thinning']}_" + \
                    f"{self.settings['distance_metric']}_" + \
                    f"nearest_neighbours{self.settings['nearest_neighbours']}"

        # Get filepath
        figure_filepath = os.path.join(
            outputs.experiment.subconfig['outputs']['directory'],
            outputs.experiment.subconfig['experiment_data'],
            output_dir,
            filename
        )

        # Plot lower dimensional embedding
        self.plot_lower_dimensional_embedding(
            figure_filepath,
            embedded_data_df
        )
            
    
    def r2_parameter_grid_plot(self):
            
        self.logger.info('Running r2_parameter_grid_plot')

        for i,output_directory in tqdm(enumerate(self.outputs_directories),total=len(self.outputs_directories)): 
            
            if not valid_experiment_type(output_directory,["rsquaredanalysis"]):
                continue

            outputs = Outputs(
                output_directory,
                self.settings,
                disable_logger=True
            )

            self.parameter_grid_plot(i,outputs.experiment_id,'r2',r'$R^2$')

    def log_target_parameter_grid_plot(self):
            
        self.logger.info('Running log_target_parameter_grid_plot')

        for i,output_directory in tqdm(enumerate(self.outputs_directories),total=len(self.outputs_directories)): 
            
            outputs = Outputs(
                output_directory,
                self.settings,
                disable_logger=True
            )

            if not valid_experiment_type(outputs.experiment_id,["logtargetanalysis"]):
                continue

            self.parameter_grid_plot(i,outputs.experiment_id,'log_target',r"$\log(\pi(\mathbf{{x}}\|\mathbf{{\theta}}))$")

    def absolute_error_parameter_grid_plot(self):
        self.logger.info('Running absolute_error_parameter_grid_plot')

        for i,output_directory in tqdm(enumerate(self.outputs_directories),total=len(self.outputs_directories)): 
            
            outputs = Outputs(
                output_directory,
                self.settings,
                disable_logger=True
            )

            if not valid_experiment_type(outputs.experiment_id,["absoluteerroranalysis"]):
                continue

            self.parameter_grid_plot(i,outputs.experiment_id,'absolute_error',r"$\log(\pi(\mathbf{{x}}\|\mathbf{{\theta}}))$")

    def parameter_grid_plot(self,index,experiment_id,sample_name,sample_symbol):
            
        # Options
        params = {
            'text.usetex' : True,
            'font.size' : 20,
            'legend.fontsize': 20,
            'legend.handlelength': 2,
            'font.family' : 'sans-serif',
            'font.sans-serif':['Helvetica']
        }

        # plt.rcParams.update(params)
        grid_size = outputs.experiment.subconfig['grid_size']
        amin,amax = outputs.experiment.subconfig['a_range']
        bmin,bmax = outputs.experiment.subconfig['b_range']
        # Get config and sim
        outputs = Outputs(
            output_directory,
            self.settings,
            disable_logger=True
        )
        dummy_config = Namespace(**{'settings':outputs.experiment.subconfig})
        sim = instantiate_sim(dummy_config)
        
        # Update max values
        bmin *= sim.bmax
        bmax *= sim.bmax
        # Create grid
        alpha_values = np.linspace(amin, amax, grid_size,endpoint=True)
        beta_values = np.linspace(bmin, bmax, grid_size,endpoint=True)
        XX, YY = np.meshgrid(alpha_values, beta_values)
        # Get values
        values = outputs.load_samples(sample_name)

        # Store values for upper and lower bounds of colorbar
        colorbar_min = None
        colorbar_max = None
        if str_in_list("colorbar_limit",self.settings.keys()):
            colorbar_min = self.settings['colorbar_limit'][0]
            colorbar_max = self.settings['colorbar_limit'][1]
        
        # Update colorbar range if necessary
        if colorbar_min is not None:
            values[values<colorbar_min] = np.nan
        if colorbar_max is not None:
            values[values>colorbar_max] = np.nan

        # Construct interpolator
        interpolated_data = interpolate.griddata((XX[np.isfinite(values)].ravel(), 
                                                YY[np.isfinite(values)].ravel()), 
                                                values[np.isfinite(values)].ravel(), 
                                                (XX[~np.isfinite(values)].ravel(), 
                                                YY[~np.isfinite(values)].ravel()), 
                                                method='nearest')
        # Interpolate zero values
        # values[~np.isfinite(values)] = interpolated_data

        # Get output directory
        output_directory = Path(self.outputs_directories[index])
        
        # Save R2 figure to file
        filepath = os.path.join(
                output_directory,
                "figures",
                f"table_{'x'.join([str(s) for s in sim.shape()])}_" + \
                f"total_{sim.total_flow}_" + \
                f"{outputs.experiment.subconfig['type']}" + \
                f"{self.settings['experiment_title']}" + \
                f"{sim.noise_regime.title()}Noise"
        )

        # plt.hist(values.ravel())
        # # Save figure to file
        # write_figure(fig,filepath,**self.settings)
        # Close figure
        # plt.close()
        # return 

        # Initialise plot
        fig = plt.figure(figsize=self.settings['figure_size'])
        # Plot values
        plt.contourf(XX, YY*(1/sim.bmax), values, cmap = self.settings['main_colormap'], vmin = colorbar_min, vmax = colorbar_max,levels=self.settings['n_bins'])
        # plt.contourf(XX, YY*(1/sim.bmax), values, cmap = self.settings['main_colormap'])
        # Plot x,y limits
        plt.xlim([np.min(XX), np.max(XX)])
        plt.ylim([np.min(YY)*(1/sim.bmax), np.max(YY)*(1/sim.bmax)])
        # Plot colorbar
        cbar = plt.colorbar()
        cbar.set_label(sample_symbol,rotation=self.settings['colorbar_label_rotation'],labelpad=self.settings['colorbar_labelpad'])
        # Plot ground truth and fitted value
        plt.scatter(outputs.experiment.subconfig['fitted_alpha'],outputs.experiment.subconfig['fitted_scaled_beta'],color='blue',s=self.settings['marker_size']*(2/3),label='fitted')
        # *(1/sim.bmax)
        if hasattr(sim,'alpha_true') and hasattr(sim,'beta_true'):
            plt.scatter(sim.alpha_true,sim.beta_true,color='red',s=self.settings['marker_size'],label='true',marker='x')
        # Axis labels
        plt.ylabel(r'$\beta$',rotation=self.settings['axis_label_rotation'],labelpad=self.settings['axis_labelpad'])
        plt.xlabel(r'$\alpha$')
        # X axis tick fontsize
        plt.xticks(fontsize=self.settings['tick_font_size'])
        plt.yticks(fontsize=self.settings['tick_font_size'])
        # Legend
        if self.settings['legend_label_size'] is not None:
            leg = plt.legend(frameon=False,prop={'size': self.settings['legend_label_size']})
        else:
            leg = plt.legend(frameon=False)
        leg._ncol = 1
        for text in leg.get_texts():
            text.set_color("white")
        # Tight layout
        plt.tight_layout()

        # Write figure
        write_figure(fig,filepath,**self.settings)

    def parameter_mixing(self):

        self.logger.info('Running parameter_mixing')

        for i,output_directory in tqdm(enumerate(self.outputs_directories),total=len(self.outputs_directories)): 
            # Load samples and SIM model
            outputs = Outputs(
                output_directory,
                self.settings,
                disable_logger=True
            )
            # Convert to path object
            output_directory = Path(output_directory)

            # Get only first n_samples as specified by settings
            parameter_samples = outputs.load_samples('theta')
            sign_samples = outputs.load_samples('sign')
            # Plot empirical mean
            parameter_mean = np.dot(parameter_samples.T,sign_samples)/np.sum(sign_samples)

            # Get filename
            filename = f"table_{outputs.experiment.subconfig['table_dim']}_" + \
                    f"{outputs.experiment.subconfig['type']}_{self.settings['experiment_title']}_parameter_mixing_thinning_{self.settings['thinning']}_"+\
                    f"N_{outputs.experiment.subconfig['mcmc']['N']}" 

            # Define filepath
            filepath = os.path.join(output_directory.parent.absolute(),outputs.experiment_id,'figures',filename)
    
            # Plot parameter mixing
            # Figure size 
            fig,axs = plt.subplots(1,2,figsize=self.settings['figure_size'])
            # Alpha plot
            axs[0].plot(parameter_samples[:, 0],label='samples')
            axs[0].set_ylabel(r'$\alpha$',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'],rotation=self.settings['axis_label_rotation'])
            if self.settings['x_label'] is None:
                axs[0].set_xlabel('MCMC iteration',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            elif self.settings['x_label'].lower() != 'none' or self.settings['x_label'].lower() != '':
                axs[0].set_xlabel(self.settings['x_label'].replace("_"," "),fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            # if hasattr(sim,'alpha_true'):
                # axs[0].axhline(y=sim.alpha_true, color='black', linestyle='-',label='true')
            axs[0].axhline(y=parameter_mean[0],color='red',label=r'$\hat{\mu(\alpha)}$')
            # X,Y limits
            x_min,x_max = self.settings['x_limit']
            if x_min is not None and x_max is not None:
                axs[0].set_xlim(x_min,min(x_max,parameter_samples.shape[0]))
            y_min,y_max = self.settings['y_limit']
            if y_min is not None and y_max is not None:
                axs[0].set_ylim(max(0.0,y_min),min(y_max,2.0))
            # X,Y label ticks
            axs[0].locator_params(axis='x', nbins=self.settings['x_tick_frequency'])
            # Legend
            leg0 = axs[0].legend()
            leg0._ncol = 1

            # Beta plot
            axs[1].plot(parameter_samples[:, 1],label='samples')
            axs[1].set_ylabel(r'$\beta$',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'],rotation=self.settings['axis_label_rotation'])
            if self.settings['x_label'] is None:
                axs[1].set_xlabel('MCMC iteration',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            elif self.settings['x_label'].lower() != 'none' or self.settings['x_label'].lower() != '':
                axs[1].set_xlabel(self.settings['x_label'].replace("_"," "),fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            # if hasattr(sim,'beta_true'):
                # axs[1].axhline(y=sim.beta_true, color='black', linestyle='-',label='true')
            axs[1].axhline(y=parameter_mean[1],color='red',label=r'$\hat{\mu(\beta)}$')
            # X,Y limits
            x_min,x_max = self.settings['x_limit']
            if x_min is not None and x_max is not None:
                axs[1].set_xlim(x_min,min(x_max,parameter_samples.shape[0]))
            y_min,y_max = self.settings['y_limit']
            if y_min is not None and y_max is not None:
                axs[1].set_ylim(max(0.0,y_min),min(y_max,2.0))
            # X,Y label ticks
            axs[1].locator_params(axis='x', nbins=self.settings['x_tick_frequency'])
            # Legend
            leg1 = axs[1].legend()
            leg1._ncol = 1
            
            # Figure title
            if self.settings['figure_title'] is not None:
                fig.suptitle(self.settings['figure_title'],fontsize=self.settings['title_label_size'])
            # Tight layout
            plt.tight_layout()
            # Save figure
            write_figure(fig,filepath,**self.settings)
            
            self.logger.info(f"Figure exported to {os.path.join(outputs.experiment_id,'figures',filename)}")
    
    def parameter_acf(self):

        self.logger.info('Running parameter_acf')

        for output_directory in tqdm(self.outputs_directories): 
            self.logger.debug(f"Experiment id {output_directory}")
            # Load contingency table
            outputs = Outputs(
                output_directory,
                self.settings,
                disable_logger=True
            )

            # Get only first n_samples as specified by settings
            parameter_samples = outputs.load_samples('theta')

            # Get filename
            filename = f"table_{outputs.experiment.subconfig['table_dim']}_" + \
                    f"{outputs.experiment.subconfig['type']}_{self.settings['experiment_title']}_" + \
                    f"parameter_acf_thinning_{self.settings['thinning']}_"+\
                    f"N_{outputs.experiment.subconfig['mcmc']['N']}"

            
            # Define filepath
            filepath = os.path.join(outputs.outputs_path,'figures',filename)

            # Plot parameter autocorrelation function
            # Figure size 
            fig,axs = plt.subplots(1,2,figsize=self.settings['figure_size'])
            # Alpha plot
            plot_acf(parameter_samples[:, 0],lags=self.settings['n_bins'],ax=axs[0],title='')
            axs[0].set_ylabel(r'$\alpha$',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'],rotation=self.settings['axis_label_rotation'])
            if self.settings['x_label'] is None:
                axs[0].set_xlabel('Lags',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            elif self.settings['x_label'].lower() != 'none' or self.settings['x_label'].lower() != '':
                axs[0].set_xlabel(self.settings['x_label'].replace("_"," "),fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            axs[0].set_ylim(0,None)
            # Plot 0.2 acf hline
            if self.settings['benchmark']:
                axs[0].axhline(y=0.2,color='red')
            # Beta plot
            plot_acf(parameter_samples[:, 1],lags=self.settings['n_bins'],ax=axs[1],title='')
            axs[1].set_ylabel(r'$\beta$',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'],rotation=self.settings['axis_label_rotation'])
            if self.settings['x_label'] is None:
                axs[1].set_xlabel('Lags',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            elif self.settings['x_label'].lower() != 'none' or self.settings['x_label'].lower() != '':
                axs[1].set_xlabel(self.settings['x_label'].replace("_"," "),fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            axs[1].set_ylim(0,None)
            if self.settings['benchmark']:
                axs[1].axhline(y=0.2,color='red')
            # Figure title
            if self.settings['figure_title'] is not None:
                fig.suptitle(self.settings['figure_title'],fontsize=self.settings['title_label_size'])
            # Tight layout
            plt.tight_layout()
            # Save figure
            write_figure(fig,filepath,**self.settings)
            
            self.logger.info(f"Figure exported to {os.path.join(outputs.experiment_id,'figures',filename)}")

    def parameter_2d_contours(self):

        self.logger.info('Running parameter_2d_contours')

        for output_directory in tqdm(self.outputs_directories): 
            self.logger.debug(f"Experiment id {output_directory}")
            # Load contingency table
            outputs = Outputs(
                output_directory,
                self.settings,
                disable_logger=True
            )
            # dummy_config = Namespace(**{'settings':outputs.experiment.subconfig})
            # sim = instantiate_sim(dummy_config)
            # Convert to path object
            output_directory = Path(output_directory)

            # Take burnin and apply thinning
            parameter_samples = outputs.load_samples('theta')
            sign_samples = outputs.load_samples('sign')
            
            # Get filename
            filename = f"table_{outputs.experiment.subconfig['table_dim']}_" + \
                    f"{outputs.experiment.subconfig['type']}_{self.settings['experiment_title']}_parameter_contours_thinning_{self.settings['thinning']}_"+\
                    f"N_{outputs.experiment.subconfig['mcmc']['N']}"

            # Define filepath
            filepath = os.path.join(outputs.outputs_path,'figures',filename)

             # Store values for upper and lower bounds of colorbar
            colorbar_min = None
            colorbar_max = None
            if str_in_list("colorbar_limit",self.settings.keys()):
                colorbar_min = self.settings['colorbar_limit'][0]
                colorbar_max = self.settings['colorbar_limit'][1]
                # Get colormap
                colorbar_kwargs = { 
                                    "levels":np.linspace(0,1.0,self.settings['n_bins']+1)
                                }
            else:
                colorbar_kwargs = {
                                    "thresh":0,
                                }

            # Plot parameter distribution
            # Figure size 
            fig,ax = plt.subplots(1,1,figsize=self.settings['figure_size'])
            # Contour plot
            kdeplot = sns.kdeplot(
                    ax=ax,
                    x=parameter_samples[:, 0],
                    y=parameter_samples[:, 1],
                    shade=True,
                    cbar=True,
                    cmap=self.settings["main_colormap"],
                    **colorbar_kwargs
            )
            
            # Axis labels
            ax.set_xlabel(
                r'$\alpha$',
                fontsize=self.settings['axis_font_size'],
                labelpad=self.settings['axis_labelpad'],
                rotation=self.settings['axis_label_rotation']
            )
            ax.set_ylabel(
                r'$\beta$',
                fontsize=self.settings['axis_font_size'],
                labelpad=self.settings['axis_labelpad'],
                rotation=self.settings['axis_label_rotation']
            )
            # Plot empirical mean
            parameter_mean = np.dot(parameter_samples.T,sign_samples)/np.sum(sign_samples)
            ax.plot(
                parameter_mean[0],
                parameter_mean[1],
                color='black',
                label=r'$\hat{\mu}$',
                marker='x', 
                markersize=int(self.settings['marker_size'])
            )
            
            # X,Y limits
            x_min,x_max = self.settings['x_limit']
            if x_min is not None and x_max is not None:
                ax.set_xlim(x_min,min(x_max,parameter_samples.shape[0]))
            y_min,y_max = self.settings['y_limit']
            if y_min is not None and y_max is not None:
                ax.set_ylim(max(0.0,y_min),min(y_max,2.0))
            # Legend
            leg = plt.legend()
            leg._ncol = 1
            
            # Figure title
            if self.settings['figure_title'] is not None:
                fig.suptitle(self.settings['figure_title'],fontsize=self.settings['title_label_size'])
            # Tight layout
            plt.tight_layout()
            # Save figure
            write_figure(fig,filepath,**self.settings)
            
            self.logger.info(f"Figure exported to {os.path.join(outputs.experiment_id,'figures',filename)}")

    def parameter_histogram(self):

        self.logger.info('Running parameter_histogram')

        for output_directory in tqdm(self.outputs_directories): 
            self.logger.debug(f"Experiment id {output_directory}")
            # Load contingency table
            outputs = Outputs(
                output_directory,
                self.settings,
                disable_logger=True
            )
            dummy_config = Namespace(**{'settings':outputs.experiment.subconfig})
            sim = instantiate_sim(dummy_config)

            # Get only first n_samples as specified by settings
            parameter_samples = outputs.load_samples('theta')
            sign_samples = outputs.load_samples('sign')

            # Get filename
            filename = f"table_{outputs.experiment.subconfig['table_dim']}_" + \
                    f"{outputs.experiment.subconfig['type']}_{self.settings['experiment_title']}_parameter_histogram_thinning_{self.settings['thinning']}_"+\
                    f"N_{outputs.experiment.subconfig['mcmc']['N']}"

            # Define filepath
            filepath = os.path.join(outputs.outputs_path,'figures',filename)
            # Plot empirical mean
            parameter_mean = np.dot(parameter_samples.T,sign_samples)/np.sum(sign_samples)

            # Plot parameter mixing
            # Figure size 
            fig,axs = plt.subplots(1,2,figsize=self.settings['figure_size'])
            # Alpha plot
            alpha_density = stats.gaussian_kde(parameter_samples[:, 0])
            _,alpha_his = np.histogram(parameter_samples[:, 0],bins=self.settings['n_bins'], density=True)
            smoothed_alpha_his = alpha_density(alpha_his)
            axs[0].plot(alpha_his,alpha_density(alpha_his),label='samples')
            axs[0].set_xlabel(r'$\alpha$',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'],rotation=self.settings['axis_label_rotation'])
            # Prior plot
            axs[0].axhline(y=0.5,color='blue',label='prior')
            if self.settings['y_label'] is None:
                axs[0].set_ylabel('Density',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            elif self.settings['y_label'].lower() != 'none' or self.settings['y_label'].lower() != '':
                axs[0].set_ylabel(self.settings['y_label'].replace("_"," "),fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            if hasattr(sim,'alpha_true') and sim.noise_regime == 'low':
                axs[0].axvline(x=sim.alpha_true, color='black', linestyle='-',label='MAP')
                axs[0].axvline(x=alpha_his[np.argmax(smoothed_alpha_his)],color='m',label='$\hat{MAP}$')
            axs[0].axvline(x=parameter_mean[0],color='red',label=r'$\hat{\mu(\alpha)}$')
            # X,Y limits
            x_min,x_max = self.settings['x_limit']
            if x_min is not None and x_max is not None:
                axs[0].set_xlim(max(0.0,x_min),min(x_max,2.0))
            axs[0].set_ylim(0.0,None)
            # Legend
            leg0 = axs[0].legend()
            leg0._ncol = 1

            # Beta plot
            beta_density = stats.gaussian_kde(parameter_samples[:, 1])
            _,beta_his = np.histogram(parameter_samples[:, 1],bins=self.settings['n_bins'], density=True)
            smoothed_beta_his = beta_density(beta_his)
            axs[1].plot(beta_his,smoothed_beta_his,label='samples')
            axs[1].set_xlabel(r'$\beta$',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'],rotation=self.settings['axis_label_rotation'])
            # Prior plot
            axs[1].axhline(y=0.5,color='blue',label='prior')
            if self.settings['y_label'] is None:
                axs[1].set_ylabel('Density',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            elif self.settings['y_label'].lower() != 'none' or self.settings['y_label'].lower() != '':
                axs[1].set_ylabel(self.settings['y_label'].replace("_"," "),fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            if hasattr(sim,'beta_true') and sim.noise_regime == 'low':
                axs[1].axvline(x=sim.beta_true, color='black', linestyle='-',label='MAP')
                axs[1].axvline(x=beta_his[np.argmax(smoothed_beta_his)],color='m',label='$\hat{MAP}$')
            axs[1].axvline(x=parameter_mean[1],color='red',label=r'$\hat{\mu(\beta)}$')
            # X,Y limits
            x_min,x_max = self.settings['x_limit']
            if x_min is not None and x_max is not None:
                axs[1].set_xlim(max(0.0,x_min),min(x_max,2.0))
            axs[1].set_ylim(0.0,None)
            # Legend
            leg1 = axs[1].legend()
            leg1._ncol = 1
            
            # Figure title
            if self.settings['figure_title'] is not None:
                fig.suptitle(self.settings['figure_title'],fontsize=self.settings['title_label_size'])
            # Tight layout
            plt.tight_layout()
            # Save figure
            write_figure(fig,filepath,**self.settings)
            
            self.logger.info(f"Figure exported to {os.path.join(outputs.experiment_id,'figures',filename)}")
    
    def destination_attraction_mixing(self):

        self.logger.info('Running destination_attraction_mixing')

        for output_directory in tqdm(self.outputs_directories): 
            self.logger.debug(f"Experiment id {output_directory}")
            # Load contingency table
            outputs = Outputs(
                output_directory,
                self.settings,
                disable_logger=True
            )
            dummy_config = Namespace(**{'settings':outputs.experiment.subconfig})
            sim = instantiate_sim(dummy_config)

            # Get only first n_samples as specified by settings
            log_destination_attraction_samples = outputs.load_samples('log_destination_attraction')
            sign_samples = outputs.load_samples('sign')

            # Get filename
            filename = f"table_{outputs.experiment.subconfig['table_dim']}_" + \
                    f"{outputs.experiment.subconfig['type']}_{self.settings['experiment_title']}_log_destination_attraction_mixing_thinning_{self.settings['thinning']}_"+\
                    f"N_{outputs.experiment.subconfig['mcmc']['N']}"

            # Define filepath
            filepath = os.path.join(outputs.outputs_path,'figures',filename)
            
            # Plot parameter mixing
            # Figure size 
            fig,axs = plt.subplots(1,sim.dims[1],figsize=self.settings['figure_size'])
            # Get axis limits
            x_min,x_max = self.settings['x_limit']
            # Get relative noise
            relative_noise = np.sqrt(sim.noise_var)/np.log(sim.dims[1])
            relative_noise_percentage = round(100*relative_noise)
            upper_bound = sim.log_destination_attraction + np.log((1.0+2*relative_noise_percentage/100))
            lower_bound = sim.log_destination_attraction - np.log((1.0+2*relative_noise_percentage/100))
            # Get mean
            mu_x = np.dot(log_destination_attraction_samples.T,sign_samples)/np.sum(sign_samples)
            for j in range(sim.dims[1]):
                axs[j].plot(log_destination_attraction_samples[:, j])
                axs[j].set_ylabel(fr'$x_{j}$',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'],rotation=self.settings['axis_label_rotation'])
                if self.settings['x_label'] is None:
                    axs[1].set_xlabel('MCMC iteration',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
                elif self.settings['x_label'].lower() != 'none' or self.settings['x_label'].lower() != '':
                    axs[1].set_xlabel(self.settings['x_label'].replace("_"," "),fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
                axs[j].axhline(y=mu_x[j], color='red', linestyle='-',label='$\hat{\mu(x_j)}$')
                axs[j].axhline(y=sim.log_destination_attraction[j], color='m', linestyle='-',label='$\mathbf{y}$')
                if hasattr(sim,'log_true_destination_attraction') and np.abs(sim.log_destination_attraction[j]-sim.log_true_destination_attraction[j]) > 1e-5:
                    axs[j].axhline(y=sim.log_true_destination_attraction[j], color='black', linestyle='-',label='true')
                    
                axs[j].axhline(y=upper_bound[j], color='m', linestyle='--',label=r'$\mathbf{Y} + 2\sigma_0$')
                axs[j].axhline(y=lower_bound[j], color='m', linestyle='--',label=r'$\mathbf{Y} - 2\sigma_0$')
                # X,Y limits
                if x_min is not None and x_max is not None:
                    axs[j].set_xlim(x_min,min(x_max,log_destination_attraction_samples.shape[0]))
                leg = axs[j].legend()
                leg._ncol = 1
            
            # Figure title
            if self.settings['figure_title'] is not None:
                fig.suptitle(self.settings['figure_title'],fontsize=self.settings['title_label_size'])
            # Tight layout
            plt.tight_layout()
            # Save figure
            write_figure(fig,filepath,**self.settings)
            
            self.logger.info(f"Figure exported to {os.path.join(outputs.experiment_id,'figures',filename)}")

    def plot_predictions(self,prediction_data):
        # Axes limits from settings (read only x limit)
        axes_lims = [self.settings.get('x_limit',[None,None])[0],self.settings.get('x_limit',[None,None])[1]]
        # Otherwise read from data
        if None in axes_lims:
            min_val,max_val = np.infty,-np.infty
            for data in prediction_data.values():
                min_val = min([np.min(data['x']),np.min(data['y']),min_val])
                max_val = max([np.max(data['x']),np.max(data['y']),max_val])
            axes_lims = [min_val-0.02,max_val+0.02]
        
        # Figure size 
        fig = plt.figure(figsize=self.settings['figure_size'])
        # ax = fig.add_subplot(111)
        # Get relative noise
        # relative_noise = np.sqrt(sim.noise_var)/np.log(sim.dims[1])
        # relative_noise_percentage = round(100*relative_noise)
        # upper_bound = sim.log_destination_attraction + np.log((1.0+2*relative_noise_percentage/100))
        # lower_bound = sim.log_destination_attraction - np.log((1.0+2*relative_noise_percentage/100))
        
        # Plot benchmark (perfect predictions)
        if self.settings['benchmark']:
            plt.plot(np.linspace(*axes_lims),np.linspace(*axes_lims),linewidth=0.2,color='black')
        if self.settings['x_label'] is None:
            plt.xlabel(r'$\mathbb{{E}}[\mathbf{X}|\mathbf{Y}]$',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
        elif self.settings['x_label'].lower() != 'none' or self.settings['x_label'].lower() != '':
            plt.xlabel(self.settings['x_label'].replace("_"," "),fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
        if self.settings['y_label'] is None:
            plt.ylabel(r'$\log{\mathbf{Y}}$',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
        elif self.settings['y_label'].lower() != 'none' or self.settings['y_label'].lower() != '':
            plt.ylabel(self.settings['y_label'].replace("_"," "),fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
        
        # Axis limits 
        plt.xlim(left=axes_lims[0], right=axes_lims[1])
        plt.ylim(bottom=axes_lims[0], top=axes_lims[1])
        
        for v in prediction_data.values():
            # Create composite label
            composite_label = v['label'] + (fr"$R^2 = {np.round(v.get('title',0.0),2)}$)" if v.get('title',0.0) > 0 else '')
            # Plot predictions against data
            plt.scatter(x=v['x'],y=v['y'],marker='o',s=int(self.settings['marker_size']),label=composite_label)
            
        # Aspect ratio equal
        # ax.set_aspect('equal', 'box')
        plt.gca().set_aspect('equal')
    
        # Legend
        try:
            leg = plt.legend()
            leg._ncol = 1
        except:
            pass
        
        # Tight layout
        plt.tight_layout()

        # Get experiment id and its data
        experiment_id = list(prediction_data.keys())[0]
        # Decide on figure output dir
        if len(self.outputs_directories) > 1:
            # Get filename
            filename = f"log_destination_attraction_predictions_"+\
                    f"burnin_{self.settings['burnin']}_" + \
                    f"thinning_{self.settings['thinning']}"
            # Define filepath
            parent_directory = Path(prediction_data[experiment_id]['outputs_path'])
            filepath = os.path.join(parent_directory.parent.absolute(),'paper_figures',filename)
        else:

            # Get filename
            filename = f"table_{prediction_data[experiment_id]['subconfig']['table_dim']}_" + \
                    f"gamma_{prediction_data[experiment_id]['subconfig']['spatial_interaction_model']['gamma']}" + \
                    f"{prediction_data[experiment_id]['subconfig']['type']}_{self.settings['experiment_title']}_log_destination_attraction_predictions_"+\
                    f"burnin_{self.settings['burnin']}_" + \
                    f"thinning_{self.settings['thinning']}_" + \
                    f"N_{prediction_data[experiment_id]['subconfig']['mcmc']['N']}"

            # Define filepath
            filepath = os.path.join(prediction_data[experiment_id]['outputs_path'],'figures',filename)

        # Write figure
        write_figure(
            fig,
            filepath,
            **self.settings
        )
        write_figure_data(
            prediction_data,
            Path(filepath).parent,
            groupby=[],
            key_type={'x':'float','y':'float'},
            **self.settings
        )
        
        self.logger.info(f"Figure exported to {filepath}")
        
    def destination_attraction_predictions(self):

        self.logger.info('Running destination_attraction_predictions')
        predictions = {}

        for output_directory in tqdm(self.outputs_directories): 
            self.logger.debug(f"Experiment id {output_directory}")
            # Load contingency table
            outputs = Outputs(
                output_directory,
                sample_names=['log_destination_attraction','sign'],
                settings=self.settings,
                slice_samples=True,
                disable_logger=True
            )
            dummy_config = Namespace(**{'settings':outputs.experiment.subconfig})
            sim = instantiate_sim(dummy_config)

            # Create label
            label,\
            _, \
            _ = create_dynamic_data_label(
                __self__=self,
                data=outputs.experiment.subconfig
            )
            
            # Get mean 
            xxs = outputs.experiment.results['log_destination_attraction']
            ss = outputs.experiment.results['sign']
            mu_x = (np.dot(xxs.T,ss)/np.sum(ss)).flatten()

            # Compute R squared
            # Total sum squares
            w_data = np.exp(sim.log_destination_attraction)
            w_pred = np.exp(mu_x)
            w_data_centred = w_data - np.mean(w_data)
            ss_tot = np.dot(w_data_centred, w_data_centred)
            # Residiual sum squares
            res = w_pred - w_data
            ss_res = np.dot(res, res)
            # Regression sum squares
            r2 = 1. - ss_res/ss_tot

            # Add data
            predictions[outputs.experiment_id] =  {
                'label':label,
                'x':mu_x,
                'y':sim.log_destination_attraction,
                'title':r2,
                'subconfig':outputs.experiment.subconfig,
                'outputs_path':outputs.outputs_path
            }

        self.plot_predictions(
            prediction_data=predictions
        )

    def destination_attraction_residuals(self):

        self.logger.info('Running destination_attraction_residuals')

        for output_directory in tqdm(self.outputs_directories): 
            self.logger.debug(f"Experiment id {output_directory}")
            # Load contingency table
            outputs = Outputs(
                output_directory,
                self.settings,
                disable_logger=True
            )
            dummy_config = Namespace(**{'settings':outputs.experiment.subconfig})
            sim = instantiate_sim(dummy_config)

            # Get only first n_samples as specified by settings
            log_destination_attraction_samples = outputs.load_samples('log_destination_attraction')
            sign_samples = outputs.load_samples('sign')

            # Get filename
            filename = f"table_{outputs.experiment.subconfig['table_dim']}_" + \
                    f"gamma_{outputs.experiment.subconfig['spatial_interaction_model']['gamma']}" + \
                    f"{outputs.experiment.subconfig['type']}_{self.settings['experiment_title']}_log_destination_attraction_residuals_thinning_{self.settings['thinning']}_"+\
                    f"N_{outputs.experiment.subconfig['mcmc']['N']}"

            # Define filepath
            filepath = os.path.join(outputs.outputs_path,'figures',filename)

            # Plot parameter mixing
            # Figure size 
            fig = plt.figure(figsize=self.settings['figure_size'])
            # Get relative noise
            # relative_noise = np.sqrt(sim.noise_var)/np.log(sim.dims[1])
            # relative_noise_percentage = round(100*relative_noise)
            # upper_bound = sim.log_destination_attraction + np.log((1.0+2*relative_noise_percentage/100))
            # lower_bound = sim.log_destination_attraction - np.log((1.0+2*relative_noise_percentage/100))
            # Get mean
            mu_x = np.dot(log_destination_attraction_samples.T,sign_samples)/np.sum(sign_samples)
            mu_x = mu_x.flatten()
            # Get residuals
            residuals = sim.log_destination_attraction-mu_x
            # Compute R squared
            # Total sum squares
            w_data = np.exp(sim.log_destination_attraction)
            w_pred = np.exp(mu_x)
            w_data_centred = w_data - np.mean(w_data)
            ss_tot = np.dot(w_data_centred, w_data_centred)
            # Residiual sum squares
            res = w_pred - w_data
            ss_res = np.dot(res, res)
            # Regression sum squares
            r2 = 1. - ss_res/ss_tot

            # Plot predictions against data
            plt.scatter(x=residuals,y=mu_x,marker='o',s=int(self.settings['marker_size']))
            # Plot true mean of errors
            if self.settings['annotate']:
                plt.axhline(y=0,color='black')
            if self.settings['x_label'] is None:
                plt.xlabel(r'$\log{\mathbf{Y}} - \mathbb{{E}}[\mathbf{X}|\mathbf{Y}]$',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            elif self.settings['x_label'].lower() != 'none' or self.settings['x_label'].lower() != '':
                plt.xlabel(self.settings['x_label'].replace("_"," "),fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            if self.settings['y_label'] is None:
                plt.ylabel(r'$\mathbb{{E}}[\mathbf{x}|\mathbf{Y}]$',fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            elif self.settings['y_label'].lower() != 'none' or self.settings['y_label'].lower() != '':
                plt.ylabel(self.settings['y_label'].replace("_"," "),fontsize=self.settings['axis_font_size'],labelpad=self.settings['axis_labelpad'])
            # Legend                
            leg = plt.legend()
            leg._ncol = 1
            # Figure title
            if self.settings['figure_title'] is not None:
                plt.title(fr"{self.settings['figure_title'].replace('_',' ').capitalize()} ($R^2 = {np.round(r2,2)}$)",fontsize=self.settings['title_label_size'])
            # Tight layout
            plt.tight_layout()
            
            # Write figure
            write_figure(fig,filepath,**self.settings)
            
            self.logger.info(f"Figure exported to {os.path.join(outputs.experiment_id,'figures',filename)}")
    
    # def origin_destination_table_spatial(self):
        
    #     self.logger.info('Running origin_destination_table_spatial')
    #     # Define arrow kwargs
    #     kw = dict(arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8")
        
    #      # Loop through experiments
    #     for output_directory in tqdm(self.outputs_directories): 
    #         self.logger.debug(f"Experiment id {output_directory}")
    #         # Load contingency table
    #         outputs = Outputs(
    #             output_directory,
    #             self.settings,
    #             disable_logger=True
    #         )

    #         # Load geometry 
    #         geometry = outputs.load_geometry(self.settings['geometry'])
            
    #         # Load table
    #         if self.settings['statistic'][0][0].lower() == 'mean_variance':
    #             table = outputs.compute_sample_statistics(
    #                 outputs.experiment.results[self.settings['sample'][0]],
    #                 self.settings['sample'][0],
    #                 'signedmean',
    #             )
    #             origin_demand = outputs.compute_sample_statistics(
    #                 outputs.experiment.results[self.settings['sample'][0]],
    #                 self.settings['sample'][0],
    #                 'rowsums',
    #             )
    #             destination_demand = outputs.compute_sample_statistics(
    #                 outputs.experiment.results[self.settings['sample'][0]],
    #                 self.settings['sample'][0],
    #                 'colsums',
    #             )

    #             table_variance = outputs.compute_sample_statistics(
    #                 outputs.experiment.results[self.settings['sample'][0]],
    #                 self.settings['sample'][0],
    #                 'signedvariance',
    #             )
    #         else:
    #             table = outputs.compute_sample_statistics(
    #                 outputs.experiment.results[self.settings['sample'][0]],
    #                 self.settings['sample'][0],
    #                 self.settings['statistic'][0][0],
    #             )
    #             origin_demand = outputs.compute_sample_statistics(
    #                 outputs.experiment.results[self.settings['sample'][0]],
    #                 self.settings['sample'][0],
    #                 'rowsums',
    #             )
    #             destination_demand = outputs.compute_sample_statistics(
    #                 outputs.experiment.results[self.settings['sample'][0]],
    #                 self.settings['sample'][0],
    #                 'colsums',
    #             )
            
    #         self.settings['viz_type'] = 'spatial'
    #         filename = outputs.create_filename()

    #         # Define filepath
    #         filepath = os.path.join(outputs.outputs_path,'figures',filename)

    #         # Add rowsums and column sums to geometry
    #         geometry.loc[geometry.geometry_type==self.settings['origin_geometry_type'],'origin_demand'] = table.sum(axis=1)
    #         geometry.loc[geometry.geometry_type==self.settings['destination_geometry_type'],'destination_demand'] = table.sum(axis=0)

    #         # Compile table records in geodataframe
    #         table_gdf = self.compile_table_records_in_geodataframe(table,geometry)
    #         # Add variance table records in geodataframe
    #         if self.settings['statistic'][0][0].lower() == 'mean_variance':
    #             # Compile variance records
    #             table_variance_gdf = self.compile_table_records_in_geodataframe(table_variance,geometry)
    #             table_variance_gdf.rename(columns={"flow":"flow_variance"},inplace=True)
    #             # Merge in current records
    #             table_gdf = table_gdf.merge(
    #                     table_variance_gdf[['origin','destination','flow_variance']],
    #                     on=['origin','destination'],
    #                     how='left'
    #             )

    #         flow_colorbar_min,flow_colorbar_max = None, None
    #         if str_in_list("main_colorbar_limit",self.settings.keys()):
    #             flow_colorbar_min,flow_colorbar_max = self.settings['main_colorbar_limit']
            
    #         origin_colorbar_min,origin_colorbar_max = None, None
    #         destination_colorbar_min,destination_colorbar_max = None, None
    #         if str_in_list("auxiliary_colorbar_limit",self.settings.keys()):
    #             if len(np.shape(self.settings['auxiliary_colorbar_limit'])) == 1:
    #                 origin_colorbar_min,origin_colorbar_max = self.settings['auxiliary_colorbar_limit']
    #             elif len(np.shape(self.settings['auxiliary_colorbar_limit'])) > 1:
    #                 origin_colorbar_min,origin_colorbar_max = self.settings['auxiliary_colorbar_limit'][0]
    #                 destination_colorbar_min,destination_colorbar_max = self.settings['auxiliary_colorbar_limit'][1]

    #         # Normalise colormaps
    #         if flow_colorbar_min is not None and flow_colorbar_max is not None:    
    #             flow_norm = mpl.colors.Normalize(vmin=flow_colorbar_min, vmax=flow_colorbar_max)
    #         else:
    #             flow_norm = mpl.colors.Normalize(vmin=np.min(table.ravel()), vmax=np.max(table.ravel()))
            
    #         if origin_colorbar_min is not None and origin_colorbar_max is not None:    
    #             origin_norm = mpl.colors.Normalize(vmin=origin_colorbar_min, vmax=origin_colorbar_max)
    #         else:
    #             origin_norm = mpl.colors.Normalize(vmin=np.min(origin_demand.ravel()), vmax=np.max(origin_demand.ravel()))
            
    #         if destination_colorbar_min is not None and destination_colorbar_max is not None:    
    #             destination_norm = mpl.colors.Normalize(vmin=destination_colorbar_min, vmax=destination_colorbar_max)
    #         else:
    #             destination_norm = mpl.colors.Normalize(vmin=np.min(destination_demand.ravel()), vmax=np.max(destination_demand.ravel()))

    #         print(flow_norm.vmin,flow_norm.vmax)
    #         print(origin_norm.vmin,origin_norm.vmax)
    #         print(destination_norm.vmin,destination_norm.vmax)

    #         # Define colorbars for flows, and margins
    #         flow_base_cmap = cm.get_cmap(self.settings['main_colormap'])
    #         # Clip colors in colorbar to specific range
    #         colors = flow_base_cmap( 
    #             np.linspace(
    #                     self.settings['color_segmentation_limits'][0], 
    #                     self.settings['color_segmentation_limits'][1], 
    #                     self.settings['x_tick_frequency']
    #             )
    #         )
    #         # flow_color_segmented_cmap = mpl.colors.LinearSegmentedColormap.from_list(self.settings['main_colormap'], colors)
    #         flow_mapper = cm.ScalarMappable(
    #                     norm=flow_norm, 
    #                     cmap=self.settings['main_colormap']
    #         )
    #         origin_flow_mapper = cm.ScalarMappable(
    #                 norm=origin_norm, 
    #                 cmap=self.settings['aux_colormap'][0]
    #         )
    #         destination_flow_mapper = cm.ScalarMappable(
    #                 norm=destination_norm, 
    #                 cmap=self.settings['aux_colormap'][1]
    #         )

    #         # Setup figure
    #         fig = plt.figure(figsize=self.settings['figure_size'])
    #         if self.settings['colorbar']:
    #             gs = GridSpec(
    #                 nrows=1,
    #                 ncols=4,
    #                 height_ratios=[1],
    #                 width_ratios=[1,1,1,30],
    #                 hspace=0.0,
    #                 wspace=0.0
    #             )
    #             ax1 = fig.add_subplot(gs[3])
    #             origin_cbar_ax = fig.add_subplot(gs[0])
    #             destination_cbar_ax = fig.add_subplot(gs[1])
    #             flow_cbar_ax = fig.add_subplot(gs[2])

    #             flow_cbar_ax.axis('off')
    #             origin_cbar_ax.axis('off')
    #             destination_cbar_ax.axis('off')
    #         else:
    #             gs = GridSpec(
    #                 nrows=1,
    #                 ncols=1,
    #                 height_ratios=[1],
    #                 width_ratios=[10],
    #                 hspace=0.0,
    #                 wspace=0.0
    #             )
    #             ax1 = fig.add_subplot(gs[0])
                
    #         ax1.margins(x=0)

    #         # Create copy of y axis
    #         ax2 = ax1.twiny()
    #         # Set orders of appearance
    #         ax1.set_zorder(2)
    #         ax2.set_zorder(1)

    #         # Turn axes off
    #         ax1.axis('off')
    #         ax2.axis('off')

    #         # Plot geometry polygons twice (side by side)
    #         if np.abs(origin_norm.vmax - origin_norm.vmin) > 1e-3:
    #             geometry.loc[geometry.geometry_type==self.settings['origin_geometry_type'],:].plot(
    #                             ax=ax1,
    #                             column='origin_demand',
    #                             legend=False,
    #                             edgecolor='none',
    #                             alpha=1.0,#self.settings['opacity'],
    #                             cmap=self.settings['aux_colormap'][0],
    #                             vmin=origin_norm.vmin,
    #                             vmax=origin_norm.vmax,
    #                             zorder=1
    #             )

    #         # Horizontally and/or vertically translate geometries
    #         x_shift = 8000.0
    #         y_shift = 0.0
    #         geometry_translated = deepcopy(geometry.loc[geometry.geometry_type==self.settings['destination_geometry_type']])
    #         geometry_translated.loc[:,"geometry"] = geometry_translated.translate(x_shift,y_shift)

    #         if np.abs(destination_norm.vmax - destination_norm.vmin) > 1e-3:
    #             geometry_translated.plot(
    #                 ax=ax2,
    #                 column='destination_demand',
    #                 legend=False,
    #                 edgecolor='none',
    #                 alpha=1.0,#self.settings['opacity'],
    #                 cmap=self.settings['aux_colormap'][1],
    #                 vmin=destination_norm.vmin,
    #                 vmax=destination_norm.vmax,
    #                 zorder=1
    #             )
            
    #         # Plot centroids and transparent boundaries
    #         geometry.loc[geometry.geometry_type==self.settings['origin_geometry_type'],:].plot(
    #                 ax=ax1,
    #                 facecolor="none",
    #                 edgecolor='black',
    #                 zorder=10
    #         )
    #         geometry_translated.loc[geometry.geometry_type==self.settings['destination_geometry_type'],:].plot(
    #                 ax=ax1,
    #                 facecolor="none",
    #                 edgecolor='black',
    #                 zorder=10
    #         )
    #         geometry.loc[geometry.geometry_type==self.settings['origin_geometry_type'],:].centroid.plot(
    #                 ax=ax1,
    #                 facecolor="black",
    #                 edgecolor='black',
    #                 zorder=10
    #         )
    #         geometry_translated.loc[geometry.geometry_type==self.settings['destination_geometry_type'],:].centroid.plot(
    #                 ax=ax1,
    #                 facecolor="black",
    #                 edgecolor='black',
    #                 zorder=10
    #         )

    #         # Axes limits
    #         ax2.set_xlim(ax1.get_xlim())
    #         ax2.set_ylim(ax1.get_ylim())

    #         # Construct colorbars for flows
    #         if self.settings['colorbar']: 
    #             flow_cbar = fig.colorbar(
    #                     flow_mapper, 
    #                     ax=flow_cbar_ax,
    #                     fraction=1.0,
    #                     pad=0.0
    #             )
    #             flow_cbar.ax.tick_params(labelsize=self.settings['tick_font_size'])
    #             if self.settings['colorbar_title'] is not None:
    #                 flow_cbar.ax.set_title(self.settings['colorbar_title'].replace("_"," ").capitalize(),fontsize=self.settings['legend_label_size'])
    #             else:
    #                 flow_cbar.ax.set_title('Flow',fontsize=self.settings['legend_label_size'])
    #             flow_cbar.locator = mpl.ticker.MaxNLocator(nbins=self.settings['x_tick_frequency'])
    #             flow_cbar.ax.yaxis.set_ticks_position('left')
    #             flow_cbar.update_ticks()
                
    #             # Construct colorbars for margins (rowsums column sums)
    #             origin_flow_mapper._A = []
    #             destination_flow_mapper._A = []

    #             if np.abs(origin_norm.vmax - origin_norm.vmin) > 1e-3:
    #                 origin_cbar = fig.colorbar(
    #                     origin_flow_mapper, 
    #                     ax=origin_cbar_ax,
    #                     fraction=1.0,
    #                     pad=0.0,
    #                 )
    #                 origin_cbar.ax.yaxis.set_ticks_position('left')
    #                 origin_cbar.ax.tick_params(labelsize=self.settings['tick_font_size'])
    #                 origin_cbar.ax.set_title('Origin',fontsize=self.settings['legend_label_size'])
    #                 origin_cbar.locator = mpl.ticker.MaxNLocator(nbins=self.settings['x_tick_frequency'])
    #                 origin_cbar.update_ticks()
                
    #             # Origin demand colorbar
    #             if np.abs(destination_norm.vmax - destination_norm.vmin) > 1e-3:
    #                 destination_cbar = fig.colorbar(
    #                     destination_flow_mapper, 
    #                     ax=destination_cbar_ax,
    #                     fraction=1.0,
    #                     pad=0.0,
    #                 )
    #                 destination_cbar.ax.yaxis.set_ticks_position('left')
    #                 destination_cbar.ax.tick_params(labelsize=self.settings['tick_font_size'])
    #                 destination_cbar.ax.set_title('Destination',fontsize=self.settings['legend_label_size'])
    #                 destination_cbar.locator = mpl.ticker.MaxNLocator(nbins=self.settings['x_tick_frequency'])
    #                 destination_cbar.update_ticks()

    #         # Plot arrows from origin to destination
    #         for i in range(len(table_gdf)):
    #             # Extact origin centroid
    #             p1 = np.array([
    #                 table_gdf.iloc[i].origin_geometry.centroid.xy[0][0],
    #                 table_gdf.iloc[i].origin_geometry.centroid.xy[1][0]
    #             ])
    #             # Extract destination centroid
    #             p2 = np.array([
    #                 table_gdf.iloc[i].destination_geometry.centroid.xy[0][0]+x_shift,
    #                 table_gdf.iloc[i].destination_geometry.centroid.xy[1][0]+y_shift
    #             ])

    #             # Add arrow if the two points are not the same
    #             if not np.all(np.abs(p1-p2)<=1e-10):
    #                 # Compute angle between points
    #                 angle = (p2[1]-p1[1])/(p2[0]-p1[0])
    #                 # If variance records are available make line as thick as variance associated with flow
    #                 # Apply sigmoid function to variance to a value in [0,1]
    #                 if self.settings['statistic'][0][0].lower() == 'mean_variance':
    #                     # Define line width
    #                     linewidth = self.settings['linewidth']*positive_sigmoid(table_gdf.iloc[i].flow_variance,30)
    #                     # (table_gdf.iloc[i].flow_variance)
    #                     # positive_sigmoid(table_gdf.iloc[i].flow_variance,30)
    #                     # Define transparency percentage
    #                     #((table_gdf.iloc[i].flow/table_gdf.iloc[i].origin_demand))
    #                     opacity = self.settings['opacity']
    #                     #min(table_gdf.iloc[i].flow/table_gdf.iloc[i].flow_variance,1.0)
    #                     #self.settings['opacity'] #positive_sigmoid((table_gdf.iloc[i].flow/table_gdf.iloc[i].origin_demand),1)# self.settings['opacity']#positive_sigmoid(table_gdf.iloc[i].flow_mean/table_gdf.iloc[i].flow_variance,30)
    #                 else:
    #                     # Define line width
    #                     linewidth = self.settings['linewidth']
    #                     # Define transparency percentage
    #                     opacity = self.settings['opacity']

    #                 # Do not show arrows if origin or destination ids are not in the subset provided
    #                 plot_arrow = True
    #                 if len(self.settings['origin_ids']) > 0 and (table_gdf.iloc[i].origin not in self.settings['origin_ids']):
    #                     plot_arrow = False
    #                 if len(self.settings['destination_ids']) > 0 and (table_gdf.iloc[i].destination not in self.settings['destination_ids']):
    #                     plot_arrow = False

    #                 if plot_arrow:
    #                     # Create arrow from origin to destination
    #                     arrow = patches.ConnectionPatch(
    #                         xyA=p1,
    #                         xyB=p2,
    #                         coordsA = "data", 
    #                         coordsB = "data", 
    #                         axesA = ax1,
    #                         axesB = ax2,
    #                         connectionstyle=f"arc3,rad={np.sign(angle)*0.3}",
    #                         color=flow_mapper.to_rgba(table_gdf.iloc[i].flow), 
    #                         linewidth=linewidth,
    #                         arrowstyle="-|>",
    #                         mutation_scale=2.0*self.settings['linewidth'], # controls arrow head size
    #                         alpha=opacity,
    #                         zorder=1,
    #                     )
    #                     # Add arrow to patches
    #                     ax1.add_artist(arrow)

    #         # Annotate
    #         if self.settings['annotate']:
    #             origin_geometry_ids = sorted(geometry[geometry.geometry_type==self.settings['origin_geometry_type']].geometry_id.values)
    #             destination_geometry_ids = sorted(geometry[geometry.geometry_type==self.settings['destination_geometry_type']].geometry_id.values)
    #             for i,oid in enumerate(origin_geometry_ids):
    #                 # Get geometry info
    #                 geom = geometry.loc[geometry.geometry_id == oid,:]
    #                 # Get annotation x,y coordinates
    #                 coords = list(geom.geometry.values[0].centroid.coords[0])
    #                 coords[1] *= 5000/4999
    #                 ax1.annotate(
    #                     text=str(i), 
    #                     color='black',
    #                     xy=coords,
    #                     horizontalalignment='center',
    #                     fontsize=self.settings['annotation_label_size'],
    #                     zorder=10
    #                 )
    #             for j,did in enumerate(destination_geometry_ids):
    #                 # Get geometry info
    #                 geom = geometry_translated.loc[geometry_translated.geometry_id == did,:]
    #                 # Get annotation x,y coordinates
    #                 coords = list(geom.geometry.values[0].centroid.coords[0])
    #                 coords[1] *= 5000/4999
    #                 ax1.annotate(
    #                     text=str(j), 
    #                     color='black',
    #                     xy=coords,
    #                     horizontalalignment='center',
    #                     fontsize=self.settings['annotation_label_size'],
    #                     zorder=10
    #                 )

    #         # Remove space
    #         gs.tight_layout(fig)
    #         # Write figure
    #         write_figure(fig,filepath,**self.settings)

    def origin_destination_table_tabular(self):
        
        self.logger.info('Running origin_destination_table_tabular')

        for output_directory in tqdm(self.outputs_directories): 
            self.logger.debug(f"Experiment id {output_directory}")
            # Load contingency table
            outputs = Outputs(
                experiment=output_directory,
                settings=self.settings,
                # order is important in sample_names
                sample_names = (['ground_truth_table']+list(self.settings['sample'])),
                slice_samples=True,
                disable_logger=True
            )

            for sample in self.settings['sample']:
                
                # Load table
                try:
                    table = outputs.apply_sample_statistics(
                        samples=outputs.experiment.results[sample],
                        sample_name=sample,
                        statistic_axes=self.settings['statistic'][0]
                    )
                    origin_demand = outputs.apply_sample_statistics(
                        samples=table,
                        sample_name=sample,
                        statistic_axes=self.settings['statistic'][1]
                    )
                    destination_demand = outputs.apply_sample_statistics(
                        samples=table,
                        sample_name=sample,
                        statistic_axes=self.settings['statistic'][2]
                    )
                except:
                    self.logger.debug(traceback.format_exc())
                    self.logger.error(f"Could not load sample {sample} for experiment {outputs.experiment_id}. Tabular heatmap could not be plotted.")
                    continue

                # Get table dimensions
                I,J = np.shape(table)
                cells = [(i,j) for i in range(I+1) for j in range(J+1)]

                # Define filepath
                self.settings['viz_type'] = 'tabular'
                filename = outputs.create_filename(sample=sample)
                filepath = os.path.join(outputs.outputs_path,'figures',filename)

                flow_colorbar_min,flow_colorbar_max = None, None
                if self.settings.get('main_colorbar_limit',None) is not None:
                    flow_colorbar_min,flow_colorbar_max = self.settings['main_colorbar_limit']
                else:
                    flow_colorbar_min,flow_colorbar_max = np.min(table.ravel()),np.max(table.ravel())
                
                origin_colorbar_min,origin_colorbar_max = None, None
                destination_colorbar_min,destination_colorbar_max = None, None
                if len(self.settings.get('auxiliary_colorbar_limit',[])) > 0:
                    if len(np.shape(self.settings['auxiliary_colorbar_limit'])) == 1:
                        origin_colorbar_min,origin_colorbar_max = self.settings['auxiliary_colorbar_limit']
                    elif len(np.shape(self.settings['auxiliary_colorbar_limit'])) > 1:
                        origin_colorbar_min,origin_colorbar_max = self.settings['auxiliary_colorbar_limit'][0]
                        destination_colorbar_min,destination_colorbar_max = self.settings['auxiliary_colorbar_limit'][1]
                else:
                    origin_colorbar_min,origin_colorbar_max = np.min(origin_demand.ravel()),np.max(origin_demand.ravel())
                    destination_colorbar_min,destination_colorbar_max = np.min(destination_demand.ravel()), np.max(destination_demand.ravel())
                    
                # Normalise colormaps
                if flow_colorbar_min < 0:
                    # If error flows are plotted, center colormap at zero
                    flow_norm = mpl.colors.SymLogNorm(vmin=flow_colorbar_min, vmax=flow_colorbar_max, linthresh=0.001)
                else:
                    flow_norm = mpl.colors.TwoSlopeNorm(vmin=flow_colorbar_min, vmax=flow_colorbar_max,vcenter=float(self.settings.get('fvcenter',np.mean(table))))
                if origin_colorbar_min < 0:
                    # If error origin demand is plotted, center colormap at zero
                    origin_norm = mpl.colors.SymLogNorm(vmin=origin_colorbar_min, vmax=origin_colorbar_max, linthresh=0.001)
                else:
                    origin_norm = mpl.colors.Normalize(vmin=origin_colorbar_min, vmax=origin_colorbar_max)
                if destination_colorbar_min < 0:
                    # If error destination demand is plotted, center colormap at zero
                    destination_norm = mpl.colors.SymLogNorm(vmin=destination_colorbar_min, vmax=destination_colorbar_max, linthresh=0.001)
                else:
                    destination_norm = mpl.colors.Normalize(vmin=destination_colorbar_min, vmax=destination_colorbar_max)
                print('\n')
                print(outputs.experiment_id)
                print(sample)
                print('\n')
                print('flow')
                print('data',np.min(table.ravel()),np.max(table.ravel()))
                print('settings',flow_norm.vmin,flow_norm.vmax)
                print('\n')
                print('origin')
                print('data',np.min(origin_demand.ravel()),np.max(origin_demand.ravel()))
                print('settings',origin_norm.vmin,origin_norm.vmax)
                print('\n')
                print('destination')
                print('data',np.min(destination_demand.ravel()), np.max(destination_demand.ravel()))
                print('settings',destination_norm.vmin,destination_norm.vmax)

                # Define colorbars for flows, and margins
                flow_base_cmap = cm.get_cmap(self.settings['main_colormap'])
                # Clip colors in colorbar to specific range
                colors = flow_base_cmap( 
                    np.linspace(
                            self.settings['color_segmentation_limits'][0], 
                            self.settings['color_segmentation_limits'][1], 
                            self.settings['x_tick_frequency']
                    )
                )
                flow_color_segmented_cmap = mpl.colors.LinearSegmentedColormap.from_list(self.settings['main_colormap'], colors)
                flow_mapper = cm.ScalarMappable(
                            norm=flow_norm, 
                            cmap=flow_color_segmented_cmap
                )
                origin_flow_mapper = cm.ScalarMappable(
                    norm = origin_norm,
                    cmap = cm.get_cmap(self.settings['aux_colormap'][0])
                )
                destination_flow_mapper = cm.ScalarMappable(
                    norm = destination_norm, 
                    cmap = cm.get_cmap(self.settings['aux_colormap'][1])
                )
                
                # Setup plot
                fig = plt.figure(figsize=self.settings['figure_size'])
                if self.settings['colorbar']:
                    widths_ratios = [2,2,2,J,1]
                    height_ratios = [1,I]
                    if self.settings['transpose']:
                        widths_ratios = [4,4,4,I,1]
                        height_ratios = [1,J]
                    
                    gs = GridSpec(
                            nrows=2,
                            ncols=5,
                            hspace=0.0,
                            wspace=0.0,
                            width_ratios=widths_ratios,
                            height_ratios=height_ratios
                    )

                    cbar_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec = gs[:,0:3], wspace = 0.5, hspace=0.0, width_ratios = [3,2,1])
                    table_ax = fig.add_subplot(gs[1,3])
                    # flow_cbar_ax = fig.add_subplot(gs[:,0])
                    # origin_cbar_ax = fig.add_subplot(gs[:,1])
                    # destination_cbar_ax = fig.add_subplot(gs[:,2])
                    flow_cbar_ax = fig.add_subplot(cbar_gs[0])
                    origin_cbar_ax = fig.add_subplot(cbar_gs[1])
                    destination_cbar_ax = fig.add_subplot(cbar_gs[2])
                    if self.settings['transpose']:
                        origin_margin_ax = fig.add_subplot(gs[0,3])
                        destination_margin_ax = fig.add_subplot(gs[1,4])
                    else:
                        origin_margin_ax = fig.add_subplot(gs[1,4])
                        destination_margin_ax = fig.add_subplot(gs[0,3])
                    # Set order of appearance
                    # flow_cbar_ax.set_zorder(2)
                    # origin_cbar_ax.set_zorder(3)
                    # destination_cbar_ax.set_zorder(3)
                    flow_cbar_ax.axis('off')
                    origin_cbar_ax.axis('off')
                    destination_cbar_ax.axis('off')
                else:
                    image_widths_ratios = [J,1]
                    image_heights_ratios = [1,I]
                    if self.settings['transpose']:
                        image_widths_ratios = [I,1]
                        image_heights_ratios = [1,J]
                    
                    image_gs = GridSpec(
                        nrows = 2,
                        ncols = 2,
                        width_ratios = image_widths_ratios,
                        height_ratios = image_heights_ratios,
                        wspace = 0.0,
                        hspace = 0.0
                    )

                    table_ax = fig.add_subplot(image_gs[1,0])
                    if self.settings['transpose']:
                        destination_margin_ax = fig.add_subplot(image_gs[1,1])                
                        origin_margin_ax = fig.add_subplot(image_gs[0,0])
                    else:
                        origin_margin_ax = fig.add_subplot(image_gs[1,1])
                        destination_margin_ax = fig.add_subplot(image_gs[0,0])
                    
                
                # Set order of appearance
                table_ax.set_zorder(1)
                origin_margin_ax.set_zorder(2)
                destination_margin_ax.set_zorder(2)
                # Set axes off
                table_ax.axis('off')
                
                # Plot table
                if self.settings['transpose']:
                    table_ax.imshow(
                        table.T,
                        cmap=flow_mapper.get_cmap(),
                        interpolation='nearest',
                        norm = flow_norm,
                        zorder = 1,
                        aspect='auto',
                    )
                else:
                    table_ax.imshow(
                        table,
                        cmap=flow_color_segmented_cmap,#flow_mapper.get_cmap(),#self.settings['main_colormap'],
                        interpolation='nearest',
                        norm = flow_norm,
                        zorder = 1,
                        aspect='auto'
                    )
                # Plot margins
                if self.settings['transpose']:
                    origin_margin_ax.imshow(
                        table.T.sum(axis=0)[np.newaxis,:], 
                        cmap=self.settings['aux_colormap'][0],
                        interpolation='nearest', 
                        # norm = origin_norm,
                        aspect='auto'
                    )
                else:
                    origin_margin_ax.imshow(
                        table.sum(axis=1)[np.newaxis,:], 
                        cmap=self.settings['aux_colormap'][0],
                        interpolation='nearest', 
                        norm = origin_norm,
                        aspect='auto'
                    )
                if self.settings['transpose']:
                    destination_margin_ax.imshow(
                        table.T.sum(axis=1)[:,np.newaxis], 
                        cmap=self.settings['aux_colormap'][1],
                        interpolation='nearest', 
                        norm = destination_norm,
                        aspect='auto'
                    )
                else:
                    destination_margin_ax.imshow(
                        table.sum(axis=0)[:,np.newaxis], 
                        cmap=self.settings['aux_colormap'][1],
                        interpolation='nearest', 
                        norm = destination_norm,
                        aspect='auto'
                    )
                    
                # X-Y ticks
                if self.settings['transpose']:

                    origin_margin_ax.set_yticks([])
                    origin_margin_ax.set_xticks(range(I))
                    origin_margin_ax.set_xticklabels(range(I),fontsize=self.settings['tick_font_size'])
                    origin_margin_ax.set_title(
                        'Origins',
                        fontsize=self.settings['legend_label_size'],
                        # labelpad=self.settings['axis_labelpad']
                    )
                    origin_margin_ax.xaxis.locator = mpl.ticker.MaxNLocator(nbins=I)
                    origin_margin_ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
                    origin_margin_ax.xaxis.set_label_position("top")
                    origin_margin_ax.xaxis.tick_top()                

                    destination_margin_ax.set_xticks([])
                    destination_margin_ax.set_yticks(range(J))
                    destination_margin_ax.set_yticklabels(range(J),fontsize=self.settings['tick_font_size'])
                    destination_margin_ax.set_ylabel(
                        'Destinations',
                        fontsize=self.settings['legend_label_size'],
                        rotation=270,
                        labelpad=self.settings['axis_labelpad']
                    )
                    destination_margin_ax.yaxis.locator = mpl.ticker.MaxNLocator(nbins=J)
                    destination_margin_ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
                    destination_margin_ax.yaxis.set_label_position("right")
                    destination_margin_ax.yaxis.tick_right()
                    
                else:
                    origin_margin_ax.set_xticks([])
                    origin_margin_ax.set_yticks(range(I))
                    origin_margin_ax.set_yticklabels(range(I),fontsize=self.settings['tick_font_size'])
                    origin_margin_ax.set_ylabel(
                        'Origins',
                        fontsize=self.settings['tick_font_size'],
                        rotation=270,
                        labelpad=self.settings['axis_labelpad']
                    )
                    origin_margin_ax.locator = mpl.ticker.MaxNLocator(nbins=I)
                    origin_margin_ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
                    origin_margin_ax.yaxis.set_label_position("right")
                    origin_margin_ax.yaxis.tick_right()

                    destination_margin_ax.set_yticks([])
                    destination_margin_ax.set_xticks(range(J))
                    destination_margin_ax.set_xticklabels(range(J),fontsize=self.settings['tick_font_size'])
                    destination_margin_ax.set_title(
                        'Destinations',
                        fontsize=self.settings['legend_label_size'],
                        # labelpad=self.settings['axis_labelpad']
                    )
                    destination_margin_ax.locator = mpl.ticker.MaxNLocator(nbins=J)
                    destination_margin_ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
                    destination_margin_ax.xaxis.set_label_position("top")
                    destination_margin_ax.xaxis.tick_top()
                
                table_ax.xaxis.set_tick_params(labelbottom=False)
                table_ax.yaxis.set_tick_params(labelleft=False)
                table_ax.set_xticks([])
                table_ax.set_yticks([])
                table_ax.get_xaxis().set_ticks([])
                table_ax.get_yaxis().set_ticks([])

                # Annotate
                covered_cells = None
                if self.settings['annotate']:
                    try:
                        # Compute coverage probabilities
                        coverage_probabilities = coverage_probability(
                                            tab=outputs.experiment.results.get(sample,None),
                                            tab0=outputs.ground_truth_table,
                                            kwargs={"region_mass":0.99}
                                        )

                        # Eliminate last axis
                        coverage_probabilities = coverage_probabilities.sum(axis=-1)
                    except:
                        self.logger.debug(traceback.format_exc())
                        self.logger.error(f"Annotation by coverage probabilities omitted")
                        continue
                    # dummy_config = Namespace(**{'settings':outputs.experiment.subconfig})
                    # ct = instantiate_ct(table=None,config=dummy_config,disable_logger=True)
                    # print('\n')
                    for (i,j),label in np.ndenumerate(coverage_probabilities):
                        # lower_bound_hpdr,upper_bound_hpdr =     calculate_min_interval(np.sort(outputs.experiment.results.get(sample,None)[(...,i,j)]),0.01)
                        # print(
                        #     'covered' if bool(label) else '',
                        #     'fixed cell' if bool((i,j) in ct.constraints['cells']) else '',
                        #     (i,j),
                        #     ct.table[i,j],
                        #     lower_bound_hpdr,
                        #     upper_bound_hpdr
                        # )
                        if bool(label):
                            if self.settings['transpose']:
                                table_ax.text(
                                    i,
                                    j,
                                    u'\u2713',
                                    ha='center',
                                    va='center',
                                    fontsize=self.settings['annotation_label_size']
                                )
                            else:
                                table_ax.text(
                                    i,
                                    j,
                                    u'\u2713',
                                    ha='center',
                                    va='center',
                                    fontsize=self.settings['annotation_label_size']
                                )
                    # Get covered cells
                    print(np.mean(coverage_probabilities))
                    covered_cell_locations = np.argwhere(coverage_probabilities==1)
                    covered_cells = {
                        "x":covered_cell_locations[:,0].astype('int32'),
                        "y":covered_cell_locations[:,1].astype('int32'),
                        "label":f"{sample}_covered_cell_coordinates",
                        "subconfig":outputs.experiment.subconfig,
                        "outputs_path":outputs.outputs_path
                    }

                if self.settings['colorbar']:
                    # Colorbar
                    flow_cbar = fig.colorbar(
                        flow_mapper,
                        ax=flow_cbar_ax,
                        fraction=1,
                        pad=0.0,
                        location='left',
                    )

                    flow_cbar.ax.tick_params(labelsize=self.settings['tick_font_size'])
                    if self.settings['colorbar_title'] is not None:
                        flow_cbar.ax.set_title(self.settings['colorbar_title'].replace("_"," ").capitalize(),fontsize=self.settings['legend_label_size'])
                    else:
                        flow_cbar.ax.set_title('Flow',fontsize=self.settings['legend_label_size'])
                    flow_cbar.locator = mpl.ticker.MaxNLocator(nbins=self.settings['x_tick_frequency'])
                    flow_cbar.ax.yaxis.set_ticks_position('left')
                    flow_cbar.update_ticks()

                    origin_margin_cbar = fig.colorbar(
                        origin_flow_mapper,
                        ax=origin_cbar_ax,
                        fraction=1, 
                        pad=0.0,
                    )
                    origin_margin_cbar.ax.tick_params(labelsize=self.settings['tick_font_size'])
                    origin_margin_cbar.ax.set_title(
                        r'$O_i$',
                        fontsize=self.settings['legend_label_size']
                    )
                    origin_margin_cbar.locator = mpl.ticker.MaxNLocator(nbins=self.settings['x_tick_frequency'])
                    origin_margin_cbar.ax.yaxis.set_ticks_position('left')
                    origin_margin_cbar.update_ticks()

                    destination_margin_cbar = fig.colorbar(
                        destination_flow_mapper,
                        ax=destination_cbar_ax,
                        fraction=1, 
                        pad=0.0,
                    )

                    destination_margin_cbar.ax.tick_params(labelsize=self.settings['tick_font_size'])
                    destination_margin_cbar.ax.set_title(
                        r'$D_j$',
                        fontsize=self.settings['legend_label_size']
                    )
                    destination_margin_cbar.locator = mpl.ticker.MaxNLocator(nbins=self.settings['x_tick_frequency'])
                    destination_margin_cbar.ax.yaxis.set_ticks_position('left')
                    destination_margin_cbar.update_ticks()

                # Title
                # if self.settings['figure_title'] is not None:
                #     table_ax.set_title(
                #         self.settings['figure_title'].replace("_"," ").capitalize(),
                #         fontsize=self.settings['title_label_size']
                #     )
                # else:
                #     table_ax.set_title(
                #         self.settings['sample'][0].replace("_"," ").capitalize(),
                #         fontsize=self.settings['title_label_size']
                #     )

                # Remove space
                plt.tight_layout()
                # Write figure
                write_figure(
                    fig,
                    filepath,
                    **self.settings
                )
                # Collect table data
                table_data = {
                    "x":np.array([cell[0] for cell in cells],dtype='int32'),
                    "y":np.array([cell[1] for cell in cells],dtype='int32'),
                    "z":np.array([table[cell[0]-1,cell[1]-1] if ((cell[0] > 0) and (cell[1] > 0)) else 0 for cell in cells],dtype='float64'),
                    "color":np.array([flow_norm(table[cell[0]-1,cell[1]-1]) if ((cell[0] > 0) and (cell[1] > 0)) else 0.5 for cell in cells],dtype='float64'),
                    "label":f"{sample}_cell_data",
                    "subconfig":outputs.experiment.subconfig,
                    "outputs_path":outputs.outputs_path
                }
                origin_demand_data = {
                    "x":np.array([cell[0] for cell in cells if (cell[1] >= J-1) and (cell[0] < I)],dtype='int32'),
                    "y":np.array([cell[1]+1 for cell in cells if (cell[1] >= J-1) and (cell[0] < I)],dtype='int32'),
                    "z":np.array([origin_demand[cell[0]-1] for cell in cells if (cell[1] >= J-1) and (cell[0] < I)],dtype='float64'),
                    "color":np.array([origin_norm(origin_demand[cell[0]-1]) for cell in cells if (cell[0] >= I-1) and (cell[1] < J)],dtype='float64'),
                    "label":f"{sample}_origin_demand_cell_data",
                    "subconfig":outputs.experiment.subconfig,
                    "outputs_path":outputs.outputs_path
                }
                destination_demand_data = {
                    "x":np.array([cell[0]+1 for cell in cells if (cell[0] >= I-1) and (cell[1] < J)],dtype='int32'),
                    "y":np.array([cell[1] for cell in cells if (cell[0] >= I-1) and (cell[1] < J)],dtype='int32'),
                    "z":np.array([destination_demand[cell[1]-1] for cell in cells if (cell[0] >= I-1) and (cell[1] < J)],dtype='float64'),
                    "color":np.array([destination_norm(destination_demand[cell[1]-1]) for cell in cells if (cell[0] >= I-1) and (cell[1] < J)],dtype='float64'),
                    "label":f"{sample}_destination_demand_cell_data",
                    "subconfig":outputs.experiment.subconfig,
                    "outputs_path":outputs.outputs_path
                }
                print(table.sum(axis=0))
                print(destination_demand_data['z'])
                print(destination_demand_data['x'].shape)
                print(destination_demand_data['y'].shape)
                print(destination_demand_data['z'].shape)

                flow_colorbar_data = {
                    "ticks":np.array([t for t in flow_cbar.ax.get_yticks()],dtype='float32'),
                    "locations":np.array([flow_norm(t) for t in flow_cbar.ax.get_yticks()],dtype='float32'),
                    "label":f"{sample}_colorbar_ticks",
                    "subconfig":outputs.experiment.subconfig,
                    "outputs_path":outputs.outputs_path
                }
                origin_demand_colorbar_data = {
                    "ticks":np.array([t for t in origin_margin_cbar.ax.get_yticks()],dtype='float32'),
                    "locations":np.array([origin_norm(t) for t in origin_margin_cbar.ax.get_yticks()],dtype='float32'),
                    "label":f"{sample}_origin_demand_colorbar_ticks",
                    "subconfig":outputs.experiment.subconfig,
                    "outputs_path":outputs.outputs_path
                }
                destination_demand_colorbar_data = {
                    "ticks":np.array([t for t in destination_margin_cbar.ax.get_yticks()],dtype='float32'),
                    "locations":np.array([destination_norm(t) for t in destination_margin_cbar.ax.get_yticks()],dtype='float32'),
                    "label":f"{sample}_destination_demand_colorbar_ticks",
                    "subconfig":outputs.experiment.subconfig,
                    "outputs_path":outputs.outputs_path
                }
                # Write figure data
                write_figure_data(
                    {outputs.experiment_id:flow_colorbar_data},
                    Path(filepath).parent,
                    groupby=[],
                    key_type={'ticks':'float','locations':'float'},
                    data_format='txt',
                    data_precision=4,
                )
                write_figure_data(
                    {outputs.experiment_id:origin_demand_colorbar_data},
                    Path(filepath).parent,
                    groupby=[],
                    key_type={'ticks':'float','locations':'float'},
                    data_format='txt',
                    data_precision=4,
                )
                write_figure_data(
                    {outputs.experiment_id:destination_demand_colorbar_data},
                    Path(filepath).parent,
                    groupby=[],
                    key_type={'ticks':'float','locations':'float'},
                    data_format='txt',
                    data_precision=4,
                )
                write_figure_data(
                    {outputs.experiment_id:table_data},
                    Path(filepath).parent,
                    groupby=[],
                    key_type={'x':'int','y':'int','z':'float','color':'float'},
                    **self.settings
                )
                write_figure_data(
                    {outputs.experiment_id:origin_demand_data},
                    Path(filepath).parent,
                    groupby=[],
                    key_type={'x':'int','y':'int','z':'float','color':'float'},
                    **self.settings
                )
                write_figure_data(
                    {outputs.experiment_id:destination_demand_data},
                    Path(filepath).parent,
                    groupby=[],
                    key_type={'x':'int','y':'int','z':'float','color':'float'},
                    **self.settings
                )
                if covered_cells is not None:
                    write_figure_data(
                        {outputs.experiment_id:covered_cells},
                        Path(filepath).parent,
                        groupby=[],
                        key_type={'x':'int','y':'int'},
                        data_format='dat',
                        precision=0
                    )
                if sample == 'table':
                    dummy_config = Namespace(**{'settings':outputs.experiment.subconfig})
                    ct = instantiate_ct(table=None,config=dummy_config,disable_logger=True)  
                    # for cell in ct.constraints['cells']:
                        # print('ground truth', ct.table[cell[0],cell[1]])
                        # print('min',min(outputs.experiment.results['table'][:,cell[0],cell[1]]))
                        # print('max',max(outputs.experiment.results['table'][:,cell[0],cell[1]]))
                        # print('\n')
                    all_cells = sorted([tuplize(cell) for cell in product(*[range(dim) for dim in ct.dims])])

                    fixed_cells = []
                    for cell in sorted(all_cells):
                        if min(outputs.experiment.results['table'][:,cell[0],cell[1]]) == max(outputs.experiment.results['table'][:,cell[0],cell[1]]):
                            fixed_cells.append(cell)
                    
                    
                    if str_in_list('cells',ct.constraints.keys()):
                        fixed_cells = {
                            "x":np.array([c[0] for c in ct.constraints['cells']],dtype='int32'),
                            "y":np.array([c[1] for c in ct.constraints['cells']],dtype='int32'),
                            "label":f"{sample}_fixed_cell_coordinates",
                            "subconfig":outputs.experiment.subconfig,
                            "outputs_path":outputs.outputs_path
                        }
                        write_figure_data(
                            {outputs.experiment_id:fixed_cells},
                            Path(filepath).parent,
                            groupby=[],
                            key_type={'x':'int','y':'int'},
                            data_format='dat',
                            data_precision=self.settings['data_precision']
                        )


    # def origin_destination_table_colorbars(self):
        
    #     self.logger.info('Running origin_destination_table_colorbars')
    #     # Define arrow kwargs
    #     kw = dict(arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8")
        
    #     for output_directory in tqdm(self.outputs_directories): 
    #         self.logger.debug(f"Experiment id {output_directory}")
    #         # Load contingency table
    #         outputs = Outputs(
    #             output_directory,
    #             self.settings,
    #             disable_logger=True
    #         )
    
    #         # Define filepath
    #         filename = f"table_{outputs.experiment.subconfig['table_dim']}_total_{self.settings['table_total']}_{self.settings['sample'][0]}_flows_colorbar"

    #         filepath = os.path.join(outputs.outputs_path,'figures',filename)

    #         flow_colorbar_min,flow_colorbar_max = None, None
    #         if str_in_list("main_colorbar_limit",self.settings.keys()):
    #             flow_colorbar_min,flow_colorbar_max = self.settings['main_colorbar_limit']
            
    #         origin_colorbar_min,origin_colorbar_max = None, None
    #         destination_colorbar_min,destination_colorbar_max = None, None
    #         if str_in_list("auxiliary_colorbar_limit",self.settings.keys()):
    #             if len(np.shape(self.settings['auxiliary_colorbar_limit'])) == 1:
    #                 origin_colorbar_min,origin_colorbar_max = self.settings['auxiliary_colorbar_limit']
    #             elif len(np.shape(self.settings['auxiliary_colorbar_limit'])) > 1:
    #                 origin_colorbar_min,origin_colorbar_max = self.settings['auxiliary_colorbar_limit'][0]
    #                 destination_colorbar_min,destination_colorbar_max = self.settings['auxiliary_colorbar_limit'][1]

    #         table = outputs.compute_sample_statistics(
    #                 outputs.experiment.results[self.settings['sample'][0]],
    #                 self.settings['sample'][0],
    #                 self.settings['statistic'][0][0]
    #         )
    #         origin_demand = outputs.compute_sample_statistics(
    #                 outputs.experiment.results[self.settings['sample'][0]],
    #                 self.settings['sample'][0],
    #                 'rowsums'
    #         )
    #         destination_demand = outputs.compute_sample_statistics(
    #             outputs.experiment.results[self.settings['sample'][0]],
    #             self.settings['sample'][0],
    #             'colsums'
    #         )
    #         self.settings['viz_type'] = 'colorbars'
    #         filename = outputs.create_filename()

    #         # Normalise colormaps
    #         if flow_colorbar_min is not None and flow_colorbar_max is not None:    
    #             flow_norm = mpl.colors.Normalize(vmin=flow_colorbar_min, vmax=flow_colorbar_max)
    #         else:
    #             flow_norm = mpl.colors.Normalize(vmin=np.min(table.ravel()), vmax=np.max(table.ravel()))
            
    #         if origin_colorbar_min is not None and origin_colorbar_max is not None:    
    #             origin_norm = mpl.colors.Normalize(vmin=origin_colorbar_min, vmax=origin_colorbar_max)
    #         else:
    #             origin_norm = mpl.colors.Normalize(vmin=np.min(origin_demand.ravel()), vmax=np.max(origin_demand.ravel()))
            
    #         if destination_colorbar_min is not None and destination_colorbar_max is not None:    
    #             destination_norm = mpl.colors.Normalize(vmin=destination_colorbar_min, vmax=destination_colorbar_max)
    #         else:
    #             destination_norm = mpl.colors.Normalize(vmin=np.min(destination_demand.ravel()), vmax=np.max(destination_demand.ravel()))

    #         print(flow_norm.vmin,flow_norm.vmax)
    #         print(origin_norm.vmin,origin_norm.vmax)
    #         print(destination_norm.vmin,destination_norm.vmax)

    #         # Define colorbars for flows, and margins
    #         flow_base_cmap = cm.get_cmap(self.settings['main_colormap'])
    #         # Clip colors in colorbar to specific range
    #         colors = flow_base_cmap( 
    #             np.linspace(
    #                     self.settings['color_segmentation_limits'][0], 
    #                     self.settings['color_segmentation_limits'][1], 
    #                     self.settings['x_tick_frequency']
    #             )
    #         )
    #         flow_color_segmented_cmap = mpl.colors.LinearSegmentedColormap.from_list(self.settings['main_colormap'], colors)
    #         flow_mapper = cm.ScalarMappable(
    #                     norm=flow_norm, 
    #                     cmap=self.settings['main_colormap']#flow_color_segmented_cmap
    #         )
    #         origin_flow_mapper = cm.ScalarMappable(
    #                 norm=origin_norm, 
    #                 cmap=self.settings['aux_colormap'][0]
    #         )
    #         destination_flow_mapper = cm.ScalarMappable(
    #                 norm=destination_norm, 
    #                 cmap=self.settings['aux_colormap'][1]
    #         )

    #         # Setup figure
    #         fig = plt.figure(figsize=(3,6))
    #         gs = GridSpec(
    #             nrows=1,
    #             ncols=3,
    #             width_ratios=[1,1,1],
    #             hspace=0.0,
    #             wspace=1.0
    #         )
    #         origin_cbar_ax = fig.add_subplot(gs[0])
    #         destination_cbar_ax = fig.add_subplot(gs[1])
    #         flow_cbar_ax = fig.add_subplot(gs[2])
            
    #         flow_cbar_ax.axis('off')
    #         origin_cbar_ax.axis('off')
    #         destination_cbar_ax.axis('off')

    #         # Construct colorbars for flows
    #         flow_cbar = fig.colorbar(
    #                 flow_mapper, 
    #                 ax=flow_cbar_ax,
    #                 fraction=1.0,
    #                 pad=0.0,
    #                 # location = 'left'
    #         )
    #         flow_cbar.ax.tick_params(labelsize=self.settings['tick_font_size'])
    #         if self.settings['colorbar_title'] is not None:
    #             flow_cbar.ax.set_title(self.settings['colorbar_title'].replace("_"," ").capitalize(),fontsize=self.settings['legend_label_size'])
    #         else:
    #             flow_cbar.ax.set_title('Flow',fontsize=self.settings['legend_label_size'])
    #         flow_cbar.locator = mpl.ticker.MaxNLocator(nbins=self.settings['x_tick_frequency'])
    #         flow_cbar.ax.yaxis.set_ticks_position('left')
    #         flow_cbar.update_ticks()
            
    #         # Construct colorbars for margins (rowsums column sums)
    #         origin_flow_mapper._A = []
    #         destination_flow_mapper._A = []
        
    #         if np.abs(origin_norm.vmax - origin_norm.vmin) > 1e-3:
    #             origin_cbar = fig.colorbar(
    #                 origin_flow_mapper, 
    #                 ax=origin_cbar_ax,
    #                 fraction=1.0,
    #                 pad=0.1,
    #                 # location = 'left'
    #             )
    #             origin_cbar.ax.yaxis.set_ticks_position('left')
    #             origin_cbar.ax.tick_params(labelsize=self.settings['tick_font_size'])
    #             origin_cbar.ax.set_title('Origin',fontsize=self.settings['legend_label_size'])
    #             origin_cbar.locator = mpl.ticker.MaxNLocator(nbins=self.settings['x_tick_frequency'])
    #             origin_cbar.update_ticks()
            
    #         # Origin demand colorbar
    #         if np.abs(destination_norm.vmax - destination_norm.vmin) > 1e-3:
    #             destination_cbar = fig.colorbar(
    #                 destination_flow_mapper, 
    #                 ax=destination_cbar_ax,
    #                 fraction=1.0,
    #                 pad=0.1,
    #                 # location = 'left'
    #             )
    #             destination_cbar.ax.yaxis.set_ticks_position('left')
    #             destination_cbar.ax.tick_params(labelsize=self.settings['tick_font_size'])
    #             destination_cbar.ax.set_title('Destination',fontsize=self.settings['legend_label_size'])
    #             destination_cbar.locator = mpl.ticker.MaxNLocator(nbins=self.settings['x_tick_frequency'])
    #             destination_cbar.update_ticks()

    #         # Remove space
    #         gs.tight_layout(fig)

    #         # Write figure
    #         write_figure(fig,filepath,**self.settings)