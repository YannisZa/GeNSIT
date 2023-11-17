import os

os.environ['USE_PYGEOS'] = '0'
import sys
import traceback
import seaborn as sns
import geopandas as gpd
import sklearn.manifold
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path
from tqdm.auto import tqdm
from itertools import product
from scipy import interpolate
from argparse import Namespace
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
from statsmodels.graphics.tsaplots import plot_acf

from multiresticodm.utils import *
from multiresticodm.colormaps import *
from multiresticodm.config import Config
from multiresticodm.global_variables import *
from multiresticodm.outputs import Outputs,OutputSummary
from multiresticodm.contingency_table import instantiate_ct
from multiresticodm.spatial_interaction_model import instantiate_sim
from multiresticodm.probability_utils import log_odds_ratio_wrt_intensity
from multiresticodm.math_utils import map_distance_name_to_function,coverage_probability

latex_preamble = r'''
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
'''

mpl.rcParams['text.latex.preamble'] = latex_preamble


class Plot(object):

    def __init__(self,plot_ids:List[str],settings:dict,**kwargs):
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__,
            console_level = level,
        ) if kwargs.get('logger',None) is None else kwargs['logger']

        # Store settings
        self.settings = settings

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

    '''
    ╔═╗┌─┐┌┐┌┌─┐┬─┐┬┌─┐  ┌─┐┬  ┌─┐┌┬┐  ┌─┐┬ ┬┌┐┌┌─┐┌┬┐┬┌─┐┌┐┌┌─┐
    ║ ╦├┤ │││├┤ ├┬┘││    ├─┘│  │ │ │   ├┤ │ │││││   │ ││ ││││└─┐
    ╚═╝└─┘┘└┘└─┘┴└─┴└─┘  ┴  ┴─┘└─┘ ┴   └  └─┘┘└┘└─┘ ┴ ┴└─┘┘└┘└─┘
    '''

    def plot_2d_scatter(self,plot_settings,**kwargs):
        # Axes limits from settings (read only x limit)
        axes_lims = {}
        for var in ['x','y']:
            axes_lims[var] = [self.settings.get(f'{var}_limit',[None,None])[0],self.settings.get(f'{var}_limit',[None,None])[1]]
            # Otherwise read from data
            if None in axes_lims[var]:

                if axes_lims[var][0] is None:
                    min_val = np.infty
                else:
                    min_val = axes_lims[var][0]
                
                if axes_lims[var][1] is None:
                    max_val = -np.infty
                else:
                    max_val = axes_lims[var][1]
                
                for plot_setting in plot_settings:
                    if self.settings.get(f'{var}_discrete',False):
                        min_val = min([1,min_val])
                        max_val = max([len(set(len(plot_setting[var])))+1,max_val])
                    else:
                        min_val = min([np.min(plot_setting[var]),min_val])
                        max_val = max([np.max(plot_setting[var]),max_val])
                # Update axis limits
                axes_lims[var] = [min_val,max_val]
        
        # Figure size 
        fig = plt.figure(figsize=self.settings['figure_size'])
        
        # Plot benchmark (perfect predictions)
        if self.settings['benchmark']:
            plt.plot(
                np.linspace(*axes_lims['x']),
                np.linspace(*axes_lims['y']),
                linewidth=0.2,
                color='black'
            )
        if self.settings.get('x_label',''):
            plt.xlabel(
                self.settings['x_label'].replace("_"," "),
                fontsize=self.settings['axis_font_size'],
                labelpad=self.settings['axis_labelpad']
            )
        if self.settings.get('y_label',''):
            plt.ylabel(
                self.settings['y_label'].replace("_"," "),
                fontsize=self.settings['axis_font_size'],
                labelpad=self.settings['axis_labelpad']
            )
        
        # Axis limits
        plt.xlim(left=axes_lims['x'][0], right=axes_lims['x'][1])
        plt.ylim(bottom=axes_lims['y'][0], top=axes_lims['y'][1])
        print(axes_lims)
        # Ticks
        if self.settings.get('x_discrete',False):
            # Get sorted unique x values
            unique_x = set(list(flatten([plot_sett['x'] for plot_sett in plot_settings])))
            unique_x = sorted(list(unique_x))
            # Create a discrete hashmap
            x_hashmap = dict(zip(unique_x,range(1,len(unique_x)+1)))
            plt.xticks(
                ticks = list(x_hashmap.values()),
                labels = list(x_hashmap.keys())
            )
        if self.settings.get('y_discrete',False):
            # Get sorted unique x values
            unique_y = set(list(flatten([plot_sett['y'] for plot_sett in plot_settings])))
            unique_y = sorted(list(unique_y))
            # Create a discrete hashmap
            y_hashmap = dict(zip(unique_y,range(1,len(unique_y)+1)))
            plt.yticks(
                ticks = list(y_hashmap.values()),
                labels = list(y_hashmap.keys())
            )
        
        # 2D scatter plot
        for plot_sett in plot_settings:
            # Extract data
            x_range = list(map(lambda v: x_hashmap[v], plot_sett['x'])) \
                    if self.settings.get('x_discrete',False) \
                    else plot_sett['x']
            y_range = list(map(lambda v: y_hashmap[v], plot_sett['y'])) \
                    if self.settings.get('y_discrete',False) \
                    else plot_sett['y']
            sizes = plot_sett.get('size',int(self.settings['marker_size']))
            colours = plot_sett.get('colour',None)
            print('colours',colours)
            alphas = plot_sett.get('visibility',np.array([1.0]))/max(plot_sett.get('visibility',np.array([1.0])))
            labels = plot_sett.get('label','')
            marker = plot_sett.get('marker','o')
            hatch = plot_sett.get('hatch',None)
            # Plot x versus y
            for i in range(len(y_range)):
                plt.scatter(
                    x = x_range[i],
                    y = y_range[i],
                    s = sizes[i] if isinstance(sizes,Iterable) else sizes,
                    c = colours[i] if isinstance(colours,Iterable) else colours,
                    alpha = alphas[i] if len(alphas) > 1 else alphas[0],
                    label = labels[i] if not isinstance(alphas,str) else labels,
                    marker = marker,
                    hatch = hatch
                )

            # Annotate data
            if self.settings.get('annotate',False):
                for i, txt in enumerate(plot_sett.get(self.settings.get('annotation_label',''),[])):
                    plt.annotate(str(string_to_numeric(txt) if str(txt).isnumeric() else str(txt)), (x_range[i], y_range[i]))


        # Aspect ratio equal
        plt.gca().set_aspect('equal')
    
        # Legend
        try:
            # Ensure no duplicate entries in legend exist
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            leg = plt.legend(
                by_label.values(), 
                by_label.keys(),
                frameon = False,
                prop = {'size': self.settings.get('legend_label_size',None)}
            )
            leg._ncol = 1
        except:
            pass

        
        # Tight layout
        plt.tight_layout()

        # Get directory path and file name
        dirpath = kwargs['dirpath']
        filename = kwargs['filename']
        filepath = os.path.join(dirpath,filename)

        # Make outputs directories (if necessary)
        makedir(dirpath)

        # Write figure
        write_figure(
            fig,
            filepath,
            **self.settings
        )
        # Write figure data
        write_figure_data(
            plot_settings,
            filepath=filepath,
            key_type={'x':'float','y':'float'},
            aux_keys=['label','marker','hatch','size','visibility','outputs'],
            **self.settings
        )
        self.logger.info(f"Figure exported to {dirpath}")
        self.logger.info(f"Filename: {filename}")

    '''
    Extracting plotting data
    '''

    def extract_plot_settings(self,vars:list,meta:dict):
        # print_json(meta)
        # print('\n')
        var_values = {}
        for var in vars:
            if var in PLOT_COORDINATES+PLOT_CORE_FEATURES:
                # Get value from metadata element
                if self.settings[var] in meta:
                    value = meta[self.settings[var]]
                # Data not found
                else:
                    value = None
                    self.logger.debug(f"Could not find data for {var} var.")
                # Convert x or y coordinate to list
                if isinstance(value,Iterable) and not isinstance(value,str):
                    value = list(value)
            elif var == 'label':
                label_str = []
                # Get label key and value
                for label_key in self.settings['label']:
                    label_str.append(
                        str(label_key) + ' = ' + \
                        str(meta[label_key]).replace('%','percent').replace(' ','_').replace(':','_').replace(',','_')
                    )
                value = ', '.join(label_str)
            elif var in ['marker','hatch']:
                # Get label key and value
                var_key = self.settings[var]
                # Get value from metadata element
                if self.settings[var_key] in meta:
                    if var_key == 'marker':
                        marker_value = PLOT_MARKERS[meta[self.settings[var_key]]]
                    elif var_key == 'hatch':
                        marker_value = PLOT_HATCHES[meta[self.settings[var_key]]]
                # Data not found
                else:
                    marker_value = None
                    self.logger.debug(f"Could not find data for {var_key} var.")
                # Convert x or y coordinate to list
                if isinstance(value,Iterable) and not isinstance(value,str):
                    marker_value = list(value)
                # Add to dictionary
                value = marker_value
            else: 
                value = meta[var]
            # Set variable value
            var_values[var] = value
        return var_values

    def merge_plot_settings(self,plot_settings:list):
        merged_settings = {}
        # Iterate through the list of dictionaries
        for d in plot_settings:
            # print_json(d,newline=True)
            # Concatenate values to the merged_dict
            for key, value in d.items():
                if value is None:
                    continue
                if key in ['x','y','z','label','colour','size']:
                    merged_settings.setdefault(key, []).append(value)
                elif key in ['marker','hatch']:
                    if key not in merged_settings:
                        merged_settings[key] = value
                    else:
                        merged_settings[key] = deep_merge(
                            merged_settings[key],
                            value
                        )
                else:
                    # Keep only the first value of this key for each
                    # plot setting
                    if key not in merged_settings:
                        merged_settings[key] = value
        for k,v in merged_settings.items():
            if isinstance(v,Iterable) and not isinstance(v,str):
                merged_settings[k] = list(flatten(v))
        return [merged_settings]

    def create_plot_filename(self,plot_setting,**kwargs):
        # Decide on figure output dir
        if not self.settings['by_experiment']:
            # Get filename
            filename = kwargs.get('name','NO_NAME')+'_'+\
                    f"burnin_{self.settings['burnin']}_" + \
                    f"thinning_{self.settings['thinning']}"
            if not plot_setting['outputs'].config['training']['N'].get('sweep',{}):
                filename += f"_N_{plot_setting['outputs'].config['training']['N']}"
            if self.settings.get('label',None) is not None:
                filename += f"_label_{'&'.join([str(elem) for elem in self.settings['label']])}"
            if self.settings.get('marker',None) is not None:
                filename += f"_marker_{self.settings['marker']}"
            if self.settings.get('hatch',None) is not None:
                filename += f"_hatch_{self.settings['hatch']}"
            if self.settings.get('colour',None) is not None:
                filename += f"_colour_{self.settings['colour']}"
            if self.settings.get('size',None) is not None:
                filename +=  f"_size_{self.settings['size']}"
            if self.settings.get('visibility',None) is not None:
                filename += f"_visibility_{self.settings['visibility']}"
            # Get dirpath
            parent_directory = Path(plot_setting['outputs'].outputs_path)
            if self.settings.get('plot_data_dir',None) is not None:
                dirpath = self.settings['plot_data_dir']
            else:
                if 'synthetic' in str(parent_directory):
                    parent_directory = plot_setting['outputs'].config.out_directory
                else:
                    parent_directory.parent.parent.absolute()
                dirpath = os.path.join(parent_directory,'paper_figures')
        else:
            
            # Get filename
            dims = unpack_dims(
                plot_setting['outputs'].inputs.data.dims,
                time_dims=False
            )
            filename = f"table_{'x'.join(list(map(str,list(dims))))}" + \
                    f"_{plot_setting['outputs'].experiment_id[:-21]}_{kwargs.get('name','NO_NAME')}"+ \
                    f"_burnin_{self.settings.get('burnin',0)}" + \
                    f"_thinning_{self.settings.get('thinning',1)}" 
            if not plot_setting['outputs'].config['training']['N'].get('sweep',{}):
                filename += f"_N_{plot_setting['outputs'].config['training']['N']}"
            if self.settings.get('label',None) is not None:
                filename += f"_label_{'&'.join([str(elem) for elem in self.settings['label']])}"
            if self.settings.get('marker',None) is not None:
                filename += f"_marker_{self.settings['marker']}"
            if self.settings.get('hatch',None) is not None:
                filename += f"_hatch_{self.settings['hatch']}"
            if self.settings.get('colour',None) is not None:
                filename += f"_colour_{self.settings['colour']}"
            if self.settings.get('size',None) is not None:
                filename +=  f"_size_{self.settings['size']}"
            if self.settings.get('visibility',None) is not None:
                filename += f"_visibility_{self.settings['visibility']}"
            # Get dirpath
            if self.settings.get('plot_data_dir',None) is not None:
                dirpath = self.settings['plot_data_dir']
            else:
                dirpath = os.path.join(
                    plot_setting['outputs'].outputs_path,
                    'figures'
                )
        
        return dirpath,filename
    
    def read_plot_data(self):
        # If directory exists and loading of plot data is instructed
        if self.settings.get('plot_data_dir','') is not None and \
            os.path.exists(self.settings.get('plot_data_dir','')) and \
            os.path.isdir(self.settings.get('plot_data_dir','')):

            # Find data in json format
            # no other format is acceptable
            files = list(glob(f"{self.settings['plot_data_dir']}/*[!settings].json",recursive=False))

            # If nothing was found return false
            if len(files) <= 0:
                return False,None
            # Try to read file
            plot_settings = read_file(files[0])

            # Canonicalise the data
            if isinstance(plot_settings,dict):
                plot_settings = [plot_settings]
            elif isinstance(plot_settings,pd.DataFrame):
                plot_settings = [dict(plot_settings.to_dict())]
            elif isinstance(plot_settings,list):
                plot_settings = plot_settings
            else:
                self.logger.warning(f"Cannot recognise plot settings of type {type(plot_settings)}")
                return False,None
            
            # Extract outputs
            for i in range(len(plot_settings)):
                if 'outputs' not in plot_settings[i]:
                    self.logger.debug(plot_settings[i])
                    self.logger.warning(f"Outputs are not included in plot settings")
                    return False,None
                else:
                    # Try to load outputs
                    if isinstance(plot_settings[i]['outputs'],dict):
                        # Instantiate config
                        config = Config(
                            settings = plot_settings[i]['outputs'],
                            logger = self.logger
                        )
                        # Instantiate outputs
                        plot_settings[i].update(dict(
                            outputs = Outputs(
                                config = config,
                                settings = self.settings,
                                data_names = self.settings['sample'],
                                logger = self.logger
                            )
                        ))
                    else:
                        self.logger.warning(f"Outputs are of type {type(plot_settings[i]['outputs'])} and not dict.")
                        return False,None
            return True,plot_settings

        else:
            return False,None

    
    '''    
    ╔═╗┬  ┌─┐┌┬┐  ┬ ┬┬─┐┌─┐┌─┐┌─┐┌─┐┬─┐┌─┐
    ╠═╝│  │ │ │   │││├┬┘├─┤├─┘├─┘├┤ ├┬┘└─┐
    ╩  ┴─┘└─┘ ┴   └┴┘┴└─┴ ┴┴  ┴  └─┘┴└─└─┘
    '''
    
    def data_plot_2d_scatter(self):
            
        self.logger.info('Running data_plot_2d_scatter')
    
        # Try to load plot data from file
        loaded, plot_settings = self.read_plot_data()
        
        if not loaded:
            # Run output handler
            outputs_summary = OutputSummary(
                settings=self.settings,
                logger=self.logger
            )
            
            # Loop through output folder
            plot_settings = []
            for i,output_folder in enumerate(outputs_summary.output_folders):
                
                self.logger.info(f"Scanning folder {i+1}/{len(outputs_summary.output_folders)}")

                # Collect outputs from folder's Data Collection
                outputs = outputs_summary.get_folder_outputs(output_folder)
                
                # Create plot settings
                plot_sett = {
                    'outputs':outputs
                }
                # Loop through each member of the data collection
                for j in range(len(outputs.data)):
                    
                    # Get metadata for this experiment and this element
                    # of the Data Collection
                    metadata = outputs_summary.get_experiment_metadata(j,outputs)

                    # Loop through each entry of metadata
                    for meta in metadata:
                        plot_sett.update(
                            self.extract_plot_settings(
                                vars = ['x','y','label','colour','size','marker','hatch','visibility'],
                                meta = meta,
                            )
                        )
                        # print_json({k:plot_sett[k] for k in ['x','y','label','colour','size','marker','hatch','visibility']})

                        # Add data
                        plot_settings.append(deepcopy(plot_sett))
                
                # If plot is by experiment
                # plot all element from data collection
                # for every output folder
                if self.settings['by_experiment']:
                    # Create output dirpath and filename
                    dirpath,filename = self.create_plot_filename(
                        plot_setting = plot_sett,
                        name = self.settings.get('figure_title','NONAME')
                    )
                    # Merge all settings into one
                    # Plot
                    self.plot_2d_scatter(
                        plot_settings = self.merge_plot_settings(plot_settings),
                        dirpath = dirpath,
                        filename = filename
                    )
                    # Reset list of plot settings
                    plot_settings = []
            
            # If plot is NOT by experiment
            # plot all elements from data collection
            # from all output folder(s)
            if not self.settings['by_experiment']:
                # Create output dirpath and filename
                dirpath,filename = self.create_plot_filename(
                    plot_setting = plot_sett,
                    name = self.settings.get('figure_title','NONAME')
                )
                # Plot
                self.plot_2d_scatter(
                    plot_settings = self.merge_plot_settings(plot_settings),
                    name = self.settings.get('title','NONAME'),
                    dirpath = dirpath,
                    filename = filename
                )
        else:
            # Get first plot setting
            plot_sett = plot_settings[0]
            # Create output dirpath and filename
            dirpath,filename = self.create_plot_filename(
                plot_setting = plot_sett,
                name = self.settings.get('figure_title','NONAME')
            )
            # Plot
            self.plot_2d_scatter(
                plot_settings = self.merge_plot_settings(plot_settings),
                name = self.settings.get('title','NONAME'),
                dirpath = dirpath,
                filename = filename
            )