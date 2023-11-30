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
from multiresticodm.config import Config
from multiresticodm.plot_variables import *
from multiresticodm.global_variables import *
from multiresticodm.outputs import Outputs,OutputSummary
from multiresticodm.contingency_table import instantiate_ct
from multiresticodm.spatial_interaction_model import instantiate_sim
from multiresticodm.probability_utils import log_odds_ratio_wrt_intensity
from multiresticodm.math_utils import map_distance_name_to_function

latex_preamble = r'''
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
'''

mpl.rcParams['text.latex.preamble'] = latex_preamble
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman"
})

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
        # Figure size 
        fig = plt.figure(figsize=self.settings['figure_size'])
        ax = fig.add_subplot(111)

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
                
                for plot_sett in plot_settings:
                    if self.settings.get(f'{var}_discrete',False):
                        min_val = min([0,min_val])
                        max_val = max([len(set(plot_sett[var+'_id']))+1,max_val])
                    else:
                        min_val = min([np.min(plot_sett[var]),min_val])
                        max_val = max([np.max(plot_sett[var]),max_val])
                # Update axis limits
                axes_lims[var] = [min_val,max_val]

        # Plot benchmark (perfect predictions)
        if self.settings['benchmark']:
            ax.plot(
                np.linspace(*axes_lims['x']),
                np.linspace(*axes_lims['y']),
                linewidth=0.2,
                color='black'
            )
        if self.settings.get('x_label',''):
            ax.set_xlabel(
                self.settings['x_label'].replace("_"," "),
                fontsize=self.settings['axis_font_size'],
                labelpad=self.settings['axis_labelpad']
            )
        if self.settings.get('y_label',''):
            ax.set_ylabel(
                self.settings['y_label'].replace("_"," "),
                fontsize=self.settings['axis_font_size'],
                labelpad=self.settings['axis_labelpad']
            )
        
        # Axis limits
        ax.set_xlim(left=axes_lims['x'][0], right=axes_lims['x'][1])
        ax.set_ylim(bottom=axes_lims['y'][0], top=axes_lims['y'][1])

        # Ticks
        if self.settings.get('x_discrete',False):
            # Sort all x values and keep their ordering
            all_x = np.array(list(flatten([plot_sett['x_id'] for plot_sett in plot_settings])))
            sorted_index = all_x.argsort(axis=0)
            all_x = all_x[sorted_index]
            # Get unique x values
            unique_x = np.unique(all_x)
            # Read x ticks from x
            xticks = np.array([[subx for subx in plot_sett['x']] for plot_sett in plot_settings]).squeeze()
            # Sort x ticks based on ordering of unique x
            xticks = xticks[sorted_index]
            # make sure ticks are at least 2-dimensional
            xticks = xticks if len(xticks.shape) > 1 else np.expand_dims(xticks,axis=-1)
            # For each subtick (up to two subticks - one for major and one for minor ticks)
            for i,xtick_labels in enumerate(xticks[:,:2].T):
                # Get tick locations
                tick_indices = np.arange(
                    self.settings['x_tick_frequency'][i][0],
                    len(all_x),
                    self.settings['x_tick_frequency'][i][1]*len(self.settings.get('sample',[None]))
                )
                tick_locations = np.arange(
                    self.settings['x_tick_frequency'][i][0]+1,
                    len(all_x)+1,
                    self.settings['x_tick_frequency'][i][1]
                )[:len(xtick_labels[tick_indices])]

                print(tick_locations)
                print(xtick_labels)
                # Decide on major/minor axis
                if i == 0:
                    minor = False
                else:
                    minor = True
                pad = self.settings["x_tick_pad"][i]
                rotation = self.settings["x_tick_rotation"][i]

                # Plot ticks
                ax.set_xticks(
                    ticks = tick_locations,
                    labels = xtick_labels[tick_indices],
                    minor = minor
                )
                ax.tick_params(
                    axis='x', 
                    which=('minor' if minor else 'major'), 
                    pad=pad,
                    bottom=True,
                    labelsize=self.settings['tick_font_size'],
                    rotation=rotation
                )
            # Set gridlines
            ax.grid(axis='x',which='both')
            ax.xaxis.remove_overlapping_locs = False

            # Create a discrete hashmap
            x_hashmap = dict(zip(
                unique_x,
                tick_locations
                # np.arange(1,len(unique_x)+1)
            ))
            print(x_hashmap)
        else:
            ax.set_xticks(fontsize = self.settings['tick_font_size'])
        
        if self.settings.get('y_discrete',False):
            # Get sorted unique x values
            unique_y = set(list(flatten([plot_sett['y'] for plot_sett in plot_settings])))
            unique_y = sorted(list(unique_y))
            # Create a discrete hashmap
            y_hashmap = dict(zip(
                unique_y,
                range(
                    1,
                    len(unique_y)*self.settings['y_tick_frequency']+1,
                    self.settings['y_tick_frequency']
                )
            ))
            plt.yticks(
                ticks = list(y_hashmap.values()),
                labels = list(y_hashmap.keys()),
                fontsize = self.settings['tick_font_size']
            )
        else:
            plt.yticks(fontsize = self.settings['tick_font_size'])
        
        # 2D scatter plot
        for plot_sett in plot_settings:
            # Extract data
            x_range = list(map(lambda v: hash_vars(x_hashmap,v), plot_sett['x'])) \
                    if self.settings.get('x_discrete',False) \
                    else plot_sett['x']
            y_range = list(map(lambda v: hash_vars(y_hashmap,v), plot_sett['y'])) \
                    if self.settings.get('y_discrete',False) \
                    else plot_sett['y']
            sizes = plot_sett.get('size',float(self.settings['marker_size']))
            sizes = [float(sizes)] if not isinstance(sizes,Iterable) else sizes
            colours = plot_sett.get('colour','black')
            colours = [colours] if isinstance(colours,str) else colours
            # Convert transparency levels to approapriate data type
            alphas = plot_sett.get('visibility','1.0')
            alphas = [float(alphas)] if isinstance(alphas,str) else alphas
            zorders = plot_sett.get('zorder',[1])
            zorders = [float(zorders)] if isinstance(zorders,str) else zorders 
            labels = plot_sett.get('label',[''])
            labels = [labels] if isinstance(labels,str) else labels
            markers = plot_sett.get('marker',['o'])
            markers = [markers] if isinstance(markers,str) else markers
            hatches = plot_sett.get('hatch','')
            hatches = [hatches] if isinstance(hatches,str) else hatches

            # print('x raw')
            # print(plot_sett['x'])
            # print('y raw')
            # print(plot_sett['y'])
            print('x')
            print(list(flatten(x_range)))
            print('y')
            print(list(flatten(y_range)))
            # print('Sizes')
            # print(sizes)
            # print('Alphas')
            # print(alphas)
            # print('Zorders')
            # print(zorders)
            print('Labels')
            print(labels)
            # print('Markers')
            # print(markers)
            # print('Hatches')
            # print(hatches)

            # Plot x versus y
            for i in range(len(y_range)):
                ax.scatter(
                    x = x_range[i],
                    y = y_range[i],
                    s = sizes[i] if len(sizes) > 1 else sizes[0],
                    c = colours[i] if len(colours) > 1 else colours[0],
                    alpha = alphas[i] if len(alphas) > 1 else alphas[0],
                    zorder = zorders[i] if len(zorders) > 1 else zorders[0],
                    label = labels[i] if len(labels) > 1 else labels[0],
                    marker = markers[i] if len(markers) > 1 else markers[0],
                    hatch = hatches[i] if len(hatches) > 1 else hatches[0]
                )

            # Annotate data
            if self.settings.get('annotate',False):
                for i, txt in enumerate(plot_sett.get(self.settings.get('annotation_label',''),[])):
                    ax.annotate(str(string_to_numeric(txt) if str(txt).isnumeric() else str(txt)), (x_range[i], y_range[i]))


        # Aspect ratio equal
        if self.settings['equal_aspect']:
            plt.gca().set_aspect('equal')
    
        # Legend
        # try:
        # Ensure no duplicate entries in legend exist
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        print(by_label)
        # print(by_label)
        leg = ax.legend(
            by_label.values(), 
            by_label.keys(),
            frameon = False,
            prop = {'size': self.settings.get('legend_label_size',None)}
        )
        leg._ncol = 1
        # except:
            # pass

        
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
            aux_keys=PLOT_VARIABLES_AND_DERIVATIVES+['outputs'],
            **self.settings
        )
        self.logger.info(f"Figure exported to {dirpath}")
        self.logger.info(f"Filename: {filename}")

    '''
    Extracting plotting data
    '''
    def extract_plot_variable(self,var:str,meta:dict,settings:dict):
        value = None
        if var in PLOT_COORDINATES+list(PLOT_AUX_FEATURES.keys()):
            # Extract variable values
            groups = settings[var]
            value = []
            for grp in groups:
                grp_values = []
                var_value = None
                for group_var in grp:
                    # Get value from metadata element
                    if group_var in meta:
                        var_value = get_value(meta,group_var)
                    # Data not found
                    else:
                        self.logger.error(f"Could not find {var} data for {group_var} var.")
                        return None
                    # Convert x or y coordinate to list
                    if isinstance(var_value,Iterable) and not isinstance(var_value,str):
                        var_value = list(var_value)
                    # add to sub-variable values
                    if len(grp) > 1:
                        # If more than one variables are provided for this coordinate
                        # convert all values to string
                        grp_values.append(str(var_value))
                    else:
                        grp_values = var_value
                # Combine all sub-variable values
                if len(grp) > 1:
                    # If more than one variables are provided for this coordinate
                    # convert value to string tuple
                    grp_values = "(" + ", ".join(grp_values) + ")"
                
                # Add group values to value
                value.append(grp_values)

        elif var == 'label':
            label_str = []
            # Get label key and value
            for label_key in settings['label']:
                label_str.append(
                    str(label_key) + ' = ' + \
                    str(meta[label_key]).replace('%','percent').replace(' ','_').replace(':','_').replace(',','_')
                )
            value = ', '.join(label_str)
        
        elif var == 'zorder':
            # Get label key and value
            order_tuple = []
            for order_var in settings['zorder']:
                
                # get ordering (ascending/descending)
                order = order_var[0]
                # get ordering variable
                var_key = order_var[1]
                # get order variable value
                var_val = get_value(meta,var_key,default = 1)
                # convert to right dtype
                var_val = PLOT_CORE_FEATURES[var]['dtype'](var_val)
                # add to ordered tuple depending on order
                if order == 'asc':
                    order_tuple.append(-var_val)
                else:
                    order_tuple.append(var_val)
            # Get values
            value = order_tuple

        elif var in PLOT_CORE_FEATURES:
            # Get label key and value
            var_key = get_value(settings,var)
            # If no settings provided return None
            if var_key is None:
                return None
            
            # Get value from metadata element
            if var_key in meta:
                # Extract value
                value = get_value(meta,var_key)
                
                # Determine plot features based on global plot settings
                if var == 'marker':
                    value = PLOT_MARKERS[var_key][str(parse(value))]
                elif var == 'hatch':
                    value = PLOT_HATCHES[var_key][str(parse(value))]
                elif var == 'colour':
                    # Try to extract colour from global settings
                    colour_value = PLOT_COLOURS.get(var_key,{}).get(value,None)
                    value = colour_value if colour_value is not None else value

            # This is the case where the value is passed to the variable
            # directly and not though a metric/metadata key
            elif var_key is not None:
                # Convert string input to relevant dtype
                value = PLOT_CORE_FEATURES[var]["dtype"](var_key)
            # Data not found
            else:
                self.logger.error(f"Could not find {var} data for {var_key} var.")
                return None
            # Convert x or y coordinate to list
            if isinstance(value,Iterable) and not isinstance(value,str):
                value = list(value)
        elif var not in PLOT_DERIVATIVES:
            value = get_value(settings,var)
            if value is None:
                self.logger.error(f"Could not find {var} data for var in settings.")
                return None
        return value
    
    def extract_plot_settings(self,vars:list,meta:dict):
        var_values = {}
        # print_json(meta)
        # print('\n')
        for variable in vars:
            # Extract variable value
            value = self.extract_plot_variable(
                var = variable,
                meta = meta,
                settings = self.settings
            )
            # If no value found move on
            if value is None:
                continue

            # Set variable value
            var_values[variable] = value
        
        # If the variable is a plot coordinate
        # add a global identifier of all sub-coordinates together
        for variable in vars:
            if variable in PLOT_COORDINATES:
                # Get all sub-coordinates and flatten them
                subcoords = [[subcoord for subcoord in coord] for coord in self.settings[variable]]
                subcoords = list(flatten(subcoords))
                # Extract global identifier
                value = self.extract_plot_variable(
                    var = variable,
                    meta = meta,
                    settings = {variable:[subcoords]}
                )
                # If no value found move on
                if value is None:
                    continue

                # Set variable value
                var_values[variable+'_id'] = value


        # print('x')
        # print(var_values.get('x','not_found'))
        # print('x_id')
        # print(var_values.get('x_id','not_found'))

        return var_values

    def merge_plot_settings(self,plot_settings:list):
        merged_settings = {}
        # Iterate through the list of dictionaries
        for d in plot_settings:
            # print_json(d,newline=True)
            # Concatenate values to the merged_dict
            for key, value in d.items():
                if value is None:
                    if key not in merged_settings:
                        merged_settings[key] = [value]
                    else:
                        merged_settings[key].append(value)

                if key in PLOT_VARIABLES:
                    merged_settings.setdefault(key, []).append(value)
                elif key in PLOT_DERIVATIVES:
                    merged_settings.setdefault(key, []).append(value)
                else:
                    # Keep only the first value of this key for each
                    # plot setting
                    if key not in merged_settings:
                        merged_settings[key] = value
                
        
        # Create ordering of data points based on provided data
        if 'zorder' in merged_settings:
            # get value to order by
            values = np.array(merged_settings['zorder'])
            # Number of elements to sort
            ndims = values.shape[1]
            # Use lexsort to argsort along each axis successively
            sorted_indices = np.lexsort([values[:,i].ravel() for i in range(ndims)])
            # Update merged settings
            # add 1.0 to avoid zorder = 0
            merged_settings['zorder'] = list(map(float,sorted_indices+1.0))

        # flatten list of lists
        for key,value in merged_settings.items():
            # No need to flatten any of these variables
            if key in PLOT_COORDINATES:
                continue
            elif isinstance(value,Iterable) and not isinstance(value,str):
                merged_settings[key] = list(flatten(value))

        # print('x')
        # print(merged_settings.get('x','not_found'))
        # print('x_id')
        # print(merged_settings.get('x_id','not_found'))

        return [merged_settings]

    def create_plot_filename(self,plot_setting,**kwargs):
        # Decide on figure output dir
        if not self.settings['by_experiment']:
            # Get filename
            filename = kwargs.get('name','NO_NAME')+'_' + \
                    f"burnin_{self.settings['burnin']}_" + \
                    f"thinning_{self.settings['thinning']}"
            if not isinstance(plot_setting['outputs'].config['training']['N'],dict):
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
                plot_settings = [{
                    k:(np.array(v) if isinstance(v,list) else v) for k,v in plot_settings.items()
                }]
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
            
            # print_json(plot_settings[0],newline=True)
            print_json({k:np.shape(plot_settings[0].get(k)) for k in plot_settings[0].keys() if k != 'outputs'},newline=True)
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
                settings = self.settings,
                logger = self.logger
            )
            
            # Loop through output folder
            plot_settings = []
            for indx,output_folder in enumerate(outputs_summary.output_folders):
                
                self.logger.info(f"Scanning folder {indx+1}/{len(outputs_summary.output_folders)}")

                # Collect outputs from folder's Data Collection
                outputs = outputs_summary.get_folder_outputs(indx,output_folder)

                # Create plot settings
                plot_sett = {'outputs':outputs}

                try:
                    # Loop through each member of the data collection
                    if self.settings.get('n_workers',1) > 1:
                        metric_data_collection = outputs_summary.get_experiment_metadata_concurrently(outputs)
                    else:
                        metric_data_collection = outputs_summary.get_experiment_metadata_sequentially(outputs)
                    
                    # Convert metric data collection to list
                    metric_data_collection = list(metric_data_collection)
                    
                    # Loop through metadata for each data collection member
                    for metadata in tqdm(
                        metric_data_collection,
                        desc='Extracting plot settings',
                        leave=False
                    ):
                        # Loop through each entry of metadata
                        for meta in metadata:
                            plot_sett.update(
                                self.extract_plot_settings(
                                    vars = PLOT_VARIABLES,
                                    meta = meta
                                )
                            )
                            # print_json({k:plot_sett[k] for k in PLOT_VARIABLES_AND_DERIVATIVES if k in plot_sett})

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
                except:
                    self.logger.error(traceback.format_exc())
                    self.logger.error(f"Plot for folder {indx+1}/{len(outputs_summary.output_folders)} failed...")
                    continue
            
            # If plot is NOT by experiment
            # plot all elements from data collection
            # from all output folder(s)
            if not self.settings['by_experiment']:
                # Create output dirpath and filename
                dirpath,filename = self.create_plot_filename(
                    plot_setting = {'outputs':outputs},
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