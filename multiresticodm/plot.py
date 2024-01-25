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
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec

from multiresticodm.config import Config
from multiresticodm.utils.misc_utils import *
from multiresticodm.fixed.plot_variables import *
from multiresticodm.fixed.global_variables import *
from multiresticodm.outputs import Outputs,OutputSummary
from multiresticodm.contingency_table import instantiate_ct
from multiresticodm.spatial_interaction_model import instantiate_sim
from multiresticodm.utils.probability_utils import log_odds_ratio_wrt_intensity
from multiresticodm.utils.math_utils import map_distance_name_to_function

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

    def __init__(self,plot_view:str,settings:dict,**kwargs):
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__,
            console_level = level,
        ) if kwargs.get('logger',None) is None else kwargs['logger']

        # Store settings
        self.settings = settings

        # self.logger.info(f"{','.join([Path(out_dir).stem for out_dir in self.outputs_directories])}")
        # Run plotting
        self.data_plot(plot_func = self.compile_plot(plot_view))

    def print_data(self,plot_setting:dict,local_vars:dict,plot_vars:list=None,index:int=None,summarise:bool=False):
        for v in ['x','y','size','colour','visibility','zorder','style','label','marker','hatch']:
            if plot_vars is None or v in plot_vars:
                if index is None:
                    if summarise:
                        self.logger.info(f"{v}: {np.shape(plot_setting[v])} {type(plot_setting[v])}")
                        try:
                            self.logger.info(f"{v}: min = {np.nanmin(plot_setting[v])}, max = {np.nanmax(plot_setting[v])}")
                        except:
                            pass
                    else:
                        self.logger.info(f"{v} = {plot_setting[v]}")
                else:
                    if summarise:
                        if isinstance(plot_setting[v],Iterable) and index < len(plot_setting[v]):
                            self.logger.info(f"{v}: {np.shape(plot_setting[v][index])} {type(plot_setting[v][index])}")
                            try:
                                self.logger.info(f"{v}: min = {np.nanmin(plot_setting[v][index])}, max = {np.nanmax(plot_setting[v][index])}")
                            except:
                                pass
                        else:
                            self.logger.info(f"{v}: {np.shape(plot_setting[v])} {type(plot_setting[v])}")
                            try:
                                self.logger.info(f"{v}: min = {np.nanmin(plot_setting[v])}, max = {np.nanmax(plot_setting[v])}")
                            except:
                                pass
                    else:
                        if isinstance(plot_setting[v],Iterable) and index < len(plot_setting[v]):
                            self.logger.info(f"{v} = {plot_setting[v][index]}")
                        else:
                            self.logger.info(f"{v} = {plot_setting[v]}")
        for v in ['x_range','y_range']:
            if plot_vars is None or v in plot_vars:
                if index is None:
                    if summarise:
                        self.logger.info(f"{v}: {np.shape(list(flatten(local_vars[v])))} {type(list(flatten(local_vars[v])))}")
                        try:
                            self.logger.info(f"{v}: min = {np.nanmin(list(flatten(local_vars[v])))}, max = {np.nanmax(list(flatten(local_vars[v])))}")
                        except:
                            pass
                    else:
                        self.logger.info(f"{v} = {list(flatten(local_vars[v]))}")
                else:
                    if summarise:
                        if isinstance(local_vars[v],Iterable) and index < len(local_vars[v]):
                            self.logger.info(f"{v}: {np.shape(list(flatten(local_vars[v][index])))} {type(list(flatten(local_vars[v][index])))}")
                            try:
                                self.logger.info(f"{v}: min = {np.nanmin(list(flatten(local_vars[v][index])))}, max = {np.nanmax(list(flatten(local_vars[v][index])))}")
                            except:
                                pass
                        else:
                            self.logger.info(f"{v}: {np.shape(list(flatten(local_vars[v])))} {type(list(flatten(local_vars[v])))}")
                            try:
                                self.logger.info(f"{v}: min = {np.nanmin(list(flatten(local_vars[v])))}, max = {np.nanmax(list(flatten(local_vars[v])))}")
                            except:
                                pass
                    else:
                        if isinstance(local_vars[v],Iterable) and index < len(local_vars[v]):
                            self.logger.info(f"{v} = {list(flatten(local_vars[v][index]))}")
                        else:
                            self.logger.info(f"{v} = {list(flatten(local_vars[v]))}")
        
    def compile_plot(self,visualiser_name):
        if hasattr(self, PLOT_VIEWS[visualiser_name]):
            return getattr(self, PLOT_VIEWS[visualiser_name])
        else:
            raise Exception(f'Experiment class {PLOT_VIEWS[visualiser_name]} not found')
    
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

    def get_axes_limits(self,plot_settings,vars:list,axes_id:str,axes_lims=None):
        # Axes limits from settings (read only x,y limits)
        if axes_lims is None:
            axes_lims = {}
        if axes_id not in axes_lims:
            axes_lims[axes_id] = {}
        
        
        for var in vars:
            # Get index of axes id
            axes_counter = self.axids[axes_id]
            if axes_counter >= len(self.settings.get(f'{var}_limit',[(None,None)])):
                axes_counter = 0

            # Get limits from settings
            settings_limits = list(self.settings.get(f'{var}_limit',[(None,None)])[axes_counter])
            # Update default axes limits 
            if var not in axes_lims[axes_id]:
                axes_lims[axes_id][var] = settings_limits
            
            # print(var,axes_id,axes_lims[axes_id][var])
            # Otherwise read from data
            if None in settings_limits or None in axes_lims[axes_id][var]:

                if axes_lims[axes_id][var][0] is None:
                    min_val = np.infty
                else:
                    min_val = axes_lims[axes_id][var][0]
                
                if axes_lims[axes_id][var][1] is None:
                    max_val = -np.infty
                else:
                    max_val = axes_lims[axes_id][var][1]

                if self.settings.get(f'{var}_discrete',False):
                    min_val = min(0,min_val)
                    max_val = max(len(set(plot_settings[var+'_id']))+1,max_val)
                else:
                    min_val = min(np.nanmin(plot_settings[var]),min_val)
                    max_val = max(np.nanmax(plot_settings[var]),max_val)
                
                # Update axis limits
                axes_lims[axes_id][var] = [min_val,max_val]

                self.logger.info(f"Set {var} limits for index {axes_counter}: {min_val}, {max_val}")
            else:
                # Update existing limits
                if self.settings.get(f'{var}_discrete',False):
                    min_val = min(0,axes_lims[axes_id][var][0])
                    max_val = max(len(set(plot_settings[var+'_id']))+1,axes_lims[axes_id][var][1])
                else:
                    min_val = axes_lims[axes_id][var][0]
                    max_val = axes_lims[axes_id][var][1]
                # Update axis limits
                axes_lims[axes_id][var] = [min_val,max_val]

                self.logger.info(f"Updated {var} limits for index {axes_counter}: {min_val}, {max_val}")

        return axes_lims[axes_id]
    
    def get_discrete_ticks(self,plot_settings,var:str,id_var:str,tickfreq_var:str):
        # Sort all var values and keep their ordering
        all_var = np.array(list(flatten(plot_settings[id_var])))
        sorted_index = all_var.argsort(axis=0)
        all_var = all_var[sorted_index]
        # Get unique var values
        unique_var = np.unique(all_var)
        # Read var ticks from var
        var_ticks = np.array([subvar for subvar in plot_settings[var]]).squeeze()
        # Sort var ticks based on ordering of unique var
        var_ticks = var_ticks[sorted_index]
        # make sure ticks are at least 2-dimensional
        var_ticks = var_ticks if len(var_ticks.shape) > 1 else np.expand_dims(var_ticks,axis=-1)

        all_ticks = {
            "unique": unique_var,
            "tick_locations":[],
            "tick_label":[]
        }
        # For each subtick (up to two subticks - one for major and one for minor ticks)
        for i,var_tick_label in enumerate(var_ticks[:,:2].T):
            # Get tick locations
            tick_indices = np.arange(
                self.settings[tickfreq_var][i][0],
                len(all_var),
                self.settings[tickfreq_var][i][1]*len(plot_settings[var])
            )
            all_ticks['tick_locations'].append(
                np.arange(
                    self.settings[tickfreq_var][i][0]+1,
                    len(all_var)+1,
                    self.settings[tickfreq_var][i][1]
                )[:len(var_tick_label[tick_indices])]
            )
            all_ticks['tick_label'].append(
                var_tick_label[tick_indices]
            )
            # print(var,i,all_ticks['tick_locations'][-1])
        return all_ticks
    

    def map_groups_to_axes(self,index,plot_settings):
        # For each variable
        axes = {}
        for var in ['y','x']:
            # Get current group
            group_id = plot_settings.get(f"{var}_group")[index]
            # Get index of group id
            if group_id in plot_settings[f"{var}_group_id"]:
                axes[var] = plot_settings[f"{var}_group_id"].tolist().index(str(group_id))
            else:
                axes[var] = 0

        return axes

    '''
    ╔═╗┌─┐┌┐┌┌─┐┬─┐┬┌─┐  ┌─┐┬  ┌─┐┌┬┐  ┌─┐┬ ┬┌┐┌┌─┐┌┬┐┬┌─┐┌┐┌┌─┐
    ║ ╦├┤ │││├┤ ├┬┘││    ├─┘│  │ │ │   ├┤ │ │││││   │ ││ ││││└─┐
    ╚═╝└─┘┘└┘└─┘┴└─┴└─┘  ┴  ┴─┘└─┘ ┴   └  └─┘┘└┘└─┘ ┴ ┴└─┘┘└┘└─┘
    '''

    def plot_wrapper(self,ax,plot_type:str,**kwargs):
        if plot_type == 'plot':
            ax.plot(
                kwargs['x'],
                kwargs['y'],
                linewidth = kwargs.get('s',None),
                linestyle = kwargs.get('style',None),
                c = kwargs.get('c',None),
                alpha = kwargs.get('alpha',None),
                zorder = kwargs.get('zorder',None),
                label = kwargs.get('label',None),
            )
                
        elif plot_type == 'scatter':
            ax.scatter(
                x = kwargs['x'],
                y = kwargs['y'],
                s = kwargs['s'],
                c = kwargs.get('c',None),
                alpha = kwargs.get('alpha',None),
                zorder = kwargs.get('zorder',None),
                label = kwargs.get('label',None),
                marker = kwargs.get('marker',None),
                hatch = kwargs.get('hatch',None)
            )
        
        return ax
    
    def plot_2d(self,plot_settings,**kwargs):

        # Get directory path and file name
        dirpath = kwargs['dirpath']
        filename = kwargs['filename']
        filepath = os.path.join(dirpath,filename)

        # Make outputs directories (if necessary)
        makedir(dirpath)

        # Flag for whether multiple subplots are plotted
        plot_settings = plot_settings[0]
        multiple_x = all([len(xgrp)>1 for xgrp in plot_settings['x_group']])
        multiple_y = all([len(ygrp)>1 for ygrp in plot_settings['y_group']])
        multiple_subplots = multiple_x or multiple_y
        
        # Get discete group values
        if multiple_x:
            all_ticks = self.get_discrete_ticks(
                var = 'x',
                id_var = "x_group",
                tickfreq_var = "subx_tick_frequency",
                plot_settings = plot_settings
            )
            # Create a discrete hashmap
            plot_settings['x_group_id'] = all_ticks['unique']
        else:
            plot_settings['x_group_id'] = []

        if multiple_y:
            all_ticks = self.get_discrete_ticks(
                var = 'y',
                id_var = "y_group",
                tickfreq_var = "suby_tick_frequency",
                plot_settings = plot_settings
            )
            # Create a discrete hashmap
            plot_settings['y_group_id'] = all_ticks['unique']
        else:
            plot_settings['y_group_id'] = []

        
        # Extract data
        x_range = list(map(lambda v: hash_vars(locals()['x_hashmap'],v), plot_settings['x'])) \
                if self.settings.get('x_discrete',False) \
                else plot_settings['x']
        y_range = list(map(lambda v: hash_vars(locals()['y_hashmap'],v), plot_settings['y'])) \
                if self.settings.get('y_discrete',False) \
                else plot_settings['y']
        size = plot_settings.get('size',float(self.settings['size']))
        size = [float(size)] if not isinstance(size,Iterable) else size
        colour = plot_settings.get('colour','black')
        colour = [colour] if isinstance(colour,str) else colour
        # Convert transparency levels to approapriate data type
        visibility = plot_settings.get('visibility','1.0')
        visibility = [float(visibility)] if isinstance(visibility,str) else visibility
        zorder = plot_settings.get('zorder',[1])
        zorder = [float(zorder)] if isinstance(zorder,str) else zorder 
        label = plot_settings.get('label',[''])
        label = [label] if isinstance(label,str) else label
        marker = plot_settings.get('marker',['o'])
        marker = [marker] if isinstance(marker,str) else marker
        style = plot_settings.get('style',['-'])
        style = [style] if isinstance(style,str) else style
        hatch = plot_settings.get('hatch','')
        hatch = [hatch] if isinstance(hatch,str) else hatch

        # Write figure data
        write_figure_data(
            [plot_settings],
            filepath=filepath,
            key_type={'x':'float','y':'float'},
            aux_keys=PLOT_VARIABLES_AND_DERIVATIVES+['outputs'],
            **self.settings,
            print_data=False
        )
        self.logger.success(f"Figure data exported to {dirpath}")
        

        # Figure size 
        fig, ax = plt.subplots(
            figsize = self.settings['figure_size'],
            ncols = max(1,len(plot_settings['x_group_id'])),
            nrows = max(1,len(plot_settings['y_group_id'])),
            squeeze = False
        )

        # Global x label
        if self.settings.get('x_label',''):
            plt.xlabel(
                self.settings['x_label'].replace("_"," "),
                fontsize=self.settings['axis_label_size'],
                labelpad=self.settings['axis_label_pad'],
                rotation=self.settings['axis_label_rotation']
            )
        # Global y label
        if self.settings.get('y_label',''):
            plt.ylabel(
                self.settings['y_label'].replace("_"," "),
                fontsize=self.settings['axis_label_size'],
                labelpad=self.settings['axis_label_pad'],
                rotation=self.settings['axis_label_rotation'],
            )
        
        # Set axes limits
        if not multiple_subplots:
            # Get global axes limits
            axes_lims = self.get_axes_limits( 
                plot_settings = plot_settings,
                vars = ['x','y'],
                axes_id = 'global'
            )
            
            plt.xlim(left=axes_lims['x'][0], right=axes_lims['x'][1])
            plt.ylim(bottom=axes_lims['y'][0], top=axes_lims['y'][1])

        # Get axes ids and arange them in order
        self.axids = {}
        # Count number of axes ids
        counter = 0
        # Set ticks
        for r in range(max(len(plot_settings['y_group_id']),1)):
            for c in range(max(len(plot_settings['x_group_id']),1)):

                self.axids[(r,c)] = counter
                counter += 1

                for var in ['x','y']:
                    if self.settings.get(f'{var}_discrete',False):
                        # Get discrete ticks
                        all_ticks = self.get_discrete_ticks(
                            var = var,
                            id_var = f"{var}_id",
                            tickfreq_var = f"{var}_tick_frequency",
                            plot_settings = plot_settings
                        )

                        for i in range(len(all_ticks['tick_locations'])):
                            tick_locations = all_ticks['tick_locations'][i]
                            tick_label = all_ticks['tick_label'][i]
                            
                            # print(tick_locations)
                            # print(locals()[f"{var}tick_label"])

                            # Decide on major/minor axis
                            if i == 0:
                                minor = False
                            else:
                                minor = True

                            # Plot ticks
                            getattr(ax[r,c],f"set_{var}ticks",ax[r,c].set_xticks)(
                                ticks = tick_locations,
                                label = tick_label,
                                minor = minor
                            )
                            ax[r,c].tick_params(
                                axis = var, 
                                which = ('minor' if minor else 'major'), 
                                pad = self.settings[f"{var}_tick_pad"][i],
                                bottom = True,
                                labelize = self.settings['tick_label_size'],
                                rotation = self.settings[f"{var}_tick_rotation"][i]
                            )
                        # Set gridlines
                        ax[r,c].grid(axis=var,which='both')
                        getattr(
                            ax[r,c],
                            f"{var}axis",
                            ax[r,c].xaxis
                        ).remove_overlapping_locs = False
                        # Create a discrete hashmap
                        var_hashmap = dict(zip(
                            all_ticks['unique'],
                            tick_locations
                        ))
                        # Rename var_hashmap accordingly
                        exec(f"{var}_hashmap = var_hashmap")
                        print(f'{var}_hashmap')
                        print(globals()[f'{var}_hashmap'])
                    else:
                        getattr(
                            plt,
                            f'{var}ticks',
                            plt.xticks
                        )(
                            fontsize = self.settings['tick_label_size']
                        )

        # Keep track of each group's axes limits
        group_axes_limits = {}
        # Loop over sweeps
        for sid in range(np.shape(y_range)[0]):
            
            # Get axes id 
            axes_id = self.map_groups_to_axes(
                sid,
                plot_settings
            )
            axes_id = tuple([axes_id[var] for var in ['y','x']])

            print(sid,axes_id)
            # Print plotting data
            self.print_data(
                index = sid,
                plot_setting = plot_settings,
                local_vars = dict(locals()),
                plot_vars = ['zorder','label'],
                summarise = False
            )
            self.print_data(
                index = sid,
                plot_setting = plot_settings,
                local_vars = dict(locals()),
                plot_vars = ['y'],
                summarise = True
            )
            # Plot x versus y
            self.plot_wrapper(
                ax = ax[axes_id],
                plot_type = PLOT_TYPES[self.settings.get('plot_type','scatter')],
                x = x_range[sid],
                y = y_range[sid],
                s = size[sid] if len(size) > 0 else size[0],
                style = style[sid] if len(style) > 0 else style[0],
                c = colour[sid] if len(colour) > 0 else colour[0],
                alpha = visibility[sid] if len(visibility) > 0 else visibility[0],
                zorder = zorder[sid] if len(zorder) > 0 else zorder[0],
                label = label[sid] if len(label) > 0 else label[0],
                marker = marker[sid] if len(marker) > 0 else marker[0],
                hatch = hatch[sid] if len(hatch) > 0 else hatch[0]
            )

            # Shade area between line and axis
            if self.settings.get('x_shade',False):
                facecolour = colour[sid] if len(colour) > 0 else colour[0]
                ax[axes_id].fill_betweenx(
                    y = y_range[sid],
                    x1 = 0,
                    x2 = x_range[sid],
                    facecolor = facecolour,
                    edgecolor = 'black' if len(hatch) > 0 else facecolour,
                    zorder = zorder[sid] if len(zorder) > 0 else zorder[0],
                    hatch = hatch[sid] if len(hatch) > 0 else hatch[0],
                    label = label[sid] if len(label) > 0 else label[0],
                    alpha = visibility[sid] if len(visibility) > 0 else visibility[0]
                )
            if self.settings.get('y_shade',False):
                facecolour = colour[sid] if len(colour) > 0 else colour[0]
                ax[axes_id].fill_between(
                    x = x_range[sid],
                    y1 = 0,
                    y2 = y_range[sid],
                    facecolor = facecolour,
                    edgecolor = 'black' if len(hatch) > 0 else facecolour,
                    zorder = zorder[sid] if len(zorder) > 0 else zorder[0],
                    hatch = hatch[sid] if len(hatch) > 0 else hatch[0],
                    label = label[sid] if len(label) > 0 else label[0],
                    alpha = visibility[sid] if len(visibility) > 0 else visibility[0]
                )

            # Annotate data
            if self.settings.get('annotate',False):
                for i, txt in enumerate(
                    plot_settings.get(self.settings.get('annotation_label',''),[])
                ):
                    ax[axes_id].annotate(
                        str(string_to_numeric(txt) 
                            if str(txt).isnumeric() 
                            else str(txt)), 
                        (x_range[sid], y_range[sid])
                    )

            # Get local group axes limits
            group_axes_limits[axes_id] = self.get_axes_limits(
                plot_settings = {
                    "x":x_range[sid],
                    "y":y_range[sid],
                    **{k:v for k,v in plot_settings.items() if k not in ['x','y']}
                },
                vars = ['x','y'],
                axes_id = axes_id,
                axes_lims = group_axes_limits
            )

        # Set local group axes limits, label etc..
        for r in range(max(len(plot_settings['y_group_id']),1)):
            for c in range(max(len(plot_settings['x_group_id']),1)):
                ax[r,c].set_ylim(
                    bottom = group_axes_limits[(r,c)]['y'][0], 
                    top = group_axes_limits[(r,c)]['y'][1]
                )
                ax[r,c].set_xlim(
                    left = group_axes_limits[(r,c)]['x'][0], 
                    right = group_axes_limits[(r,c)]['x'][1]
                )
                if r < len(plot_settings['y_group_id']):
                    print('y',r,plot_settings['y_group_id'][r])
                    ax[r,c].set_ylabel(
                        plot_settings['y_group_id'][r],
                        fontsize = self.settings['subaxis_label_size'],
                        labelpad = self.settings['subaxis_label_pad'],
                        rotation = self.settings['subaxis_label_rotation']
                    )
                if c < len(plot_settings['x_group_id']):
                    print('x',c,plot_settings['x_group_id'][c])
                    ax[r,c].set_xlabel(
                        plot_settings['x_group_id'][c],
                        fontsize = self.settings['subaxis_label_size'],
                        labelpad = self.settings['subaxis_label_pad'],
                        rotation = self.settings['subaxis_label_rotation']
                    )

        # Aspect ratio equal
        if self.settings['equal_aspect']:
            plt.gca().set_aspect('equal')
    
        # Legend
        # try:
        # Create dictionary of labels
        by_label = {}
        handles, label, label_split = [],[],[]
        # Find each label from each plot
        for axid in self.axids.keys():
            
            # Ensure no duplicate entries in legend exist
            ax_handles, ax_label = ax[axid].get_legend_handles_labels()
            # Convert everything to numpy arrays
            ax_label_split = np.array([lab.split(', ') for lab in ax_label],dtype='str')
            ax_handles = np.array(ax_handles)
            ax_label = np.array(ax_label)

            # If no legend axis specified
            # Create a legend for each axis
            if self.settings['legend_axis'] is None:
                # Sort label first by first label, then by second etc.
                index_sorted = np.lexsort(ax_label_split.T)
                # Create dictionary of label
                by_label = dict(zip(
                    ax_label[index_sorted].tolist(), 
                    ax_handles[index_sorted].tolist()
                ))

                leg = ax[axid].legend(
                    by_label.values(), 
                    by_label.keys(),
                    frameon = False,
                    prop = {'size': self.settings.get('legend_label_size',None)},
                    bbox_to_anchor=self.settings.get('legend_location',(1.0, 1.0))
                )
                leg._ncol = 1
            else:
                handles.append(ax_handles)
                label_split.append(ax_label_split)
                label.append(ax_label)

        # Create a legend for legend axis provided
        if self.settings['legend_axis'] is not None:
            
            # Merge all legend labels
            handles = np.concatenate(handles)
            label_split = np.concatenate(label_split)
            label = np.concatenate(label)
            
            # Sort label first by first label, then by second etc.
            index_sorted = np.lexsort(label_split.T)
            # Create dictionary of label
            by_label = dict(zip(
                label[index_sorted].tolist(), 
                handles[index_sorted].tolist()
            ))
    
            leg = ax[tuple(list(self.settings['legend_axis']))].legend(
                by_label.values(), 
                by_label.keys(),
                frameon = False,
                prop = {'size': self.settings.get('legend_label_size',None)},
                bbox_to_anchor=self.settings.get('legend_location',(1.0, 1.0))
            )
            leg._ncol = 1

        
        # Tight layout
        plt.tight_layout()

        # Write figure
        write_figure(
            fig,
            filepath,
            **self.settings
        )
        self.logger.info(f"Filename: {filename}")
        self.logger.success(f"Figure exported to {dirpath}")

    '''
    Extracting plotting data
    '''
    def extract_plot_variable(self,var:str,meta:dict,settings:dict):
        value = None
        if var in PLOT_ALL_COORDINATES:
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
                    # Convert coordinate to list
                    if isinstance(var_value,Iterable) \
                        and not isinstance(var_value,str):
                        if len(var_value) > 0:
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

        elif var in list(PLOT_AUX_FEATURES.keys()):
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
            label_strs = []
            # Get label key and value
            for label_key in settings['label']:
                # If label key has a math expression replace it with math expression
                if label_key in MATH_EXPRESSIONS:
                    if isinstance(MATH_EXPRESSIONS[label_key],dict):
                        label_strs.append(
                            MATH_EXPRESSIONS[label_key][str(meta[label_key])]
                        )
                    else:
                        label_strs.append(
                            r'$'+MATH_EXPRESSIONS[label_key]+'='+str(parse(meta[label_key],'learned',ndigits=3))+r'$'
                        )
                else:
                    label_strs.append(
                        # str(label_key) + ' = ' + tidy_label(str(meta[label_key]))
                        tidy_label(str(meta[label_key]))
                    )
            value = ', '.join(label_strs)
        
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
                    value = PLOT_MARKERS[var_key].get(
                        str(parse(value,'.',ndigits=3)),
                        PLOT_MARKERS[var_key]['else']
                    )
                elif var == 'hatch':
                    value = PLOT_HATCHES[var_key].get(
                        str(parse(value,'+++',ndigits=3)),
                        PLOT_HATCHES[var_key]['else']
                    )
                elif var == 'style':
                    value = PLOT_LINESTYLES[var_key].get(
                        str(parse(value,'-',ndigits=3)),
                        PLOT_LINESTYLES[var_key]['else']
                    )
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
        
        # print(var,type(value),np.shape(value))
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
            if value is None or (
                isinstance(value,Iterable) and \
                 any([v is None for v in list(value)])
            ):
                continue

            # Set variable value
            var_values[variable] = to_json_format(value)
        
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
                var_values[variable+'_id'] = to_json_format(value)

        return var_values

    def merge_plot_settings(self,plot_settings:list):
        merged_settings = {}
        # Iterate through the list of dictionaries
        if len(plot_settings) > 1:
            for d in plot_settings:
                # print_json(d,newline=True)
                # Concatenate values to the merged_dict
                for key, value in d.items():
                    if value is None:
                        if key not in merged_settings:
                            merged_settings[key] = [np.squeeze(value).tolist()]
                        else:
                            merged_settings[key].append(
                                np.squeeze(value).tolist()
                            )

                    if key in PLOT_VARIABLES or key in PLOT_DERIVATIVES:
                        # Append squeezed version of data
                        merged_settings.setdefault(key, []).append(
                            np.squeeze(value).tolist()
                        )
                    else:
                        # Keep only the first value of this key for each
                        # plot setting
                        if key not in merged_settings:
                            merged_settings[key] = np.squeeze(value).tolist()
        else:
            merged_settings = plot_settings[0]
                
        # Create ordering of data points based on provided data
        if 'zorder' in merged_settings:
            # get value to order by
            values = np.array(merged_settings['zorder'])
            if len(values.shape) > 1:
                # Number of elements to sort
                ndims = values.shape[1]
                # Use lexsort to argsort along each axis successively
                sorted_indices = np.lexsort([values[:,i].ravel() for i in range(ndims)])
            else:
                sorted_indices = np.argsort(values)
            # Update merged settings
            # add 1.0 to avoid zorder = 0
            merged_settings['zorder'] = list(map(float,sorted_indices+1.0))
            print(values)
            print(merged_settings['zorder'])

        # flatten list of lists
        # for key,value in merged_settings.items():
            # print(key,np.shape(value),type(value))
        #     # No need to flatten any of these variables
        #     if key in PLOT_COORDINATES:
        #         continue
        #     elif isinstance(value,Iterable) and \
        #         not isinstance(value,str):
        #         # Flatten
        #         merged_settings[key] = list(flatten(value))
        #         # Squeeze
        #         merged_settings[key] = np.squeeze(merged_settings[key]).tolist()
        #     elif key in PLOT_CORE_FEATURES:
        #         # Store as list
        #         merged_settings[key] = [value]
            # print(key,type(merged_settings[key]),np.shape(merged_settings[key]))

        return [merged_settings]

    def create_plot_filename(self,plot_setting,**kwargs):
        # Get filename
        filename = kwargs.get('name','NO_NAME')+'_'+'_'.join([
                    f"({var},burnin{setts['burnin']},thinning{setts['thinning']},trimming{setts['trimming']})"
                    for coord_slice in self.settings['burnin_thinning_trimming']
                    for var,setts in coord_slice.items()
                ])
        if not isinstance(plot_setting['outputs'].config['training']['N'],dict):
            filename += f"_N_{plot_setting['outputs'].config['training']['N']}"
        if self.settings.get('label',['']) != ['']:
            filename += f"_label_{'&'.join([str(elem) for elem in self.settings['label']])}"
        if self.settings.get('marker','.') != '.':
            filename += f"_marker_{self.settings['marker']}"
        if self.settings.get('linestyle','-') != '-':
            filename += f"_linestyle_{self.settings['linestyle']}"
        if self.settings.get('hatch','') != '':
            filename += f"_hatch_{self.settings['hatch']}"
        if self.settings.get('colour','') != '':
            filename += f"_colour_{self.settings['colour']}"
        if self.settings.get('size','') != '':
            filename +=  f"_size_{self.settings['size']}"
        if self.settings.get('visibility','') != '':
            filename += f"_visibility_{self.settings['visibility']}"
           
        # Decide on figure output dir
        if len(self.settings.get('plot_data_dir',[])) == 1:
            dirpath = self.settings['plot_data_dir'][0]
        elif not self.settings['by_experiment']:
            # Get dirpath
            parent_directory = Path(plot_setting['outputs'].outputs_path)
            if 'synthetic' in str(parent_directory):
                parent_directory = plot_setting['outputs'].config.out_directory
            else:
                parent_directory = parent_directory.parent.absolute()
            dirpath = os.path.join(parent_directory,'paper_figures')        
        else:
            # Get dirpath
            dirpath = os.path.join(
                plot_setting['outputs'].outputs_path,
                'figures'
            )
        
        return dirpath,filename
    
    def read_plot_data(self):
        # If directory exists and loading of plot data is instructed
        if len(self.settings.get('plot_data_dir',[])) > 0:
            
            plot_settings = []
            for plot_data_dir in self.settings.get('plot_data_dir',[]):
                
                if os.path.exists(plot_data_dir) and \
                    os.path.isdir(plot_data_dir):

                    # Find data in json format
                    # no other format is acceptable
                    files = list(glob(os.path.join(plot_data_dir,"*data.json"),recursive=False))

                    # If nothing was found return false
                    if len(files) <= 0:
                        continue
                    # Try to read file
                    plot_sett = read_file(files[0])

                    # Canonicalise the data
                    if isinstance(plot_sett,pd.DataFrame):
                        plot_sett = dict(plot_sett.to_dict())
                    elif isinstance(plot_sett,list):
                        plot_sett = plot_sett[0]
                    
                    # Extract outputs
                    if 'outputs' not in plot_sett:
                        continue
                    else:
                        # Try to load outputs
                        if isinstance(plot_sett['outputs'],dict):
                            # Instantiate config
                            config = Config(
                                settings = plot_sett['outputs'],
                                logger = self.logger
                            )
                            # Get sweep-related data
                            config.get_sweep_data()
                        elif isinstance(plot_sett['outputs'],str):
                            # Instantiate config
                            config = Config(
                                settings = plot_sett['outputs'],
                                logger = self.logger
                            )
                            # Get sweep-related data
                            config.get_sweep_data()
                        else:
                            self.logger.warning(f"Outputs are of type {type(plot_sett['outputs'])} and not dict.")
                            continue
                    
                        # Instantiate outputs
                        plot_sett.update(dict(
                            outputs = Outputs(
                                config = config,
                                settings = self.settings,
                                data_names = self.settings['sample'],
                                logger = self.logger,
                                print_slice = False
                            )
                        ))
                    
                    plot_settings.append(plot_sett)

            return True,plot_settings

        else:
            return False,None

    
    '''    
    ╔═╗┬  ┌─┐┌┬┐  ┬ ┬┬─┐┌─┐┌─┐┌─┐┌─┐┬─┐┌─┐
    ╠═╝│  │ │ │   │││├┬┘├─┤├─┘├─┘├┤ ├┬┘└─┐
    ╩  ┴─┘└─┘ ┴   └┴┘┴└─┴ ┴┴  ┴  └─┘┴└─└─┘
    '''
    
    def data_plot(self,plot_func):
            
        self.logger.info('Running data_plot')
    
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
                
                # Get metadata collection for this 
                metadata_collection,outputs = outputs_summary.collect_folder_metadata(indx,output_folder)

                # Create plot settings
                plot_sett = {'outputs':outputs}

                try:
                    # Loop through metadata for each data collection member
                    for metadata in tqdm(
                        metadata_collection,
                        desc = 'Extracting plot settings',
                        leave = False
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
                    if self.settings.get('by_experiment',False):
                        # Create output dirpath and filename
                        dirpath,filename = self.create_plot_filename(
                            plot_setting = plot_sett,
                            name = self.settings.get('figure_title','NONAME')
                        )
                        # Merge all settings into one
                        # Plot
                        plot_func(
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
            if not self.settings.get('by_experiment',False):
                # Create output dirpath and filename
                dirpath,filename = self.create_plot_filename(
                    plot_setting = {'outputs':outputs},
                    name = self.settings.get('figure_title','NONAME')
                )
                print(dirpath)
                # Plot
                plot_func(
                    plot_settings = self.merge_plot_settings(plot_settings),
                    name = self.settings.get('title','NONAME'),
                    dirpath = dirpath,
                    filename = filename
                )
        else:
            # Merge plot settings
            merged_plot_settings = self.merge_plot_settings(plot_settings)
            # Create output dirpath and filename
            dirpath,filename = self.create_plot_filename(
                plot_setting = merged_plot_settings[0],
                name = self.settings.get('figure_title','NONAME')
            )
            # Plot
            plot_func(
                plot_settings = merged_plot_settings,
                name = self.settings.get('title','NONAME'),
                dirpath = dirpath,
                filename = filename
            )