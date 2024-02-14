import os
os.environ['USE_PYGEOS'] = '0'

import matplotlib as mpl
mpl.use('ps')
import sys
import traceback
import seaborn as sns
import geopandas as gpd
import sklearn.manifold
import scipy.stats as stats
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

from gensit.config import Config
from gensit.utils.misc_utils import *
from gensit.static.plot_variables import *
from gensit.static.global_variables import *
from gensit.outputs import Outputs,OutputSummary


# LaTeX font configuration
mpl.rcParams.update(LATEX_RC_PARAMETERS)


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
        self.data_plot(plot_func = self.compile_plot(plot_view), **kwargs)

    def print_data(self,plot_setting:dict,local_vars:dict,plot_vars:list = None,index:int = None,summarise:bool = False):
        if index:
            self.logger.info(f"index: {index}")
        for v in ['x','y','marker_size','colour','line_width','opacity','hatch_opacity','zorder','annotate','line_style','label','marker','hatch']:
            if v not in plot_setting:
                self.logger.debug(f"{v} not found in plot settings.")
                continue
            if plot_vars is None or v in plot_vars:
                if index is None:
                    if summarise:
                        try:
                            self.logger.info(f"{v}: {np.shape(plot_setting[v])} {type(plot_setting[v])}")
                            self.logger.info(f"{v}: min = {np.nanmin(plot_setting[v])}, max = {np.nanmax(plot_setting[v])}")
                        except:
                            try:
                                self.logger.info(f"{v}: {np.shape(plot_setting[v])} {type(plot_setting[v])}")
                            except:
                                if isinstance(plot_setting[v],Iterable):
                                    self.logger.info(f"{v}: {len(plot_setting[v])} {type(plot_setting[v])}")
                                else:
                                    self.logger.info(f"{v}: {type(plot_setting[v])}")
                    else:
                        self.logger.info(f"{v} = {plot_setting[v]}")
                else:
                    if summarise:
                        if plot_setting[v] and isinstance(plot_setting[v],Iterable) and index < len(plot_setting[v]):
                            try:
                                self.logger.info(f"{v}: {np.shape(plot_setting[v][index])} {type(plot_setting[v][index])}")
                                self.logger.info(f"{v}: min = {np.nanmin(plot_setting[v][index])}, max = {np.nanmax(plot_setting[v][index])}")
                            except:
                                try:
                                    self.logger.info(f"{v}: {np.shape(plot_setting[v][index])} {type(plot_setting[v][index])}")
                                except:
                                    if isinstance(plot_setting[v],Iterable):
                                        self.logger.info(f"{v}: {len(plot_setting[v][index])} {type(plot_setting[v][index])}")
                                    else:
                                        self.logger.info(f"{v}: {type(plot_setting[v][index])}")
                        else:
                            try:
                                self.logger.info(f"{v}: {np.shape(plot_setting[v])} {type(plot_setting[v])}")
                                self.logger.info(f"{v}: min = {np.nanmin(plot_setting[v])}, max = {np.nanmax(plot_setting[v])}")
                            except:
                                try:
                                    self.logger.info(f"{v}: {np.shape(plot_setting[v])} {type(plot_setting[v])}")
                                except:
                                    if isinstance(plot_setting[v],Iterable):
                                        self.logger.info(f"{v}: {len(plot_setting[v])} {type(plot_setting[v])}")
                                    else:
                                        self.logger.info(f"{v}: {type(plot_setting[v])}")
                    else:
                        if plot_setting[v] and isinstance(plot_setting[v],Iterable) and index < len(plot_setting[v]):
                            self.logger.info(f"{v} = {plot_setting[v][index]}")
                        else:
                            self.logger.info(f"{v} = {plot_setting[v]}")
        for v in ['x_range','y_range']:
            if v not in local_vars:
                self.logger.debug(f"{v} not found in local variables.")
                continue
            if (plot_vars is None or v in plot_vars) :
                if index is None:
                    if summarise:
                        try:
                            self.logger.info(f"{v}: {np.shape(list(flatten(local_vars[v])))} {type(list(flatten(local_vars[v])))}")
                            self.logger.info(f"{v}: min = {np.nanmin(list(flatten(local_vars[v])))}, max = {np.nanmax(list(flatten(local_vars[v])))}")
                        except:
                            try:
                                self.logger.info(f"{v}: {np.shape(local_vars[v])} {type(local_vars[v])}")
                            except:
                                if isinstance(local_vars[v],Iterable):
                                    self.logger.info(f"{v}: {len(local_vars[v])} {type(local_vars[v])}")
                                else:
                                    self.logger.info(f"{v}: {type(local_vars[v])}")
                    else:
                        try:
                            self.logger.info(f"{v} = {list(flatten(local_vars[v]))}")
                        except:
                            self.logger.info(f"{v} = {local_vars[v]}")
                else:
                    if summarise:
                        if local_vars[v] and isinstance(local_vars[v],Iterable) and index < len(local_vars[v]):
                            try:
                                self.logger.info(f"{v}: {np.shape(list(flatten(local_vars[v][index])))} {type(list(flatten(local_vars[v][index])))}")
                                self.logger.info(f"{v}: min = {np.nanmin(list(flatten(local_vars[v][index])))}, max = {np.nanmax(list(flatten(local_vars[v][index])))}")
                            except:
                                try:
                                    self.logger.info(f"{v}: {np.shape(local_vars[v][index])} {type(local_vars[v][index])}")
                                except:
                                    if isinstance(plot_setting[v][index],Iterable):
                                        self.logger.info(f"{v}: {len(plot_setting[v][index])} {type(plot_setting[v][index])}")
                                    else:
                                        self.logger.info(f"{v}: {type(plot_setting[v][index])}")
                        else:
                            try:
                                self.logger.info(f"{v}: {np.shape(list(flatten(local_vars[v])))} {type(list(flatten(local_vars[v])))}")
                                self.logger.info(f"{v}: min = {np.nanmin(list(flatten(local_vars[v])))}, max = {np.nanmax(list(flatten(local_vars[v])))}")
                            except:
                                try:
                                    self.logger.info(f"{v}: {np.shape(local_vars[v])} {type(local_vars[v])}")
                                except:
                                    if isinstance(local_vars[v],Iterable):
                                        self.logger.info(f"{v}: {len(local_vars[v])} {type(local_vars[v])}")
                                    else:
                                        self.logger.info(f"{v}: {type(local_vars[v])}")
                    else:
                        if local_vars[v] and isinstance(local_vars[v],Iterable) and index < len(local_vars[v]):
                            try:
                                self.logger.info(f"{v} = {list(flatten(local_vars[v][index]))}")
                            except:
                                self.logger.info(f"{v} = {local_vars[v][index]}")
                        else:
                            try:
                                self.logger.info(f"{v} = {list(flatten(local_vars[v]))}")
                            except:
                                self.logger.info(f"{v} = {local_vars[v]}")
        
    def compile_plot(self,visualiser_name):
        if hasattr(self, PLOT_VIEWS[visualiser_name]):
            return getattr(self, PLOT_VIEWS[visualiser_name])
        else:
            raise Exception(f"Experiment class {PLOT_VIEWS[visualiser_name]} not found")
    
    def compile_table_records_in_geodataframe(self,table,geometry):
        # Extract ids from geometry
        origin_geometry_ids = geometry[geometry.geometry_type == self.settings['origin_geometry_type']].geometry_id.tolist()
        destination_geometry_ids = geometry[geometry.geometry_type == self.settings['destination_geometry_type']].geometry_id.tolist()
        # Create dataframe
        table_df = pd.DataFrame(table,index = origin_geometry_ids,columns = destination_geometry_ids)
        # Create pairs of flow records instead of 2D flows
        table_df = table_df.stack().reset_index()
        # Rename columns
        table_df.rename(columns={"level_0":"origin","level_1":"destination",0:"flow"},inplace = True)
        # Attach origin geometry
        table_df = table_df.merge(
                        geometry[['geometry_id','LONG','LAT','geometry','origin_demand']].set_index('geometry_id'),
                        left_on='origin',
                        right_index = True,
                        how='left'
        )
        # Rename geometry
        table_df.rename(columns={"LONG":"origin_long","LAT":"origin_lat","geometry":"origin_geometry"},inplace = True)
        # Attach destination geometry
        table_df = table_df.merge(
                        geometry[['geometry_id','LONG','LAT','geometry','destination_demand']].set_index('geometry_id'),
                        left_on='destination',
                        right_index = True,
                        how='left'
        )
        # Rename geometry
        table_df.rename(
                columns={
                    "LONG":"destination_long",
                    "LAT":"destination_lat",
                    "geometry":"destination_geometry"
                },inplace = True)

        # Convert to geopandas
        return gpd.GeoDataFrame(table_df,geometry='origin_geometry')

    def get_axes_limits(self,plot_settings,vars:list,axes_id:str,axes_lims=None,discrete_hashmaps:dict={}):
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
                
                # If variable is discrete set limits from hashmap
                if self.settings.get(f'{var}_discrete',False):
                    min_val = np.min(list(discrete_hashmaps[var].values())).astype('int32')
                    max_val = np.max(list(discrete_hashmaps[var].values())).astype('int32')
                else:
                    # If var is not discrete read only major axis data, i.e. axis 0
                    if isinstance(plot_settings[var],Iterable):
                        try:
                            min_val = min(np.nanmin(np.ravel(plot_settings[var])),min_val)
                        except:
                            min_val = min(np.nanmin(np.ravel([dt[0] for dt in plot_settings[var]])),min_val)
                    else:
                        min_val = min(plot_settings[var],min_val)
                    if isinstance(plot_settings[var],Iterable):
                        try:
                            max_val = max(np.nanmax(np.ravel(plot_settings[var])),max_val)
                        except:
                            max_val = max(np.nanmax(np.ravel([dt[0] for dt in plot_settings[var]])),max_val)
                    else:
                        max_val = max(plot_settings[var],max_val)
                
                # Update axis limits
                axes_lims[axes_id][var] = [min_val,max_val]

                self.logger.progress(f"Set {var} limits for index {axes_counter}: {min_val}, {max_val}")
            else:
                # Update existing limits
                if self.settings.get(f'{var}_discrete',False):
                    min_val = min(0,axes_lims[axes_id][var][0])
                    max_val = max(np.max(list(discrete_hashmaps[var].values())).astype('int32')+1,axes_lims[axes_id][var][1])
                else:
                    min_val = axes_lims[axes_id][var][0]
                    max_val = axes_lims[axes_id][var][1]
                # Update axis limits
                axes_lims[axes_id][var] = [min_val,max_val]

                self.logger.progress(f"Updated {var} limits for index {axes_counter}: {min_val}, {max_val}")

        return axes_lims[axes_id]
    
    def get_discrete_ticks(self,plot_settings,var:str,tick_locator_var:str=None):
        # All ticks (element 0 for major and element 1 for minor)
        ticks = [
            {
                "data": [],
                "unique":[],
                "locations":[],
                "labels":[]
            },
            {
                "data": [],
                "unique":[],
                "locations":[],
                "labels":[]
            }
        ]
        # First, figure out tick locations and labels for major/minor ticks
        # For each subtick (one for major and one for minor ticks)
        for i, tick_type in reversed(list(enumerate(['major','minor']))):
            try:
                # print(var,i,len(plot_settings[var]),len(plot_settings[var][0]))
                # Get major or minor ticks
                ticks[i]['data'] = [v[i] for v in plot_settings[var]]
                # print(var,tick_type,ticks[i]['data'])
                # Get number of dimensions in minor or major tick tuple/list
                # There must be exactly 2 dimensions 
                # (number of data points, major or minor axis tick tuple/list)
                if len(np.shape(ticks[i]['data'])) == 1:
                    ticks[i]['data'] = [[dt] for dt in ticks[i]['data']]
                # print(np.shape(ticks[i]['data']))
                D = np.shape(ticks[i]['data'])[1]
                # Sort each of their sub-dimensions and merge them into one
                ticks[i]['data'] = sorted(ticks[i]['data'], key = lambda x: tuple([unstringify(x[di]) for di in range(D)]))
                # print('sorted',ticks[i]['data'])
                # Create string id over all &-separated dims in column 1
                ticks[i]['data'] = np.array([
                    stringify(
                        td,
                        scientific = self.settings.get(f"{var}_scientific",False)
                    ) 
                    for td in ticks[i]['data'] if td
                ])
                # print(tick_type,np.shape(ticks[i]['data']),ticks[i]['data'])
                # Get unique tick string labels
                unique_indices = np.unique(ticks[i]['data'], return_index=True)[1]
                # Assert that there is at least one tick label
                assert len(unique_indices) > 0
                ticks[i]['unique'] = [ticks[i]['data'][ind] for ind in sorted(unique_indices)]
                # Stringify and latex if required
                # print(tick_type,'unique',ticks[i]['data'])
                # Make sure tick labels are repeated to match the length of the data
                if (i+1) < len(ticks):
                    n_repetitions = len(ticks[i+1]['unique'])
                else:
                    n_repetitions = 1
                # Repeat labels if required
                ticks[i]['labels'] = ticks[i]['unique']*n_repetitions
                # print(var,'labels',ticks[i]['labels'])
                
                # Get tick locations
                if self.settings.get(tick_locator_var,None):
                    tick_start_loc = self.settings[tick_locator_var][i][0]
                    tick_step_loc = self.settings[tick_locator_var][i][1]
                    tick_end_loc = tick_start_loc + len(ticks[i]['labels'])*tick_step_loc
                    ticks[i]['locations'] = np.arange(
                        tick_start_loc,
                        tick_end_loc,
                        tick_step_loc
                    )[:len(ticks[i]['labels'])]
                # print(var,'locations',tick_type,ticks[i]['locations'])
            except (IndexError,AssertionError):
                traceback.print_exc()
                ticks[i]['unique'] = [None]
                continue
            except Exception:
                traceback.print_exc()
                sys.exit()
            # Remove data from ticks
            ticks[i].pop('data')
        
        # Second, create a hashmap
        hashmap = {}
        # print(self.settings[tick_locator_var])
        # Get major and minor unique ticks
        for major,minor in list(product(ticks[0]['unique'],ticks[1]['unique'])):
            # First find major index (there should be only one)
            major_index = ticks[0]['unique'].index(major)
            # Get tick location of minor if minor exists
            if minor:
                # Second minor index (there should be only one)
                minor_index = ticks[1]['unique'].index(minor)
                # Use both to get the major tick location
                # print(major,minor,major_index,minor_index,len(ticks[0]['unique']),len(ticks[1]['unique']))
                # tick_location = major_index + \
                #     minor_index*len(ticks[0]['unique'])*self.settings[tick_locator_var][0][1] + \
                #     self.settings[tick_locator_var][0][0]
                if self.settings.get(tick_locator_var,None):
                    tick_location = minor_index*self.settings[tick_locator_var][1][1] + \
                        major_index*self.settings[tick_locator_var][0][1] + \
                        self.settings[tick_locator_var][0][0]
                    # print(tick_location)
                    # Create entry on hashmap
                    hashmap[stringify(
                        [major,minor],
                        scientific = self.settings.get(f"{var}_scientific",False)
                    )] = tick_location
            else:
                # print(major,major_index,len(ticks[0]['unique']))
                # If there is no minor tick then the location is simply the 
                # major's first (and only location)
                if self.settings.get(tick_locator_var,None):
                    tick_location = major_index*self.settings[tick_locator_var][0][1] + \
                        self.settings[tick_locator_var][0][0]
                    hashmap[stringify(
                        major,
                        scientific = self.settings.get(f"{var}_scientific",False)
                    )] = tick_location
                else:
                    tick_location = major_index
                    hashmap[stringify(
                        major,
                        scientific = self.settings.get(f"{var}_scientific",False)
                    )] = tick_location
                    # print('default major tick loc')

                # print(tick_location)
                # Delete minor ticks from list
                if len(ticks) == 2:
                    ticks.pop()
    
        # print(var,hashmap,'\n')
        return ticks,hashmap
    

    def map_groups_to_axes(self,index,plot_settings:dict,group_hashmap:dict={}):
        # For each variable
        axes = {}
        for var in ['y','x']:
            # If group exists
            if group_hashmap[var]:
                # Get current group
                if index < len(plot_settings.get(f"{var}_group")): 
                    # print(plot_settings.get(f"{var}_group"))
                    # Get group id and stringify it
                    # print(plot_settings.get(f"{var}_group")[index])
                    group_id = stringify(
                        plot_settings.get(f"{var}_group")[index],
                        scientific = self.settings.get(f"{var}_scientific",False)
                    )
                    # print(var,group_id)
                    # Get index of group id
                    if group_id in group_hashmap[var]:
                        axes[var] = group_hashmap[var][group_id]
            # Set axes to zero if no group was found
            if var not in axes:
                axes[var] = 0
            # print(axes[var])
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
                linewidth = kwargs.get('line_width',None),
                markersize = kwargs.get('s',None),
                linestyle = kwargs.get('line_style',None),
                c = kwargs.get('c',None),
                alpha = kwargs.get('alpha',None),
                zorder = kwargs.get('zorder',None),
                label = kwargs.get('label',None),
            )
                
        elif plot_type == 'scatter':
            # print(kwargs['s'])
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

        # Flag for whether multiple subplots are plotted
        plot_settings = plot_settings[0]
        # Store whether either x or y are grouped by 
        group_hashmap = {}
        for var in ['x','y']:
            # Get flag for whether multiple groups exist for var
            if len(plot_settings[f'{var}_group']) <= 0:
                group_hashmap[var] = {}
            else:
                # Get group hashmap
                _,group_hashmap[var] = self.get_discrete_ticks(
                    var = var+'_group',
                    plot_settings = plot_settings
                )
                # print(var,group_hashmap[var])
        # Store whether subplots will be created
        subplots_exist = any([v for v in group_hashmap.values()])
        # Get axes ids and arange them in order
        self.axids = {}
        # Count number of axes ids
        counter = 0
        # Set ticks
        for r in range(max(len(group_hashmap['y']),1)):
            for c in range(max(len(group_hashmap['x']),1)):
                self.axids[(r,c)] = counter
                counter += 1

        # Get discrete variable hashmaps
        discrete_hashmaps = {}
        discrete_ticks = {}
        for var in ['x','y']:
            if self.settings.get(f'{var}_discrete',False):
                # Get discrete ticks
                discrete_ticks[var],discrete_hashmaps[var] = self.get_discrete_ticks(
                    var = var,
                    tick_locator_var = f"{var}_tick_locations",
                    plot_settings = plot_settings
                )
        
        # print(plot_settings['x'])
        print(discrete_hashmaps['x'])
        # print(group_hashmap)
        # print(discrete_ticks['x'])
        
        # Extract data
        x_range = list(map(lambda v: hash_major_minor_var(
            discrete_hashmaps['x'],
            v,
            scientific=self.settings.get('x_scientific',False)
        ), plot_settings['x'])) \
        if self.settings.get('x_discrete',False) \
        else [dt[0] for dt in plot_settings['x']] # keep only major axis data
        y_range = list(map(lambda v: hash_major_minor_var(
            discrete_hashmaps['y'],
            v,
            scientific=self.settings.get('y_scientific',False)
        ), plot_settings['y'])) \
        if self.settings.get('y_discrete',False) \
        else [dt[0] for dt in plot_settings['y']] # keep only major axis data
        marker_size = plot_settings.get('marker_size',1.0)
        marker_size = [float(marker_size)] if not isinstance(marker_size,Iterable) else marker_size
        colour = plot_settings.get('colour','black')
        colour = [colour] if isinstance(colour,str) else colour
        # Convert transparency levels to approapriate data type
        opacity = plot_settings.get('opacity','1.0')
        opacity = [float(opacity)] if isinstance(opacity,str) else opacity
        # Convert hatch pattern transparency levels to approapriate data type
        hatch_opacity = plot_settings.get('hatch_opacity','1.0')
        hatch_opacity = [float(hatch_opacity)] if isinstance(hatch_opacity,str) else hatch_opacity
        zorder = plot_settings.get('zorder',[1])
        zorder = [float(zorder)] if isinstance(zorder,str) else zorder 
        label = plot_settings.get('label',[''])
        label = [label] if isinstance(label,str) else label
        marker = plot_settings.get('marker',['o'])
        marker = [marker] if isinstance(marker,str) else marker
        line_style = plot_settings.get('line_style',['-'])
        line_style = [line_style] if isinstance(line_style,str) else line_style
        line_width = plot_settings.get('line_width','1.0')
        line_width = [float(line_width)] if isinstance(line_width,str) else line_width
        annotate = plot_settings.get('annotate','')
        annotate = [annotate] if isinstance(annotate,str) else annotate
        hatch = plot_settings.get('hatch','')
        hatch = [hatch] if isinstance(hatch,str) else hatch

        # print(y_range)
        # print(x_range)
        

        # Figure size 
        fig, ax = plt.subplots(
            figsize = self.settings['figure_size'],
            ncols = max(1,len(group_hashmap['x'])),
            nrows = max(1,len(group_hashmap['y'])),
            squeeze = False
        )

        # Global axes label
        for var in ['x','y']:
            if self.settings.get(f'{var}_label',''):
                axis_label = self.settings[f'{var}_label'].replace("_"," ")
                getattr(plt,f"{var}label",plt.xlabel)(
                    axis_label,
                    fontsize = self.settings[f'{var}_label_size'],
                    labelpad = self.settings[f'{var}_label_pad'],
                    rotation = self.settings[f'{var}_label_rotation']
                )
        
        # Global axes limits
        if not subplots_exist:
            # Get global axes limits
            axes_lims = self.get_axes_limits( 
                plot_settings = plot_settings,
                vars = ['x','y'],
                axes_id = (0,0),
                discrete_hashmaps = discrete_hashmaps
            )
            plt.xlim(left = axes_lims['x'][0], right = axes_lims['x'][1])
            plt.ylim(bottom = axes_lims['y'][0], top = axes_lims['y'][1])

        # Count number of axes ids
        counter = 0
        # Set ticks
        print(group_hashmap)
        for r in range(max(len(group_hashmap['y']),1)):
            for c in range(max(len(group_hashmap['x']),1)):
                for j,var in enumerate(['x','y']):
                    # Count number of discrete tick types
                    discrete_tick_types = 0
                    for i,tick_type in enumerate(['major','minor']):
                        if self.settings.get(f'{var}_discrete',False) and i < len(discrete_ticks[var]):
                            # Get discrete ticks
                            tick_locations = discrete_ticks[var][i]['locations']
                            tick_labels = discrete_ticks[var][i]['labels']
                            # print(tick_labels)
                            # print(tick_locations)
                            # Plot ticks
                            getattr(ax[r,c],f"set_{var}ticks",ax[r,c].set_xticks)(
                                ticks = tick_locations,
                                labels = tick_labels,
                                minor = (tick_type == 'minor')
                            )
                            # Get tick label pad
                            tick_pad = safe_list_get(self.settings[f"{var}_tick_pad"],i,self.settings[f"{var}_tick_pad"][0])
                            # Get tick label pad
                            tick_rotation = safe_list_get(self.settings[f"{var}_tick_rotation"],i,self.settings[f"{var}_tick_rotation"][0])
                            # Get tick label size
                            tick_size = safe_list_get(self.settings[f"{var}_tick_size"],i,self.settings[f"{var}_tick_size"][0])
                            # print(tick_type,tick_pad,tick_rotation,tick_size)
                            ax[r,c].tick_params(
                                axis = var, 
                                which = tick_type, 
                                pad = tick_pad,
                                bottom = True,
                                labelsize = tick_size,
                                rotation = tick_rotation
                            )
                            # Increment discrete tick types
                            discrete_tick_types += 1
                        else:
                            # Set tick parameters for continuous ticks if there are properly specified
                            if self.settings[f"{var}_tick_locations"] and self.settings[f"{var}_tick_locations"][0]:
                                # Read axis limits
                                start, end = getattr(
                                    ax[r,c],
                                    f"get_{var}lim()",
                                    ax[r,c].get_xlim()
                                )
                                # Change frequency at which continuous ticks appear
                                getattr(
                                    ax[r,c],
                                    f"{var}axis",
                                    ax[r,c].xaxis
                                ).set_ticks(np.arange(
                                    self.settings[f"{var}_tick_locations"][0][0], 
                                    end,
                                    self.settings[f"{var}_tick_locations"][0][1]
                                ))
                            tick_params = {
                                "pad": self.settings[f"{var}_tick_pad"][0] if self.settings[f"{var}_tick_pad"] else None,
                                "labelsize": self.settings[f"{var}_tick_size"][0] if self.settings[f"{var}_tick_size"] else None,
                                "rotation": self.settings[f"{var}_tick_rotation"][0] if self.settings[f"{var}_tick_rotation"] else None
                            }
                            ax[r,c].tick_params(
                                axis = var, 
                                which = 'both',
                                bottom = True,
                                **{k:v for k,v in tick_params.items() if v}
                            )

                    
                    # Set gridlines for discrete ticks (major or (major and minor))
                    if discrete_tick_types > 0:
                        ax[r,c].grid(
                            axis = var,
                            which = 'major'# if discrete_tick_types == 1 else 'both')
                        )
                        ax[r,c].set_axisbelow(True)
                        getattr(
                            ax[r,c],
                            f"{var}axis",
                            ax[r,c].xaxis
                        ).remove_overlapping_locs = False
                    
        # Keep track of each group's axes limits
        group_axes_limits = {}
        group_axes_data = {}
        
        # Loop over sweeps
        for sid in range(len(y_range)):
            # Get axes id 
            axes_id = self.map_groups_to_axes(
                index = sid,
                plot_settings = plot_settings,
                group_hashmap = group_hashmap
            )
            axes_id = tuple([int(axes_id[var]) for var in ['y','x']])
            
            # Initialise group axes data (if required)
            group_axes_data.setdefault(axes_id,{})
            for feature in ['x','y','s','linestyle', 'linewidth', 'c', 'alpha', 
                            'zorder', 'label', 'marker', 'hatch', 'hatch_opacity',
                            'facecolor', 'edgecolor', 'annotate']:
                group_axes_data[axes_id].setdefault(feature,[])
            
            # Unpack data
            group_axes_data[axes_id]['x'] = x_range[sid]
            group_axes_data[axes_id]['y'] = y_range[sid]
            group_axes_data[axes_id]['s'] = unpack_data(marker_size,sid)
            group_axes_data[axes_id]['linestyle'] = unpack_data(line_style,sid)
            group_axes_data[axes_id]['linewidth'] = unpack_data(line_width,sid)
            group_axes_data[axes_id]['c'] = unpack_data(colour,sid)
            group_axes_data[axes_id]['alpha'] = unpack_data(opacity,sid)
            group_axes_data[axes_id]['zorder'] = unpack_data(zorder,sid)
            group_axes_data[axes_id]['label'] = unpack_data(label,sid)
            group_axes_data[axes_id]['marker'] = unpack_data(marker,sid)
            group_axes_data[axes_id]['hatch'] = unpack_data(hatch,sid)
            group_axes_data[axes_id]['hatch_opacity'] = unpack_data(hatch_opacity,sid)
            group_axes_data[axes_id]['annotate'] = unpack_data(annotate,sid)
            group_axes_data[axes_id]['facecolor'] = group_axes_data[axes_id]['c']
            group_axes_data[axes_id]['edgecolor'] = ('black',group_axes_data[axes_id]['hatch_opacity']) \
                if group_axes_data[axes_id]['hatch_opacity'] \
                else (group_axes_data[axes_id]['facecolor'],group_axes_data[axes_id]['hatch_opacity'])
            
            # print(sid,axes_id)
            # Print plotting data
            self.print_data(
                index = None,
                plot_setting = plot_settings,
                local_vars = group_axes_data[axes_id],
                #dict(locals().vars)
                plot_vars = [],
                summarise = False
            )
            self.print_data(
                index = None,
                plot_setting = plot_settings,
                local_vars = group_axes_data[axes_id],
                #dict(locals().vars)
                plot_vars = [],
                summarise = True
            )
            
            # Plot x versus y
            self.plot_wrapper(
                ax = ax[axes_id],
                plot_type = PLOT_TYPES[self.settings.get('plot_type','scatter')],
                **group_axes_data[axes_id]
            )

            # Shade area between line and axis
            if self.settings.get('x_shade',False):
                plt.rcParams['hatch.linewidth'] = self.settings.get('hatch_linewidth',1.0)
                ax[axes_id].fill_betweenx(
                    y = group_axes_data[axes_id]['y'],
                    x1 = 0,
                    x2 = group_axes_data[axes_id]['x'],
                    **{
                        k:v for k,v in group_axes_data[axes_id].items()
                        if k in ['facecolor', 'edgecolor', 'zorder', 'hatch', 'label', 'alpha']
                    }
                )
            if self.settings.get('y_shade',False):
                plt.rcParams['hatch.linewidth'] = self.settings.get('hatch_linewidth',1.0)
                ax[axes_id].fill_between(
                    x = group_axes_data[axes_id]['x'],
                    y1 = 0,
                    y2 = group_axes_data[axes_id]['y'],
                    **{ 
                        k:v for k,v in group_axes_data[axes_id].items()
                        if k in ['facecolor', 'edgecolor', 'zorder', 'hatch', 'label', 'alpha']
                    }
                )

            # Annotate data
            if group_axes_data[axes_id]['annotate']:
                txt = group_axes_data[axes_id]['annotate']
                ax[axes_id].annotate(
                    str(string_to_numeric(txt) 
                        if str(txt).isnumeric() 
                        else str(txt)), 
                    (group_axes_data[axes_id]['x'], group_axes_data[axes_id]['y']),
                    zorder = 10000 # bring annotation data to front
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
                axes_lims = group_axes_limits,
                discrete_hashmaps = discrete_hashmaps
            )

        # Set local group axes limits, label etc..
        for r in range(max(len(group_hashmap['y']),1)):
            for c in range(max(len(group_hashmap['x']),1)):
                for var in ['x','y']:
                    if var in group_axes_limits[(r,c)]:
                        # Set axis limits for var that is grouped
                        getattr(
                            ax[r,c],
                            f"set_{var}lim",
                            ax[r,c].set_ylim
                        )(
                            group_axes_limits[(r,c)][var][0], 
                            group_axes_limits[(r,c)][var][1]
                        )
                    # Set axis scaling
                    if self.settings.get(f"{var}_scale",''):
                        getattr(
                            ax[r,c],
                            f"set_{var}scale",
                            ax[r,c].set_yscale(self.settings[f"{var}_scale"])
                        )

                    # Try to set `axes label for var that is grouped
                    try:
                        if group_hashmap[var]:
                            ax[r,c].set_ylabel(
                                list(group_hashmap[var].keys())[r],
                                fontsize = self.settings[f'{var}_label_size'],
                                labelpad = self.settings[f'{var}_label_pad'],
                                rotation = self.settings[f'{var}_label_rotation']
                            )
                    except:
                        # print(r,c, 'y label skipped')
                        continue

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
            ax_handles, ax_labels = ax[axid].get_legend_handles_labels()
            # Convert everything to numpy arrays
            ax_label_split = [lab.split(', ') for lab in ax_labels]
            
            # Add legend handles and labels to list
            handles += ax_handles
            label_split += ax_label_split
            label += ax_labels
        
        # Create a legend for specific or all axes
        plot_legend = self.settings['legend_axis'] in list(self.axids.keys())
        if plot_legend:
            label_split = np.array(label_split,dtype='str')
            # Sort label first by first label, then by second etc.
            index_sorted = np.lexsort(label_split.T)
            # Do not worry about duplicates. These will be handled here
            # Create dictionary of label
            by_label = dict(zip(
                np.array(label)[index_sorted].tolist(), 
                np.array(handles)[index_sorted].tolist()
            ))

            # Gather legend kwargs
            leg_kwargs = {
                'bbox_to_anchor': self.settings.get('bbox_to_anchor',None),
                'ncols': self.settings.get('legend_cols',1),
                'columnspacing': self.settings.get('legend_col_spacing',None),
                'handletextpad': self.settings.get('legend_pad',None),
                'loc': self.settings.get('legend_location','best')
            }
            print(leg_kwargs)
            # Remove nulls 
            leg_kwargs = {k:v for k,v in leg_kwargs.items() if v}
            # If more than one column are provided split legend patches and keys
            # into sublists of length ncols
            leg = ax[axid].legend(
                flip(list(by_label.values()), self.settings.get('legend_cols',1)), 
                flip(list(by_label.keys()), self.settings.get('legend_cols',1)),
                frameon = False,
                prop = {'size': self.settings.get('legend_label_size',None)},
                **leg_kwargs
            )
            leg._ncol = 1
        

        
        # Tight layout
        if self.settings.get('figure_format','pdf') == 'ps':
            fig.tight_layout(rect=(0, 0, 0.7, 1.1))
        else:
            fig.tight_layout()

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
                # If the group is not iterable -make it so
                if not isinstance(grp, Iterable) or isinstance(grp,str):
                    grp = [grp]

                # If group is empty append empty list
                if len(grp) <= 0:
                    # Add group values to value
                    value.append(grp_values)
                    continue

                for group_var in grp:
                    # Get value from metadata element
                    if group_var in meta:
                        # Get value
                        var_value = get_value(
                            meta,
                            group_var,
                            default = DATA_SCHEMA.get(group_var,{}).get('default',value),
                            apply_latex = (var in LATEX_COORDINATES)
                        )
                    # Data not found
                    else:
                        self.logger.error(f"Could not find {var} data for {group_var} var.")
                        return None
                    # TODO: Fix the fact that default sigma is none and not learned

                    # Convert coordinate to list
                    if isinstance(var_value,Iterable) and \
                        not isinstance(var_value,str):
                        var_value = list(var_value)
                    # add to sub-variable values
                    if len(grp) > 1:
                        if var not in (PLOT_COORDINATES+PLOT_COORDINATE_DERIVATIVES):
                            # If more than one variables are provided for this coordinate
                            # convert all values to string
                            grp_values.append(str(var_value))
                        else:
                            grp_values.append(var_value)
                    else:
                        grp_values = var_value
    
                # Combine all sub-variable values iff they are not part of the core coordinates
                if len(grp) > 1 and var not in (PLOT_COORDINATES+PLOT_COORDINATE_DERIVATIVES):
                    # If more than one variables are provided for this coordinate
                    # convert value to string tuple
                    grp_values = "(" + ", ".join(grp_values) + ")"
                
                # Add group values to value
                value.append(grp_values)

        elif var == 'label':
            label_strs = []
            # Get label key and value
            for label_key in settings['label']:
                label_strs.append(
                    latex_it(
                        key = label_key,
                        value = meta[label_key],
                        default = DATA_SCHEMA.get(label_key,{}).get('default',None)
                    )
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
                var_val = get_value(
                    meta,
                    var_key,
                    default = 1,
                    apply_latex = False
                )
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
            var_key = get_value(
                settings,
                var,
                default = None,
                apply_latex = False
            )
            # If no settings provided return None
            if not var_key:
                return None
            # Get value from metadata element
            if var_key in meta:
                # Extract value
                value = get_value(
                    meta,
                    var_key,
                    default = None,
                    apply_latex = (var in LATEX_COORDINATES)
                )
                
                # Determine plot features based on global plot settings
                if var == 'marker':
                    value = PLOT_MARKERS[var_key].get(
                        str(parse(value,'.',ndigits = 3)),
                        PLOT_MARKERS[var_key]['else']
                    )
                elif var == 'hatch':
                    value = PLOT_HATCHES[var_key].get(
                        str(parse(value,'+++',ndigits = 3)),
                        PLOT_HATCHES[var_key]['else']
                    )
                elif var == 'line_style':
                    value = PLOT_LINESTYLES[var_key].get(
                        str(parse(value,'-',ndigits = 3)),
                        PLOT_LINESTYLES[var_key]['else']
                    )
                elif var == 'colour':
                    # Try to extract colour from global settings
                    colour_value = PLOT_COLOURS.get(var_key,{}).get(value,None)
                    value = colour_value if colour_value else value
            # This is the case where the value is passed to the variable
            # directly and not though a metric/metadata key
            elif var_key:
                # Convert string input to relevant dtype
                value = PLOT_CORE_FEATURES[var]["dtype"](var_key)
            # Data not found
            else:
                self.logger.error(settings)
                self.logger.error(f"Could not find '{var}' data for '{var_key}' var.")
                return None
            # Convert x or y coordinate to list
            if isinstance(value,Iterable) and not isinstance(value,str):
                value = list(value)
            # print(var,var_key,value)
        elif var not in PLOT_COORDINATE_DERIVATIVES:
            value = get_value(
                settings,
                var,
                default = None,
                apply_latex = (var in LATEX_COORDINATES)
            )
            if value is None:
                self.logger.error(f"Could not find '{var}' data for var in settings.")
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
            # print(variable,var_values[variable])
        
        # If the variable is a plot coordinate
        # add a global identifier of all sub-coordinates together
        for variable in vars:
            if variable in PLOT_COORDINATES:
                for derivative in PLOT_DERIVATIVES:
                    # Extract global identifier
                    value = self.extract_plot_variable(
                        var = variable,
                        meta = meta,
                        settings = {variable:self.settings[variable]}
                    )
                    # If no value found move on
                    if value is None:
                        # print('skipping',variable+derivative)
                        continue

                    # Set variable value
                    # var_values[variable+derivative] = to_json_format(value)
                    # print(variable+derivative,var_values[variable+derivative])
                    
        return var_values

    def merge_plot_settings(self,plot_settings:list,apply_zorder:bool=True):
        merged_settings = {}
        # Iterate through the list of dictionaries
        if isinstance(plot_settings,list):
            for d in plot_settings:
                # print_json(d,newline = True)
                # Concatenate values to the merged_dict
                for key, value in d.items():
                    if value is None:
                        # Append squeezed version of data
                        merged_settings.setdefault(key, []).append([])
                        continue
                    if key in (PLOT_VARIABLES+PLOT_AUX_COORDINATES+PLOT_COORDINATE_DERIVATIVES):
                        if isinstance(value,Iterable) and not isinstance(value,str):
                            # x,y,z vary across both major/minor axes AND &-separated dimensions
                            # so do NOT squeeze them
                            data = value if isinstance(value,list) else value.tolist()
                        else: 
                            data = value
                        # Append squeezed version of data
                        merged_settings.setdefault(key, []).append(data)
                    else:
                        # Keep only the first value of this key for each
                        # plot setting
                        if key not in merged_settings:
                            merged_settings[key] = value
            
            # print_json(merged_settings,newline=True)
            # print(merged_settings['y_group'])
        else:
            merged_settings = plot_settings
        
        if apply_zorder:
            # print(merged_settings['y_group'])
            # Create ordering of data points based on provided data
            if 'zorder' in merged_settings:
                # get value to order by
                values = np.array(merged_settings['zorder'])
                if len(values.shape) > 1:
                    # Number of elements to sort
                    ndims = values.shape[1]
                    print('lexsort')
                    # Use lexsort to argsort along each axis successively
                    sorted_indices = np.lexsort([values[:,i].ravel() for i in range(ndims)])
                else:
                    print('argsort')
                    sorted_indices = np.argsort(values,axis=0)
                # Update merged settings
                # add 1.0 to avoid zorder = 0
                merged_settings['zorder'] = np.zeros(len(sorted_indices))
                for i,j in enumerate(sorted_indices):
                    merged_settings['zorder'][j] = len(sorted_indices) - i
                # Convert back to list
                merged_settings['zorder'] = merged_settings['zorder'].tolist()

        # Remove nulls from certain data
        for key in PLOT_AUX_COORDINATES:
            if key in merged_settings:
                value = deepcopy(merged_settings[key])
                # All variable groups should NOT contain nulls
                if isinstance(value,Iterable) and \
                    not isinstance(value,str):
                    # All groups must list of lists (2d lists)
                    # contrary to x,y,z which can (nonhomogeneous lists of 2d lists)
                    merged_settings[key] = [list(flatten(v)) for v in value if v]
                elif value:
                    merged_settings[key] = [value]
                # print(key,merged_settings[key])

        return [merged_settings]
    
    def flatten_merged_settings(self,merged_data):
        for key in merged_data.keys():
            if key in ['outputs']:
                continue
            merged_data[key] = [x for xs in merged_data[key] for x in xs]
        return merged_data
    
    def create_plot_filename(self,plot_setting,**kwargs):
        # Get filename
        filename = [
            kwargs.get('name','NO_NAME'),
            '_'.join([
                f"({var},burnin{setts['burnin']},thinning{setts['thinning']},trimming{setts['trimming']})"
                for coord_slice in self.settings['burnin_thinning_trimming']
                for var,setts in coord_slice.items()
            ])
        ]
        if self.settings.get('label',['']) != ['']:
            filename += [f"label_{'&'.join([str(elem) for elem in self.settings['label']])}"]
        if self.settings.get('marker','.') != '.':
            filename += [f"marker_{self.settings['marker']}"]
        if self.settings.get('marker_size','') != '':
            filename += [f"markersize_{self.settings['marker_size']}"]
        if self.settings.get('line_style','-') != '-':
            filename += [f"linestyle_{self.settings['line_style']}"]
        if self.settings.get('line_width','') != '':
            filename += [f"linewidth_{self.settings['line_width']}"]
        if self.settings.get('hatch','') != '':
            filename += [f"hatch_{self.settings['hatch']}"]
        if self.settings.get('colour','') != '':
            filename += [f"colour_{self.settings['colour']}"]
        if self.settings.get('opacity','') != '':
            filename += [f"opacity_{self.settings['opacity']}"]
        if self.settings.get('hatch_opacity','') != '':
            filename += [f"hatchopacity_{self.settings['hatch_opacity']}"]
        
        # Joint into string filename
        filename = '_'.join([f for f in filename if f])
           
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
                    files = list(glob(os.path.join(plot_data_dir,"*data.json"),recursive = False))
                    # If nothing was found return false
                    if len(files) <= 0:
                        continue
                    # Try to read first file
                    for fl in files:
                        plot_sett = read_file(fl)
                    
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
                                    slice = False
                                )
                            ))
                    
                        plot_settings.append(plot_sett)
                
                # merged_plot_settings = self.merge_plot_settings(
                #     plot_settings,
                #     apply_zorder = False
                # )
                # merged_plot_settings = self.flatten_merged_settings(
                #     merged_data = merged_plot_settings[0]
                # )
                # print_json({k:(v.config.settings if isinstance(v,Outputs) else v) for k,v in merged_plot_settings.items()},indent=2)
                # sys.exit()
            # print(plot_settings)
            return True,plot_settings

        else:
            return False,None

    
    '''    
    ╔═╗┬  ┌─┐┌┬┐  ┬ ┬┬─┐┌─┐┌─┐┌─┐┌─┐┬─┐┌─┐
    ╠═╝│  │ │ │   │││├┬┘├─┤├─┘├─┘├┤ ├┬┘└─┐
    ╩  ┴─┘└─┘ ┴   └┴┘┴└─┴ ┴┴  ┴  └─┘┴└─└─┘
    '''
    
    def data_plot(self,plot_func,**kwargs):
            
        self.logger.info('Running data_plot')
    
        # Try to load plot data from file
        self.loaded, plot_settings = self.read_plot_data()
        
        if not self.loaded:

            # Run output handler
            outputs_summary = OutputSummary(
                settings = self.settings,
                logger = self.logger
            )
            
            # Loop through output folder
            plot_settings = []
            for indx,output_folder in enumerate(outputs_summary.output_folders):
                
                # Get metadata collection for this 
                metadata_collection,outputs = outputs_summary.collect_folder_metadata(
                    indx,
                    output_folder,
                    **kwargs
                )

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
                            print_json({k:plot_sett[k] for k in ['x','y','marker_size'] if k in plot_sett})

                            # Add data
                            plot_settings.append(deepcopy(plot_sett))
                    

                    # If plot is by experiment
                    # plot all element from data collection
                    # for every output folder
                    if self.settings.get('by_experiment',False):
                        # Export all plot settings
                        # Create output dirpath and filename
                        dirpath,filename = self.create_plot_filename(
                            plot_setting = plot_sett,
                            name = self.settings.get('figure_title','NONAME')
                        )
                        merged_plot_settings = self.merge_plot_settings(
                            plot_sett,
                            apply_zorder = True
                        )
                        # Make outputs directories (if necessary)
                        makedir(dirpath)
                        # Write figure data
                        write_figure_data(
                            plot_data = merged_plot_settings,
                            filepath = os.path.join(dirpath,filename),
                            keys = PLOT_COORDINATES_AND_CORE_FEATURES+['outputs'],
                            figure_settings = self.settings
                        )
                        self.logger.success(f"Figure data exported to {dirpath}")
                        # Merge all settings into one
                        # Plot
                        plot_func(
                            merged_plot_settings,
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
                merged_plot_settings = self.merge_plot_settings(
                    plot_settings,
                    apply_zorder = True
                )
                # Make outputs directories (if necessary)
                makedir(dirpath)
                # Write figure data
                write_figure_data(
                    plot_data = merged_plot_settings,
                    filepath = os.path.join(dirpath,filename),
                    keys = PLOT_COORDINATES_AND_CORE_FEATURES+['outputs'],
                    figure_settings = self.settings
                )
                self.logger.success(f"Figure data across experiments exported to {dirpath}")
                # Plot
                plot_func(
                    plot_settings = merged_plot_settings,
                    name = self.settings.get('title','NONAME'),
                    dirpath = dirpath,
                    filename = filename
                )
        else:
            # Merge plot settings
            # merged_plot_settings = self.merge_plot_settings(
            #     plot_settings,
            #     apply_zorder = False
            # )
            # if 'zorder' in plot_settings[0]:
            #     # get value to order by
            #     values = np.array(plot_settings[0]['zorder'])
            #     if len(values.shape) > 1:
            #         # Number of elements to sort
            #         ndims = values.shape[1]
            #         # Use lexsort to argsort along each axis successively
            #         sorted_indices = np.lexsort([values[:,i].ravel() for i in range(ndims)])
            #     else:
            #         sorted_indices = np.argsort(values,axis=0)
            #     # Update merged settings
            #     # add 1.0 to avoid zorder = 0
            #     plot_settings[0]['zorder'] = np.zeros(len(sorted_indices))
            #     for i,j in enumerate(sorted_indices):
            #         plot_settings[0]['zorder'][i] = len(sorted_indices) - j
            #     print(values)
            #     print(plot_settings[0]['zorder'])
            #     sys.exit()
            # Create output dirpath and filename
            dirpath,filename = self.create_plot_filename(
                plot_setting = plot_settings,
                name = self.settings.get('figure_title','NONAME')
            )
            # Plot
            plot_func(
                plot_settings = plot_settings,
                name = self.settings.get('title','NONAME'),
                dirpath = dirpath,
                filename = filename
            )