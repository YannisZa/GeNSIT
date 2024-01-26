import os
os.environ['USE_PYGEOS'] = '0'
import h5py
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt

from copy import deepcopy
from tqdm.auto import tqdm
from datetime import datetime
from functools import partial, update_wrapper
from joblib import Parallel, delayed
from shapely.geometry import Point,LineString, Polygon, MultiPoint

from multiresticodm.utils.misc_utils import *
from multiresticodm.utils.math_utils import *
from multiresticodm.static.global_variables import SIM_TYPE_CONSTRAINTS

def mean_euclidean_distance_from_centroid(multipoly1,multipoly2):
    distances = []
    center = multipoly1.centroid
    for poly in multipoly2.geoms:
        for poi in poly.exterior.coords[:-1]:
            distances.append(center.distance(Point(poi)))
    return np.mean(distances)

def compute_individual_facility_shortest_path(index,
                                              G,
                                              origin_locs,
                                              destination_locs,
                                              num_cores,
                                              n_batches,
                                              store:bool=False):
    
    # Compute all origin-destination combinations
    od_pairs = np.asarray(list(itertools.product(origin_locs,destination_locs)))

    # Compute shortest paths
    print('', end="\r",flush=True)
    sps = np.array([float(nx.bidirectional_dijkstra(G,od[0],od[1],weight='length')[0]) for od in tqdm(od_pairs,desc=f"Processing batch {index}",leave=False,position=(index%num_cores+1))])
    
    if store:
        with open(f"./data/raw/cambridge_commuter/progress/shortest_path_batch_{index}_outof_{n_batches}.npy", 'wb') as f:
            np.save(f, np.hstack((od_pairs,np.array(sps)[:,np.newaxis])))

    return np.hstack((od_pairs,np.array(sps)[:,np.newaxis]))

def compute_individual_facility_euclidean_distance(index,
                                                origin_locs,
                                                destination_locs,
                                                _print):
        
    # Compute all origin-destination combinations
    od_pairs = np.asarray(list(itertools.product(list(origin_locs.keys()),list(destination_locs.keys()))))

    # Compute shortest paths
    if _print:
        print('', end="\r",flush=True)
        euclidean_dis = np.array([float(origin_locs[od[0]].distance(destination_locs[od[1]])) for od in tqdm(od_pairs,desc=f"Processing batch {index}",leave=False)])
    else:
        euclidean_dis = np.array([float(origin_locs[od[0]].distance(destination_locs[od[1]])) for od in od_pairs])
    
    return np.hstack((od_pairs,np.array(euclidean_dis)[:,np.newaxis]))

def normalise_data(data,ntype,nfactor):
    if 'max' in ntype.lower():
        return data / data.values.max()
    elif 'sum' in ntype.lower():
        return data / data.values.sum()
    elif 'factor' in ntype.lower():
        return data / nfactor
    elif 'unit' in ntype.lower():
        return (data - data.values.min()) / data.values.max()
    else:
        return data
    
def angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + np.degrees(np.arctan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + np.degrees(np.arctan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

def compute_centroid_boundary_enclosed_angle(regions):
    points = MultiPoint(regions.centroid).convex_hull.boundary.coords[:-1]
    angles = dict(zip(regions.geometry_id.values,np.ones(regions.shape[0])))
    for i in range(len(points)):
        p1 = points[i]
        ref = points[i-1]
        p2 = points[i-2]
        # Find angle
        angle = angle_between(p1,ref,p2)
        # Find relevant geometry
        gid = regions.loc[regions.centroid.geom_almost_equals(Point(ref)),'geometry_id'].values[0]
        # Append angle to dict
        angles[gid] = angle/360
    return angles

def compute_sampled_facility_boundary_enclosed_angle(
    origins,
    boundary_destinations,
    statistic:str='mean'
):

    results = []
    for _,bd in boundary_destinations.iterrows():
        # Extend origin locations to include boundary destination
        origins_extended = gpd.GeoDataFrame( pd.concat( [origins,gpd.GeoDataFrame(bd).T], ignore_index=True) )
        # Compute convex hull
        convex_hull = MultiPoint(origins_extended.geometry.values).convex_hull.boundary.coords
        for index,chp in enumerate(convex_hull):
            if bd.geometry.equals_exact(Point(chp),tolerance=0.0):
                results.append({
                    "geometry_id":bd.geometry_id,
                    "destination_id": bd.facility_id,
                    "destination_geometry": bd.geometry,
                    "closest_origin1_id": origins[origins.geometry.geom_almost_equals(Point(convex_hull[index-1]))].facility_id.values,
                    "closest_origin1_geometry": LineString([bd.geometry,Point(convex_hull[index-1])]),
                    "closest_origin2_id":origins[origins.geometry.geom_almost_equals(Point(convex_hull[index+1]))].facility_id.values,
                    "closest_origin2_geometry": LineString([bd.geometry,Point(convex_hull[index+1])]),
                })
                break
    # Convert to geodataframe
    results = gpd.GeoDataFrame.from_dict(results)
    # Compute angles enclosed between boundary destination and nearby origins
    for index,row in results.iterrows():
#         print(row['closest_origin1_geometry'].coords[1])
#         print(row['destination_geometry'].coords[0])
#         print(row['closest_origin2_geometry'].coords[1])
#         print('\n')
        results.loc[results.index[index],'angle_covered'] = angle_between(
                row['closest_origin1_geometry'].coords[1],
                row['destination_geometry'].coords[0],
                row['closest_origin2_geometry'].coords[1]
        )/360
    
    # Compute summary statistics of enclosed angles by geometry id
    return results[['geometry_id','angle_covered']].groupby('geometry_id').agg(statistic).to_dict('dict')['angle_covered']
    


def compute_sampled_facility_neighbourhood_enclosed_angle(
    origin_locations,
    destination_locations,
    origin_boundary_geometries,
    destination_boundary_geometries,
    statistic:str='mean',
    n_workers:int=1
):  
    boundary_angles = {}
    for geo_index,geo in tqdm(destination_boundary_geometries.iterrows(),total=destination_boundary_geometries.shape[0]):
        # Get destination ids
        destinations = destination_locations[destination_locations.geometry_id == geo.geometry_id].centroid.values
    
        # Get boundary neighbourhood
        nearby_origin_boundary_geometries = origin_boundary_geometries[
                                                (~origin_boundary_geometries.geometry.disjoint(geo.geometry)) & \
                                                (origin_boundary_geometries.geometry_id != geo.geometry_id)
                                            ]
    
        # Find neighbour geographies
        neighbour_ids = nearby_origin_boundary_geometries.geometry_id.values
        
        # Neighbour data
        neighbour_data = {}
        # Add destinations to data
        neighbour_data['dest'] = destinations
        # Only two boundary geographies should be nearby any given geography
        # If not merge nearby geometries adjacent to each other
        if len(neighbour_ids) > 2:
            # print('Multiple neighbours')
            # Compute distances to neighbours
            neighbours = deepcopy(nearby_origin_boundary_geometries.loc[(nearby_origin_boundary_geometries.geometry_id.isin(neighbour_ids))])
            neighbours.loc[:,'distance_to_neighbourhood'] = neighbours.distance(geo.geometry)
            # Group neighbouring polygons by distance and get the two closest
            neighbouring_polygons = neighbours.sort_values('distance_to_neighbourhood',ascending=True).head(2).geometry.values
            # There should be maximum two neighbourhoods of polygons
            assert len(neighbouring_polygons) == 2
            # print('Extracted two closest neighbours')
            # Extract origin locations in each neighbourhood
            for index,neighbour in enumerate(neighbouring_polygons):
                # Gather geometry ids 
                gids = nearby_origin_boundary_geometries.loc[nearby_origin_boundary_geometries.geometry.intersects(neighbour)].geometry_id.values
                neighbour_data[f"origins{str(index)}"] = origin_locations.loc[origin_locations.geometry_id.isin(gids)].centroid.values
        else:
            # print('Two neighbours')
            # For each neighbour
            for index,neighbour in enumerate(neighbour_ids):
                # Get all origin facility points
                neighbour_data[f"origins{str(index)}"] = origin_locations.loc[origin_locations.geometry_id == neighbour].centroid.values
    
        # Create all possible combinations of origin-destination-origin tuples
        facility_tuples = [neighbour_data['origins0'],neighbour_data['dest'],neighbour_data['origins1']]
        candidate_angles = list(itertools.product(*facility_tuples))
        
        # Compute angle between in parallel
        data = np.asarray(Parallel(n_jobs=n_workers,backend='multiprocessing')(delayed(angle_between)([p1.x,p1.y],[p2.x,p2.y],[p3.x,p3.y]) for p1,p2,p3 in candidate_angles))
        # Store statistic over angles
        boundary_angles[geo.geometry_id] = convert_string_to_numpy_function(statistic)(data)/360
    
    return boundary_angles

def enumerated_product(*args):
    yield from zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args))

def apply_edge_corrections(geodata,edge_correction_df,origin_id_name,destination_id_name,correction_methods:list=[],axes:list=[]):
    assert len(correction_methods) == len(axes)
    geodata = geodata.copy(deep=True)
    # Apply edge corrections along axes
    for i in range(len(axes)):
        geodata = apply_edge_correction(geodata,edge_correction_df,origin_id_name,destination_id_name,correction_methods[i],axes[i])
    return geodata
        

def apply_edge_correction(geodata,edge_correction_df,origin_id_name,destination_id_name,correction_method:str='',axis:int=1):
    data = geodata.copy(deep=True)
    if correction_method.endswith('boundary_only'):
        # Extract boundary areas
        edge_correction_df = deepcopy(edge_correction_df[edge_correction_df.geometry.intersects(edge_correction_df.unary_union.exterior)])
        # Crop boundary only suffix
        correction_method = correction_method.split('_boundary_only')[0]
    if 'inverse_ripleys_k' in correction_method:
        inverse = True
        correction_method = correction_method.replace("inverse_","")
    else:
        inverse = False

    # Filter corrections geometries depending on axis they are beying applied to
    if axis == 1:
        edge_correction_df = edge_correction_df[edge_correction_df.geometry_type == destination_id_name]
    elif axis == 0:
        edge_correction_df = edge_correction_df[edge_correction_df.geometry_type == origin_id_name]
    # Correction method must exist
    try:
        assert correction_method in edge_correction_df.columns.values
    except:
        print(correction_method)
    # Ensure correction method is supplied
    if correction_method != '':
        for _,row in edge_correction_df.iterrows():
            if axis == 1:
                if inverse:
                    data.loc[:,row['geometry_id']] /= row[correction_method]#.values[0]
                else:
                    data.loc[:,row['geometry_id']] *= row[correction_method]#.values[0]
            elif axis == 0:
                if inverse:
                    data.loc[row['geometry_id'],:] /= row[correction_method]#.values#[0]
                else:
                    data.loc[row['geometry_id'],:] *= row[correction_method]#.values#[0]
    return data


def save_cost_matrices(
    cost_matrices:list=None,
    cost_matrix_names:list=None,
    norm:str='',
    mode:str='',
    correction_method:str='',
    multiresticodm_format:bool=None,
    dataset:str='',
    geometry_name:str=''
):
    assert len(cost_matrices) == len(cost_matrix_names)

    main_directory = os.path.join(f'../data/inputs/{dataset}/cost_matrices')
    neural_net_directory = os.path.join(f'../../NeuralABM/data/HarrisWilson/Cambridge_data/')
    if mode != '':
        mode += '_'
    
    # If there is no correction applied just store normalised matrix
    if correction_method == '':
        for index,cm in enumerate(cost_matrices):
            if multiresticodm_format is None or multiresticodm_format:
                np.savetxt(os.path.join(main_directory,f'{mode}{cost_matrix_names[index]}_cost_matrix{norm}.txt'),cm.values)
            if multiresticodm_format is None or not multiresticodm_format:
                np.exp(-cm).reset_index().to_csv(os.path.join(neural_net_directory,f'{geometry_name}_{mode}{cost_matrix_names[index]}_exp_cost_matrix{norm}.csv'),index=False)
    # Otherwise store edge-corrected normalised matrix
    else:
        for index,cm in enumerate(cost_matrices):
            if multiresticodm_format is None or multiresticodm_format:
                np.savetxt(os.path.join(main_directory,f'{mode}{cost_matrix_names[index]}_{correction_method}_edge_corrected_cost_matrix{norm}.txt'),cm.values)
            if multiresticodm_format is None or not multiresticodm_format:
                np.exp(-cm).reset_index().to_csv(os.path.join(neural_net_directory,f'{geometry_name}_{mode}{cost_matrix_names[index]}_{correction_method}_edge_corrected_exp_cost_matrix{norm}.csv'),index=False)
        
        
def load_cost_matrices(
    cost_matrix_names:list = None,
    norm:str='',
    mode:str='',
    correction_method:str='',
    dataset:str=''
):
    main_directory = os.path.join(f'../data/inputs/{dataset}/cost_matrices')
    neural_net_directory = os.path.join(f'../../NeuralABM/data/HarrisWilson/Cambridge_data/')

    if mode != '':
        mode += '_'
    
    cost_matrices = []
    for index,cm in enumerate(cost_matrix_names):
        cm = np.loadtxt(os.path.join(main_directory,f'{mode}{cost_matrix_names[index]}_cost_matrix{norm}.txt'))
    # Append to results
    cost_matrices.append(cm)
        
    return dict(zip(cost_matrix_names,cost_matrices))

        
def plot_destination_attraction_by_cost_margins(
    destination_attraction,
    cms:list=None,
    cm_names:list=None,
    fig_size:list=None,
    fig_title:str=''
):
    assert len(cms) == len(cm_names)
    
    fig,axs = plt.subplots(len(cms),1,figsize=fig_size)
    
    if fig_title != '':
        fig.suptitle(fig_title)
    
    if len(cms) <= 1:
        im = axs.imshow(destination_attraction[sorted(range(cms[0].shape[1]), key=lambda k: cms[0].values.sum(axis=0)[k])][np.newaxis,:], cmap=plt.cm.coolwarm, interpolation='nearest')
        axs.set_title(f"{cm_names[0].replace('_',' ').capitalize()} cost matrix",fontsize=16)
        axs.set_xlabel('Destinations',fontsize=16)
        axs.set_xticks(range(cms[0].shape[1]))
        axs.set_xticklabels(sorted(range(cms[0].shape[1]), key=lambda k: cms[0].values.sum(axis=0)[k]))
        axs.set_yticklabels([])
    
    else:
        for i in range(len(cms)):
            im = axs[i].imshow(destination_attraction[sorted(range(cms[i].shape[1]), key=lambda k: cms[i].values.sum(axis=0)[k])][np.newaxis,:], cmap=plt.cm.coolwarm, interpolation='nearest')
            axs[i].set_title(f"{cm_names[i].replace('_',' ').capitalize()} cost matrix",fontsize=16)
            axs[i].set_xlabel('Destinations',fontsize=16)
            axs[i].set_xticks(range(cms[i].shape[1]))
            axs[i].set_xticklabels(sorted(range(cms[i].shape[1]), key=lambda k: cms[i].values.sum(axis=0)[k]))
            axs[i].set_yticklabels([])
    
    plt.colorbar(im,location='bottom',pad=0.6)
    plt.show()

def read_distance_matrix_api_data(
    origin_geo_ids:list,
    destination_geo_ids:list,
    mode_of_transport:str,
    resolution:str,
    date_ingested:str,
    origin_geo_name:str,
    destination_geo_name:str,
    origin_agg_statistic:str='sum',
    destination_agg_statistic:str='sum',
    origin_geo_map:dict=None,
    destination_geo_map:dict=None,
):
    if date_ingested != "":
        date_ingested += "_"
    temporal_matrix = pd.read_csv(f'../data/raw/cambridge_commuter/google/{origin_geo_name}_to_{destination_geo_name}_{resolution}_{mode_of_transport}_metrics_{date_ingested}times.csv',index_col=0)
    distance_matrix = pd.read_csv(f'../data/raw/cambridge_commuter/google/{origin_geo_name}_to_{destination_geo_name}_{resolution}_{mode_of_transport}_metrics_{date_ingested}distances.csv',index_col=0)

    # Change row and column names
    temporal_matrix.columns = destination_geo_ids
    temporal_matrix.index = origin_geo_ids
    temporal_matrix.index.name = 'row: origin/col: dest/entries: meters'

    distance_matrix.columns = destination_geo_ids
    distance_matrix.index = origin_geo_ids
    distance_matrix.index.name = 'row: origin/col: dest/entries: meters'
    # Get index name
    index_name = temporal_matrix.index.name
    if destination_geo_map is not None:
        # Map column indices
        temporal_matrix = temporal_matrix.rename(destination_geo_map,axis=1)
        temporal_matrix = temporal_matrix.groupby(temporal_matrix.columns, axis=1).aggregate(destination_agg_statistic).reset_index().set_index(index_name)
    if origin_geo_map is not None:
        # Map row indices 
        if temporal_matrix.index.name == index_name:
            temporal_matrix = temporal_matrix.reset_index()
            
        temporal_matrix[index_name] = temporal_matrix[index_name].map(origin_geo_map)
        temporal_matrix = temporal_matrix.set_index(index_name)

        temporal_matrix = temporal_matrix.groupby(level=0).aggregate(origin_agg_statistic)
    # Sort rows and columns
    temporal_matrix = temporal_matrix.sort_index()
    temporal_matrix = temporal_matrix[sorted(temporal_matrix.columns.values)]
    
    # Get index name
    index_name = distance_matrix.index.name
    if destination_geo_map is not None:
        # Map column indices
        distance_matrix = distance_matrix.rename(destination_geo_map,axis=1)
        distance_matrix = distance_matrix.groupby(distance_matrix.columns, axis=1).aggregate(destination_agg_statistic)
    if origin_geo_map is not None:
        # Map row indices
        if distance_matrix.index.name == index_name:
            distance_matrix = distance_matrix.reset_index()

        distance_matrix[index_name] = distance_matrix[index_name].map(origin_geo_map)
        distance_matrix = distance_matrix.set_index(index_name)

        distance_matrix = distance_matrix.groupby(level=0).aggregate(origin_agg_statistic).reset_index().set_index(index_name)
    # Sort rows and columns
    distance_matrix = distance_matrix.sort_index()
    distance_matrix = distance_matrix[sorted(distance_matrix.columns.values)]
        

    return temporal_matrix,distance_matrix

def shorten_filename(filenames):
    if len(filenames) == 1:
        return filenames[0]
    
    first_filename = filenames[0].split('_')
    for fi,f in enumerate(filenames):
        words = f.split('_')
        for wi,w in enumerate(words):
            if w != first_filename[wi]:
                first_filename[wi] += '_' + w
    first_filename = '_'.join(first_filename)
    return first_filename

def prepare_cost_matrices(
    cost_matrices,
    cost_matrix_names,
    corrections:gpd.GeoDataFrame,
    normalisation_types:list,
    edge_correction_types:list,
    edge_correction_axes:list,
    origin_id_name:str,
    destination_id_name:str,
    transport_mode:str='',
    data:str='cambridge_work_commuter_lsoas',
    geo_name:str='lsoas'
):
    # Make sure inputs have correct dimensions
    assert len(cost_matrices) == len(cost_matrix_names)
    assert len(edge_correction_types) == len(edge_correction_axes)
    
    # Get all combinations of normalisation and edge correction types
    total_norm_correction_type_pairs = len(list(itertools.product(normalisation_types, edge_correction_types)))
    
    for i,cm in tqdm(enumerate(cost_matrices),total=len(cost_matrices)):
        
        for idx, norm_correction in tqdm(enumerated_product(normalisation_types, edge_correction_types),total=total_norm_correction_type_pairs,leave=False):
            # Read normalisation and correction types and factors
            norm_type = norm_correction[0]
            norm_factor = 1
            if 'factor' in norm_type:
                norm_factor = float(norm_type.split('_factor_')[1].split('_normalised')[0])
            correction_methods = norm_correction[1]
            
            # Apply edge correction
            cm_edge_corrected = apply_edge_corrections(
                cm,
                corrections,
                origin_id_name,
                destination_id_name,
                correction_methods,
                edge_correction_axes[idx[1]],
            )
            # Normalise
            cm_edge_corrected_normalised = normalise_data(
                cm_edge_corrected,
                norm_type,
                norm_factor
            )
            
            # Export to file
            save_cost_matrices(
                cost_matrices = [cm_edge_corrected_normalised],
                cost_matrix_names = [cost_matrix_names[i]],
                norm = norm_type,
                mode = transport_mode,
                correction_method = shorten_filename(correction_methods),
                multiresticodm_format = True,
                dataset = data,
                geometry_name = geo_name
            )

def ripleys_k(
    location,
    aoi_ids,
    G,
    network_vertices,
    region_of_interest,
    radius,
    neighbourhood_method,
    conditional_probabilty_method
):    
    if neighbourhood_method == 'graph_shortest_path':
        # Deepcopy graph
        G_copy = G.copy()
        # Remove all network nodes that are not in the vincinity of the point of interest
        removed_network_nodes = network_vertices[~network_vertices.within(location.geometry.buffer(2*radius))].id.values

        # Remove nodes outside of buffer zone
        G_copy.remove_nodes_from(removed_network_nodes)

        # Find subgraph within radius and count number of nodes
        neighbourhood_graph = nx.ego_graph(
            G=G_copy,
            n=location.facility_id,
            radius=radius,
            distance='weight',
            undirected=False
        )
        nearby_points = [n for n in neighbourhood_graph.nodes() if  n.startswith(location.main_activity)]
        num_nearby_points_within_aoi = len(set(aoi_ids).intersection(set(nearby_points)))
                            
        
    elif neighbourhood_method == 'euclidean':
        # Location circle buffer
        location_buffer = location.geometry.buffer(radius)
        # Find all nodes in vincinity
        nearby_points = network_vertices[
                                            (network_vertices.within(location_buffer)) & \
                                            (network_vertices.id.str.startswith(location.main_activity))
                                        ].id.values
        
        # Extract origin nearby nodes and count them
        num_nearby_points_within_aoi = len(set(aoi_ids).intersection(set(nearby_points)))

    else:
        raise ValueError(f'Neighbourhood method {neighbourhood_method} not found!')

    if conditional_probabilty_method == 'area%':
        # Total circle area
        total_area = location_buffer.area
        # Total circle area that intesects region of interest
        observed_area = location_buffer.intersection(region_of_interest).area
        conditional_probability_of_point_within_radius = (observed_area/total_area)

    elif conditional_probabilty_method == 'points%':
        # Compute percentage of points lying within boundary
        # Denominator is number of points lying within radius buffer (including outside boundary)
        conditional_probability_of_point_within_radius = num_nearby_points_within_aoi/len(nearby_points)

    else:
        raise ValueError(f'Conditional probability of point within radius "{conditional_probabilty_method}" not found!')
    
    return pd.Series({"facility_id":location.facility_id, 
                      "ripleys_k":num_nearby_points_within_aoi/conditional_probability_of_point_within_radius},
                    index=['facility_id','ripleys_k'])

def apply_ripleys_k_edge_correction(
    index,
    aoi_locations,
    network_vertices,
    geographies,
    G,
    location_type,
    radius:float=1.0,
    neighbourhood_method:str='euclidean',
    conditional_probabilty_method:str='area%'
):
    
    # Initialise tqdm for pandas
    tqdm.pandas(leave=True,position=index)
    
    boundary_corrections = {}
    for _,geo in geographies.iterrows():
        boundary_corrections[geo.geometry_id] = {"origin":1,"destination":1}
        
    # Compute area of interest
    area_of_interest = Polygon(geographies.unary_union.exterior).buffer(radius)
    
    # Get all network vertices inside geography boundary
    network_vertices = network_vertices[network_vertices.geometry.within(area_of_interest)]
    # Copy graph
    if G is not None:
        G_copy = G.copy()
    else:
        G_copy = None
    
    if neighbourhood_method == 'graph_shortest_path':
        try:
            assert G_copy is not None
        except:
            raise ValueError('Empty graph provided...')

        # Remove all nodes and edges outside geography boundary
        removed_nodes = network_vertices[~network_vertices.geometry.within(area_of_interest)].id.values
        G_copy.remove_nodes_from(removed_nodes)
    
    # Create partial function(s)
    ripleys_k_partial = partial(ripleys_k,
                                        G=G_copy,
                                        network_vertices=network_vertices,
                                        region_of_interest=area_of_interest,
                                        radius=radius,
                                        neighbourhood_method=neighbourhood_method,
                                        conditional_probabilty_method=conditional_probabilty_method
                                )

    # Apply ripley's k function to data
    aoi_ids = aoi_locations.facility_id.values

    # Computre ripleys k
    ripleys_k_adjusted = aoi_locations[['geometry','facility_id','main_activity']].progress_apply(
        ripleys_k_partial,
        aoi_ids=aoi_ids,
            axis=1
        )
    
    aoi_locations = pd.merge(aoi_locations,
                            ripleys_k_adjusted,
                            on='facility_id',
                            how='left')
    # Compute Ripley's for every geography
    for name, group in tqdm(aoi_locations.groupby('geometry_id')):
        boundary_corrections[name][location_type] = ((group.region_geometry.head(1).area/group.shape[0]) * np.mean(group.ripleys_k)).values[0]

    return boundary_corrections, aoi_locations

def fix(df,a_map):
    index_name = df.index.name
    df = df.rename(columns=a_map)
    df = df.drop(columns=["work_1065800744","work_1829418524","work_1991326470","work_327229438"])
    df = df.groupby(df.columns, axis=1).aggregate('sum').reset_index().set_index(index_name)
    df = df.sort_index()
    df = df[sorted(df.columns.values)]
    return df

def printcols(df):
    for c in sorted(df.columns.values):
        print(c)

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def read_h5_data(file,keys:list=['loss']):
    dataset = h5py.File(os.path.join(file,"data.h5"), "r").get('HarrisWilson')
    output_data = {}
    for k in keys:
        output_data[k] = dataset.get(k)[:]
        if k == 'pred_dest_sizes':
            output_data[k] = output_data[k].T
    return output_data

def prepare_for_export(dataset,take_mean:bool=True):
    thetas,signs,log_destination_attractions = None,None,None
    for k in tqdm(list(dataset.keys())): #[list(dataset.keys())[0]]:
        # Merge alpha,beta to theta
        theta = np.vstack([dataset[k]['alpha'],dataset[k]['beta']]).T.astype('float32')
        # Normalise predicted destination sizes to one and take logs
        log_destination_attraction= np.log(
            dataset[k]['pred_dest_sizes'],#/np.sum(dataset[k]['pred_dest_sizes'],axis=1)[:,np.newaxis],
            dtype='float32'
        )
        # Create sign array
        sign = np.ones((theta.shape[0],1),dtype='int8')
        # Append all sample arrays
        if thetas is None:
            thetas = theta[np.newaxis,:,:]
        else:
            thetas = np.append(
                thetas,
                theta[np.newaxis,:,:],
                axis=0
            )
        if signs is None:
            signs = sign[np.newaxis,:,:]
        else:
            signs = np.append(
                signs,
                sign[np.newaxis,:,:],
                axis=0
            )
        if log_destination_attractions is None:
            log_destination_attractions = log_destination_attraction[np.newaxis,:,:]
        else:
            log_destination_attractions = np.append(
                log_destination_attractions,
                log_destination_attraction[np.newaxis,:,:],
                axis=0
            )
    if take_mean:
        return np.mean(signs,axis=1),np.mean(thetas,axis=1),np.mean(log_destination_attractions,axis=1)
    else:
        return np.vstack(signs),np.vstack(thetas),np.vstack(log_destination_attractions)

def prepare_config_for_export(barebone_config,metadata_config):
    new_config = dict(zip(metadata_config.keys(),{}))
    for sim_type in metadata_config.keys():
        new_config[sim_type] = deepcopy(barebone_config)
        sim_metadata_config = metadata_config[sim_type]
        sigma = sim_metadata_config['HarrisWilson']['Training']['true_parameters'].get('sigma',-1)
        if sigma > 0.05:
            noise_regime = 'high'
            gamma = int(np.round(2/sigma**2,0))
        elif sigma <= 0.05 and sigma >= 0:
            noise_regime = 'low'
            gamma = int(np.round(2/sigma**2,0))
        else:
            noise_regime = 'learned'
            gamma = -1
        updated_config = {
                'N':sim_metadata_config['num_epochs']*(sim_metadata_config['seed']+1),
                'gamma':gamma,
                'name':sim_type,
                'axes': [[1]] if sim_type == 'ProductionConstrained' else [[0,1]],
                'noise_regime':str(noise_regime),
                'experiment_title':SIM_TYPE_CONSTRAINTS[sim_type],
                'datetime':datetime.strptime(
                        sim_metadata_config['output_dir'].split('HarrisWilson/')[1].split('_Cambridge_dataset')[0], 
                        '%y%m%d-%H%M%S'
                    )
                    .strftime("%d_%m_%Y_%H_%M_%S")
        }
        for k,v in updated_config.items():
            deep_update(
                new_config[sim_type],
                k,
                v
            )
    return new_config

def export_competitive_method_outputs(
        metadata,
        signs,
        thetas,
        log_destination_attractions
):
    output_folder = metadata['experiment_id']+'_'+\
        metadata['noise_regime'].capitalize()+'Noise_'+\
        metadata['outputs']['experiment_title']+'_'+\
        metadata['datetime']
    output_path = os.path.join(
        f".{metadata['outputs']['directory']}",
        metadata['inputs']['dataset'],
        output_folder
    )
    print('Exporting to',output_path)
    makedir(output_path)
    makedir(os.path.join(output_path,'samples'))
    makedir(os.path.join(output_path,'figures'))

    write_npy(signs,os.path.join(output_path,'samples','signs_samples.npy'))
    write_npy(thetas,os.path.join(output_path,'samples','theta_samples.npy'))
    write_npy(log_destination_attractions,os.path.join(output_path,'samples','log_destination_attraction_samples.npy'))
    write_json(metadata,os.path.join(output_path,output_folder+'_metadata.json'))