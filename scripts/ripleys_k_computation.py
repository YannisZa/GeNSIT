import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
# Use pygeos in geopandas
os.environ['USE_PYGEOS'] = '0'

import geopandas as gpd

from multiresticodm.utils.misc_utils import *
from dask.distributed import Client, LocalCluster
from multiresticodm.utils.notebook_functions import *


if __name__ == '__main__':
    # Expertiment id
    geometry_name = 'lsoas_to_msoas'
    origin_geometry_name = 'lsoa'
    destination_geometry_name = 'msoa'
    # 'msoa'
    # 'lsoa'
    # 'oa'
    dataset = f'cambridge_work_commuter_{geometry_name}'
    table_filename = 'lower_super_output_areas_to_medium_super_output_areas_work_flows_cambridge_2011'
    # 'lower_super_output_areas_work_flows_cambridge_2011'
    # 'middle_super_output_areas_work_flows_cambridge_2011'
    # 'lower_super_output_areas_work_flows_cambridge_2011'
    # 'output_areas_work_flows_cambridge_2011'
    # 'lower_super_output_areas_to_medium_super_output_areas_work_flows_cambridge_2011'

    # Define directory
    table_path = f'./data/raw/cambridge_commuter/{table_filename}.csv'
    geometries_path = f'./data/raw/cambridge_commuter/cambridge_{geometry_name}.geojson'

    # Read table
    table = pd.read_csv(table_path,index_col = 0)
    # Store first column
    origin_geometry_ids = sorted(table.index.values)
    destination_geometry_ids = sorted(table.columns.values)
    geometry_ids = np.append(origin_geometry_ids,destination_geometry_ids)

    # Read geometries
    geometries = gpd.read_file(geometries_path)
    # Reproject
    geometries = geometries.set_crs('epsg:27700',allow_override = True)

    # Number of facilities per geographical unit
    n_facilities = None #20
    random_seed = 1234 #None
    facility_sample_name = f'sample_{n_facilities}' if (n_facilities is not None) else 'all'
    facility_name = 'facilities'
    import_sample = True

    home_locs = gpd.read_file(f'./data/inputs/{dataset}/{facility_sample_name}_home_facilities_seed_{random_seed}.geojson')
    home_ids = np.unique(home_locs.facility_id.values)
    work_locs = gpd.read_file(f'./data/inputs/{dataset}/{facility_sample_name}_work_facilities_seed_{random_seed}.geojson')
    work_ids = np.unique(work_locs.facility_id.values)

    # Read graph pickle from file
    # graph = nx.read_gpickle(os.path.join(f'./data/raw/cambridge_commuter/{geometry_name}_graph.gpickle'))

    network_and_origin_destination_vertices = gpd.read_file(f'./data/raw/cambridge_commuter/{geometry_name}_graph_nodes_network_and_origin_destination.geojson',index_col = 0)

    # Read edge corrections
    edge_corrections = gpd.read_file(f'./data/inputs/{dataset}/edge_corrections.geojson')
    edge_corrections.crs = geometries.crs

    # Parallelisation parameters
    num_cores = 4
    buffer_radius = 1000
    neighbourhood_method_name = 'euclidean'
    conditional_probabilty_method_names = ['area%','points%']
    #'area%' 'points%', ''

    # Initialise cluster
    with LocalCluster(
        n_workers = num_cores,
        processes = True,
        threads_per_worker = 2,
        memory_limit='50GB',
        dashboard_address=':49053',
        # worker_dashboard_address=':19971998',
        # ip='tcp://localhost:19971999'
    ) as my_cluster, Client(my_cluster) as my_client:

        # Add region geometry to origins
        orig_locs = pd.merge(home_locs[home_locs.facility_id.isin(home_ids)],
                            geometries[['geometry_id','geometry']].rename(columns={"geometry":"region_geometry"}),
                            on='geometry_id',
                            how='left')[['geometry','geometry_id','region_geometry','facility_id','main_activity']]
        # Add region geometry to destinations
        dest_locs = pd.merge(work_locs[work_locs.facility_id.isin(work_ids)],
                            geometries[['geometry_id','geometry']].rename(columns={"geometry":"region_geometry"}),
                            on='geometry_id',
                            how='left')[['geometry','geometry_id','region_geometry','facility_id','main_activity']]
        dest_locs.loc[dest_locs.facility_id.str.startswith('E'),'main_activity'] = 'work'
        
        # Scatter big data
        big_data_futures = my_client.scatter([
                                    orig_locs,
                                    dest_locs
                            ])
        [network_and_origin_destination_vertices_future] = my_client.scatter(
            [network_and_origin_destination_vertices],
            broadcast = True
        )
        # Compile params
        params = []
        for cpmn in ['area%','points%']: 
            for loc_type in ['origin','destination']: 
                params.append(
                    {
                        "geographies":geometries,
                        "G":None,
                        "location_type":loc_type,
                        "radius":buffer_radius,
                        "neighbourhood_method":neighbourhood_method_name,
                        "conditional_probabilty_method":cpmn
                    }
                )
        
        # Submit jobs
        ripleys_k_origin_area_percentage_edge_correction_job, \
        ripleys_k_destination_area_percentage_edge_correction_job,\
        ripleys_k_origin_points_percentage_edge_correction_job, \
        ripleys_k_destination_points_percentage_edge_correction_job = [
                    my_client.submit(
                            apply_ripleys_k_edge_correction, 
                            *[i,big_data_futures[i % 2],network_and_origin_destination_vertices_future],
                            **param
                    ) for i,param in enumerate(params)
        ]
        # Gather results
        ripleys_k_origin_area_percentage_edge_correction, \
        ripleys_k_destination_area_percentage_edge_correction,\
        ripleys_k_origin_points_percentage_edge_correction, \
        ripleys_k_destination_points_percentage_edge_correction  = \
        ripleys_k_origin_area_percentage_edge_correction_job.result()[0], \
        ripleys_k_destination_area_percentage_edge_correction_job.result()[0],\
        ripleys_k_origin_points_percentage_edge_correction_job.result()[0], \
        ripleys_k_destination_points_percentage_edge_correction_job.result()[0]
        
        # Pass it to geometries df
        edge_corrections.loc[:,f'{facility_sample_name}_{facility_name}_ripleys_k_{buffer_radius}_{neighbourhood_method_name}_area%_prob_origin_adjusted'] = edge_corrections.loc[:,'geometry_id'].map(
                {k:v['origin'] for k,v in ripleys_k_origin_area_percentage_edge_correction.items()}
        )
        edge_corrections.loc[:,f'{facility_sample_name}_{facility_name}_ripleys_k_{buffer_radius}_{neighbourhood_method_name}_area%_prob_destination_adjusted'] = edge_corrections.loc[:,'geometry_id'].map(
                {k:v['destination'] for k,v in ripleys_k_destination_area_percentage_edge_correction.items()}
        )
        edge_corrections.loc[:,f'{facility_sample_name}_{facility_name}_ripleys_k_{buffer_radius}_{neighbourhood_method_name}_points%_prob_origin_adjusted'] = edge_corrections.loc[:,'geometry_id'].map(
                {k:v['origin'] for k,v in ripleys_k_origin_points_percentage_edge_correction.items()}
        )
        edge_corrections.loc[:,f'{facility_sample_name}_{facility_name}_ripleys_k_{buffer_radius}_{neighbourhood_method_name}_points%_prob_destination_adjusted'] = edge_corrections.loc[:,'geometry_id'].map(
                {k:v['destination'] for k,v in ripleys_k_destination_points_percentage_edge_correction.items()}
        )

        # Normalise it 
        ripleys_l_edge_correction_normalised = {}
        for geomid in geometries.geometry_id.values:
            ripleys_l_edge_correction_normalised[geomid] = {'origin':1.0,'destination':1.0}
        for k,v in ripleys_k_origin_area_percentage_edge_correction.items():
            ripleys_l_edge_correction_normalised[k]['origin'] = np.pi*buffer_radius**2/v['origin']
        for k,v in ripleys_k_destination_area_percentage_edge_correction.items():
            ripleys_l_edge_correction_normalised[k]['destination'] = np.pi*buffer_radius**2/v['destination']
        # Pass it to geometries df
        edge_corrections.loc[:,f'{facility_sample_name}_{facility_name}_ripleys_k_{buffer_radius}_{neighbourhood_method_name}_area%_prob_origin_adjusted_normalised'] = edge_corrections.loc[:,'geometry_id'].map({k:v['origin'] for k,v in ripleys_l_edge_correction_normalised.items()})
        edge_corrections.loc[:,f'{facility_sample_name}_{facility_name}_ripleys_k_{buffer_radius}_{neighbourhood_method_name}_area%_prob_destination_adjusted_normalised'] = edge_corrections.loc[:,'geometry_id'].map({k:v['destination'] for k,v in ripleys_l_edge_correction_normalised.items()})
        
        # Normalise it 
        ripleys_l_edge_correction_normalised = {}
        for geomid in geometries.geometry_id.values:
            ripleys_l_edge_correction_normalised[geomid] = {'origin':1.0,'destination':1.0}
        
        for k,v in ripleys_k_origin_points_percentage_edge_correction.items():
            ripleys_l_edge_correction_normalised[k]['origin'] = np.pi*buffer_radius**2/v['origin']
        for k,v in ripleys_k_destination_points_percentage_edge_correction.items():
            ripleys_l_edge_correction_normalised[k]['destination'] = np.pi*buffer_radius**2/v['destination']

        # Pass it to geometries df
        edge_corrections.loc[:,f'{facility_sample_name}_{facility_name}_ripleys_k_{buffer_radius}_{neighbourhood_method_name}_points%_prob_origin_adjusted_normalised'] = edge_corrections.loc[:,'geometry_id'].map({k:v['origin'] for k,v in ripleys_l_edge_correction_normalised.items()})
        edge_corrections.loc[:,f'{facility_sample_name}_{facility_name}_ripleys_k_{buffer_radius}_{neighbourhood_method_name}_points%_prob_destination_adjusted_normalised'] = edge_corrections.loc[:,'geometry_id'].map({k:v['destination'] for k,v in ripleys_l_edge_correction_normalised.items()})


        edge_corrections.to_file(
                f'./data/inputs/{dataset}/edge_corrections.geojson',
                driver='GeoJSON',
                index = False
            )

        # Close client
        my_client.close()
        my_cluster.close()
