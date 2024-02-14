import warnings
warnings.filterwarnings("ignore")

import os
os.environ['USE_PYGEOS'] = '0'
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import date
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from gensit.utils.notebook_functions import compute_individual_facility_shortest_path
from gensit.utils.misc_utils import read_json

# Suppress warnings
warnings.filterwarnings("ignore")

num_cores = 7
n_batches = 10000
distance_method = 'shortest_path'
n_facilities_filename = f'all'
random_seed = 1234
store = True
starting_batch = 0

# Check that starting batch exists
if starting_batch > 0:
    assert os.path.isfile(f"./data/raw/cambridge_commuter/progress/shortest_path_batch_{starting_batch}_outof_{n_batches}.npy")

# Read graph
graph = nx.read_gpickle(os.path.join(f'./data/raw/cambridge_commuter/lsoas_graph.gpickle'))
# Read origin and destination locations
home_locs = gpd.read_file(f'./data/inputs/cambridge_work_commuter_lsoas/{n_facilities_filename}_home_facilities_seed_{random_seed}.geojson')
work_locs = gpd.read_file(f'./data/inputs/cambridge_work_commuter_lsoas/{n_facilities_filename}_work_facilities_seed_{random_seed}.geojson')
# Read origin and destination location ids
home_ids = sorted(home_locs.facility_id.values)
work_ids = sorted(work_locs.facility_id.values)

# Split inputs into batches/chunks
home_ids_batches = np.array_split(home_ids,n_batches)
# Start from last batch
home_ids_batches[starting_batch:] = home_ids_batches

individual_facility_distance_matrices = np.asarray(Parallel(n_jobs = num_cores,
                                                backend="multiprocessing")(
                                        delayed(compute_individual_facility_shortest_path)(i,
                                                graph,
                                                home_ids_batches[i],
                                                work_ids,
                                                num_cores,
                                                n_batches,
                                                store) for i in tqdm(range(starting_batch,n_batches),desc='Shortest path computation',leave = True,position = 0)),dtype = object)

# Take loaded batches into account
if starting_batch > 0:
    # Load stored batches
    individual_facility_distance_matrices_stored = []
    for j in range(starting_batch):
        temp = np.load(f"./data/raw/cambridge_commuter/progress/shortest_path_batch_{j}_outof_{n_batches}.npy",)
        individual_facility_distance_matrices_stored.append(temp)
    # Convert to numpy array
    individual_facility_distance_matrices_stored = np.array(individual_facility_distance_matrices_stored,dtype='object')
    # Reshape stored batches appropriately
    individual_facility_distance_matrices_stored = individual_facility_distance_matrices_stored.reshape(-1, individual_facility_distance_matrices_stored.shape[-1])
    # Reshape computed batches appropriately
    individual_facility_distance_matrices = individual_facility_distance_matrices.reshape(-1, individual_facility_distance_matrices_stored.shape[-1])
    # Stack loaded batches with computed batches
    individual_facility_distance_matrices = np.vstack((individual_facility_distance_matrices,individual_facility_distance_matrices_stored))
    
else:
    # Merge all the computed batches
    individual_facility_distance_matrices = np.concatenate(individual_facility_distance_matrices,axis = 0)
    
# Convert to df
print('Convert to df')
individual_facility_distance_matrix_gdf = pd.DataFrame(individual_facility_distance_matrices, columns = ['origin','destination',distance_method],index = None)

print('Merge origin geometry')
# Merge origin geometry
individual_facility_distance_matrix_gdf = pd.merge(individual_facility_distance_matrix_gdf,home_locs[['facility_id','geometry_id','geometry']],left_on='origin',right_on='facility_id',how='left')
# Rename columns
individual_facility_distance_matrix_gdf = individual_facility_distance_matrix_gdf.rename(columns={'geometry_id':'origin_geometry_id','geometry':'origin_geometry'})
# Drop columns
individual_facility_distance_matrix_gdf.drop(columns=['facility_id'],inplace = True)
print('Merge destination geometry')
# Merge destination geometry
individual_facility_distance_matrix_gdf = pd.merge(individual_facility_distance_matrix_gdf,work_locs[['facility_id','geometry_id','geometry']],left_on='destination',right_on='facility_id',how='left')
# Rename columns
individual_facility_distance_matrix_gdf = individual_facility_distance_matrix_gdf.rename(columns={'geometry_id':'destination_geometry_id','geometry':'destination_geometry'})
# Drop columns
individual_facility_distance_matrix_gdf.drop(columns=['facility_id'],inplace = True)

print('Convert to geopandas df')
# Convert data types
individual_facility_distance_matrix_gdf[distance_method] = individual_facility_distance_matrix_gdf[distance_method].astype('float32')
# Convert to geopandas
individual_facility_distance_matrix_gdf = gpd.GeoDataFrame(individual_facility_distance_matrix_gdf,geometry='origin_geometry')

# Get output filename
facility_cost_filename = f"facility_{distance_method}_{n_facilities_filename}_facilities_{date.today().strftime('%d_%m_%Y')}"
print('Save to file')
# Save pickle to file
individual_facility_distance_matrix_gdf.to_pickle(f"./data/raw/cambridge_commuter/{facility_cost_filename}.pickle")

print(individual_facility_distance_matrix_gdf)