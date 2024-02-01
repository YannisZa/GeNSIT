import os
os.environ['USE_PYGEOS'] = '0'
import warnings
warnings.filterwarnings("ignore")
import itertools
import googlemaps
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle

from tqdm import tqdm
from datetime import datetime
from copy import deepcopy

""" Get distances and travel times for origin and destination zones """

# Add your Google Maps API key here:
with open('.api_key', 'r') as file:
    API_key = file.read().replace('\n', '')
gmaps = googlemaps.Client(key = API_key)

# ==== Functions =======================================================================================================
def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))

def get_times_distances(origin_coords: list,
                        dest_coords: list,
                        *,
                        mode: str = None,
                        dep_time: int = None,
                        write_every: int = 50):

    """ Collect travel times and distances from the API, using a list of origin zones and destination zones.
    See https://developers.google.com/maps/documentation/distance-matrix/distance-matrix

    :param origin_coords: the coordinates of the origin zones
    :param dest_coords: the coordinates fo the destination zones
    :param mode: (optional) the transportation mode. Must be one of the accepted transportation modes 'driving' (default),
        'walking', 'bicycling', 'transit'. Defaults to 'driving' if None.
    :param dep_time: (optional) the departure time in Unix time. Required if mode is transit, and must lie in the future
    :param write_every: number of requests to collect before dumping to file.
    :return:
    """

    # Collect values in a dictionary
    res = {'values': []}

    # Count number of requests
    num_requests = 0

    # Get all origin-destination pairs
    origin_batches = split_given_size(origins,10)
    destination_batches = split_given_size(destinations,10)
    origin_destination_pair_batches = list(itertools.product(origin_batches,destination_batches))

    print('Total number of requests',len(origin_destination_pair_batches))
    
    # Create output filename
    output_filename = f"./data/raw/cambridge_commuter/{mode_of_transport}_{resolution}_origs_{n_origins}_dests_{n_dests}_metrics_{datetime.today().strftime('%d_%m_%Y')}.pkl"

    # Get the values from the Google Distance Matrix
    for od_pairs in tqdm(origin_destination_pair_batches):
        # Get origins and destinations
        origs = od_pairs[0]
        dests = od_pairs[1]
        
        if mode == 'transit':
            res['values'].append(
            gmaps.distance_matrix(origs, dests, mode = mode, departure_time = dep_time,transit_routing_preference='fewer_transfers'))
        else:
            res['values'].append(
            gmaps.distance_matrix(origs, dests, mode = mode, departure_time = dep_time))

        num_requests += 1
        # Print status and write every n requests
        if num_requests % write_every == 0:
            with open(output_filename, 'wb') as f:
                pickle.dump(res, f)

            print(f'  Completed run {num_requests} of {len(origin_destination_pair_batches)} for mode {mode}.')

    # Dump results to pickle
    with open(output_filename, 'wb') as f:
        pickle.dump(res, f)


def eval_distance_matrix(route_data: dict,
                         origin_zones: pd.DataFrame,
                         dest_zones: pd.DataFrame,
                         *,
                         mode: str,
                         resolution:str):
    """ Function to evaluate the distance matrix data. Splits the data into spatial and temporal distances,
        saving the data as formatted csv files. This way they can be loaded by the NeuralABM model.
        Trips where the origin and destination zone are identical are processed to have distance 0.

    :param route_data: the dictionary of distance values
    :param origin_zones: the DataFrame of origin zone names and coordinates
    :param dest_zones: the DataFrame of destination zone names and coordinates
    :param mode: the chosen transit mode
    :param resolution: resolution of points queried
    """

    # Get the number of origin and destination zones
    I,J = len(origin_zones), len(dest_zones)

    # Split the values into temporal and spatial values
    time_data, distance_data, traffic_duraction_data = [], [], []

    print(f'Evaluation distance matrix for mode {mode}')
    # Discard trips where origin and destination are identical
    for idx, entry in enumerate(route_data['values']):
        for row in entry['rows']:
            for elem in row['elements']:
                if elem['status'] == 'OK':
                    time_data.append(elem['duration']['value'])
                    distance_data.append(elem['distance']['value'])
                    try:
                        traffic_duraction_data.append(elem['duration_in_traffic']['value'])
                    except:
                        traffic_duraction_data.append(-1)
                else:
                    # these are trips where the origin and destination are identical
                    time_data.append(0)
                    distance_data.append(0)
                    traffic_duraction_data.append(0)

    # Create DataFrame
    times = pd.DataFrame(np.reshape(time_data, (I,J)), index = origin_zones.index, columns = dest_zones.index)
    distances = pd.DataFrame(np.reshape(distance_data, (I,J)), index = origin_zones.index, columns = dest_zones.index)
    traffic_duration = pd.DataFrame(np.reshape(distance_data, (I,J)), index = origin_zones.index, columns = dest_zones.index)

    # Write to csv
    times.to_csv(f"./data/raw/cambridge_commuter/{resolution}_origs_{n_origins}_dests_{n_dests}_{mode}_metrics_{date_string}_times.csv", index_label='row: origin/col: dest/entries: seconds')
    distances.to_csv(f"./data/raw/cambridge_commuter/{resolution}_origs_{n_origins}_dests_{n_dests}_{mode}_metrics_{date_string}_distances.csv", index_label='row: origin/col: dest/entries: meters')
    traffic_duration.to_csv(f"./data/raw/cambridge_commuter/{resolution}_origs_{n_origins}_dests_{n_dests}_{mode}_metrics_{date_string}_traffic_duraction.csv", index_label='row: origin/col: dest/entries: seconds')

# ==== Get data ========================================================================================================

# Collect the origin zone and destination zone coordinates
# origin_zones = pd.read_csv('./data/raw/london_commuter/GLA_data/origin_sizes.csv', header = 0, index_col = 0)
# dest_zones = pd.read_csv('./data/raw/london_commuter/GLA_data/dest_sizes.csv', header = 0, index_col = 0)
# origins = list(zip(origin_zones['Latitude'], origin_zones['Longitude']))
# destinations = list(zip(dest_zones['Latitude'], dest_zones['Longitude']))
# origin_coords = [{'lat': val[0], 'lng': val[1]} for val in origins]
# dest_coords = [{'lat': val[0], 'lng': val[1]} for val in destinations]


# Read LSOAS
# origin_zones = gpd.read_file('./data/raw/cambridge_commuter/cambridge_lsoas.geojson')
# origin_zones = origin_zones.sort_values('LSOA11CD')
# destination_zones = deepcopy(origin_zones)

# origins = list(zip(origin_zones['LSOA11CD'],origin_zones['LAT'], origin_zones['LONG']))
# destinations = list(zip(destination_zones['LSOA11CD'],destination_zones['LAT'], destination_zones['LONG']))
# origin_coords = [{'lat': val[1], 'lng': val[2]} for val in origins] #'place_id':val[0],
# destination_coords = [{'lat': val[1], 'lng': val[2]} for val in destinations]

# Departure time: GMT (in seconds). If specified, the departure time must lie in the future.
dep_time = 'now' # 'now'
resolution = 'clustered_facility_sample'
id_name = "facility_id"
n_origins = 22
n_dests = 21
date_string = '18_01_2023' #datetime.today().strftime('%d_%m_%Y')

# Read origins
origin_zones = gpd.read_file(f'./data/inputs/cambridge_work_commuter_lsoas/sample_{n_origins}_home_clustered_facilities_seed_1234.geojson')
# Reproject to global coordinates and sort by id
origin_zones = origin_zones.set_crs('EPSG:27700',allow_override = True)
origin_zones = origin_zones.to_crs('EPSG:4326')
origin_zones = origin_zones.sort_values(id_name)
origin_zones = origin_zones.set_index("facility_id")

# Read destinations
destination_zones = gpd.read_file(f'./data/inputs/cambridge_work_commuter_lsoas/sample_{n_dests}_work_clustered_facilities_seed_1234.geojson')
# Reproject to global coordinates and sort by id
destination_zones = destination_zones.set_crs('EPSG:27700',allow_override = True)
destination_zones = destination_zones.to_crs('EPSG:4326')
destination_zones = destination_zones.sort_values(id_name)
destination_zones = destination_zones.set_index("facility_id")

origins = list(zip(origin_zones['geometry'].y, origin_zones['geometry'].x))
destinations = list(zip(destination_zones['geometry'].y, destination_zones['geometry'].x))
origins = [{'lat': val[0], 'lng': val[1]} for val in origins]
destinations = [{'lat': val[0], 'lng': val[1]} for val in destinations]

for mode_of_transport in ['driving','transit','bicycling']:
    

    output_filename = f"./data/raw/cambridge_commuter/{resolution}_origs_{n_origins}_dests_{n_dests}_{mode_of_transport}_metrics_{date_string}.pkl"

    # get_times_distances(origins, destinations, mode = mode_of_transport, dep_time = dep_time, write_every = 30)

    with open(output_filename, 'rb') as f:
        metrics = pickle.load(f)

    eval_distance_matrix(metrics, origin_zones, destination_zones, mode = mode_of_transport,resolution = resolution)
