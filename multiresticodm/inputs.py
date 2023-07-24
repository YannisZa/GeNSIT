import os
import torch
import logging
import h5py as h5
import numpy as np

from pathlib import Path

from multiresticodm.config import Config
from multiresticodm.utils import str_in_list
from multiresticodm.probability_utils import random_tensor
from multiresticodm.global_variables import PARAMETER_DEFAULTS

class Inputs:
    def __init__(
            self,
            config:Config,
            model:str = 'neural_net',
            synthetic_data:bool = False
    ):
        # Read logging
        self.logger = logging.getLogger(__name__)

        # Store config
        self.config = config

        if model == 'neural_net':
            if not synthetic_data:
                self.read_neural_net_data()

    
    def read_sim_data(
        self,
    ):
        self.logger.note("   Loading SIM data ...")
        if not str_in_list('dataset',self.config.settings['inputs']):
            raise Exception('Input dataset NOT provided. SIM cannot be loaded.')
            

        if str_in_list('origin_demand',self.config.settings['inputs']['data_files'].keys()):
            origin_demand_filepath = os.path.join(
                self.config.settings['inputs']['dataset'],
                self.config.settings['inputs']['data_files']['origin_demand']
            )
            if os.path.isfile(origin_demand_filepath):
                # Import rowsums
                origin_demand = np.loadtxt(origin_demand_filepath,dtype='float32')
                # Store size of rows
                self.dims[0] = len(origin_demand)
                # Check to see see that they are all positive
                if (origin_demand <= 0).any():
                    raise Exception(f'Origin demand {origin_demand} are NOT strictly positive')
            else:
                raise Exception(f"Origin demand file {origin_demand_filepath} NOT found")
        else:
            raise Exception(f"Origin demand filepath NOT provided")

        if len(self.config.settings['inputs']['data_files'].get('log_destination_attraction',[])) >= 0:
            log_destination_attraction_filepath = os.path.join(self.config.settings['inputs']['dataset'],self.config.settings['inputs']['data_files']['log_destination_attraction'])
            if os.path.isfile(log_destination_attraction_filepath):
                # Import log destination attractions
                self.log_destination_attraction = np.loadtxt(log_destination_attraction_filepath,dtype='float32')
                # Store size of rows
                self.dims[1] = len(self.log_destination_attraction)
                # Check to see see that they are all positive
                # if (self.log_destination_attraction <= 0).any():
                    # raise Exception(f'Log destination attraction {self.log_destination_attraction} are NOT strictly positive')
            else:
                raise Exception(f"Log destination attraction file {log_destination_attraction_filepath} NOT found")
        else:
            raise Exception(f"Log destination attraction filepath NOT provided")

        if str_in_list('cost_matrix',self.config.settings['inputs']['data_files'].keys()):
            cost_matrix_filepath = os.path.join(self.config.settings['inputs']['dataset'],self.config.settings['inputs']['data_files']['cost_matrix'])
            if os.path.isfile(cost_matrix_filepath):
                # Import rowsums
                cost_matrix = np.loadtxt(os.path.join(self.config.settings['inputs']['dataset'],self.config.settings['inputs']['data_files']['cost_matrix']),dtype='float32')
                # Make sure it has the right dimension
                try:
                    assert np.shape(cost_matrix) == tuple(self.dims)
                except:
                    self.dims = np.asarray(np.shape(cost_matrix))
                    raise Exception(f"Cost matrix does NOT have the required dimension {tuple(self.dims)} but has {self.dims}")
            else:
                raise Exception(f"Cost matrix file {cost_matrix_filepath} NOT found")

            # Check to see see that they are all positive
            if (cost_matrix < 0).any():
                raise Exception(f'Cost matrix {cost_matrix} is NOT non-negative.')
        else:
            raise Exception(f"Cost matrix filepath NOT provided")

        # Reshape cost matrix if necessary
        if self.dims[0] == 1:
            cost_matrix = np.reshape(cost_matrix[:,np.newaxis],tuple(self.dims))
        if self.dims[1] == 1:
            cost_matrix = np.reshape(cost_matrix[np.newaxis,:],tuple(self.dims))

        # Update noise level
        if str_in_list('noise_percentage',self.config.settings['spatial_interaction_model'].keys()):
            self.noise_var = ((self.config.settings['spatial_interaction_model']['noise_percentage']/100)*np.log(self.dims[1]))**2
        else:
            self.noise_var = (0.03*np.log(self.dims[1]))**2
            self.config.settings['spatial_interaction_model']['noise_percentage'] = 3

        # Normalise origin demand
        self.origin_demand = origin_demand

        # Normalise cost matrix
        self.cost_matrix = cost_matrix

        # Determine if true data exists
        if (hasattr(self,'alpha_true') and self.alpha_true is not None) and (hasattr(self,'beta_true') and self.beta_true is not None):
            self.ground_truth_known = True
        else:
            self.ground_truth_known = False

        # Store additional sim-specific parameters
        # Update delta and kappa
        if self.delta is None and self.kappa is None:
            self.kappa = (np.sum(self.origin_demand))/(np.sum(np.exp(self.log_destination_attraction))-np.min(np.exp(self.log_destination_attraction))*self.dims[1])
            self.delta = np.min(np.exp(self.log_destination_attraction)) * self.kappa
            # Update config
            self.config.settings['spatial_interaction_model']['delta'] = self.delta
            self.config.settings['spatial_interaction_model']['kappa'] = self.kappa
        elif self.kappa is None and self.delta is not None:
            self.kappa = (np.sum(self.origin_demand) + self.delta*self.dims[1])/np.sum(np.exp(self.log_destination_attraction))
            # Update config
            self.config.settings['spatial_interaction_model']['kappa'] = self.kappa
        elif self.kappa is not None and self.delta is None:
            self.delta = self.kappa * np.min(np.exp(self.log_destination_attraction))
            # Update config
            self.config.settings['spatial_interaction_model']['delta'] = self.delta

    
    
    
    def read_neural_net_data(
        self,
    ):
        self.logger.note("   Loading Harris Wilson data ...")
        if not str_in_list('dataset',self.config.settings['inputs']):
            raise Exception('Input dataset NOT provided. Harris Wilson model cannot be created.')
        
        if str_in_list('origin_demand',self.config.settings['inputs']['data_files'].keys()):
            origin_demand_filepath = os.path.join(
                self.config.settings['inputs']['dataset'],
                self.config.settings['inputs']['data_files']['origin_demand']
            )
            if os.path.isfile(origin_demand_filepath):
                # Import origin demand
                origin_demand = np.loadtxt(origin_demand_filepath,dtype='float32')
                # Check to see see that they are all positive
                if (origin_demand <= 0).any():
                    raise Exception(f'Origin demand {origin_demand} are NOT strictly positive')
            else:
                raise Exception(f"Origin demand file {origin_demand_filepath} NOT found")
        else:
            raise Exception(f"Origin demand filepath NOT provided")

        if len(self.config.settings['inputs']['data_files'].get('destination_attraction_ts',[])) >= 0:
            destination_attraction_ts_filepath = os.path.join(self.config.settings['inputs']['dataset'],self.config.settings['inputs']['data_files']['destination_attraction_ts'])
            if os.path.isfile(destination_attraction_ts_filepath):
                # Import destination attraction time series
                destination_attraction_ts = np.loadtxt(destination_attraction_ts_filepath,dtype='float32', ndmin=2)
            else:
                raise Exception(f"Destination attraction time series file {destination_attraction_ts_filepath} NOT found")

            # Check to see see that they are all positive
            if (destination_attraction_ts < 0).any():
                raise Exception(f'Destination attraction time series {destination_attraction_ts} is NOT non-negative.')
        else:
            raise Exception(f"Destination attraction filepath NOT provided")

        if str_in_list('cost_matrix',self.config.settings['inputs']['data_files'].keys()):
            cost_matrix_filepath = os.path.join(self.config.settings['inputs']['dataset'],self.config.settings['inputs']['data_files']['cost_matrix'])
            if os.path.isfile(cost_matrix_filepath):
                # Import rowsums
                cost_matrix = np.loadtxt(os.path.join(self.config.settings['inputs']['dataset'],self.config.settings['inputs']['data_files']['cost_matrix']),dtype='float32',ndmin=2)
            else:
                raise Exception(f"Cost matrix file {cost_matrix_filepath} NOT found")

            # Check to see see that they are all positive
            if (cost_matrix < 0).any():
                raise Exception(f'Cost matrix {cost_matrix} is NOT non-negative.')
        else:
            raise Exception(f"Cost matrix filepath NOT provided")

        # Read dimensions
        self.dims = [np.shape(origin_demand)[0],np.shape(destination_attraction_ts)[0]]
        # Store necessary quantities
        self.origin_demand = origin_demand
        self.destination_attraction_ts = destination_attraction_ts
        self.cost_matrix = cost_matrix
        # Extract parameters to learn
        params_to_learn = self.config.settings['spatial_interaction_model']['sim_to_learn'] + self.config.settings['harris_wilson_model']['hw_to_learn']
        # Extract the underlying parameters from the config
        parameter_settings = {**self.config.settings['spatial_interaction_model']['parameters'],**self.config.settings['harris_wilson_model']['parameters']}
        self.true_parameters = {
            k: parameter_settings.get(k,v) \
                for k,v in PARAMETER_DEFAULTS.items() \
                if not k in params_to_learn
        }

    def pass_to_device(self):
        # Define device to set 
        device = self.config.settings['inputs']['device']

        self.origin_demand = torch.from_numpy(self.origin_demand).float().to(device)
        self.destination_attraction_ts = torch.unsqueeze(
            torch.from_numpy(self.destination_attraction_ts).float(),
            -1
        ).to(device)
        self.cost_matrix = torch.reshape(
            torch.from_numpy(self.cost_matrix).float(),
            (self.dims)
        ).to(device)
        
        self.data_in_device = True
    
    def receive_from_device(self):

        self.origin_demand = self.origin_demand.cpu().detach().numpy()
        self.destination_attraction_ts = self.destination_attraction_ts.cpu().detach().numpy()
        self.cost_matrix = self.cost_matrix.cpu().detach().numpy().reshape(self.dims)
        
        self.data_in_device = False

    def prepare_neural_net_outputs(
            self,
            origin_demand,
            destination_attraction_ts,
            cost_matrix,
            dims,
            h5file: h5.File, 
            h5group: h5.Group,
    ):

        # If time series has a single frame, double it to enable visualisation.
        # This does not affect the training data
        training_data_size = self.config.settings.get("training_data_size", destination_attraction_ts.shape[0])
        if time_series.shape[0] == 1:
            time_series = torch.concat((destination_attraction_ts, destination_attraction_ts), axis=0)

        # Extract the training data from the time series data
        training_data = destination_attraction_ts[-training_data_size:]

        # Set up dataset for complete synthetic time series
        dset_time_series = h5group.create_dataset(
            "time_series",
            time_series.shape[:-1],
            maxshape=time_series.shape[:-1],
            chunks=True,
            compression=3,
        )

        write_start = self.config.settings['outputs']['neural_net']['write_start']
        write_every = self.config.settings['outputs']['neural_net']['write_every']
        dset_time_series.attrs["dim_names"] = ["time", "zone_id"]
        dset_time_series.attrs["coords_mode__time"] = "start_and_step"
        dset_time_series.attrs["coords__time"] = [write_start, write_every]
        dset_time_series.attrs["coords_mode__zone_id"] = "values"
        dset_time_series.attrs["coords__zone_id"] = np.arange(
            dims[0], sum(dims), 1
        )

        # Write the time series data
        dset_time_series[:, :] = torch.flatten(time_series, start_dim=-2)

        # Save the training time series
        dset_training_data = h5group.create_dataset(
            "training_data",
            training_data.shape[:-1],
            maxshape=training_data.shape[:-1],
            chunks=True,
            compression=3,
        )
        dset_training_data.attrs["dim_names"] = ["time", "zone_id"]
        dset_training_data.attrs["coords_mode__time"] = "trivial"
        dset_training_data.attrs["coords_mode__zone_id"] = "values"
        dset_training_data.attrs["coords__zone_id"] = np.arange(
            dims[0], sum(dims), 1
        )
        dset_training_data[:, :] = torch.flatten(training_data, start_dim=-2)

        # Set up chunked dataset to store the state data in
        # Origin zone sizes
        dset_origin_sizes = h5group.create_dataset(
            "origin_sizes",
            origin_demand.shape,
            maxshape=origin_demand.shape,
            chunks=True,
            compression=3,
        )
        dset_origin_sizes.attrs["dim_names"] = ["zone_id", "dim_name__0"]
        dset_origin_sizes.attrs["coords_mode__zone_id"] = "trivial"
        dset_origin_sizes[:] = origin_demand

        # Create a network group
        nw_group = h5file.create_group("network")
        nw_group.attrs["content"] = "graph"
        nw_group.attrs["is_directed"] = True
        nw_group.attrs["allows_parallel"] = False

        # Add vertices
        vertices = nw_group.create_dataset(
            "_vertices",
            (sum(dims),),
            maxshape=(sum(dims),),
            chunks=True,
            compression=3,
            dtype=int,
        )
        vertices.attrs["dim_names"] = ["vertex_idx"]
        vertices.attrs["coords_mode__vertex_idx"] = "trivial"
        vertices[:] = np.arange(0, sum(dims), 1)
        vertices.attrs["node_type"] = [0] * dims[0] + [1] * dims[1]

        # Add edges. The network is a complete bipartite graph
        edges = nw_group.create_dataset(
            "_edges",
            (np.prod(dims), 2),
            maxshape=(np.prod(dims), 2),
            chunks=True,
            compression=3,
        )
        edges.attrs["dim_names"] = ["edge_idx", "vertex_idx"]
        edges.attrs["coords_mode__edge_idx"] = "trivial"
        edges.attrs["coords_mode__vertex_idx"] = "trivial"
        edges[:,] = np.reshape(
            [
                [[i, j] for i in range(dims[0])]
                for j in range(dims[0], sum(dims))
            ],
            (np.prod(dims), 2),
        )

        # Edge weights
        edge_weights = nw_group.create_dataset(
            "_edge_weights",
            (np.prod(dims),),
            maxshape=(np.prod(dims),),
            chunks=True,
            compression=3,
        )
        edge_weights.attrs["dim_names"] = ["edge_idx"]
        edge_weights.attrs["coords_mode__edge_idx"] = "trivial"
        edge_weights[:] = torch.reshape(cost_matrix, (np.prod(dims),))


#     def generate_synthetic_data(
#     *, cfg, device: str
# ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

#     """Generates synthetic Harris-Wilson using a numerical solver.

#     :param cfg: the configuration file
#     :returns the origin sizes, cost_matrix, and the time series
#     """

#     log.note("   Generating synthetic data ...")

#     # Get run configuration properties
#     data_cfg = cfg["synthetic_data"]
#     N_origin, N_destination = data_cfg["N_origin"], data_cfg["N_destination"]
#     num_steps = data_cfg["num_steps"]

#     # Generate the initial origin sizes
#     or_sizes = torch.abs(
#         random_tensor(
#             **data_cfg.get("origin_sizes"), size=(N_origin, 1), device=device
#         )
#     )

#     # Generate the edge weights
#     cost_matrix = torch.exp(
#         -1
#         * torch.abs(
#             random_tensor(
#                 **data_cfg.get("init_weights"),
#                 size=(N_origin, N_destination),
#                 device=device
#             )
#         )
#     )

#     # Generate the initial destination zone sizes
#     init_dest_sizes = torch.abs(
#         random_tensor(
#             **data_cfg.get("init_dest_sizes"), size=(N_destination, 1), device=device
#         )
#     )

#     # Extract the underlying parameters from the config
#     true_parameters = {
#         "alpha": data_cfg["alpha"],
#         "beta": data_cfg["beta"],
#         "kappa": data_cfg["kappa"],
#         "sigma": data_cfg["sigma"],
#     }

#     # Initialise the ABM
#     ABM = HarrisWilson(
#         origin_sizes=or_sizes,
#         cost_matrix=cost_matrix,
#         true_parameters=true_parameters,
#         M=data_cfg["N_destination"],
#         epsilon=data_cfg["epsilon"],
#         dt=data_cfg["dt"],
#         device="cpu",
#     )

#     # Run the ABM for n iterations, generating the entire time series
#     dset_sizes_ts = ABM.run(
#         init_data=init_dest_sizes,
#         input_data=None,
#         n_iterations=num_steps,
#         generate_time_series=True,
#         requires_grad=False,
#     )

#     # Return all three
#     return or_sizes, dset_sizes_ts, cost_matrix

    def __str__(self):
        return f"""
            Dataset: {Path(self.config.settings['inputs']['dataset']).stem}
            Cost matrix: {Path(self.config.settings['inputs']['data_files']['cost_matrix']).stem}
        """