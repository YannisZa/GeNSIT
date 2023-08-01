import os
import torch
import logging
import h5py as h5
import numpy as np

from pathlib import Path
from pandas import read_csv

from multiresticodm.config import Config
from multiresticodm.utils import setup_logger, str_in_list
from multiresticodm.probability_utils import random_tensor
from multiresticodm.global_variables import PARAMETER_DEFAULTS

class Inputs:
    def __init__(
            self,
            config:Config,
            synthetic_data:bool = False,
            **kwargs
    ):  
        # Setup logger
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            config.level,
            log_to_file=True,
            log_to_console=True
        )

        # Store config
        self.config = config

        # Store attributes and their associated dims
        self.schema = {
            "origin_demand":{"axes":[0],"dtype":"float32", "ndmin":1},
            "destination_demand":{"axes":[1],"dtype":"float32", "ndmin":1},
            "origin_attraction":{"axes":[0],"dtype":"float32", "ndmin":1},
            "destination_attraction_ts":{"axes":[1],"dtype":"float32", "ndmin":2},
            "cost_matrix":{"axes":[0,1],"dtype":"float32", "ndmin":2},
            "table":{"axes":[0,1],"dtype":"int32", "ndmin":1},
            "dims":{},
            "grand_total":{}
        }

        if not synthetic_data:
            self.read_data()

    def validate_dims(self):
        for attr,schema in self.schema.items():
            if len(schema) > 0:
                for ax in schema["axes"]:
                    if getattr(self,attr) is not None:
                        try:
                            assert np.shape(getattr(self,attr))[ax] == self.dims[ax]
                        except:
                            raise Exception(f"{attr.replace('_',' ').capitalize()} has dim {np.shape(getattr(self,attr))[ax]} instead of {self.dims[ax]}.")
    
    def read_data(
        self,
    ):
        self.logger.note("Loading Harris Wilson data ...")
        if not str_in_list('dataset',self.config.settings['inputs']):
            raise Exception('Input dataset NOT provided. Harris Wilson model cannot be created.')

        # Initialise all
        for attr in self.schema.keys():
            setattr(self,attr,None)

        # Import dims
        self.dims = tuple(self.config.settings['inputs'].get('dims',[None,None]))
        self.time_dim = (1,)
        
        # Import all data
        for attr,schema in self.schema.items():
            if len(schema) > 0:
                if str_in_list(attr,self.config.settings['inputs']['data_files'].keys()):
                    filepath = os.path.join(
                        self.config.settings['inputs']['dataset'],
                        self.config.settings['inputs']['data_files'][attr]
                    )
                    if os.path.isfile(filepath):
                        # Import data
                        data = np.loadtxt(filepath,dtype=schema['dtype'], ndmin=schema['ndmin'])
                        setattr(self,attr,data)
                        # Check to see see that they are all positive
                        if (getattr(self,attr) <= 0).any():
                            raise Exception(f"{attr.replace('_',' ').capitalize()} {self.origin_demand} are NOT strictly positive")
                        # Update dims
                        for subax in schema['axes']:
                            if getattr(self,attr) is not None and self.dims[subax] is None:
                                self.dims[subax] = int(np.shape(getattr(self,attr)))[subax]
                    else:
                        raise Exception(f"{attr.replace('_',' ').capitalize()} file {filepath} NOT found")
                # else:
                    # raise Exception(f"{attr.replace('_',' ').capitalize()} filepath NOT provided")

        # Validate dimensions
        self.validate_dims()

        # Update grand total if it is not specified
        if self.grand_total is None:
            # Compute grand total
            self.grand_total = np.sum(self.table)
            self.grand_total = self.config.settings['spatial_interaction_model'].get('grand_total',1.0)
            

        # Extract parameters to learn
        if hasattr(self.config.settings['inputs'],'to_learn'):
            params_to_learn = self.config.settings['inputs']['to_learn']
        else:
            params_to_learn = ['alpha','beta']
        # Extract the underlying parameters from the config
        parameter_settings = {**self.config.settings['spatial_interaction_model']['parameters'],**self.config.settings['harris_wilson_model']['parameters']}
        self.true_parameters = {
            k: parameter_settings.get(k,v) \
                for k,v in PARAMETER_DEFAULTS.items() \
                if not k in params_to_learn
        }
        print(self.true_parameters)

        self.data_in_device = False

    def pass_to_device(self):
        # Define device to set 
        device = self.config.settings['inputs']['device']

        if not self.data_in_device:
            if self.origin_demand is not None:
                self.origin_demand = torch.from_numpy(self.origin_demand).float().to(device)
            if self.destination_attraction_ts is not None:
                self.destination_attraction_ts = torch.unsqueeze(
                    torch.from_numpy(self.destination_attraction_ts).float(),
                    -1
                ).to(device)
            if self.cost_matrix is not None:
                self.cost_matrix = torch.reshape(
                    torch.from_numpy(self.cost_matrix).float(),
                    (self.dims)
                ).to(device)
            if self.table is not None:
                self.cost_matrix = torch.reshape(
                    torch.from_numpy(self.cost_matrix).int(),
                    (self.dims)
                ).to(device)
            if self.dims is not None:
                self.dims = torch.tensor(self.dims).int().to(device)
            if self.grand_total is not None:
                self.grand_total = torch.tensor(self.grand_total).int().to(device)

        self.data_in_device = True
    
    def receive_from_device(self):
        
        if self.data_in_device:
            self.origin_demand = self.origin_demand.cpu().detach().numpy() if hasattr(self,'origin_demand') else None
            self.destination_attraction_ts = self.destination_attraction_ts.cpu().detach().numpy() if hasattr(self,'destination_attraction_ts') else None
            self.cost_matrix = self.cost_matrix.cpu().detach().numpy().reshape(self.dims) if hasattr(self,'cost_matrix') else None
            self.dims = self.dims.cpu().detach().numpy() if hasattr(self,'dims') else None
            self.grand_total = self.grand_total.cpu().detach().numpy() if hasattr(self,'grand_total') else None

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

        write_start = self.config.settings['outputs']['neural_network']['write_start']
        write_every = self.config.settings['outputs']['neural_network']['write_every']
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

#     log.note("Generating synthetic data ...")

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
        od = Path(self.config.settings['inputs']['data_files'].get('origin_demand',''))
        dd = Path(self.config.settings['inputs']['data_files'].get('destination_demand',''))
        loa = Path(self.config.settings['inputs']['data_files'].get('log_origin_attraction',''))
        lda = Path(self.config.settings['inputs']['data_files'].get('log_destination_attraction',''))
        dats = Path(self.config.settings['inputs']['data_files'].get('destination_attraction_time_series',''))
        return f"""
            Dataset: {Path(self.config.settings['inputs']['dataset']).stem}
            Cost matrix: {Path(self.config.settings['inputs']['data_files']['cost_matrix']).stem}
            Origin demand: {od.stem if len(od) > 0 else ''}
            Destination demand: {dd.stem if len(dd) > 0 else ''}
            Log origin attraction: {loa.stem if len(loa) > 0 else ''}
            Log destination attraction: {lda.stem if len(lda) > 0 else ''}
            Destination attraction time series: {dats.stem if len(dats) > 0 else ''}
        """