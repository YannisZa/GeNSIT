import os
import gc
import sys
import torch
import h5py as h5
import numpy as np
import xarray as xr

from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from functools import partial
from torch import float32,int32
import torch.multiprocessing as mp

from copy import deepcopy

from gensit.config import Config
from gensit.utils.exceptions import *
from gensit.utils.misc_utils import create_mask,write_json
from gensit.utils.math_utils import torch_optimize
from gensit.utils.probability_utils import random_vector
from gensit.contingency_table import *
from gensit.physics_models.HarrisWilsonModel import HarrisWilson
from gensit.intensity_models import instantiate_intensity_model
from gensit.static.global_variables import TRAIN_SCHEMA, PARAMETER_DEFAULTS, TRAIN_SCHEMA, VALIDATION_SCHEMA, INPUT_SCHEMA, Dataset
from gensit.utils.misc_utils import makedir, read_json, safe_delete, set_seed, setup_logger, tuplize, unpack_dims, write_txt, deep_call, ndims, eval_dtype, read_file

class Inputs:
    def __init__(
            self,
            config:Config,
            synthetic_data:bool = False,
            **kwargs
    ):  
        # Import logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__,
            console_level = level, 
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevels( console_level = level )

        # Store config
        self.config = config

        # Create data
        self.data = Dataset()

        # Define device to set 
        self.device = self.config.settings['inputs']['device']

        # Flag for whether data are stored in device
        self.data_in_device = False

        # Get instance of experiment 
        # this is relevant in the case of the DataGeneration experiment
        self.instance = kwargs.get('instance',1)

        self.seed = None
        if 'data_generation_seed' in list(self.config.settings['inputs']['data'].keys()):
            self.seed = int(self.config.settings['inputs']['data']['data_generation_seed'])
        
        if synthetic_data:
            self.generate_synthetic_data()
        else:
            self.read_data()

    def data_vars(self):
        return {k:v for k,v in vars(self.data).items() if k in INPUT_SCHEMA}
    
    def validate_dims(self):
        for attr,schema in TRAIN_SCHEMA.items():
            if len(schema) > 0:
                for ax,dim in zip(schema.get("axes",[]),schema.get("dims",[])):
                    if hasattr(self.data,attr) and getattr(self.data,attr) is not None:
                        try:
                            assert list(getattr(self.data,attr).shape)[ax] == self.data.dims[dim]
                        except:
                            raise Exception(f"{attr.replace('_',' ').capitalize()} has dim {list(getattr(self.data,attr).shape)[ax]} instead of {getattr(self.data,'dims')[dim]}.")
                        
    def copy(self,datasets:list=None):
        # Try to cast all data to tensor from xarray
        keys_of_interest = [
            sam_name for sam_name in dict({**INPUT_SCHEMA}).keys() \
            if getattr(self.data,sam_name,None) is not None
        ]
        datasets = list(set(datasets).intersection(set(keys_of_interest))) \
            if datasets \
            else list(keys_of_interest)
        
        # Copy everything
        new_copy = deepcopy(self)
        for dataset_name in self.data_vars().keys():
            if dataset_name not in datasets:
                delattr(new_copy.data,dataset_name)
        # Garbage collect
        gc.collect()

        return new_copy

    def read_data(self):
        
        self.logger.note("Loading Harris Wilson data ...")
        if not 'dataset' in list(self.config.settings['inputs']):
            raise Exception('Input dataset NOT provided. Harris Wilson model cannot be created.')

        # Initialise all
        # for attr in TRAIN_SCHEMA.keys():
            # setattr(self.data,attr,None)

        # Import dims
        setattr(
            self.data,
            "dims",
            self.config.settings['inputs'].get("dims",{"origin":None,"destination":None,"time":None})
        )

        for attr,schema in INPUT_SCHEMA.items():
            if len(schema) > 0 and attr in list(self.config.settings['inputs']['data'].keys()):
                filename = self.config.settings['inputs']['data'][attr]
                filename = filename.get('file','') if isinstance(filename,dict) else filename
                filepath = os.path.join(
                    self.config.in_directory,
                    self.config.settings['inputs']['dataset'],
                    filename
                )
                if os.path.isfile(filepath):
                    # Import data
                    data = read_file(filepath, dtype = schema['dtype'], ndmin = schema['ndmin'])
                    setattr(self.data,attr,data)
                    # Check to see see that they are all positive
                    if schema.get('positive',True) and (getattr(self.data,attr) < 0).any():
                        raise Exception(f"{attr.replace('_',' ').capitalize()} {getattr(self.data,attr)} are NOT positive")
                    # Update dims
                    for ax,dim in zip(schema.get("axes",[]),schema.get("dims",[])):
                        if getattr(self.data,attr) is not None:
                            self.data.dims[dim] = int(np.shape(getattr(self.data,attr))[ax])
                elif attr in VALIDATION_SCHEMA:
                    self.logger.debug(f"{attr} set to empty numpy array")
                    if attr == 'test_cells':
                        setattr(
                            self.data,attr,np.array([[i,j] \
                            for i in range(self.data.dims['origin']) \
                            for j in range(self.data.dims['destination'])])
                        )
                    else:
                        setattr(self.data,attr,np.array([]))
                else:
                    raise Exception(f"{attr.replace('_',' ').capitalize()} file {filepath} NOT found")
        # Initialise test cells if none are provided
        if getattr(self.data,'test_cells',None) is None:
            setattr(
                self.data,'test_cells',np.array([[i,j] \
                for i in range(self.data.dims['origin']) \
                for j in range(self.data.dims['destination'])])
            )
        
        # Import table margin data
        self.import_margins()
        # Import partial table cell data
        self.import_cells_subset()
        # Validate dimensions
        self.validate_dims()
        # Create masks based on train/test/validation cells
        self.create_cell_masks()
        
        # Update grand total if it is not specified
        if hasattr(self.data,'ground_truth_table') and self.data.ground_truth_table is not None:
            # Compute grand total
            self.data.grand_total = np.sum(self.data.ground_truth_table,dtype='float32')
            if 'spatial_interaction_model' in self.config.settings:
                self.config.settings['spatial_interaction_model']['grand_total'] = float(self.data.grand_total)
        if hasattr(self.data,"dims") and self.data.dims is not None and len(self.data.dims) > 0:
            self.config.settings['inputs']["dims"] = self.data.dims

        # Extract parameters to learn
        if 'to_learn' in list(self.config.settings['training'].keys()):
            params_to_learn = self.config.settings['training']['to_learn']
        else:
            params_to_learn = ['alpha','beta']
        # Extract the underlying parameters from the config
        parameter_settings = {
            **self.config.settings.get('spatial_interaction_model',{}).get('parameters',{}),
            **self.config.settings.get('harris_wilson_model',{}).get('parameters',{})
        }
        self.true_parameters = {
            k: parameter_settings.get(k,v) \
                for k,v in PARAMETER_DEFAULTS.items() \
                if k not in params_to_learn
        }
        # Update kappa and delta based on data (if not specified)
        self.update_delta_and_kappa()

    def update_delta_and_kappa(self):
        self.true_parameters['delta'] = self.config.settings.get('harris_wilson_model',{}).get('parameters',{}).get('delta',-1.0)
        self.true_parameters['delta'] = self.true_parameters['delta'] if self.true_parameters['delta'] >= 0 else None
        self.true_parameters['kappa'] = self.config.settings.get('harris_wilson_model',{}).get('parameters',{}).get('kappa',-1.0)
        self.true_parameters['kappa'] = self.true_parameters['kappa'] if self.true_parameters['kappa'] >= 0 else None
        
        if hasattr(self.data,'destination_attraction_ts'):
            smallest_zone_size = np.min(self.data.destination_attraction_ts)
            total_zone_sizes = np.sum(self.data.destination_attraction_ts)
        else:
            smallest_zone_size = (1/self.data.dims['destination'])*0.9
            total_zone_sizes = 1.0

        if hasattr(self.data,'origin_demand'):
            total_flow = np.sum(self.data.origin_demand,dtype='float32')
        elif hasattr(self.data,'grand_total'):
            total_flow = np.sum(self.data.grand_total,dtype='float32')
        else:
            total_flow = 1.0

        # Delta and kappa
        if self.true_parameters['delta'] is None and self.true_parameters['kappa'] is None:
            self.true_parameters['kappa'] = total_flow / (total_zone_sizes-smallest_zone_size*self.data.dims['destination'])
            self.true_parameters['delta'] = smallest_zone_size * self.true_parameters['kappa']
        elif self.true_parameters['kappa'] is None and self.true_parameters['delta'] is not None:
            self.true_parameters['kappa'] = (total_flow + self.true_parameters['delta']*self.data.dims['destination'])/total_zone_sizes
        elif self.true_parameters['kappa'] is not None and self.true_parameters['delta'] is None:
            self.true_parameters['delta'] = self.true_parameters['kappa'] * smallest_zone_size

    def create_cell_masks(self,datasets:list=None):
        # Try to cast all data to tensor from xarray
        keys_of_interest = [
            sam_name for sam_name in VALIDATION_SCHEMA.keys() \
            if getattr(self.data,sam_name,None) is not None
        ]
        datasets = list(set(datasets).intersection(set(keys_of_interest))) \
            if datasets \
            else list(keys_of_interest)
        
        for sample_name in datasets:
            if getattr(self.data,sample_name,None) is not None:
                setattr(
                    self.data,
                    (sample_name+'_mask'),
                    create_mask(
                        shape = unpack_dims(self.data.dims,time_dims=False),
                        index = getattr(self.data,sample_name)
                    )
                )

    def cast_from_xarray(self,datasets:list=None):

        # Try to cast all data to tensor from xarray
        keys_of_interest = [
            sam_name for sam_name,sam_schema in INPUT_SCHEMA.items() \
            if sam_schema.get('cast_to_xarray',False)
        ]
        datasets = list(set(datasets).intersection(set(keys_of_interest))) \
            if datasets \
            else list(keys_of_interest)
        
        for sample_name in datasets:
            if getattr(self.data,sample_name,None) is not None:

                # if data is already a tensor do not convert
                if torch.is_tensor(getattr(self.data,sample_name)):
                    continue

                # Data must be of torch type and in cpu
                try:
                    assert isinstance(getattr(self.data,sample_name),xr.DataArray)
                except:
                    raise CastingException(
                        data_name = sample_name,
                        from_type = type(getattr(self.data,sample_name,None)),
                        to_type = str(torch.tensor)
                    )

                setattr(
                    self.data,
                    sample_name,
                    torch.tensor(
                        getattr(self.data,sample_name).values,
                        dtype = int32,
                        device = self.device
                    )
                )
            else:
                raise MissingData(
                    missing_data_name = sample_name,
                    data_names = ', '.join([k for k in self.data_vars().keys()]),
                    location = 'Inputs'
                )  

    def cast_to_xarray(self, datasets:list=None):

        # Try to cast all data to xarray from numpy/tensor
        keys_of_interest = [
            sam_name for sam_name,sam_schema in INPUT_SCHEMA.items() \
            if sam_schema.get('cast_to_xarray',False)
        ]
        datasets = list(set(datasets).intersection(set(keys_of_interest))) \
            if datasets \
            else list(keys_of_interest)

        for sample_name in datasets:
            if getattr(self.data,sample_name,None) is not None:

                # Get input,validation schema
                schema = INPUT_SCHEMA[sample_name]

                # if data is already a tensor do not convert
                if isinstance(getattr(self.data,sample_name),xr.DataArray):
                    continue

                # Data must be of torch type and in cpu
                try:
                    assert (torch.is_tensor(getattr(self.data,sample_name)) and \
                        not getattr(self.data,sample_name).is_cuda) or \
                        isinstance(getattr(self.data,sample_name),np.ndarray)
                except:
                    raise CastingException(
                        data_name = sample_name,
                        from_type = type(getattr(self.data,sample_name,None)),
                        to_type = str(xr.DataArray)
                    )

                
                # Get data dims
                dims = [self.data.dims[dim_name] for dim_name in schema["dims"]]
                coordinates = {}
                # For each dim create coordinate
                for i,d in enumerate(dims):
                    obj,func = schema['funcs'][i]
                    # Create coordinate ranges based on schema
                    coordinates[schema["dims"][i]] = deep_call(
                        input = globals()[obj],
                        expressions = func,
                        defaults = None,
                        start = 1,
                        stop = d+1,
                        step = 1
                    ).astype(schema['args_dtype'][i])

                setattr(
                    self.data,
                    sample_name,
                    xr.DataArray(
                        data = getattr(self.data,sample_name),
                        name = sample_name,
                        dims = schema["dims"],
                        coords = coordinates
                    )
                )
    def pass_to_device(self):

        if not self.data_in_device:
            for input,schema in TRAIN_SCHEMA.items():
                if input not in ["dims",'grand_total','margins','true_parameters'] and \
                    getattr(self.data,input,None) is not None:
                        if "dims" in schema:
                            data = torch.reshape(
                                torch.from_numpy(
                                    getattr(self.data,input)).to(dtype = eval_dtype(
                                        schema['dtype'],
                                        numpy_format = False
                                    )
                                ),
                                tuple([self.data.dims[name] for name in schema["dims"]])
                            ).to(self.device)
                        else:
                            data = torch.from_numpy(
                                getattr(self.data,input)).to(dtype = eval_dtype(
                                    schema['dtype'],
                                    numpy_format = False
                                )
                            )
                        # Store to data
                        setattr(
                            self.data,
                            input,
                            data
                        )
            if hasattr(self.data,'log_destination_attraction') and getattr(self.data,'log_destination_attraction') is not None:
                self.data.log_destination_attraction = torch.squeeze(self.data.log_destination_attraction)
            if hasattr(self.data,'grand_total') and getattr(self.data,'grand_total') is not None:
                self.data.grand_total = torch.tensor(self.data.grand_total).float().to(self.device)
            if hasattr(self.data,'margins') and getattr(self.data,'margins') is not None:
                self.data.margins = {axis: torch.tensor(margin,dtype = int32,device = self.device) for axis,margin in self.data.margins.items()}
            if hasattr(self,'true_parameters') and len(getattr(self,'true_parameters')) > 0:
                for param in self.true_parameters.keys():
                    self.true_parameters[param] = torch.tensor(self.true_parameters[param]).float().to(self.device)

        self.data_in_device = True
    
    def import_margins(self):
        if 'margins' in list(self.config.settings['inputs']['data'].keys()):
            self.data.margins = {}
            for margin in self.config.settings['inputs']['data'].get('margins',[]):
                # Make sure that imported filename is not empty
                axis = margin['axis']
                filepath = os.path.join(
                    self.config.in_directory,
                    self.config.settings['inputs']['dataset'],
                    margin.get('file','')
                )
                if os.path.isfile(filepath):
                    # Import margin
                    margin = np.loadtxt(filepath, dtype='int32')
                    # Convert to tensor
                    self.data.margins[tuplize(axis)] = margin
                    # Check to see see that they are all positive
                    if torch.any(self.data.margins[tuplize(axis)] <= 0):
                        self.logger.error(f'margin {self.data.margins[tuplize(axis)]} for axis {axis} is not strictly positive')
                        del self.data.margins[tuplize(axis)]
                elif 'value' in list(margin.keys()):
                    self.data.margins[tuplize(axis)] = margin['value']
                else:
                    raise Exception(f"margin for axis {axis} not found in {filepath}.")
        else:
            self.logger.note('Margins not provided')
    
    def import_cells_subset(self):
        if 'cells_subset' in self.config.settings['inputs']['data'] and \
            getattr(self.data,'ground_truth_table',None) is None:
            
            cell_filename = os.path.join(
                self.config.settings['inputs']['dataset'],
                self.config.settings['inputs']['data']['cells_subset']
            )
            if os.path.isfile(cell_filename):
                # Initialise ground truth table
                self.data.ground_truth_table = -np.ones(
                    tuple([d for d in TRAIN_SCHEMA['ground_truth_table']['dims']]),
                    dtype = TRAIN_SCHEMA['ground_truth_table']['dtype']
                )
                # Import all cells
                cells = read_json(cell_filename)
                # Check to see see that they are all positive
                if (cells.values() < 0).any():
                    self.logger.error(f'Cell values{cells.values()} are not strictly positive')
                # Check that no cells exceed any of the margins
                for cell_str, value in cells.items():
                    # Convert string key to tuple
                    cell = tuple(eval(cell_str))
                    try:
                        assert len(cell) == ndims(self) and cell < np.asarray(list(unpack_dims(self.data.dims,time_dims = False)))
                    except:
                        self.logger.error(f"Cell has length {len(cell)}. The number of table dims are {ndims(self)}")
                        self.logger.error(f"Cell is equal to {cell}. The cell bounds are {np.asarray(list(unpack_dims(self.data.dims,time_dims = False)))}")
                    for ax in cell:
                        if tuplize(ax) in self.margins.keys():
                            try:
                                # Store flag for whether to allow sparse margins or not
                                if self.config.settings['contingency_table'].get('sparse_margins',False):
                                    assert self.data.margins[tuplize(ax)][cell[ax]] >= value
                                else:
                                    assert self.data.margins[tuplize(ax)][cell[ax]] > value
                            except:
                                self.logger.error(f"margin for ax = {','.join([str(a) for a in ax])} is less than specified cell value {value}")
                                raise Exception('Cannot import cells subset.')
                    # Update table
                    self.data.ground_truth_table[cell] = value
                else:
                    raise Exception(f"Cells subset values not found in {cell_filename}.")
        else:
            self.logger.note(f"Cells subset values file not provided")

    def receive_from_device(self):
        
        if self.data_in_device:
            # Input data
            if hasattr(self.data,'origin_demand') and getattr(self.data,'origin_demand') is not None:
                self.data.origin_demand = self.data.origin_demand.cpu().detach().numpy()
            if hasattr(self.data,'destination_attraction_ts') and getattr(self.data,'destination_attraction_ts') is not None:
                self.data.destination_attraction_ts = self.data.destination_attraction_ts.cpu().detach().numpy()
            if hasattr(self.data,'log_destination_attraction') and getattr(self.data,'log_destination_attraction') is not None:
                self.data.log_destination_attraction = self.data.log_destination_attraction.cpu().detach().numpy()
            if hasattr(self.data,'origin_attraction_ts') and getattr(self.data,'origin_attraction_ts') is not None:
                self.data.origin_attraction_ts = self.data.origin_attraction_ts.cpu().detach().numpy()
            if hasattr(self.data,'cost_matrix') and getattr(self.data,'cost_matrix') is not None:
                self.data.cost_matrix = self.data.cost_matrix.cpu().detach().numpy().reshape(unpack_dims(self.data,time_dims = False))
            if hasattr(self.data,'grand_total') and getattr(self.data,'grand_total') is not None:
                self.data.grand_total = self.data.grand_total.cpu().detach().numpy()
            if hasattr(self.data,'margins') and getattr(self.data,'margins') is not None:
                self.data.margins = {axis: margin.cpu().detach().numpy() for axis,margin in self.data.margins.items()}
            if hasattr(self.data,'total_cost_by_origin') and getattr(self.data,'total_cost_by_origin') is not None:
                self.data.total_cost_by_origin = self.data.total_cost_by_origin.cpu().detach().numpy()
            if hasattr(self.data,'ground_truth_table') and getattr(self.data,'ground_truth_table') is not None:
                self.data.ground_truth_table = self.data.ground_truth_table.cpu().detach().numpy().reshape(unpack_dims(self.data,time_dims = False))
            if hasattr(self,'true_parameters') and len(getattr(self,'true_parameters')) > 0:
                for param in self.true_parameters.keys():
                    self.true_parameters[param] = self.true_parameters[param].cpu().detach().item()

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
        data_size = self.config.settings.get("data_size", destination_attraction_ts.shape[0])
        if time_series.shape[0] == 1:
            time_series = torch.concat((destination_attraction_ts, destination_attraction_ts), axis = 0)

        # Extract the training data from the time series data
        training_data = destination_attraction_ts[-data_size:]

        # Set up dataset for complete synthetic time series
        dset_time_series = h5group.create_dataset(
            "time_series",
            time_series.shape[:-1],
            maxshape = time_series.shape[:-1],
            chunks = True,
            compression = 3,
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
            maxshape = training_data.shape[:-1],
            chunks = True,
            compression = 3,
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
            maxshape = origin_demand.shape,
            chunks = True,
            compression = 3,
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
            chunks = True,
            compression = 3,
            dtype = int,
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
            chunks = True,
            compression = 3,
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
            chunks = True,
            compression = 3,
        )
        edge_weights.attrs["dim_names"] = ["edge_idx"]
        edge_weights.attrs["coords_mode__edge_idx"] = "trivial"
        edge_weights[:] = torch.reshape(cost_matrix, (np.prod(dims),))


    def generate_synthetic_data(self,**kwargs) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        """Generates synthetic Harris-Wilson using a numerical solver.

        :param cfg: the configuration file
        :returns the origin sizes, cost_matrix, and the time series
        """

        self.logger.info("Generating synthetic data ...")

        # Set seed
        set_seed(self.seed)

        # Get run configuration properties
        self.data.dims = self.config.settings['inputs']['data']["dims"]
        synthetis_method = self.config.settings['inputs']['data']['synthesis_method']
        num_samples = self.config['inputs']['data']['synthesis_n_samples']
        num_steps = self.config.settings['training']['num_steps']
        
        # Generate the initial origin sizes
        for key,value in self.config.settings['inputs']['data'].items():
            if key in list(TRAIN_SCHEMA.keys()) and not (key in ['cost_matrix','dims']):
                # Randomly generate data
                data = np.abs(
                    random_vector(
                        **value, size = tuple([self.data.dims[name] for name in TRAIN_SCHEMA[key]["dims"]])
                    )
                ).astype('float32')
                # Normalise to sum to one
                data /= np.sum(data)
                # Store dataset
                setattr(self.data,key,data)
        
        # Generate the edge weights
        self.data.cost_matrix = np.abs(
            random_vector(
                **self.config.settings['inputs']['data']["cost_matrix"],
                size=(self.data.dims['origin'], self.data.dims['destination']),
            )
        ).astype('float32')
        # Make matrix symmetric if origins = destinations
        if self.data.dims['origin'] ==  self.data.dims['destination']:
            self.data.cost_matrix += self.data.cost_matrix.T
        self.data.total_cost_by_origin = self.data.cost_matrix.sum(axis = 1)

        # Normalise to sum to one
        cost_matrix_max = deepcopy(self.data.cost_matrix.max())
        self.data.cost_matrix = (self.data.cost_matrix/cost_matrix_max)
        cost_matrix_max = 1
        # self.data.cost_matrix /= self.data.cost_matrix.sum()
        self.data.total_cost_by_origin /= self.data.total_cost_by_origin.sum()

        # Extract the underlying parameters from the config
        parameter_settings = {
            **self.config.settings['spatial_interaction_model']['parameters'],
            **self.config.settings['harris_wilson_model']['parameters']
        }
        self.true_parameters = {
            k: parameter_settings.get(k,v) for k,v in PARAMETER_DEFAULTS.items()
        }
        # Update kappa and delta based on data (if not specified)
        self.update_delta_and_kappa()
        # Update true parameters to multiply
        self.true_parameters['beta'] *= self.true_parameters['bmax']

        # Get grand total
        grand_total = self.config.settings['spatial_interaction_model'].get('grand_total',1)
        self.data.grand_total = grand_total

        # Pass all data to device
        self.pass_to_device()

        # Initialise spatial interaction model
        intensity_model = instantiate_intensity_model(
            config = self.config,
            instance = kwargs.get('instance',''),
            **vars(self.data),
            logger = self.logger
        )

        # Initialise the Harris Wilson model
        HWM = HarrisWilson(
            trial = None,
            intensity_model = intensity_model,
            config = self.config,
            dt = self.config['harris_wilson_model'].get('dt',0.001),
            true_parameters = self.true_parameters,
            instance = kwargs.get('instance',''),
            logger = self.logger
        )

        if synthetis_method == 'sde_potential':
            destination_attraction_ts = self.sde_potential_in_sequence(HWM)
        elif synthetis_method == 'sde_solver':
            destination_attraction_ts = self.sde_solver_in_sequence(HWM)
        else:
            raise Exception(f'Cannot synthetise data using method {synthetis_method}')

        # Get training data size from dimensions
        data_time_steps = self.config.settings['inputs']["dims"].get('time',destination_attraction_ts.shape[0])
        # Extract the training data from the time series data
        self.data.destination_attraction_ts = destination_attraction_ts[-data_time_steps:].T
        # self.data.log_destination_attraction = 

        # Create synthetic table
        # Get log probability (normalised intensity)
        if self.data.dims['origin'] != self.data.dims['destination']:
            log_probability = HWM.intensity_model.log_intensity(
                log_destination_attraction = torch.log(destination_attraction_ts[-1:]),
                grand_total = torch.tensor(1.0),
                **HWM.hyperparams
            )
        else:
            log_intensity = torch.log(destination_attraction_ts[-1:].repeat(self.data.dims['origin'],1)/self.data.cost_matrix)
            log_probability = log_intensity - torch.logsumexp(log_intensity.ravel(),dim=(0))

        # Initialise ground truth table
        ground_truth_table = torch.zeros(self.data.dims['origin']*self.data.dims['destination'],dtype=int32)

        # Sample from multinomial with grand total fixed
        continue_loop = True
        while continue_loop:
            # Sample free cells from multinomial
            updated_cells = distr.multinomial.Multinomial(
                total_count = int(grand_total),
                probs = torch.exp(log_probability).ravel()
            ).sample()
            # Update free cells
            ground_truth_table = updated_cells.to(dtype = float32,device = self.device)
            # Reshape table to match original dims
            ground_truth_table = torch.reshape(ground_truth_table, tuplize(list(unpack_dims(self.data.dims))))
            # Continue loop only if table is not sparse admissible
            continue_loop = False

        # Store ground truth table to input data
        self.data.ground_truth_table = ground_truth_table.to(dtype = float32,device = self.device)

        # Receive all data from device
        self.receive_from_device()

        # Set seed to None
        set_seed(None)

        # Set dataset name
        dataset_name = f"./data/inputs/synthetic/" + \
        f"synthetic_{'x'.join([str(self.data.dims[name]) for name in TRAIN_SCHEMA[key]['dims']])}_total_{int(grand_total)}"
        # f"_using_{synthetis_method}_samples_{str(num_samples)}_steps_{str(num_steps)}_sigma_{str(np.round(self.true_parameters['sigma'],2))}"

        # Create inputs folder
        if os.path.isdir(dataset_name) and not self.config.settings['experiments'][0]['overwrite']:
            self.logger.warning(f"Synthetic dataset {dataset_name} already exists.")
            # raise Exception(f"Synthetic dataset {dataset_name} already exists.")
        else:
            makedir(dataset_name)

        # Write config used to synthesize the data
        write_json(
            self.config.settings,
            os.path.join(dataset_name,'config.json'),
            indent = 2
        )
        
        # Write data to inputs folder
        for key in vars(self.data):
            if key not in ["dims","grand_total"]:
                self.logger.note(f"Dataset {os.path.join(dataset_name,f'{key}.txt')} created")
                write_txt(
                    getattr(self.data,key),
                    os.path.join(dataset_name,f'{key}.txt')
                )
            if key == 'cost_matrix':
                write_txt(
                    self.data.cost_matrix*cost_matrix_max,
                    os.path.join(dataset_name,f'{key}_max_normalised.txt')
                )
        self.logger.success(f"Created and populated synthetic dataset {dataset_name}")

    def sde_solver_in_sequence(self,physics_model):
        samples = []
        # Run the ABM for n iterations, generating the entire time series
        num_samples = self.config['inputs']['data']['synthesis_n_samples']
        num_steps = self.config.settings['training']['num_steps']
        for instance in tqdm(
            range(num_samples),
            desc = f"Generating time series in sequence: Instance {self.instance}",
            leave = False,
            position = self.instance+1
        ):
            samples.append(
                physics_model.run(
                    init_destination_attraction = self.data.destination_attraction_ts,
                    n_iterations = num_steps,
                    free_parameters = physics_model.hyperparams,
                    generate_time_series = True,
                    requires_grad = False,
                    seed = instance
                )
            )

        # Take mean over seed
        return torch.stack(samples).mean(dim = 0)
    
    def sde_potential_in_sequence(self,physics_model):
        # Delete destination attraction
        safe_delete(self.data.destination_attraction_ts)
        # Create partial function
        torch_optimize_partial = partial(
            torch_optimize,
            function= physics_model.sde_potential_and_gradient,
            method='L-BFGS-B',
            **self.true_parameters
        )
        # Create initialisations
        Ndests = self.data.dims['destination']
        g = np.log(physics_model.hyperparams['delta'].item())*np.ones((Ndests,Ndests)) - \
            np.log(physics_model.hyperparams['delta'].item())*np.eye(Ndests) + \
            np.log(1+physics_model.hyperparams['delta'].item())*np.eye(Ndests)
        g = g.astype('float32')
        
        # Get minimum across different initialisations in parallel
        xs = [torch_optimize_partial(g[i,:]) for i in range(Ndests)]
        # Compute potential
        fs = np.asarray([
            physics_model.sde_potential(
                torch.tensor(xs[i]
            ).to(
                dtype = float32,
                device = physics_model.intensity_model.device
            ),
            **self.true_parameters
            ).detach().cpu().numpy() for i in range(Ndests)
        ])
        # Get arg min
        arg_min = np.argmin(fs)
        minimum = xs[arg_min]
        # Update log destination attraction
        destination_attraction_ts = torch.exp(torch.tensor(minimum,dtype = float32,device = self.config['inputs']['device']))
        
        return destination_attraction_ts

    def __str__(self):
        od = self.config.settings['inputs']['data'].get('origin_demand','not-found')
        od = Path(od['file']).stem if isinstance(od,dict) else od
        dd = self.config.settings['inputs']['data'].get('destination_demand','not-found')
        dd = Path(dd['file']) if isinstance(dd,dict) else dd
        oats = self.config.settings['inputs']['data'].get('origin_attraction_ts','not-found')
        oats = Path(oats['file']) if isinstance(oats,dict) else oats
        dats = self.config.settings['inputs']['data'].get('destination_attraction_ts','not-found')
        dats = Path(dats['file']).stem if isinstance(dats,dict) else dats
        cost = self.config.settings['inputs']['data'].get('cost_matrix','not-found')
        cost = Path(cost['file']).stem if isinstance(cost,dict) else cost
        table = self.config.settings['inputs']['data'].get('table','not-found')
        table = Path(table['file']).stem if isinstance(table,dict) else table
        total_cost_by_origin = self.config.settings['inputs']['data'].get('total_cost_by_origin','not-found')
        total_cost_by_origin = Path(total_cost_by_origin['file']).stem if isinstance(total_cost_by_origin,dict) else total_cost_by_origin
        margins = self.config.settings['inputs']['data'].get('margins','not-found')
        margins = Path(margins['file']).stem if isinstance(margins,dict) else margins
        cells_subset = self.config.settings['inputs']['data'].get('cells_subset','not-found')
        cells_subset = Path(cells_subset['file']).stem if isinstance(cells_subset,dict) else cells_subset

        return f"""
            Dataset: {Path(self.config.settings['inputs']['dataset']).stem}
            Cost matrix: {cost}
            Total cost by origin: {total_cost_by_origin}
            Origin demand: {od}
            Destination demand: {dd}
            Origin attraction time series: {oats}
            Destination attraction time series: {dats}
            Ground truth table: {table}
            Ground truth table margins: {margins}
            Ground truth table cell subset: {cells_subset}
        """