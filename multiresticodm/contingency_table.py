import os
import sys
import torch
import logging
import numpy as np
import torch.distributions as distr

from copy import deepcopy
from pandas import read_csv
from pandas import DataFrame
from itertools import product
from tabulate import tabulate
from torch import int32, uint8
from torch import bool as tbool
from num2words import num2words
from torch.masked import masked_tensor
from typing import List, Union, Callable

from multiresticodm.config import Config
from multiresticodm.global_variables import *
from multiresticodm.math_utils import powerset
from multiresticodm.utils import setup_logger, str_in_list, write_txt, extract_config_parameters, makedir, read_json, tuplize, flatten, tuple_contained, depth

# -> Union[ContingencyTable,None]:
def instantiate_ct(table, config: Config, **kwargs):
    if hasattr(sys.modules[__name__], config.settings['contingency_table']['ct_type']):
        return getattr(sys.modules[__name__], config.settings['contingency_table']['ct_type'])(
                table=table, 
                config=config,
                **kwargs
        )
    else:
        raise Exception(
            f"Input class {config.settings['contingency_table']['ct_type']} not found")


class ContingencyTable(object):

    def __init__(self, table = None, config: Config = None, **kwargs: dict):
        # Setup logger
        self.level = config.level if hasattr(config,'level') else kwargs.get('level','INFO')
        self.logger = setup_logger(
            __name__,
            level=self.level,
            log_to_file=False,
            log_to_console=kwargs.get('log_to_console',True),
        )
        # Config
        self.config = config
        # Table
        self.ground_truth_table = table
        # Device name
        self.device = kwargs.get('device','cpu')
        # Instantiate dataset 
        self.data = Dataset()
        # Markov basis class
        self.markov_basis_class = None
        # Flag for whether to allow sparse margins or not
        self.sparse_margins = False
        # Random seed
        self.seed = None
        # margins
        self.margins = {}
        self.residual_margins = {}

    def build(self, table, config, **kwargs):
        # Build contingency table
        if config is not None:
            # Set configuration file
            self.config = extract_config_parameters(
                config,
                {
                    "log_mode", "sweep_mode", 
                    "inputs", "contingency_table",
                    "outputs"
                }
            )
            # Store flag for whether to allow sparse margins or not
            if str_in_list('sparse_margins', self.config.settings['contingency_table'].keys()):
                self.sparse_margins = self.config.settings['contingency_table']['sparse_margins']
            if str_in_list('seed', self.config.settings['inputs'].keys()):
                self.seed = int(self.config.settings['inputs']['seed'])
            
            # Read tabular data
            self.import_tabular_data()

        elif table is not None:
            # Read table and dimensions
            if torch.is_tensor(table):
                self.ground_truth_table = table.int().to(dtype=int32,device=self.device)
            else:
                self.ground_truth_table = torch.from_numpy(table).int().to(dtype=int32,device=self.device)
            self.dims = np.asarray(np.shape(self.ground_truth_table), dtype='uint8')
            # Read seed
            self.seed = int(kwargs['seed']) if str_in_list('seed', kwargs.keys()) else None
            # Read margin sparsity
            self.sparse_margins = bool(kwargs['sparse_margins']) if str_in_list('sparse_margins', kwargs.keys()) else False
            # Update table properties
            self.update_table_properties_from_table()

        else:
            self.logger.error('Neither table nor config provided.')
            raise Exception('Cannot initialise Contingency table.')
        
        # Read constraints
        self.read_constraints(**kwargs)

        # Create cell_locations
        self.cells = sorted([tuplize(cell) for cell in product(*[range(dim) for dim in self.dims])])
        
        # Fill in missing margins that can be recovered from existing (user input) margins
        self.distribution_name = self.update_and_map_margin_constraints_to_distribution_name()

        # Fill in missing cell margins and 
        # Update residual margins based on exhaustive list of cell constraints
        self.propagate_cell_constraints()

        # Update type of table to determine the type of Markov basis to use
        self.update_table_type_and_markov_basis_class()

        self.logger.info(f"margin constraints over axes {','.join([str(ax) for ax in self.constraints['constrained_axes']])} provided")


    def axes_complement(self, axes, same_length:bool=False):
        if hasattr(axes,'__len__') and depth(axes) > 1:
            axes_set = set([tuplize(ax) for ax in axes])
        else:
            axes_set = {tuplize(axes)}
        # Define all axes
        all_axes_set = set(powerset(range(self.ndims())))
        # Take set difference with all axes
        set_diff = all_axes_set - axes_set
        # If same length gather all lengths of axes
        if same_length:
            # Gather all unique lengths
            axes_lens = set([len(ax) for ax in list(axes_set)])
            # Slice set difference to take all axes with same length
            # as axes provided
            set_diff = {elem for elem in set_diff if len(elem) in list(axes_lens)}
        
        return list(set_diff)
    
    # Compute summary statistics for given function
    def table_margins_summary_statistic(self, tab: np.ndarray) -> List:
        return torch.cat(tuple([self.table_axis_margin_summary_statistic(tab=tab,ax=ax) for ax in self.constraints['all_axes']]), dim=0)
    
    def table_axis_margin_summary_statistic(self, tab: np.ndarray, ax:int = None) -> List:
        return tab.sum(dim=tuplize(ax), keepdims=True, dtype=int32).flatten()

    # Compute margin summary statistics for given function
    def table_constrained_margin_summary_statistic(self, tab: np.ndarray) -> List:
        if len(self.constraints['constrained_axes']) > 0:
            return torch.cat(tuple([tab.sum(dim=tuplize(ax), keepdims=True, dtype=int32).flatten() for ax in sorted(self.constraints['constrained_axes'])]), dim=0)
        else:
            return torch.empty(1)

    def table_unconstrained_margin_summary_statistic(self, tab: np.ndarray) -> List:
        if len(self.constraints['unconstrained_axes']) > 0:
            return torch.cat(tuple([tab.sum(dim=tuplize(ax), keepdims=True, dtype=int32).flatten() for ax in sorted(self.constraints['unconstrained_axes'])]), dim=0)
        else:
            return torch.empty(1)

    # Compute cell summary statistics for given function
    def table_cell_constraint_summary_statistic(self, tab: np.ndarray, cells: list=None) -> List:
        if cells is None:
            return torch.tensor(
                [tab[cell] for cell in sorted(self.constraints['cells'])], 
                dtype=int32
            )
        else:
            return torch.tensor(
                [tab[cell] for cell in sorted(self.constraints['cells'])], 
                dtype=int32
            )

    # Check if margin constraints are satisfied
    def table_margins_admissible(self, tab) -> bool:
        if len(self.constraints['constrained_axes']) > 0:
            return torch.equal(
                self.table_constrained_margin_summary_statistic(tab),
                torch.cat(
                    [self.margins[tuplize(ax)] \
                        for ax in sorted(self.constraints['constrained_axes'])],
                    dim=0
                )
            )
        else:
            return True

    # Check if cell constraints are satisfied
    def table_cells_admissible(self, tab) -> bool:
        return all([self.ground_truth_table[cell] == tab[cell] for cell in sorted(self.constraints['cells'])])

    # Check if function (or table equivalently) is admissible
    def table_admissible(self, tab) -> bool:
        return self.table_margins_admissible(tab) and self.table_cells_admissible(tab)
    
    def table_sparse_admissible(self, tab) -> bool:
        return all(self.table_unconstrained_margin_summary_statistic(tab) > 0) if not self.sparse_margins else True
    
    def margin_sparse_admissible(self, margin) -> bool:
        return torch.torch.all(margin > 0) if not self.sparse_margins else True

    # Checks if function f proposed is positive
    def table_nonnegative(self, tab) -> bool:
        return not torch.any(tab < 0)

    def table_positive(self, tab) -> bool:
        return not torch.any(tab <= 0)

    def table_difference_histogram(self, tab: np.ndarray, tab0: np.ndarray):
        # Take difference of tables
        difference = tab - tab0
        # Values classification
        values = {"+": torch.count_nonzero(difference > 0), "-": torch.count_nonzero(
            difference < 0), "0": torch.count_nonzero(difference == 0)}
        return values

    def check_table_validity(self, table: np.ndarray) -> bool:
        return self.table_nonnegative(table)

    
    def __eq__(self, other_ct):
        equality_holds = True
        try:
            # Test if class types are the same
            assert type(other_ct) == type(self)
        except Exception:
            self.logger.error(f"Contingency table type {type(other_ct)} is not the same as {type(self)}")
            equality_holds = False
        try:
            # Test if dimensions are the same
            assert self.ndims() == other_ct.ndims()
        except Exception:
            self.logger.error(f"Contingency table dims {other_ct.ndims()} are not the same as {self.ndims()}")
            equality_holds = False
        try:
            # Test if constrained margins are the same
            assert set(self.constraints['constrained_axes']) == set(other_ct.constraints['constrained_axes'])
        except Exception:
            self.logger.error(f"Contingency table margin constraints {other_ct.constrained_margins()} are not the same as {self.constrained_margins()}")
            equality_holds = False
        try:
            # Test if table distribution is the same
            assert self.distribution_name == other_ct.distribution_name
        except Exception:
            self.logger.error(f"Contingency table distribution {other_ct.distribution_name} are not the same as {self.distribution_name}")
            equality_holds = False
        try:
            # Test if cell constraints are the same
            assert set(self.constraints['cells']) == set(other_ct.constraints['cells'])
        except Exception:
            self.logger.error(f"Contingency table cell constraints {other_ct.constraints['cells']} are not the same as {self.constraints['cells']}")
            equality_holds = False

        return equality_holds
    
    def __str__(self):
        if self.ground_truth_table is not None:
            # Convert to numpy
            ground_truth_table = self.ground_truth_table.cpu().detach().numpy()
            return ',\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in ground_truth_table])
        else:
            return 'Empty Table'

    def __repr__(self):
        return f"{'x'.join([str(dim) for dim in self.dims])} ContingencyTable{self.ndims()}(Config)"

    def __len__(self):
        return np.prod([dim for dim in self.dims if dim > 0])
    
    def ndims(self):
        return np.sum([1 for dim in self.dims if dim > 1],dtype='int32')
    
    def reset(self, ground_truth_table, config, **kwargs) -> None:
        # Rebuild table
        self.build(ground_truth_table=ground_truth_table, config=config, **kwargs)
    
    def constraint_table(self,with_margins:bool=False):
        if self.ground_truth_table is not None:
            table_constraints = -np.ones(self.ground_truth_table.shape)
            table_mask = np.zeros(self.ground_truth_table.shape,dtype=tbool)
            for cell in self.constraints['cells']:
                table_constraints[cell] = self.ground_truth_table[cell].cpu().detach().numpy()
                table_mask[cell] = True
            table_constraints = table_constraints.astype(torch.int32)
            
            # Printing margins for visualisation purposes only
            if with_margins:
                assert self.ndims() == 2

                # Flip table for viz purposes (clear visibility)
                row_margin,col_margin = (0,),(1,)
                if self.dims[1] >= self.dims[0]:
                    table_constraints = table_constraints.T
                    row_margin,col_margin = (1,),(0,)
                
                # Create pandas dataframe from constraints
                table_constraints_ct = DataFrame(table_constraints)

                # Add margins to the contingency table
                table_constraints_ct.loc['Total',:] = self.residual_margins[row_margin].to(dtype=int32)
                table_constraints_ct['Total'] = list(self.residual_margins[col_margin].to(dtype=int32)) + list(self.residual_margins[(0,1)].to(dtype=int32))
                
                # Replace free cells with empty string
                table_constraints_ct[table_constraints_ct < 0] = ''
                table_constraints_ct = table_constraints_ct.applymap(str).replace('\.0', '', regex=True)
                
                # Convert the contingency table to a pretty printed string
                table_constraints = tabulate(table_constraints_ct, headers=[], showindex=False, tablefmt='plain')

            return table_constraints
        else:
            return 'Empty Table'

    def import_tabular_data(self) -> None:
        # Based on filepath read txt or csv
        if str_in_list('ground_truth_table', self.config.settings['inputs']['data_files'].keys()) and \
                bool(self.config.settings['inputs']['data_files']['ground_truth_table']) and \
                os.path.isfile(os.path.join(self.config.settings['inputs']['dataset'], self.config.settings['inputs']['data_files']['ground_truth_table'])):
            
            table_filepath = os.path.join(
                self.config.settings['inputs']['dataset'], 
                self.config.settings['inputs']['data_files']['ground_truth_table']
            )
            # Read ground truth table
            ground_truth_table = np.loadtxt(table_filepath, dtype='float32').astype('int32')
            self.ground_truth_table = torch.from_numpy(ground_truth_table).to(dtype=int32,device=self.device)

            # Check that all entries are non-negative
            try:
                assert self.check_table_validity(self.ground_truth_table)
            except:
                print(self.ground_truth_table)
                raise Exception("Imported table contains negative values! Import aborted...")
            
            # Update table properties
            self.update_table_properties_from_table()
            # Copy residal margins
            self.residual_margins = deepcopy(self.margins)
        else:
            self.logger.warning('Valid contingency table was not provided. One will be randomly generated.')
            if hasattr(self,'ground_truth_table') and self.ground_truth_table is not None:
                self.dims = np.asarray(np.shape(self.ground_truth_table), dtype='uint8')
            elif str_in_list('dims', self.config.settings['inputs'].keys()):
                self.dims = np.asarray(self.config.settings['inputs']['dims'], dtype='uint8')
            else:
                raise Exception("Cannot read table dimensions.")
            
            # Import margins
            self.import_margins()
            # Ensure margin validity
            for axis, margin in self.margins.items():
                # Assert all margins are unidimensional
                try:
                    assert np.sum([1 for dim in np.shape(margin) if dim > 1]) == 1
                except:
                    raise Exception(f"margin for axis {axis} is not unidimensional.")
                # Assert all margin axis match dimensions
                if np.sum(self.dims) > 0:
                    try:
                        assert len(margin) == self.dims[axis]
                    except:
                        self.logger.error(f"margin for axis {axis} has dim {len(margin)} and not {self.dims[axis]}.")
                        raise Exception('Imported inconsistent margins.')
                else:
                    self.dims[axis] = len(margin)
                # Assert all margin sums are the same
                try:
                    assert np.array_equal(
                                margin.sum(keepdims=True,dtype=int32).flatten(),
                                self.margins[tuplize(range(self.ndims()))]
                    )
                except:
                    self.logger.error(f"margin for axis {axis} has total {margin.sum(keepdims=True).ravel().flatten()} and not {self.margins[tuplize(range(self.ndims()))]}.")
                    raise Exception('Imported inconsistent margins.')
            # No dim must be zero
            try:
                assert (self.__len__() > 0) and (len(self.dims) > 0)
            except AssertionError:
                self.logger.error('Dims provided are not valid.')
                raise Exception('Tabular data could not be imported')
            # Randomly initialise margins not provided
            for axis in sorted(list(powerset(range(self.ndims()))), key=len, reverse=True):
                if tuplize(axis) not in self.margins.keys():
                    if tuplize(axis) == tuplize(range(self.ndims())):
                        self.margins[tuplize(axis)] = torch.randint(
                            low=1,
                            high=None,
                            size=1,
                            dtype=int32
                        )
                    else:
                        dim = self.dims[self.axes_complement(axis,same_length=True)[0]]
                        self.margins[tuplize(axis)] = distr.multinomial.Multinomial(
                            n=self.margins[tuplize(range(self.ndims()))], 
                            pvals=np.repeat(1/dim, dim)
                        ).sample().to(dtype=int32)
                        self.logger.debug(f"Randomly initialising margin for axis", tuplize(axis))
            # Copy residal margins
            self.residual_margins = deepcopy(self.margins)
            # Initialise table
            self.ground_truth_table = -torch.ones(self.dims)
            # Import and update cells of table
            self.import_cells()
            # Initialise constraints
            self.constraints = {}
            self.constraints['constrained_axes'] = []
            self.constraints['all_axes'] = sorted(list(set(powerset(range(self.ndims())))))
            self.constraints['unconstrained_axes'] = sorted(list(set(powerset(range(self.ndims())))))

    def import_margins(self) -> None:
        if str_in_list('margins', self.config.settings['inputs']['data_files'].keys()):
            for axis, margin_filename in enumerate(self.config.settings['inputs']['data_files']['margins']):
                # Make sure that imported filename is not empty
                if len(margin_filename) > 0:
                    margin_filepath = os.path.join(
                        self.config.settings['inputs']['dataset'], 
                        margin_filename
                    )
                    if os.path.isfile(margin_filepath):
                        # Import margin
                        margin = np.loadtxt(margin_filepath, dtype='int32')
                        # Convert to tensor
                        self.margins[tuplize(axis)] = torch.tensor(margin).int().to(dtype=int32,device=self.device)
                        # Check to see see that they are all positive
                        if torch.any(self.margins[tuplize(axis)] <= 0):
                            self.logger.error(f'margin {self.margins[tuplize(axis)]} for axis {axis} is not strictly positive')
                            self.margins[tuplize(axis)] = None
                    else:
                        raise Exception(f"margin for axis {axis} not found in {margin_filepath}.")
            # Copy residual margins
            self.residual_margins = deepcopy(self.margins)

        else:
            self.logger.warning(f"margins file not provided")
            self.margins = {}
            self.residual_margins = {}
            # raise Exception(f"Importing margins failed.")

    def import_cells(self) -> None:
        if str_in_list('cell_values', self.config.settings['inputs']['data_files'].keys()):
            cell_filename = os.path.join(
                self.config.settings['inputs']['dataset'],
                self.config.settings['inputs']['data_files']['cell_values']
            )
            if os.path.isfile(cell_filename):
                # Import all cells
                cells = read_json(cell_filename)
                # Check to see see that they are all positive
                if (cells.values() < 0).any():
                    self.logger.error(f'Cell values{cells.values()} are not strictly positive')
                # Check that no cells exceed any of the margins
                for cell, value in cells.items():
                    try:
                        assert len(cell) == self.ndims() and cell < self.dims
                    except:
                        self.logger.error(f"Cell has length {len(cell)}. The number of table dims are {self.ndims()}")
                        self.logger.error(f"Cell is equal to {cell}. The cell bounds are {self.dims}")
                    for ax in cell:
                        if tuplize(ax) in self.margins.keys():
                            try:
                                if self.sparse_margins:
                                    assert self.margins[tuplize(ax)][cell[ax]] >= value
                                else:
                                    assert self.margins[tuplize(ax)][cell[ax]] > value
                            except:
                                self.logger.error(f"margin for ax = {','.join([str(a) for a in ax])} is less than specified cell value {value}")
                                raise Exception('Cannot import cells.')
                    # Update table
                    self.ground_truth_table[cell] = value
                else:
                    raise Exception(f"Cell values not found in {cell_filename}.")
        else:
            self.logger.debug(f"Cells file not provided")
            # raise Exception(f"Importing cells failed.")

    def read_constraints(self,**kwargs):
        self.constraints = {}
        constrained_axes,cell_constraints = None, None
        # margin and cell constraints
        ## Reading config if provided

        if self.config is not None:
            if str_in_list('constraints', self.config.settings['contingency_table'].keys()):
                if str_in_list('axes', self.config.settings['contingency_table']['constraints'].keys()):
                    constrained_axes = [tuplize(ax) for ax in self.config.settings['contingency_table']['constraints']['axes'] if len(ax) > 0]
                if str_in_list('cells', self.config.settings['contingency_table']['constraints'].keys()):
                    if isinstance(self.config.settings['contingency_table']['constraints']['cells'],str):
                        cell_constraints = os.path.join(
                            self.config.settings['inputs']['dataset'],
                            self.config.settings['contingency_table']['constraints']['cells']
                        )
                    elif isinstance(self.config.settings['contingency_table']['constraints']['cells'],(list,np.ndarray)):
                        cell_constraints = self.config.settings['contingency_table']['constraints']['cells']
        ## Reading kwargs if provided
        elif str_in_list('constraints', kwargs.keys()):
            if str_in_list('axes', kwargs['constraints'].keys()):
                constrained_axes = [tuplize(ax) for ax in kwargs['constraints']['axes'] if len(ax) > 0]
            if str_in_list('cells', kwargs['constraints'].keys()):
                if isinstance(kwargs['constraints']['cells'],str):
                    cell_constraints = os.path.join(
                        kwargs['inputs']['dataset'],
                        kwargs['constraints']['cells']
                    )
                elif isinstance(kwargs['constraints']['cells'],(list,np.ndarray)):
                    cell_constraints = kwargs['constraints']['cells']
        ## Storing constraints
        if constrained_axes is not None:
            # Get the set of constrained axes
            constrained_axes = set(constrained_axes)
            # Determine axes over which margins are defined
            self.constraints['all_axes'] = sorted(list(set(powerset(range(self.ndims())))))
            # Infer derivative margins from the ones provided
            constrained_axes = self.infer_derivative_margin_axes(constrained_axes)
            # Update constrained axes
            self.constraints['constrained_axes'] = sorted(list(constrained_axes))
            # Determine complement of these axes
            self.constraints['unconstrained_axes'] = self.axes_complement(self.constraints['constrained_axes'])
        else:
            self.constraints['all_axes'] = sorted(list(set(powerset(range(self.ndims())))))
            self.constraints['constrained_axes'] = []
            self.constraints['unconstrained_axes'] = sorted(list(set(powerset(range(self.ndims())))))
        
        # Cell constraints
        ## Reading config if provided
        if cell_constraints is not None:
            if isinstance(cell_constraints,str):
                if os.path.isfile(cell_constraints):
                    # Load cell constraints from file
                    cells = np.loadtxt(cell_constraints, dtype='uint8')
                    # Reshape cells
                    cells = cells.reshape(np.size(self.constraints['cells'])//self.ndims(), self.ndims()).tolist()
                    # Remove invalid cells (i.e. cells than do not have the right number of dims or are out of bounds)
                    cells = sorted([tuplize(c) for c in cells if (len(c) == self.ndims() and (c < self.dims).all())])
                    self.constraints['cells'] = torch.tensor(cells).to(dtype=uint8,device=self.device)
                    self.logger.info(f"Cell constraints {','.join([str(c) for c in cells])} provided")
                else:
                    self.constraints['cells'] = torch.tensor([]).to(dtype=uint8,device=self.device)
            elif isinstance(cell_constraints,(list,np.ndarray)):
                self.constraints['cells'] = torch.tensor([tuplize(ax) for ax in cell_constraints],dtype=uint8).to(device=self.device)
            else:
                print('cell constraints')
                print(cell_constraints)
                self.logger.error(f"Cell constraints of type {type(cell_constraints)}")
                raise Exception('Cell constraints could not be recognised')
        else:
            self.constraints['cells'] = torch.empty(1).to(device=self.device)

    def update_table_type_and_markov_basis_class(self):
        # Check if there are is at lest one active cell
        try:
            assert len(self.cells) > 0
        except:
            # Groudn truth
            print('Ground truth table')
            print(self.ground_truth_table)
            print('Constraint table')
            print(self.constraint_table())
            self.logger.error('No activate cells found. Table can be deterministically elicited based on provided constraints.')
            raise Exception(f"Table inference is halted.")

        # Count number of non zero dims
        if self.ndims() == 0:
            self.ct_type = 'empty_table'
            self.markov_basis_class = None
        elif self.ndims() <= 2:
            self.markov_basis_class = f'MarkovBasis{self.ndims()}DTable'
            if isinstance(self, ContingencyTableDependenceModel):
                self.ct_type = f'{num2words(self.ndims())}_way_contingency_table_dependence_model'
            else:
                self.ct_type = f'{num2words(self.ndims())}_way_contingency_table_independence_model'
        else:
            self.logger.error(f'No implementation found for ndims = {self.ndims()}')
            raise Exception('Table type and markov basis class could not be identified.')
        
    def update_margins(self, margins: dict = {}):
        for ax, margin in margins.items():
            # Update margins
            if not torch.is_tensor(margin):
                self.margins[tuplize(ax)] = torch.from_numpy(margin).int().to(dtype=int32,device=self.device)
                self.residual_margins[tuplize(ax)] = torch.from_numpy(margin).int().to(dtype=int32,device=self.device)
            else:
                self.margins[tuplize(ax)] = margin.int().to(dtype=int32,device=self.device)
                self.residual_margins[tuplize(ax)] = margin.int().to(dtype=int32,device=self.device)

    def update_residual_margins_from_cell_constraints(self, constrained_cells=None):
        constrained_cells = self.constraints['cells'] if constrained_cells is None or len(constrained_cells) <= 0 else constrained_cells
        for ax in self.constraints['all_axes']:
            ax_complement = self.axes_complement([ax])
            for cell in constrained_cells:
                cell_list = np.asarray(cell)
                for subax in ax_complement:
                    # Subtract non-total margins
                    if len(subax) < self.ndims() and len(ax) < self.ndims():
                        self.residual_margins[subax][cell_list[ax]] -= self.ground_truth_table[tuplize(cell)]
                if len(ax) == self.ndims():
                    # Subtract cell values from grand total (only once)
                    self.residual_margins[ax] -= self.ground_truth_table[tuplize(cell)]

        # Update cells though
        self.cells = sorted(list(set(self.cells) - set(constrained_cells)))


    def update_cell_constraints_deterministically(self):
        # Get latest constraint table
        constraint_table = self.constraint_table()
        # Get flags for whether cells are constrained or not
        cell_unconstrained = (constraint_table == -1)
        # Repeat process if there any margins with only one unknown
        while any([any(cell_unconstrained.sum(dim=ax,keepdims=True).flatten() == 1) for ax in self.constraints['constrained_axes']]):
            # If cells need to be filled in deterministically the entire table must be available
            try:
                assert self.table_nonnegative(self.ground_truth_table)
            except:
                self.logger.error('Table has negative entries')
                raise Exception('Cannot fill in remainder of cells determinstically as table values are not provided for these cells.')
            # Go through each margin
            for ax in self.constraints['constrained_axes']:
                # If there any margins with only one unknown cell
                if any(cell_unconstrained.sum(dim=tuplize(ax),keepdims=True).flatten() == 1):
                    # Get axis complement
                    ax_complement = list(flatten(self.axes_complement(ax, same_length=True)))
                    # Create cell index
                    cell_index = [{
                        "axis": ax,
                        "index": list(flatten(np.where(cell_unconstrained.sum(dim=tuplize(ax)) == 1)))
                    }]
                    # Update last index
                    last_axis_index = cell_index[-1]['index']
                    for subax in ax_complement:
                        # Find indices
                        cell_index.append({
                            "axis": subax,
                            "index": np.where(cell_unconstrained.take(indices=last_axis_index[0], dim=subax))
                        })
                        # Update last index
                        last_axis_index = cell_index[-1]['index']
                    # Construct cell index
                    query_index = np.zeros(self.ndims(), dtype='uint8')
                    for c in cell_index:
                        query_index[c['axis']] = c['index'][0]
                    # Reverse order
                    query_index = np.flip(query_index)
                    # Update cell
                    self.constraints['cells'].append(tuple(query_index))
                    # Update residual margins
                    self.update_residual_margins_from_cell_constraints([tuple(query_index)])
                    # Update constraint table
                    constraint_table = self.constraint_table()
                    # Update flags
                    cell_unconstrained = (constraint_table == -1)
                    # print('added constraint', tuple(query_index))
                    # print(constraint_table)
                    # print(cell_unconstrained)
                    # print('\n')

    def propagate_cell_constraints(self):
        if str_in_list('cells',self.constraints.keys()) and len(self.constraints['cells']) > 0:
            # Update cell constraints if there are any deterministic solutions
            # based on the cell and margin constraints provided
            self.update_cell_constraints_deterministically()
            # Update residual margins
            self.update_residual_margins_from_cell_constraints(self.constraints['cells'])

    def constrained_margins(self,table=None):
        if table is None:
            return {ax:self.margins[ax] for ax in self.constraints['constrained_axes']}
        else:
            return {ax:table.sum(dim=tuplize(ax), keepdims=True, dtype=int32).flatten() for ax in self.constraints['constrained_axes']}

    def unconstrained_margins(self,table=None):
        if table is None:
            return {ax:self.margins[ax] for ax in self.constraints['unconstrained_axes']}
        else:
            return {ax:table.sum(dim=tuplize(ax), keepdims=True, dtype=int32).flatten() for ax in self.constraints['unconstrained_axes']}

    def update_unconstrained_margins_from_table(self, table):

        for ax in self.constraints['unconstrained_axes']:
            self.margins[tuplize(ax)] = table.sum(
                dim=tuplize(ax), 
                keepdims=True, 
                dtype=int32
            ).flatten()
            self.residual_margins[tuplize(ax)] = table.sum(
                dim=tuplize(ax), 
                keepdims=True, 
                dtype=int32
            ).flatten()

    def update_constrained_margins_from_table(self, table):
        for ax in self.constraints['constrained_axes']:
            self.margins[tuplize(ax)] = table.sum(
                dim=tuplize(ax), 
                keepdims=True, 
                dtype=int32
            ).flatten()
            self.residual_margins[tuplize(ax)] = table.sum(
                dim=tuplize(ax), 
                keepdims=True, 
                dtype=int32
            ).flatten()

    def update_margins_from_table(self, table):
        for axes in powerset(range(self.ndims())):
            self.margins[tuplize(axes)] = table.sum( 
                dim=tuplize(axes), 
                keepdims=True, 
                dtype=int32
            ).flatten()
            print(axes,self.margins[tuplize(axes)].shape)
            self.residual_margins[tuplize(axes)] = table.sum(
                dim=tuplize(axes), 
                keepdims=True, 
                dtype=int32
            ).flatten()
    
    def update_unconstrained_margins(self, margins: dict = {}):
        return self.update_margins({
            tuplize(ax): val for ax, val in margins.items() \
                if ax in self.constraints['unconstrained_axes']
        })

    def update_constrained_margins(self, margins: dict = {}):
        return self.update_margins({
            tuplize(ax): val for ax, val in margins.items() \
                if ax in self.constraints['constrained_axes']
        })
    
    def update_table(self, tab: np.ndarray) -> None:
        try:
            assert np.shape(tab) == self.dims
        except:
            self.logger(f"Cannot update table of dims {self.dims} to table of dims {np.shape(tab)}")
            raise Exception('Table update failed.')
        # Update table
        if not torch.is_tensor(tab):
            self.ground_truth_table = torch.from_numpy(tab).int().to(dtype=int32,device=self.device)
        else:
            self.ground_truth_table = tab.int().to(dtype=int32,device=self.device)
        # Update table properties
        self.update_table_properties_from_table()
        # Copy residal margins
        self.residual_margins = deepcopy(self.margins)

    def update_table_properties_from_table(self) -> None:
        # Update dimensions
        self.dims = np.asarray(np.shape(self.ground_truth_table), dtype='uint8')
        # Update margins
        self.update_margins_from_table(self.ground_truth_table)

    
    def export(self, dirpath: str = './synthetic_dummy', overwrite: bool = False) -> None:
        # Make directory if it does not exist
        makedir(dirpath)

        if self.ground_truth_table is not None:
            # Get filepath experiment filepath
            filepath = os.path.join(dirpath, 'table.txt')

            # Write experiment summaries to file
            if (not os.path.exists(filepath)) or overwrite:
                write_txt(self.ground_truth_table.cpu().detach().numpy(), filepath)

        for axis, margin in self.margins.items():
            # Get filepath experiment filepath
            filepath = os.path.join(
                dirpath, 
                f"margin_sum_axis{','.join([str(a) for a in axis])}.txt"
            )

            # Write experiment summaries to file
            if (not os.path.exists(filepath)) or overwrite:
                write_txt(margin.cpu().detach().numpy(), filepath)

        self.logger.info(f'Table successfully exported to {dirpath}')

    def infer_derivative_margin_axes(self,constrained_axes):
        updated_constrained_axes = set(constrained_axes)
        for n in range(1,self.ndims()):
            # Get all unique n-dimensional constrained axes
            unique_constrained_axes = set([ax for ax in updated_constrained_axes if len(ax) == n])
            # Get all possible axes with dimensional at least n
            all_axes = set([ax for ax in self.constraints['all_axes'] if len(ax) >= n])
            # Loop over all axes
            for ax in all_axes:
                # Loop over unique axes
                for cax in unique_constrained_axes:
                    # If unique n-dimensional constrained axes are contained entirely
                    # within a tuple of axes in the set of all axes combinations
                    # it means that the margins of the later axes can be inferred 
                    # from the former axes
                    if tuple_contained(cax, ax):
                        # Add axes to set
                        updated_constrained_axes.add(tuplize(ax))
        return updated_constrained_axes

    def map_table_solver_name_to_function(self, solver_name: str) -> Union[Callable, None]:
        
        if not solver_name.startswith('table_'):
            solver_name = 'table_' + solver_name

        # Check that such as solver is defined
        if not hasattr(self, solver_name):
            raise Exception(f'Solver {solver_name} not found')
        else:
            return getattr(self, solver_name)

    def map_margin_solver_name_to_function(self, solver_name: str) -> Union[Callable, None]:

        # Update solver name
        solver_name = 'margin_'+solver_name

        # Check that such as solver is defined
        if not hasattr(self, solver_name):
            raise Exception(f'Solver {solver_name} not found')
        # elif not solver_name.startswith(unconstrained):
            # raise Exception(f'Solver {solver_name} is in appropriate for {unconstrained} initialisation')
        else:
            return getattr(self, solver_name)

class ContingencyTableIndependenceModel(ContingencyTable):
    def __init__(self, table=None, config: Config = None, **kwargs: dict):
        # Set configuration file
        super().__init__(table, config, **kwargs)


class ContingencyTableDependenceModel(ContingencyTable):
    def __init__(self, table=None, config: Config = None, **kwargs: dict):
        # Set configuration file
        super().__init__(table, config, **kwargs)


class ContingencyTable2D(ContingencyTableIndependenceModel, ContingencyTableDependenceModel):

    def __init__(self, table=None, config: Config = None, **kwargs: dict):
        # Set configuration file
        super().__init__(table, config, **kwargs)
        # Build
        self.build(table=table, config=config, **kwargs)        

    def admissibility_debugging(self,sampler_name,table0):
        try:
            assert self.table_admissible(table0)
        except:
            constraint_table = self.constraint_table()
            table0_copy = deepcopy(table0)
            table0_copy[constraint_table>=0] = -1
            print(f'Generated table (free cells)')
            print(table0_copy)
            print('True table')
            print(self.ground_truth_table)
            if not self.table_cells_admissible(table0):
                table0_copy = deepcopy(table0)
                table0_copy[constraint_table<0] = -1
                print("Cell constraints equal")
                print(np.array_equal(table0_copy,constraint_table))
                print('Cell constraint difference')
                print(table0_copy-constraint_table)
                print("table0_fixed_cells")
                print(table0_copy)
                # print("constraints")
                # print(constraint_table)
                raise Exception(f'{sampler_name} sampler yielded a cell-inadmissible contingency table')
            elif not self.table_margins_admissible(table0):
                table0_copy = deepcopy(table0)
                table0_copy[constraint_table<0] = 0
                print("total value of constrained cells")
                print(constraint_table[constraint_table>0].sum())
                print("total value of residual margins")
                print([v.sum() for v in self.residual_margins.values()])
                print('table total')
                print(torch.ravel(self.ground_truth_table.sum()))
                for ax in sorted(self.constraints['constrained_axes']):
                    if len(ax) < self.ndims():
                        print(f"True summary statistics")
                        print(self.margins[tuplize(ax)])
                        print(f"Residual summary statistics")
                        print(self.residual_margins[tuplize(ax)])
                        print(f"Fixed cell summary statistics")
                        print(table0_copy.sum(dim=tuplize(ax),keepdims=True).flatten())
                        print(f'Current summary statistics')
                        print(table0.sum(dim=tuplize(ax),keepdims=True).flatten())
                raise Exception(f'{sampler_name} sampler yielded a margin-inadmissible contingency table')
            else:
                raise Exception('Unrecognized error encountered.')
    
    def update_and_map_margin_constraints_to_distribution_name(self):
        # TODO: Generalise this

        # Get all unique axes that appear as singletons
        unique_axes = set([ax for ax in self.constraints['constrained_axes'] if len(ax) == 1])
        
        if len(self.constraints['constrained_axes']) == 0:
            return 'poisson'
        elif len(unique_axes) == 0:
            return 'multinomial'
        elif len(unique_axes) == 1:
            return 'product_multinomial'
        elif len(unique_axes) == 2:
            return 'fishers_hypergeometric'
        else:
            raise ValueError(f"Distribution not found for constraints {str(self.constraints['constrained_axes'])}")
    
    def margin_import(self, axis, intensity: list = None):

        margin0 = None
        # Read initialised table
        if str_in_list('margin0', self.config.settings['mcmc']['contingency_table'].keys()):
            initialisation = self.config.settings['mcmc']['contingency_table']['margin0']
            # If it is a path read file
            if isinstance(initialisation, str):
                # Extract filepath
                if os.path.isfile(os.path.join(self.config.settings['inputs']['dataset'], initialisation)):
                    margin0 = np.loadtxt(
                        os.path.join(
                            self.config.settings['inputs']['dataset'],
                            initialisation
                        ), 
                        dtype='float32'
                    )
            # If it is an array/list make sure it has the
            elif isinstance(initialisation, (list, np.ndarray, np.generic)) and \
                    np.any([(len(initialisation) == dim) for dim in self.dims]):
                margin0 = initialisation
            else:
                self.logger.error(f"Initial table margin {initialisation} is neither a list nor a valid filepath.")
            
        else:
            self.logger.error(f"No margin 0 provided in config.")
        
        if margin0 is None:
            raise Exception('margin 0 could not be found.')
        else:
            return margin0
        
    def margin_multinomial(self, axis, intensity: list = None):
        # This is a 1-dimensional margin sampling scheme 
        # That is why all margins sum to the grand total 
        margin = np.array([0],dtype='int32')
        # This is to make sure I sample at least once
        firstIteration = True
        # Create uniform probs if none are provided
        if intensity is None:
            intensity = np.repeat(1./np.prod(self.dims), np.prod(self.dims)).reshape(self.dims)
        # Sum probabilities to one in the case of multinomial sampling
        if len(axis) < self.ndims():
            intensity /= intensity.sum()
        
        while (not self.margin_sparse_admissible(margin)) or (firstIteration):
            if len(axis) < self.ndims():
                margin = np.random.multinomial(
                    n=self.margins[tuplize(range(self.ndims()))], 
                    pvals=intensity.sum(dim=axis)
                )
            else:
                # This is sampling the grand total
                margin = np.array([np.random.poisson(intensity.sum(dim=axis))])
            # The loop can now end
            firstIteration = False
        return margin
    
    def table_monte_carlo_sample(self,intensity:list=None, margins: dict = {}, **__) -> Union[np.ndarray, None]:
        # Update margins
        if margins is not None:
            self.update_margins(margins)

        # Fix random seed

        # Define axes
        constrained_axis1 = (1,)
        constrained_axis2 = (0,)
        
        # Initialise table to zero
        table0 = torch.zeros(tuple(self.dims), dtype=int32)

        # Get N (sum or row or column sums)
        N = self.residual_margins[tuplize(range(self.ndims()))]

        # Generate permutation of N items
        permutation = torch.randperm(N) + torch.tensor(1,dtype=int32)

        # Loop through first axis
        for i in range(self.dims[constrained_axis2]):
            # Look at first r_i items
            if i == 0:
                permutation_subset = permutation[0 : (self.residual_margins[constrained_axis1][i])]
            else:
                permutation_subset = permutation[torch.sum(self.residual_margins[constrained_axis1][:(i)]):
                                                torch.sum(self.residual_margins[constrained_axis1][:(i+1)])]
                    
            # Loop through columns
            for j in range(self.dims[constrained_axis1]):
                # Create index appropriately
                query_index = torch.zeros(tuple(self.ndims()), dtype=uint8)
                query_index[constrained_axis2] = i
                query_index[constrained_axis1] = j
                query_index = tuplize(query_index)

                if query_index in self.constraints['cells']:
                    # Initialise cell at constraint
                    table0[query_index] = self.ground_truth_table[query_index]
                else:
                    # Count number of entries between c_1+...+c_{j-1}+1 and c_1+...+c_j
                    if j == 0:
                        table0[query_index] = torch.int32(
                            ((1 <= permutation_subset) & \
                             (permutation_subset <= self.residual_margins[constrained_axis2][j])).sum()
                        )
                    else:
                        table0[query_index] = torch.int32(
                            (((torch.sum(self.residual_margins[constrained_axis2][:j])+1) <= permutation_subset) & \
                            (permutation_subset <= torch.sum(self.residual_margins[constrained_axis2][:(j+1)]))).sum()
                        )

        self.admissibility_debugging('Monte Carlo',table0)

        return table0.to(device=self.device,dtype=int32)
    
    def table_import(self,intensity:list=None, margins: dict = {}, **__) -> Union[np.ndarray, None]:
        # Read initial table
        table0 = None
        if str_in_list('table0', self.config.settings['mcmc']['contingency_table'].keys()):
            initialisation = self.config.settings['mcmc']['contingency_table']['table0']
            # If it is a path read file
            if isinstance(initialisation, str):
                # Extract filepath
                if os.path.isfile(os.path.join(self.config.settings['inputs']['dataset'], initialisation)):
                    tab0 = np.loadtxt(
                        os.path.join(
                            self.config.settings['inputs']['dataset'], 
                            initialisation
                        ), 
                        dtype='float32'
                    )
            # If it is an array/list make sure it has the
            elif isinstance(initialisation, (list, np.ndarray, np.generic)) and \
                    (np.shape(initialisation) == self.dims):
                table0 = initialisation
            else:
                self.logger.warning(
                    f"Initial table {initialisation} is neither a list nor a valid filepath.")
        else:
            self.logger.error(f"No table 0 provided in config.")
        
        if table0 is None:
            raise Exception('Table 0 could not be found.')
        else:
            return table0.to(device=self.device,dtype=int32)

    def table_random_sample(self,intensity:list=None, margins: dict = {}, **__) -> Union[np.ndarray, None]:

        try:
            assert len(self.constraints['constrained_axes']) == 0
        except:
            raise Exception ('table_random_sample is only used by tables with no margin constraints')
        # Fix random seed
        
        # Initialise table to zero
        table0 = torch.zeros(tuple(self.dims), dtype=int32)
        min_cell_value,max_cell_value = 0,1
        for cell in self.constraints['cells']:
            table0[cell] = torch.int32(self.ground_truth_table[cell])
            min_cell_value = min(min_cell_value,table0[cell])
            max_cell_value = max(max_cell_value,table0[cell])
        
        # Define support to sample from
        cell_value_range = [min(min_cell_value,1), max(max_cell_value,100)]
        # Update remaining table cells
        for cell in self.cells:
            # Sample value randomly
            table0[cell] = torch.randint(low=cell_value_range[0],high=cell_value_range[1],size=(1,),dtype=int32)

        self.admissibility_debugging('Random solution',table0)
        return table0.to(device=self.device,dtype=int32)

    def table_direct_sampling(self, intensity:list=None, margins:dict = {}, **kwargs):
        # Get direct sampling proposal
        direct_sampling_proposal = getattr(kwargs['ct_mcmc'],f"direct_sampling_proposal_{self.ndims()}way_table")
        table0,_,_,_,_ = direct_sampling_proposal(
            table_prev=None,
            log_intensity=torch.log(intensity)
        )
        return table0.to(device=self.device,dtype=int32)

    def table_maximum_entropy_solution(self, intensity:list=None, margins: dict = {}, **__) -> Union[np.ndarray, None]:
        '''
        This is the solution of the maximum entropy in the non-integer case
        X_ij = (r_i*c_j) / T
        '''
        # Update margins
        if margins is not None:
            self.update_margins(margins)

        # Fix random seed

        # Initialise table to zero
        table0 = torch.zeros(tuple(self.dims), dtype=int32)
        table0_learned = torch.zeros(tuple(table0.shape),dtype=tbool)

        # Non fixed (free) indices
        free_cells = np.array(self.cells)
        free_indices = [ free_cells[:,i] for i in range(self.ndims()) ]

        table0[free_indices] = self.ground_truth_table[free_indices]
        table0_learned[free_indices] = True
        # Create masked array
        table0 = masked_tensor(table0, table0_learned)

        # Define axes
        constrained_axis1 = (1,)
        constrained_axis2 = (0,)

        # Keep a copy of row and column sums
        rowsums0 = self.residual_margins[tuplize(constrained_axis1)].detach().clone().numpy()
        colsums0 = self.residual_margins[tuplize(constrained_axis2)].detach().clone().numpy()

        # Minimum residual table
        min_residual = np.zeros(tuple(self.dims),dtype='int32')
        min_residual_mask = np.zeros(tuple(self.dims),dtype=bool)

        min_residual[free_indices] = np.minimum(rowsums0[free_cells[:,0]],colsums0[free_cells[:,1]])
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                if (i,j) in self.constraints['cells']:
                    min_residual[i,j] = 0
                    min_residual_mask[i,j] = True
                else:
                    min_residual[i,j] = int(min(rowsums0[i], colsums0[j]))
                    if min_residual[i,j] == 0:
                        min_residual_mask[i,j] = True
        min_residual = np.ma.array(min_residual, mask=min_residual_mask)

        # Get N (sum or row or column sums)
        N = self.margins[tuplize(range(self.ndims()))]
        counter = 0

        while not self.table_margins_admissible(table0):

            # In first iteration
            if counter == 0:
                # Loop through every table cell
                for cell in self.cells:
                    # Use stochastic rounding to round up maximum entropy solution
                    # Max-entropy solution is to set all cells (i,j) to value c_j*r_i/N
                    max_ent_solution = (rowsums0[cell[constrained_axis2]]*colsums0[cell[constrained_axis1]]) / N

                    # Determine amount to set by stochastically rounding it up or down
                    # + int(add_delta)*delta)
                    amount = int(np.floor(max_ent_solution))

                    # Update table
                    table0[cell] += np.int32(amount)

                    # Update row,column sums
                    rowsums0[cell[constrained_axis2]] -= amount
                    colsums0[cell[constrained_axis1]] -= amount
                    min_residual[cell] = int(min(rowsums0[cell[constrained_axis2]], colsums0[cell[constrained_axis1]]))

                    # Assert that residual col and rowsums are non-negative
                    assert rowsums0[cell[constrained_axis2]] >= 0
                    assert colsums0[cell[constrained_axis1]] >= 0

                    if rowsums0[cell[constrained_axis2]] == 0:
                        table0.mask[cell[constrained_axis2],:] = True
                        min_residual.mask[cell[constrained_axis2],:] = True
                    if colsums0[cell[constrained_axis1]] == 0:
                        table0.mask[:,cell[constrained_axis1]] = True
                        min_residual.mask[:,cell[constrained_axis1]] = True

            else:
                k = 0
                # Fill in residual rowsums or colsums
                while (any(rowsums0 > 0) or any(colsums0 > 0)):
                    
                    # Order min residual values
                    sorted_indices = np.ma.argsort(
                                min_residual,
                                dim=None,
                                kind='quicksort', 
                                fill_value=np.int32(self.margins[tuplize(range(self.ndims()))]+1)
                    )
                    # sorted_indices = np.unravel_index(sorted_indices,table0.shape)
                    # Get indices of unmasked elements
                    unmasked_indices = np.ma.where(~table0.mask)
                    # unmasked_indices = np.asarray(list(zip(*unmasked_indices)))
                    # Create index in same dims ars sorted_indices
                    unmasked_indices = np.ravel_multi_index(unmasked_indices,table0.shape)
                    # Maintain index order and get only unmasked indices
                    sorted_unmasked_indices_2d = np.unravel_index(
                        sorted_indices[np.isin(sorted_indices, unmasked_indices[np.isin(unmasked_indices, sorted_indices)])], table0.shape
                    )
                    # ordered_vals = []
                    # for j in range(np.shape(sorted_unmasked_indices_2d)[1]):
                    #     ordered_cell = (sorted_unmasked_indices_2d[0][j],sorted_unmasked_indices_2d[1][j])
                    #     ordered_vals.append(min_residual[ordered_cell])
                    # print(ordered_vals)
                    # sys.exit()


                    try:
                        smallest_value_cell = (sorted_unmasked_indices_2d[0][k], sorted_unmasked_indices_2d[1][k])
                    except:
                        print(np.shape(sorted_unmasked_indices_2d))
                        print(k)
                        print(min_residual.data)
                        print(rowsums0)
                        print(colsums0)
                        raise Exception()
                    # print(smallest_value_cell)
                    # print(table0[smallest_value_cell])
                    # print(table0[smallest_value_cell])
                    # sys.exit()
                    assert not table0.mask[smallest_value_cell]

                    # Try to add maximum amount possible
                    amount = int(min(rowsums0[smallest_value_cell[0]], colsums0[smallest_value_cell[1]]))
                    table0[smallest_value_cell] += np.int32(amount)
                    # Update row,column sums
                    rowsums0[smallest_value_cell[0]] -= amount
                    colsums0[smallest_value_cell[1]] -= amount

                    min_residual[cell] = int(min(rowsums0[smallest_value_cell[0]], colsums0[smallest_value_cell[1]]))
                    # Get query index
                    # query_index = [None] * self.ndims()
                    # if smallest_value_cell[0] == 20 or smallest_value_cell[1] == 6:
                        # print('smallest cell val',smallest_value_cell,'amount',amount)
                        # print('table val',table0[smallest_value_cell])
                        # print('rowsums',rowsums0[smallest_value_cell[0]],'colsums',colsums0[smallest_value_cell[1]])

                    if rowsums0[smallest_value_cell[0]] <= 0:
                        # print('To be learned',table0[smallest_value_cell])
                        # print('Deactivating row',smallest_value_cell[0])
                        # print(table0.mask[smallest_value_cell],table0[smallest_value_cell])
                        # Update deactivated (fixed) cells
                        table0.mask[smallest_value_cell[0],:] = True
                        min_residual.mask[smallest_value_cell[0],:] = True
                        
                    if colsums0[smallest_value_cell[1]] <= 0:
                        # print('To be learned',table0[smallest_value_cell])
                        # print('Deactivating col',smallest_value_cell[1])
                        # print(table0.mask[smallest_value_cell],table0[smallest_value_cell])
                        # Update deactivated (fixed) cells
                        table0.mask[:,smallest_value_cell[1]] = True
                        min_residual.mask[:,smallest_value_cell[1]] = True

            # Increment counter
            counter += 1

        self.admissibility_debugging('Maximum entropy solution',table0)

        return table0.to(device=self.device,dtype=int32)

    def table_iterative_residual_filling_solution(self,intensity:list=None, margins: dict = {}, **__) -> Union[np.ndarray, None]:

        # Update margins
        if margins is not None:
            self.update_margins(margins)
            
        # Initialise table to zero
        table0 = np.zeros(self.dims, dtype='int32')
        table0_learned = np.zeros(table0.shape,dtype=bool)
        for cell in self.constraints['cells']:
            table0[cell] = self.ground_truth_table[cell]
            table0_learned[cell] = True
        # Create masked array
        table0 = np.ma.array(table0, mask=table0_learned)

        # Define axes
        constrained_axis1 = (1,)
        constrained_axis2 = (0,)
            
        # Keep a copy of row and column sums
        rowsums0 = np.zeros(self.dims[constrained_axis2])
        rowsums0[:] = self.residual_margins[tuplize(constrained_axis1)]
        colsums0 = np.zeros(self.dims[constrained_axis1])
        colsums0[:] = self.residual_margins[tuplize(constrained_axis2)]

        # Minimum residual table
        min_residual = np.zeros(self.dims,dtype='int32')
        min_residual_mask = np.zeros(self.dims,dtype=bool)
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                if (i,j) in self.constraints['cells']:
                    min_residual[i,j] = 0
                    min_residual_mask[i,j] = True
                else:
                    min_residual[i,j] = int(min(rowsums0[i], colsums0[j]))
                    if min_residual[i,j] == 0:
                        min_residual_mask[i,j] = True
        min_residual = np.ma.array(min_residual, mask=min_residual_mask)

        while not self.table_margins_admissible(table0.data):

            # Order min residual values
            min_residual_index_flat = np.ma.argmin(
                        min_residual,
                        dim=None,
                        fill_value=np.int32(self.margins[tuplize(range(self.ndims()))]+1)
            )
            min_residual_index = np.unravel_index(min_residual_index_flat,shape=self.dims)
            smallest_value_cell = tuplize(min_residual_index)
            
            assert not table0.mask[smallest_value_cell]
        
            # Try to add maximum amount possible
            amount = int(min(rowsums0[smallest_value_cell[0]], colsums0[smallest_value_cell[1]]))
            table0[smallest_value_cell] += np.int32(amount)
            # Update row,column sums
            rowsums0[smallest_value_cell[0]] -= amount
            colsums0[smallest_value_cell[1]] -= amount
            for i in range(self.dims[0]):
                if rowsums0[i] <= 0:
                    table0.mask[i,:] = True
                    min_residual.mask[i,:] = True
            for j in range(self.dims[1]):
                if colsums0[j] <= 0:
                    table0.mask[:,j] = True
                    min_residual.mask[:,j] = True
            # Update residual
            for i in range(self.dims[0]):
                for j in range(self.dims[1]):
                    # Update minimum residual table
                    if not min_residual.mask[i,j]:
                        min_residual[i,j] = int(min(rowsums0[i], colsums0[j]))
                    if min_residual[i,j] == 0:
                        min_residual.mask[i,j] = True

        self.admissibility_debugging('Iterative residual filling solution',table0.data)

        return table0.to(device=self.device,dtype=int32)

    def table_iterative_uniform_residual_filling_solution(self,intensity:list=None, margins: dict = {}, **__) -> Union[np.ndarray, None]:

        # Update margins
        if margins is not None:
            self.update_margins(margins)

        # Fix random seed
        
        # Initialise table to zero
        table0 = np.zeros(self.dims, dtype='int32')
        table0_learned = np.zeros(table0.shape,dtype=bool)
        for cell in self.constraints['cells']:
            table0[cell] = self.ground_truth_table[cell]
            table0_learned[cell] = True
        # Create masked array
        table0 = np.ma.array(table0, mask=table0_learned)

        # Define axes
        constrained_axis1 = 1
        constrained_axis2 = 0
            
        # Keep a copy of row and column sums
        rowsums0 = np.zeros(self.dims[constrained_axis2])
        rowsums0[:] = self.residual_margins[tuplize(constrained_axis1)]
        colsums0 = np.zeros(self.dims[constrained_axis1])
        colsums0[:] = self.residual_margins[tuplize(constrained_axis2)]

        for cell in self.cells:
            # Initalise table values with minimum possible row or col sum or N/IJ
            amount = int(min(np.floor(np.sum(rowsums0)/np.sum(table0.mask)),rowsums0[cell[0]],colsums0[cell[1]]))

            # Update table
            table0[cell] += np.int32(amount)

            # Update row,column sums
            # This method allows for negative residuals
            rowsums0[cell[0]] -= amount
            colsums0[cell[1]] -= amount

            if table0[cell] <= 0:
                table0.mask[cell] = True
            if rowsums0[cell[0]] <= 0:
                table0.mask[cell[0],:] = True
            if colsums0[cell[1]] <= 0:
                table0.mask[:,cell[1]] = True

        while not self.table_margins_admissible(table0.data):

            # IF no solution found rerun
            if np.sum(table0.mask) == np.prod(self.ndims()):
                self.table_iterative_uniform_residual_filling_solution(margins)
            
            # Get indices of unmasked elements
            unmasked_indices = np.ma.where(~table0.mask)
            # Create index in same dims ars sorted_indices
            # unmasked_indices = np.ravel_multi_index(unmasked_indices,table0.shape)
            # Maintain index order and get only unmasked indices
            # print(table0)
            # print('unmasked_indices',unmasked_indices)
            cell_index = np.random.randint(len(unmasked_indices))
            # print(cell_index,list(zip(*unmasked_indices)))
            try:
                cell = list(zip(*unmasked_indices))[cell_index]
            except:
                self.table_iterative_uniform_residual_filling_solution(margins)
            # try:
            # assert not table0.mask[cell]
            # except:
            #     print('cell',cell, 'out of index')
            #     print(table0)
            # print('\n')
            # Try to add maximum amount possible
            amount = int(min(
                rowsums0[cell[0]], 
                colsums0[cell[1]]
            ))
            table0[cell] += np.int32(amount)
            # Update row,column sums
            rowsums0[cell[0]] -= amount
            colsums0[cell[1]] -= amount
            
            if table0[cell] <= 0:
                table0.mask[cell] = True
            if rowsums0[cell[0]] <= 0:
                table0.mask[cell[0],:] = True           
            if colsums0[cell[1]] <= 0:
                table0.mask[:,cell[1]] = True

        self.admissibility_debugging('Iterative uniform residual filling solution',table0.data)

        return table0.data.to(device=self.device,dtype=int32)
    