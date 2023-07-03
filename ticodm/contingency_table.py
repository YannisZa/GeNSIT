import os
import sys
import logging
import numpy as np

from copy import deepcopy
from pandas import read_csv
from pandas import DataFrame
from itertools import product
from tabulate import tabulate
from typing import List, Union, Callable

from ticodm.config import Config
from ticodm.global_variables import *
from ticodm.utils import str_in_list, write_txt, extract_config_parameters, makedir, read_json, tuplize, flatten, tuple_contained, depth
from ticodm.math_utils import logsumexp, powerset

# -> Union[ContingencyTable,None]:
def instantiate_ct(table, config: Config, disable_logger: bool = False, **kwargs):
    if hasattr(sys.modules[__name__], config.settings['inputs']['contingency_table']['ct_type']):
        return getattr(sys.modules[__name__], config.settings['inputs']['contingency_table']['ct_type'])(
                table=table, 
                config=config, 
                disable_logger=disable_logger, 
                **kwargs
        )
    else:
        raise Exception(
            f"Input class {config.settings['inputs']['contingency_table']['ct_type']} not found")


class ContingencyTable(object):

    def __init__(self, table=None, config: Config = None, disable_logger: bool = False, **kwargs: dict):
        # Import logger
        self.logger = logging.getLogger(__name__)
        self.logger.disabled = disable_logger
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)
        # Config
        self.config = None
        # Update Markov basis class
        self.markov_basis_class = None
        # Store flag for whether to allow sparse margins or not
        self.sparse_margins = False
        # Random seed
        self.seed = None
        # Table
        self.table = None
        # margins
        self.margins = {}
        self.residual_margins = {}

    def build(self, table, config, **kwargs):
        # Build contingency table
        if config is not None:
            # Set configuration file
            self.config = extract_config_parameters(config,
                                                    {"seed": "",
                                                     "inputs": {
                                                        "contingency_table": "", 
                                                        "seed": "", 
                                                        "n_workers": "", 
                                                        "n_threads": "", 
                                                        "directory": "", 
                                                        "spatial_interaction_model": {"sim_type": "","sim_name":""}
                                                        },
                                                        "mcmc": {"table_inference": "", "N": "", "contingency_table": ""},
                                                        "outputs": ""
                                                     })
            # Store flag for whether to allow sparse margins or not
            if str_in_list('sparse_margins', self.config.settings['inputs']['contingency_table'].keys()):
                self.sparse_margins = self.config.settings['inputs']['contingency_table']['sparse_margins']
            if str_in_list('seed', self.config.settings['inputs'].keys()):
                self.seed = int(self.config.settings['inputs']['seed'])
            
            # Read tabular data
            self.import_tabular_data()

        elif table is not None:
            # Read table and dimensions
            self.table = table.astype('int32')
            self.dims = np.asarray(np.shape(self.table), dtype='uint8')
            # Read seed
            if str_in_list('seed', kwargs.keys()):
                self.seed = int(kwargs['seed'])
            # Read margin sparsity
            if str_in_list('sparse_margins', kwargs.keys()):
                self.sparse_margins = bool(kwargs['sparse_margins'])
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
        if str_in_list('cells',self.constraints.keys()) and len(self.constraints['cells']) > 0:
            # Update cell constraints if there are any deterministic solutions
            # based on the cell and margin constraints provided
            self.update_cell_constraints_deterministically()
            # Update residual margins
            self.update_residual_margins_from_cell_constraints(self.constraints['cells'])

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
        return np.concatenate([self.table_axis_margin_summary_statistic(tab=tab,ax=ax) for ax in self.constraints['all_axes']], axis=0)
    
    def table_axis_margin_summary_statistic(self, tab: np.ndarray, ax:int = None) -> List:
        return np.sum(tab, axis=tuplize(ax), keepdims=True, dtype='int32').flatten()

    # Compute margin summary statistics for given function
    def table_constrained_margin_summary_statistic(self, tab: np.ndarray) -> List:
        if len(self.constraints['constrained_axes']) > 0:
            return np.concatenate([np.sum(tab, axis=tuplize(ax), keepdims=True, dtype='int32').flatten() for ax in sorted(self.constraints['constrained_axes'])], axis=0)
        else:
            return np.array([])

    def table_unconstrained_margin_summary_statistic(self, tab: np.ndarray) -> List:
        if len(self.constraints['unconstrained_axes']) > 0:
            return np.concatenate([np.sum(tab, axis=tuplize(ax), keepdims=True, dtype='int32').flatten() for ax in sorted(self.constraints['unconstrained_axes'])], axis=0)
        else:
            return np.array([])

    # Compute cell summary statistics for given function
    def table_cell_constraint_summary_statistic(self, tab: np.ndarray, cells: list=None) -> List:
        if cells is None:
            return np.asarray([tab[cell] for cell in sorted(self.constraints['cells'])], dtype='int32')
        else:
            return np.asarray([tab[cell] for cell in sorted(self.constraints['cells'])], dtype='int32')

    # Check if margin constraints are satisfied
    def table_margins_admissible(self, tab) -> bool:
        if len(self.constraints['constrained_axes']) > 0:
            return np.array_equal(
                self.table_constrained_margin_summary_statistic(tab),
                np.concatenate([self.margins[tuplize(ax)] for ax in sorted(self.constraints['constrained_axes'])], axis=0)
            )
        else:
            return True

    # Check if cell constraints are satisfied
    def table_cells_admissible(self, tab) -> bool:
        return np.all([self.table[cell] == tab[cell] for cell in sorted(self.constraints['cells'])])

    # Check if function (or table equivalently) is admissible
    def table_admissible(self, tab) -> bool:
        return self.table_margins_admissible(tab) and self.table_cells_admissible(tab)
    
    def table_sparse_admissible(self, tab) -> bool:
        return np.all(self.table_unconstrained_margin_summary_statistic(tab) > 0) if not self.sparse_margins else True
    
    def margin_sparse_admissible(self, margin) -> bool:
        return np.all(margin > 0) if not self.sparse_margins else True

    # Checks if function f proposed is positive
    def table_nonnegative(self, tab) -> bool:
        return not np.any(tab < 0)

    def table_positive(self, tab) -> bool:
        return not np.any(tab <= 0)

    def table_difference_histogram(self, tab: np.ndarray, tab0: np.ndarray):
        # Take difference of tables
        difference = tab - tab0
        # Values classification
        values = {"+": np.count_nonzero(difference > 0), "-": np.count_nonzero(
            difference < 0), "0": np.count_nonzero(difference == 0)}
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
        if self.table is not None:
            return ','.join([str(x) for x in self.table])
        else:
            return 'Empty Table'

    def __repr__(self):
        return f"{'x'.join([str(dim) for dim in self.dims])} ContingencyTable{self.ndims()}(Config)"

    def __len__(self):
        return np.prod([dim for dim in self.dims if dim > 0])
    
    def ndims(self):
        return np.sum([1 for dim in self.dims if dim > 1],dtype='int32')
    
    def reset(self, table, config, **kwargs) -> None:
        # Rebuild table
        self.build(table=table, config=config, **kwargs)
    
    def constraint_table(self,with_margins:bool=False):
        if self.table is not None:
            table_constraints = -np.ones(self.table.shape)
            table_mask = np.zeros(self.table.shape,dtype=bool)
            for cell in self.constraints['cells']:
                table_constraints[cell] = self.table[cell]
                table_mask[cell] = True
            table_constraints = table_constraints.astype('int32')
            
            # Printing margins for visualisation purposes only
            if with_margins:
                assert self.ndims() == 2

                # Flip table for viz purposes (clear visibility)
                row_margin,col_margin = (0,),(1,)
                if self.table.shape[1] >= self.table.shape[0]:
                    table_constraints = table_constraints.T
                    row_margin,col_margin = (1,),(0,)
                
                # Create pandas dataframe from constraints
                table_constraints_ct = DataFrame(table_constraints)

                # Add margins to the contingency table
                table_constraints_ct.loc['Total',:] = self.residual_margins[row_margin].astype('int32')
                table_constraints_ct['Total'] = list(self.residual_margins[col_margin].astype('int32')) + list(self.residual_margins[(0,1)].astype('int32'))
                
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
        if str_in_list('table', self.config.settings['inputs']['contingency_table']['import'].keys()) and \
                bool(self.config.settings['inputs']['contingency_table']['import']['table']) and \
                os.path.isfile(os.path.join(self.config.settings['inputs']['dataset'], self.config.settings['inputs']['contingency_table']['import']['table'])):
            
            table_filepath = os.path.join(
                self.config.settings['inputs']['dataset'], 
                self.config.settings['inputs']['contingency_table']['import']['table']
            )
            if self.config.settings['inputs']['contingency_table']['import']['table'].endswith('.csv'):
                self.table = read_csv(table_filepath, index_col=0, header=0)
                # Convert to numpy
                self.table = self.table.values
            elif self.config.settings['inputs']['contingency_table']['import']['table'].endswith('.txt'):
                self.table = np.loadtxt(
                    table_filepath, dtype='float32').astype('int32')
            else:
                raise Exception(
                    f"Extension {table_filepath.split('.')[1]} cannot be imported")

            # Check that all entries are non-negative
            try:
                assert self.check_table_validity(self.table)
            except:
                self.logger.error(self.table.to_string())
                raise Exception(
                    "Imported table contains negative values! Import aborted...")

            # Cast to int
            self.table = self.table.astype('int32')
            # Update table properties
            self.update_table_properties_from_table()
            # Copy residal margins
            self.residual_margins = deepcopy(self.margins)
        else:
            self.logger.warning('Valid contingency table was not provided. One will be randomly generated.')
            if hasattr(self,'table') and self.table is not None:
                self.dims = np.asarray(np.shape(self.table), dtype='uint8')
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
                    assert np.sum(
                        [1 for dim in np.shape(margin) if dim > 1]) == 1
                except:
                    raise Exception(
                        f"margin for axis {axis} is not unidimensional.")
                # Assert all margin axis match dimensions
                if np.sum(self.dims) > 0:
                    try:
                        assert len(margin) == self.dims[axis]
                    except:
                        self.logger.error(
                            f"margin for axis {axis} has dim {len(margin)} and not {self.dims[axis]}.")
                        raise Exception('Imported inconsistent margins.')
                else:
                    self.dims[axis] = len(margin)
                # Assert all margin sums are the same
                try:
                    assert np.array_equal(
                                np.sum(margin,keepdims=True,dtype='int32').flatten(),
                                self.margins[tuplize(range(self.ndims()))]
                    )
                except:
                    self.logger.error(
                        f"margin for axis {axis} has total {np.sum(margin.ravel(),keepdims=True).flatten()} and not {self.margins[tuplize(range(self.ndims()))]}.")
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
                        self.margins[tuplize(axis)] = np.random.randint(
                            low=1,
                            high=None
                        )
                    else:
                        dim = self.dims[self.axes_complement(axis,same_length=True)[0]]
                        self.margins[tuplize(axis)] = np.random.multinomial(
                            n=self.margins[tuplize(range(self.ndims()))], 
                            pvals=np.repeat(1/dim, dim)
                        )
                        self.logger.debug(f"Randomly initialising margin for axis", tuplize(axis))
            # Copy residal margins
            self.residual_margins = deepcopy(self.margins)
            # Initialise table
            self.table = -np.ones(self.dims)
            # Import and update cells of table
            self.import_cells()
            # Initialise constraints
            self.constraints = {}
            self.constraints['constrained_axes'] = []
            self.constraints['all_axes'] = sorted(list(set(powerset(range(self.ndims())))))
            self.constraints['unconstrained_axes'] = sorted(list(set(powerset(range(self.ndims())))))

    def import_margins(self) -> None:
        if str_in_list('margins', self.config.settings['inputs']['contingency_table']['import'].keys()):
            for axis, margin_filename in enumerate(self.config.settings['inputs']['contingency_table']['import']['margins']):
                # Make sure that imported filename is not empty
                if len(margin_filename) > 0:
                    margin_filepath = os.path.join(
                        self.config.settings['inputs']['dataset'], margin_filename)
                    if os.path.isfile(margin_filepath):
                        # Import margin
                        self.margins[tuplize(axis)] = np.loadtxt(
                            margin_filepath, dtype='int32')
                        # Check to see see that they are all positive
                        if (self.margins[tuplize(axis)] <= 0).any():
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
        if str_in_list('cell_values', self.config.settings['inputs']['contingency_table']['import'].keys()):
            cell_filename = os.path.join(
                self.config.settings['inputs']['dataset'],
                self.config.settings['inputs']['contingency_table']['import']['cell_values']
            )
            if os.path.isfile(cell_filename):
                # Import all cells
                cells = read_json(cell_filename)
                # Check to see see that they are all positive
                if (cells.values() < 0).any():
                    self.logger.error(
                        f'Cell values{cells.values()} are not strictly positive')
                # Check that no cells exceed any of the margins
                for cell, value in cells.items():
                    try:
                        assert len(cell) == self.ndims() and cell < self.dims
                    except:
                        self.logger.error(
                            f"Cell has length {len(cell)}. The number of table dims are {self.ndims()}")
                        self.logger.error(
                            f"Cell is equal to {cell}. The cell bounds are {self.dims}")
                    for ax in cell:
                        if tuplize(ax) in self.margins.keys():
                            try:
                                if self.sparse_margins:
                                    assert self.margins[tuplize(ax)][cell[ax]] >= value
                                else:
                                    assert self.margins[tuplize(ax)][cell[ax]] > value
                            except:
                                self.logger.error(
                                    f"margin for ax = {','.join([str(a) for a in ax])} is less than specified cell value {value}")
                                raise Exception('Cannot import cells.')
                    # Update table
                    self.table[cell] = value
                else:
                    raise Exception(
                        f"Cell values not found in {cell_filename}.")
        else:
            self.logger.debug(f"Cells file not provided")
            # raise Exception(f"Importing cells failed.")

    def read_constraints(self,**kwargs):
        self.constraints = {}
        constrained_axes,cell_constraints = None, None
        # margin and cell constraints
        ## Reading config if provided

        if self.config is not None:
            if str_in_list('constraints', self.config.settings['inputs']['contingency_table'].keys()):
                if str_in_list('axes', self.config.settings['inputs']['contingency_table']['constraints'].keys()):
                    constrained_axes = [tuplize(ax) for ax in self.config.settings['inputs']['contingency_table']['constraints']['axes'] if len(ax) > 0]
                if str_in_list('cells', self.config.settings['inputs']['contingency_table']['constraints'].keys()):
                    if isinstance(self.config.settings['inputs']['contingency_table']['constraints']['cells'],str):
                        cell_constraints = os.path.join(
                            self.config.settings['inputs']['dataset'],
                            self.config.settings['inputs']['contingency_table']['constraints']['cells']
                        )
                    elif isinstance(self.config.settings['inputs']['contingency_table']['constraints']['cells'],(list,np.ndarray)):
                        cell_constraints = self.config.settings['inputs']['contingency_table']['constraints']['cells']
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
                    # Find cell constraints
                    self.constraints['cells'] = np.loadtxt(cell_constraints, dtype='uint8')
                    self.constraints['cells'] = self.constraints['cells'].reshape(np.size(self.constraints['cells'])//2, 2).tolist()
                    # Remove invalid cells (i.e. cells than do not have the right number of dims or are out of bounds)
                    self.constraints['cells'] = sorted([tuplize(c) for c in self.constraints['cells'] if (len(c) == self.ndims() and (c < self.dims).all())])
                    self.logger.info(f"Cell constraints {','.join([str(c) for c in self.constraints['cells']])} provided")
                else:
                    self.constraints['cells'] = []
            elif isinstance(cell_constraints,(list,np.ndarray)):
                self.constraints['cells'] = [tuplize(ax) for ax in cell_constraints]
            else:
                print('cell constraints')
                print(cell_constraints)
                self.logger.error(f"Cell constraints of type {type(cell_constraints)}")
                raise Exception('Cell constraints could not be recognised')
        else:
            self.constraints['cells'] = []

    def update_table_type_and_markov_basis_class(self):
        # Check if there are is at lest one active cell
        try:
            assert len(self.cells) > 0
        except:
            # Groudn truth
            print('Ground truth table')
            print(self.table)
            print('Constraint table')
            print(self.constraint_table())
            self.logger.error('No activate cells found. Table can be deterministically elicited based on provided constraints.')
            raise Exception(f"Table inference is halted.")

        # Count number of non zero dims
        if self.ndims() == 0:
            self.ct_type = 'empty_table'
            self.markov_basis_class = None
        elif self.ndims() == 1:
            self.markov_basis_class = f'MarkovBasis{self.ndims()}DTable'
            if isinstance(self, ContingencyTableDependenceModel):
                self.ct_type = 'one_way_contingency_table_dependence_model'
            else:
                self.ct_type = 'one_way_contingency_table_independence_model'
        elif self.ndims() == 2:
            self.markov_basis_class = f'MarkovBasis{self.ndims()}DTable'
            if isinstance(self, ContingencyTableDependenceModel):
                self.ct_type = 'two_way_contingency_table_dependence_model'
            else:
                self.ct_type = 'two_way_contingency_table_independence_model'
        else:
            self.logger.error(f'No implementation found for ndims = {self.ndims()}')
            raise Exception('Table type and markov basis class could not be identified.')
        
    def update_margins(self, margins: dict = {}):
        for ax, margin in margins.items():
            # Update margins
            self.margins[tuplize(ax)] = np.array(margin, dtype='int32')
            self.residual_margins[tuplize(ax)] = np.array(margin, dtype='int32')

    def update_residual_margins_from_cell_constraints(self, constrained_cells):
        for ax in self.constraints['constrained_axes']:
            ax_complement = self.axes_complement([ax])
            for cell in constrained_cells:
                cell_list = np.asarray(cell)
                for subax in ax_complement:
                    # Subtract non-total margins
                    if len(subax) < self.ndims() and len(ax) < self.ndims():
                        self.residual_margins[subax][cell_list[ax]] -= self.table[tuplize(cell)]
                if len(ax) == self.ndims():
                    # Subtract cell values from grand total (only once)
                    self.residual_margins[ax] -= self.table[tuplize(cell)]
        # print(self.table)
        # print(self.constraints['cells'])
        # print(self.margins)
        # print(self.residual_margins)

        # Update cells though
        self.cells = sorted(list(set(self.cells) - set(constrained_cells)))


    def update_cell_constraints_deterministically(self):
        # Get latest constraint table
        constraint_table = self.constraint_table()
        # Get flags for whether cells are constrained or not
        cell_unconstrained = (constraint_table == -1)
        # Repeat process if there any margins with only one unknown
        while np.any([np.any(cell_unconstrained.sum(axis=ax,keepdims=True).flatten() == 1) for ax in self.constraints['constrained_axes']]):
            # If cells need to be filled in deterministically the entire table must be available
            try:
                assert self.table_nonnegative(self.table)
            except:
                self.logger.error('Table has negative entries')
                raise Exception('Cannot fill in remainder of cells determinstically as table values are not provided for these cells.')
            # Go through each margin
            for ax in self.constraints['constrained_axes']:
                # If there any margins with only one unknown cell
                if np.any(cell_unconstrained.sum(axis=tuplize(ax),keepdims=True).flatten() == 1):
                    # Get axis complement
                    ax_complement = list(flatten(self.axes_complement(ax, same_length=True)))
                    # Create cell index
                    cell_index = [{
                        "axis": ax,
                        "index": list(flatten(np.where(cell_unconstrained.sum(axis=tuplize(ax)) == 1)))
                    }]
                    # Update last index
                    last_axis_index = cell_index[-1]['index']
                    for subax in ax_complement:
                        # Find indices
                        cell_index.append({
                            "axis": subax,
                            "index": np.where(cell_unconstrained.take(indices=last_axis_index[0], axis=subax))
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

    def constrained_margins(self,table=None):
        if table is None:
            return {ax:self.margins[ax] for ax in self.constraints['constrained_axes']}
        else:
            return {ax:table.sum(axis=tuplize(ax), keepdims=True, dtype='int32').flatten() for ax in self.constraints['constrained_axes']}

    def unconstrained_margins(self,table=None):
        if table is None:
            return {ax:self.margins[ax] for ax in self.constraints['unconstrained_axes']}
        else:
            return {ax:table.sum(axis=tuplize(ax), keepdims=True, dtype='int32').flatten() for ax in self.constraints['unconstrained_axes']}

    def update_unconstrained_margins_from_table(self, table):
        for ax in self.constraints['unconstrained_axes']:
            self.margins[tuplize(ax)] = np.sum(
                table, axis=tuplize(ax), keepdims=True, dtype='int32').flatten()
            self.residual_margins[tuplize(ax)] = np.sum(
                table, axis=tuplize(ax), keepdims=True, dtype='int32').flatten()

    def update_constrained_margins_from_table(self, table):
        for ax in self.constraints['constrained_axes']:
            self.margins[tuplize(ax)] = np.sum(
                table, axis=tuplize(ax), keepdims=True, dtype='int32').flatten()
            self.residual_margins[tuplize(ax)] = np.sum(
                table, axis=tuplize(ax), keepdims=True, dtype='int32').flatten()

    def update_margins_from_table(self, table):
        for axes in powerset(range(self.ndims())):            
            self.margins[tuplize(axes)] = np.sum(
                table, 
                axis=tuplize(axes), 
                keepdims=True, 
                dtype='int32'
            ).flatten()
            self.residual_margins[tuplize(axes)] = np.sum(
                table, 
                axis=tuplize(axes), 
                keepdims=True, 
                dtype='int32'
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
            self.logger(
                f"Cannot update table of dims {self.dims} to table of dims {np.shape(tab)}")
            raise Exception('Table update failed.')
        # Update table
        self.table = tab.astype('int32')
        # Update table properties
        self.update_table_properties_from_table()
        # Copy residal margins
        self.residual_margins = deepcopy(self.margins)

    def update_table_properties_from_table(self) -> None:
        # Update dimensions
        self.dims = np.asarray(np.shape(self.table), dtype='uint8')
        # Update margins
        self.update_margins_from_table(self.table)

    
    def export(self, dirpath: str = './synthetic_dummy', overwrite: bool = False) -> None:
        # Make directory if it does not exist
        makedir(dirpath)

        if self.table is not None:
            # Get filepath experiment filepath
            filepath = os.path.join(dirpath, 'table.txt')

            # Write experiment summaries to file
            if (not os.path.exists(filepath)) or overwrite:
                write_txt(self.table, filepath)

        for axis, margin in self.margins.items():
            # Get filepath experiment filepath
            filepath = os.path.join(
                dirpath, f"margin_sum_axis{','.join([str(a) for a in axis])}.txt")

            # Write experiment summaries to file
            if (not os.path.exists(filepath)) or overwrite:
                write_txt(margin, filepath)

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
    def __init__(self, table=None, config: Config = None, disable_logger: bool = False, **kwargs: dict):
        # Set configuration file
        super().__init__(table, config, disable_logger, **kwargs)


class ContingencyTableDependenceModel(ContingencyTable):
    def __init__(self, table=None, config: Config = None, disable_logger: bool = False, **kwargs: dict):
        # Set configuration file
        super().__init__(table, config, disable_logger, **kwargs)


class ContingencyTable2D(ContingencyTableIndependenceModel, ContingencyTableDependenceModel):

    def __init__(self, table=None, config: Config = None, disable_logger: bool = False, **kwargs: dict):
        # Set configuration file
        super().__init__(table, config, disable_logger, **kwargs)
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
            print(self.table)
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
                print([np.sum(v) for v in self.residual_margins.values()])
                print('table total')
                print(self.table.ravel().sum())
                for ax in sorted(self.constraints['constrained_axes']):
                    if len(ax) < self.ndims():
                        print(f"True summary statistics")
                        print(self.margins[tuplize(ax)])
                        print(f"Residual summary statistics")
                        print(self.residual_margins[tuplize(ax)])
                        print(f"Fixed cell summary statistics")
                        print(np.sum(table0_copy,axis=tuplize(ax),keepdims=True).flatten())
                        print(f'Current summary statistics')
                        print(np.sum(table0,axis=tuplize(ax),keepdims=True).flatten())
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
        if str_in_list('margin0', self.ct.config.settings['mcmc']['contingency_table'].keys()):
            initialisation = self.ct.config.settings['mcmc']['contingency_table']['margin0']
            # If it is a path read file
            if isinstance(initialisation, str):
                # Extract filepath
                if os.path.isfile(os.path.join(self.ct.config.settings['inputs']['dataset'], initialisation)):
                    margin0 = np.loadtxt(
                        os.path.join(
                            self.ct.config.settings['inputs']['dataset'],
                            initialisation
                        ), 
                        dtype='float32'
                    )
            # If it is an array/list make sure it has the
            elif isinstance(initialisation, (list, np.ndarray, np.generic)) and \
                    np.any([(len(initialisation) == dim) for dim in self.ct.dims]):
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
                    pvals=intensity.sum(axis=axis)
                )
            else:
                # This is sampling the grand total
                margin = np.array([np.random.poisson(intensity.sum(axis=axis))])
            # The loop can now end
            firstIteration = False
        return margin
    
    def table_monte_carlo_sample(self,intensity:list=None, margins: dict = {}) -> Union[np.ndarray, None]:
        # Update margins
        if margins is not None:
            self.update_margins(margins)

        # Fix random seed
        # np.random.seed(self.seed)
        # Define axes
        constrained_axis1 = (1,)
        constrained_axis2 = (0,)
        
        # Initialise table to zero
        table0 = np.zeros(shape=self.dims, dtype='int32')

        # Get N (sum or row or column sums)
        N = self.residual_margins[tuplize(range(self.ndims()))][0]

        # Generate permutation of N items
        permutation = np.random.permutation(range(1, N+1))

        # Loop through first axis
        for i in range(self.dims[constrained_axis2]):
            # Look at first r_i items
            if i == 0:
                permutation_subset = permutation[0 : (self.residual_margins[constrained_axis1][i])]
            else:
                permutation_subset = permutation[int(np.sum(self.residual_margins[constrained_axis1][:(i)])):
                                                int(np.sum(self.residual_margins[constrained_axis1][:(i+1)]))]
                    
            # Loop through columns
            for j in range(self.dims[constrained_axis1]):
                # Create index appropriately
                query_index = np.zeros(self.ndims(), dtype='uint8')
                query_index[constrained_axis2] = i
                query_index[constrained_axis1] = j
                query_index = tuplize(query_index)

                if query_index in self.constraints['cells']:
                    # Initialise cell at constraint
                    table0[query_index] = self.table[query_index]
                else:
                    # Count number of entries between c_1+...+c_{j-1}+1 and c_1+...+c_j
                    if j == 0:
                        table0[query_index] = np.int32(
                            ((1 <= permutation_subset) & \
                             (permutation_subset <= self.residual_margins[constrained_axis2][j])).sum()
                        )
                    else:
                        table0[query_index] = np.int32(
                            (((np.sum(self.residual_margins[constrained_axis2][:j])+1) <= permutation_subset) & \
                            (permutation_subset <= np.sum(self.residual_margins[constrained_axis2][:(j+1)]))).sum()
                        )

        self.admissibility_debugging('Monte Carlo',table0)

        return table0.astype('int32')
    
    def table_import(self,intensity:list=None, margins: dict = {}) -> Union[np.ndarray, None]:
        # Read initial table
        table0 = None
        if str_in_list('table0', self.ct.config.settings['mcmc']['contingency_table'].keys()):
            initialisation = self.ct.config.settings['mcmc']['contingency_table']['table0']
            # If it is a path read file
            if isinstance(initialisation, str):
                # Extract filepath
                if os.path.isfile(os.path.join(self.ct.config.settings['inputs']['dataset'], initialisation)):
                    tab0 = np.loadtxt(
                        os.path.join(
                            self.ct.config.settings['inputs']['dataset'], 
                            initialisation
                        ), 
                        dtype='float32'
                    )
            # If it is an array/list make sure it has the
            elif isinstance(initialisation, (list, np.ndarray, np.generic)) and \
                    (np.shape(initialisation) == self.ct.dims):
                table0 = initialisation
            else:
                self.logger.warning(
                    f"Initial table {initialisation} is neither a list nor a valid filepath.")
        else:
            self.logger.error(f"No table 0 provided in config.")
        
        if table0 is None:
            raise Exception('Table 0 could not be found.')
        else:
            return table0

    def table_random_sample(self,intensity:list=None, margins: dict = {}) -> Union[np.ndarray, None]:

        try:
            assert len(self.constraints['constrained_axes']) == 0
        except:
            raise Exception ('table_random_sample is only used by tables with no margin constraints')
        # Fix random seed
        # np.random.seed(self.seed)
        
        # Initialise table to zero
        table0 = np.zeros(self.dims, dtype='int32')
        min_cell_value,max_cell_value = 0,1
        for cell in self.constraints['cells']:
            table0[cell] = np.int32(self.table[cell])
            min_cell_value = min(min_cell_value,table0[cell])
            max_cell_value = max(max_cell_value,table0[cell])
        
        # Define support to sample from
        cell_value_range = [min(min_cell_value,1), max(max_cell_value,100)]
        
        # Update remaining table cells
        for cell in self.cells:
            # Sample value randomly
            table0[cell] = np.int32(np.random.randint(*cell_value_range))

        self.admissibility_debugging('Random solution',table0)
        # np.random.seed(None)

        return table0.astype('int32')

    def table_maximum_entropy_solution(self, intensity:list=None, margins: dict = {}) -> Union[np.ndarray, None]:
        '''
        This is the solution of the maximum entropy in the non-integer case
        X_ij = (r_i*c_j) / T
        '''
        # Update margins
        if margins is not None:
            self.update_margins(margins)

        # Fix random seed
        # np.random.seed(self.seed)

        # Initialise table to zero
        table0 = np.zeros(self.dims, dtype='int32')
        table0_learned = np.zeros(table0.shape,dtype=bool)
        for cell in self.constraints['cells']:
            table0[cell] = self.table[cell]
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

        # Get N (sum or row or column sums)
        N = self.margins[tuplize(range(self.ndims()))][0]
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
                                axis=None,
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

                    print('smallest cell val',smallest_value_cell,'amount',amount)
                    # print('table val',table0[smallest_value_cell])
                    print('rowsums',rowsums0[smallest_value_cell[0]],'colsums',colsums0[smallest_value_cell[1]])
                    print('\n')

            # Increment counter
            counter += 1

        self.admissibility_debugging('Maximum entropy solution',table0)
        # np.random.seed(None)

        return table0.astype('int32')

    def table_iterative_residual_filling_solution(self,intensity:list=None, margins: dict = {}) -> Union[np.ndarray, None]:

        # Update margins
        if margins is not None:
            self.update_margins(margins)
            
        # Initialise table to zero
        table0 = np.zeros(self.dims, dtype='int32')
        table0_learned = np.zeros(table0.shape,dtype=bool)
        for cell in self.constraints['cells']:
            table0[cell] = self.table[cell]
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
        k = 0
        while not self.table_margins_admissible(table0.data):

            # Order min residual values
            sorted_indices = np.ma.argsort(
                        min_residual,
                        axis=None,
                        kind='quicksort', 
                        fill_value=np.int32(self.margins[tuplize(range(self.ndims()))]+1)
            )
            # Get indices of unmasked elements
            unmasked_indices = np.ma.where(~table0.mask)
            # Create index in same dims ars sorted_indices
            unmasked_indices = np.ravel_multi_index(unmasked_indices,table0.shape)
            # Maintain index order and get only unmasked indices
            sorted_unmasked_indices_2d = np.unravel_index(
                sorted_indices[np.isin(sorted_indices, unmasked_indices[np.isin(unmasked_indices, sorted_indices)])], 
                table0.shape
            )
            smallest_value_cell = np.zeros(self.ndims(), dtype='uint8')
            for i in range(self.ndims()):
                smallest_value_cell[i] = sorted_unmasked_indices_2d[i][k]
            smallest_value_cell = tuplize(smallest_value_cell)

            assert not table0.mask[smallest_value_cell]

            # Try to add maximum amount possible
            amount = int(min(rowsums0[smallest_value_cell[0]], colsums0[smallest_value_cell[1]]))
            table0[smallest_value_cell] += np.int32(amount)
            # Update row,column sums
            rowsums0[smallest_value_cell[0]] -= amount
            colsums0[smallest_value_cell[1]] -= amount
            # Update minimum residual table
            min_residual[smallest_value_cell] = amount

            if rowsums0[smallest_value_cell[0]] <= 0:
                table0.mask[smallest_value_cell[0],:] = True
                min_residual.mask[smallest_value_cell[0],:] = True                
            if colsums0[smallest_value_cell[1]] <= 0:
                table0.mask[:,smallest_value_cell[1]] = True
                min_residual.mask[:,smallest_value_cell[1]] = True

        self.admissibility_debugging('Iterative residual filling solution',table0.data)

        return table0.data.astype('int32')

    def table_iterative_uniform_residual_filling_solution(self,intensity:list=None, margins: dict = {}) -> Union[np.ndarray, None]:

        # Update margins
        if margins is not None:
            self.update_margins(margins)

        # Fix random seed
        # np.random.seed(self.seed)
        
        # Initialise table to zero
        table0 = np.zeros(self.dims, dtype='int32')
        table0_learned = np.zeros(table0.shape,dtype=bool)
        for cell in self.constraints['cells']:
            table0[cell] = self.table[cell]
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

        # Get N (sum or row or column sums)
        N = self.residual_margins[tuplize(range(self.ndims()))][0]

        # Minimum residual table
        min_residual = np.zeros(self.dims,dtype='int32')
        min_residual_mask = np.zeros(self.dims,dtype=bool)
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                if (i,j) in self.constraints['cells']:
                    min_residual[i,j] = 0
                    min_residual_mask[i,j] = True
                else:
                    min_residual[i,j] = int(min(N/len(self.cells),rowsums0[i], colsums0[j]))
                    if min_residual[i,j] == 0:
                        min_residual_mask[i,j] = True
        min_residual = np.ma.array(min_residual, mask=min_residual_mask)

        k = 0
        counter = 0

        for cell in self.cells:
            # Initalise table values with minimum possible row or col sum or N/IJ
            amount = int(min(np.floor(np.sum(rowsums0)/np.sum(table0.mask)),rowsums0[cell[0]],colsums0[cell[1]]))

            # Update table
            table0[cell] += np.int32(amount)

            # Update row,column sums
            # This method allows for negative residuals
            rowsums0[cell[0]] -= amount
            colsums0[cell[1]] -= amount

            # Update minimum residual table
            min_residual[cell] = amount

            if rowsums0[cell[0]] <= 0:
                table0.mask[cell[0],:] = True
                min_residual.mask[cell[0],:] = True                
            if colsums0[cell[1]] <= 0:
                table0.mask[:,cell[1]] = True
                min_residual.mask[:,cell[1]] = True

        while not self.table_margins_admissible(table0.data):

            # IF no solution found rerun
            if np.sum(table0.mask) == np.prod(self.ndims()):
                self.table_iterative_uniform_residual_filling_solution(margins)
            
            # Get indices of unmasked elements
            unmasked_indices = np.ma.where(~table0.mask)
            # Create index in same dims ars sorted_indices
            # unmasked_indices = np.ravel_multi_index(unmasked_indices,table0.shape)
            # Maintain index order and get only unmasked indices
            cell_index = np.random.randint(len(unmasked_indices))
            try:
                cell = list(zip(*unmasked_indices))[cell_index]
            except:
                self.table_iterative_uniform_residual_filling_solution(margins)
            
            assert not table0.mask[cell]

            # Try to add maximum amount possible
            amount = int(min(
                rowsums0[cell[0]], 
                colsums0[cell[1]]
            ))
            table0[cell] += np.int32(amount)
            # Update row,column sums
            rowsums0[cell[0]] -= amount
            colsums0[cell[1]] -= amount
            # Update minimum residual table
            min_residual[cell] = amount

            if rowsums0[cell[0]] <= 0:
                table0.mask[cell[0],:] = True
                min_residual.mask[cell[0],:] = True                
            if colsums0[cell[1]] <= 0:
                table0.mask[:,cell[1]] = True
                min_residual.mask[:,cell[1]] = True

            counter += 1
            print(rowsums0)
            print(colsums0)
            print('\n')

        self.admissibility_debugging('Iterative uniform residual filling solution',table0.data)
        # np.random.seed(None)

        return table0.data.astype('int32')
    