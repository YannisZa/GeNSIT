import os
import sys
import torch
import numpy as np
import torch.distributions as distr

from copy import deepcopy
from pprint import pprint
from pandas import DataFrame
from itertools import product
from tabulate import tabulate
from torch import int32, float32
from num2words import num2words
from typing import List, Union, Callable

from gensit.config import Config
from gensit.static.global_variables import *
from gensit.utils.math_utils import powerset
from gensit.utils.exceptions import MissingData
from gensit.utils.misc_utils import ndims, setup_logger, unpack_dims, write_txt, makedir, tuplize, flatten, tuple_contained, depth, broadcast2d, print_json

# -> Union[ContingencyTable,None]:
def instantiate_ct(config: Config, **kwargs):
    ct_type = f"ContingencyTable{len(unpack_dims(kwargs,time_dims = False))}D"
    if hasattr(sys.modules[__name__], ct_type):
        return getattr(sys.modules[__name__], ct_type)(
                config = config,
                **kwargs
        )
    else:
        raise Exception(f"Input class {ct_type} not found")

def map_constraints_to_distribution_name(constraints):
    # TODO: Generalise this

    # Get all unique axes that appear as singletons
    unique_axes = set([ax for ax in constraints['constrained_axes'] if len(ax) == 1])
    
    if len(constraints['constrained_axes']) == 0:
        return 'poisson'
    elif len(unique_axes) == 0:
        return 'multinomial'
    elif len(unique_axes) == 1:
        return 'product_multinomial'
    elif len(unique_axes) == 2:
        return 'fishers_hypergeometric'
    else:
        raise ValueError(f"Distribution not found for constraints {str(constraints['constrained_axes'])}")


class ContingencyTable(object):

    def __init__(
            self, 
            config: Config = None, 
            **kwargs: dict
        ):
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__,
            console_level = level,
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevels( console_level = level )
        
        # Config
        self.config = config
        # Device name
        self.device = self.config['inputs']['device']

        # Instantiate dataset 
        self.data = Dataset()
        # Read data passed
        for attr in ['ground_truth_table','grand_total','dims','margins','cells']:
            if kwargs.get(attr,None) is not None:
                setattr(self.data,attr,kwargs.get(attr,None))
            else:
                if attr in ['dims','margins']:
                    setattr(self.data,attr,{})
                elif attr == 'cells':
                    self.data.cells = []
                elif attr == 'grand_total':
                    self.data.grand_total = torch.tensor(+1,dtype = float32,device = self.device)
                else:
                    setattr(self.data,attr,None)
        # CAUTION: Exclude time from table dimensions
        if hasattr(self.data,'dims'):
            self.data.dims = {k:v for k,v in self.data.dims.items() if k != 'time'}
        self.dim_names = np.array(TRAIN_SCHEMA['ground_truth_table']['dims'])
        # Markov basis class
        self.markov_basis_class = None
        # Flag for whether to allow sparse margins or not
        self.sparse_margins = False
        # Initialise residual margins
        self.residual_margins = {}

    def build(self, config, **kwargs):
        # Build contingency table
        if config is not None:
            # Store flag for whether to allow sparse margins or not
            self.sparse_margins = self.config['contingency_table'].get('sparse_margins',False)
            # Read tabular data
            self.import_tabular_data()

        elif self.data.ground_truth_table is not None:
            # Read table and dimensions
            if torch.is_tensor(self.data.ground_truth_table):
                self.data.ground_truth_table = self.data.ground_truth_table.int().to(dtype = int32,device = self.device)
            else:
                self.data.ground_truth_table = torch.from_numpy(self.data.ground_truth_table).int().to(dtype = int32,device = self.device)
            # Read margin sparsity
            self.sparse_margins = bool(kwargs['sparse_margins']) if 'sparse_margins' in list(kwargs.keys()) else False
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
        self.distribution_name = map_constraints_to_distribution_name(self.constraints)
        self.config['contingency_table']['distribution_name'] = self.distribution_name

        # Fill in missing cell margins and 
        # Update residual margins based on exhaustive list of cell constraints
        self.propagate_cell_constraints()

        # Update type of table to determine the type of Markov basis to use
        self.update_table_type_and_markov_basis_class()

        # Get total from table
        if 'spatial_interaction_model' in list(self.config.keys()) and tuplize(range(ndims(self))) in list(self.data.margins.keys()):
            self.config['spatial_interaction_model']['grand_total'] = int(self.data.margins[tuplize(range(ndims(self)))])

        self.logger.info(f"margin constraints over axes {','.join([str(ax) for ax in self.constraints['constrained_axes']])} provided")

    @property
    def dims(self):
        return unpack_dims(self.data.dims,time_dims = False)

    def axes_complement(self, axes, same_length:bool = False):
        if hasattr(axes,'__len__') and depth(axes) > 1:
            axes_set = set([tuplize(ax) for ax in axes])
        else:
            axes_set = {tuplize(axes)}
        # Define all axes
        all_axes_set = set(powerset(range(ndims(self))))
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
    def table_margins_summary_statistic(self, tab: torch.tensor) -> List:
        return torch.cat(tuple([
            self.table_axis_margin_summary_statistic(tab = tab.to(int32),ax = ax) 
            for ax in self.constraints['all_axes']
        ]), dim = 0)
    
    def table_axis_margin_summary_statistic(self, tab: torch.tensor, ax:int = None) -> List:
        return tab.to(int32).sum(dim = tuplize(ax), keepdims = True, dtype = int32).flatten()

    # Compute margin summary statistics for given function
    def table_constrained_margins_summary_statistic(self, tab: torch.tensor) -> List:
        if len(self.constraints['constrained_axes']) > 0:
            return torch.cat(tuple([
                tab.to(int32).sum(
                        dim = tuplize(ax), 
                        keepdims = True, 
                        dtype = int32
                ).flatten() for ax in sorted(self.constraints['constrained_axes'])
            ]), dim = 0)
        else:
            return torch.empty(1)

    def table_unconstrained_margin_summary_statistic(self, tab: torch.tensor) -> List:
        if len(self.constraints['unconstrained_axes']) > 0:
            return torch.cat(tuple([tab.to(int32).sum(dim = tuplize(ax), keepdims = True, dtype = int32).flatten() for ax in sorted(self.constraints['unconstrained_axes'])]), dim = 0)
        else:
            return torch.empty(1)

    # Compute cell summary statistics for given function
    def table_cell_constraint_summary_statistic(self, tab: torch.tensor, cells: list = None) -> List:
        if cells is None:
            return torch.tensor(
                [tab[cell] for cell in sorted(self.constraints['cells'])], 
                dtype = int32
            )
        else:
            return torch.tensor(
                [tab[cell] for cell in sorted(self.constraints['cells'])], 
                dtype = int32
            )

    # Check if margin constraints are satisfied
    def table_margins_admissible(self, tab) -> bool:
        if len(self.constraints['constrained_axes']) > 0:
            return torch.equal(
                self.table_constrained_margins_summary_statistic(tab),
                torch.cat(
                    [self.data.margins[tuplize(ax)] for ax in sorted(self.constraints['constrained_axes'])],
                    dim = 0
                )
            )
        else:
            return True

    # Check if cell constraints are satisfied
    def table_cells_admissible(self, tab) -> bool:
        return all([self.data.ground_truth_table[cell] == tab[cell].to(int32) for cell in sorted(self.constraints['cells'])])

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

    def table_difference_histogram(self, tab: torch.tensor, tab0: torch.tensor):
        # Take difference of tables
        difference = tab.to(int32) - tab0.to(int32)
        # Values classification
        values = {"+": torch.count_nonzero(difference > 0), "-": torch.count_nonzero(
            difference < 0), "0": torch.count_nonzero(difference == 0)}
        return values

    def check_table_validity(self, table: torch.tensor) -> bool:
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
            assert ndims(self) == ndims(other_ct)
        except Exception:
            self.logger.error(f"Contingency table dims {ndims(other_ct)} are not the same as {ndims(self)}")
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
        if hasattr(self.data,'ground_truth_table') and self.data.ground_truth_table is not None:
            # Convert to numpy
            ground_truth_table = self.data.ground_truth_table.cpu().detach().numpy()
            return ',\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in ground_truth_table])
        else:
            return 'Empty Table'

    def __repr__(self):
        return f"{'x'.join([str(dim) for dim in self.dims])} ContingencyTable{ndims(self)}D(Config)"

    def __len__(self):
        return np.prod([dim for dim in self.dims if dim > 0])
    
    def reset(self, config, **kwargs) -> None:
        # Rebuild table
        self.build(config = config, **kwargs)
    
    def constraint_table(self,with_margins:bool = False):
        if self.data.ground_truth_table is not None:
            table_constraints = -np.ones(self.data.ground_truth_table.shape)
            table_mask = np.zeros(self.data.ground_truth_table.shape,dtype = bool)
            ground_truth_table = self.data.ground_truth_table.cpu().detach().numpy()
            for cell in self.constraints['cells']:
                table_constraints[cell] = ground_truth_table[cell]
                table_mask[cell] = True
            table_constraints = table_constraints.astype(np.int32)
            
            # Printing margins for visualisation purposes only
            if with_margins:
                assert ndims(self) == 2

                # Flip table for viz purposes (clear opacity)
                row_margin,col_margin = (0,),(1,)
                if self.data.dims['destination'] >= self.data.dims['origin']:
                    table_constraints = table_constraints.T
                    row_margin,col_margin = (1,),(0,)
                
                # Create pandas dataframe from constraints
                table_constraints_ct = DataFrame(table_constraints)

                # Add margins to the contingency table
                table_constraints_ct.loc['Total',:] = self.residual_margins[row_margin].to(dtype = int32)
                table_constraints_ct['Total'] = list(self.residual_margins[col_margin].to(dtype = int32)) + list(self.residual_margins[(0,1)].to(dtype = int32))
                
                # Replace free cells with empty string
                table_constraints_ct[table_constraints_ct < 0] = ''
                table_constraints_ct = table_constraints_ct.applymap(str).replace('\.0', '', regex = True)
                
                # Convert the contingency table to a pretty printed string
                table_constraints = tabulate(table_constraints_ct, headers=[], showindex = False, tablefmt='plain')

            return table_constraints
        else:
            return 'Empty Table'

    def import_tabular_data(self) -> None:
        # Inherit data from margins
        partial_ground_truth = True
        if hasattr(self.data,'ground_truth_table') and \
            self.data.ground_truth_table is not None and \
            torch.is_tensor(self.data.ground_truth_table):

            # Check to see if partial data are provided
            if torch.any(self.data.ground_truth_table < 0):
                partial_ground_truth = True
                # Update only dimensions
                self.data.dims = dict(zip(self.dim_names,np.asarray(np.shape(self.data.ground_truth_table), dtype=CORE_COORDINATES_DTYPES['origin'])))
            else:
                partial_ground_truth = False
                # Update all table properties
                self.update_table_properties_from_table()
        
        if partial_ground_truth:
            self.logger.warning('Fully valid contingency table was not provided. One will be randomly generated.')
            if not hasattr(self.data,'dims') or len(self.data.dims) <= 0:
                raise Exception("Cannot read table dimensions.")
            # Copy to residual margins
            if hasattr(self.data,'margins') and self.data.margins is not None:
                self.data.residual_margins = deepcopy(self.data.margins)
            else:
                self.logger.warning(f"margins file not provided")
                self.data.margins = {}
                self.residual_margins = {}
                # raise Exception(f"Importing margins failed.")
            # Ensure margin validity
            for axis, margin in self.data.margins.items():
                # Assert all margins are unidimensional
                if tuplize(axis) != tuplize(range(ndims(self))):
                    try:
                        assert np.sum([1 for dim in np.shape(margin) if dim >= 1]) == 1
                    except:
                        raise Exception(f"margin for axis {axis} is not unidimensional.")
                # Assert all margin axis match dimensions
                if np.sum(list(self.dims)) > 0:
                    axis_inverse = np.array([i for i in range(ndims(self)) if i not in list(tuplize(axis))])
                    if len(axis_inverse) > 0:
                        try:
                            assert np.prod(np.shape(margin)) == np.prod([self.data.dims[dim_name] for dim_name in self.dim_names[axis_inverse]])
                        except:
                            self.logger.error(f"Margin for axis {','.join(axis_inverse)} has length {np.prod(np.shape(margin))} and not {np.prod([self.data.dims[dim_name] for dim_name in self.dim_names[axis_inverse]])}.")
                            raise Exception('Imported inconsistent margins.')
                else:
                    raise Exception(f'Zero dimensions {self.data.dims} provided.')
                # Assert all margin sums are the same
                try:
                    assert torch.equal(
                        margin.sum(dtype = int32).flatten(),
                        self.data.margins[tuplize(range(ndims(self)))].flatten()
                    )
                except:
                    self.logger.error(f"margin for axis {axis} has total {margin.sum(dtype = int32).flatten()} and not {self.data.margins[tuplize(range(ndims(self)))].flatten()}.")
                    raise Exception('Imported inconsistent margins.')
            # No dim must be zero
            try:
                assert (self.__len__() > 0) and (len(self.dims) > 0)
            except AssertionError:
                self.logger.error('Dims provided are not valid.')
                raise Exception('Tabular data could not be imported')
            # Randomly initialise margins not provided
            for axis in sorted(list(powerset(range(ndims(self)))), key = len, reverse = True):
                if not tuplize(axis) in self.data.margins.keys():
                    if tuplize(axis) == tuplize(range(ndims(self))):
                        self.data.margins[tuplize(axis)] = torch.randint(
                            size=(1,),
                            low = 1,
                            high = 50_000,
                            dtype = int32,
                            device = self.device
                        )
                    else:
                        ax = self.axes_complement(axis,same_length = True)[0][0]
                        dim = self.data.dims[self.dim_names[ax]]
                        distribution = distr.multinomial.Multinomial(
                            total_count = self.data.margins[tuplize(range(ndims(self)))].cpu().detach().item(),
                            probs = torch.repeat_interleave(torch.tensor(1/dim), dim),
                        )
                        self.data.margins[tuplize(axis)] = distribution.sample()
                        # self.data.margins[tuplize(axis)].to(dtype = int32,device = self.device)
                        self.logger.debug(f"Randomly initialising margin for axis {tuplize(axis)}")
            # Copy residal margins
            self.residual_margins = deepcopy(self.data.margins)
            # Initialise table only if none is provided
            if self.data.ground_truth_table is not None:
                self.data.ground_truth_table = -torch.ones(tuple(list(self.dims)))
            # Otherwise a partially observed table has already been provided
            # Initialise constraints
            self.constraints = {}
            self.constraints['constrained_axes'] = []
            self.constraints['all_axes'] = sorted(list(set(powerset(range(ndims(self))))))
            self.constraints['unconstrained_axes'] = sorted(list(set(powerset(range(ndims(self))))))

    def read_constraints(self,**kwargs):
        self.constraints = {}
        constrained_axes,cell_constraints = None, None
        # margin and cell constraints
        if self.config is not None:
            if 'constraints' in list(self.config.settings['contingency_table'].keys()):
                if 'axes' in list(self.config.settings['contingency_table']['constraints'].keys()):
                    constrained_axes = [tuplize(ax) for ax in self.config.settings['contingency_table']['constraints']['axes'] if len(ax) > 0]
                # read training cells from input data
                cell_constraints = kwargs.get('train_cells',np.array([]))
                if self.config.settings['contingency_table']['constraints'].get('cells',False):
                    # Train cells must be provided if the contingency table has cell constraints
                    if len(cell_constraints) == 0:
                        raise MissingData(
                            missing_data_name = 'train_cells',
                            data_names = ', '.join(list(kwargs.keys())),
                            location = 'Input Data'
                        )
        ## Reading kwargs if provided
        elif 'constraints' in list(kwargs.keys()):
            if 'axes' in list(kwargs['constraints'].keys().keys()):
                constrained_axes = [tuplize(ax) for ax in kwargs['constraints']['axes'] if len(ax) > 0]
            if kwargs['constraints'].get('cells',False):
                # Train cells must be provided if the contingency table has cell constraints
                if kwargs['inputs']['data'].get('train_cells','') == '':
                    raise MissingData(
                        missing_data_name = 'train_cells',
                        data_names = ', '.join(list(kwargs['inputs']['data'].keys())),
                        location = 'Input (Kwargs) Data'
                    )
                    
                if kwargs['inputs'].get('data',{}).get('train_cells','') != '':
                    cell_constraints = os.path.join(
                        kwargs['inputs']['in_directory'],
                        kwargs['inputs']['dataset'],
                        kwargs['inputs']['data']['train_cells']
                    )
        ## Storing constraints
        if constrained_axes is not None:
            # Get the set of constrained axes
            constrained_axes = set(constrained_axes)
            # Determine axes over which margins are defined
            self.constraints['all_axes'] = sorted(list(set(powerset(range(ndims(self))))))
            # Infer derivative margins from the ones provided
            constrained_axes = self.infer_derivative_margin_axes(constrained_axes)
            # Update constrained axes
            self.constraints['constrained_axes'] = sorted(list(constrained_axes))
            # Determine complement of these axes
            self.constraints['unconstrained_axes'] = self.axes_complement(self.constraints['constrained_axes'])
        else:
            self.constraints['all_axes'] = sorted(list(set(powerset(range(ndims(self))))))
            self.constraints['constrained_axes'] = []
            self.constraints['unconstrained_axes'] = sorted(list(set(powerset(range(ndims(self))))))
        
        # Cell constraints
        ## Reading config if provided
        if cell_constraints is not None:
            if isinstance(cell_constraints,str):
                if os.path.isfile(cell_constraints):
                    # Load cell constraints from file
                    cells = np.loadtxt(cell_constraints, dtype='uint16')
                    # Reshape cells
                    cells = cells.reshape(np.size(cells)//ndims(self), ndims(self)).tolist()
                    # Remove invalid cells (i.e. cells than do not have the right number of dims or are out of bounds)
                    cells = sorted([tuplize(c) for c in cells if (len(c) == ndims(self) and (c < np.asarray(list(self.dims))).all())])
                    self.constraints['cells'] = [tuplize(c) for c in cells]
                    # self.logger.info(f"Cell constraints {','.join([str(c) for c in cells])} provided")
                    self.logger.info(f"Cell constraints {len(cells)} provided")
                else:
                    self.constraints['cells'] = []
            elif isinstance(cell_constraints,(list,np.ndarray)):
                self.constraints['cells'] = [tuplize(c) for c in cell_constraints]
            else:
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
            print(self.data.ground_truth_table)
            print('Constraint table')
            print(self.constraint_table())
            self.logger.error('No activate cells found. Table can be deterministically elicited based on provided constraints.')
            raise Exception(f"Table inference is halted.")

        # Count number of non zero dims
        if ndims(self) == 0:
            self.ct_type = 'empty_table'
            self.markov_basis_class = None
        elif ndims(self) <= 2:
            self.markov_basis_class = f'MarkovBasis{ndims(self)}DTable'
            if isinstance(self, ContingencyTableDependenceModel):
                self.ct_type = f'{num2words(ndims(self))}_way_contingency_table_dependence_model'
            else:
                self.ct_type = f'{num2words(ndims(self))}_way_contingency_table_independence_model'
        else:
            self.logger.error(f'No implementation found for ndims = {ndims(self)}')
            raise Exception('Table type and markov basis class could not be identified.')
        
    def update_margins(self, margins: dict = {}):
        for ax, margin in margins.items():
            # Update margins
            if not torch.is_tensor(margin):
                self.data.margins[tuplize(ax)] = torch.from_numpy(margin).int().to(dtype = int32,device = self.device)
                self.residual_margins[tuplize(ax)] = torch.from_numpy(margin).int().to(dtype = int32,device = self.device)
            else:
                self.data.margins[tuplize(ax)] = margin.int().to(dtype = int32,device = self.device)
                self.residual_margins[tuplize(ax)] = margin.int().to(dtype = int32,device = self.device)

    def update_residual_margins_from_cell_constraints(self, constrained_cells = None):
        # Update residual margins
        self.residual_margins = self.update_margins_from_cells(
            margins = self.residual_margins,
            axes = self.constraints['all_axes'],
            constrained_cells = constrained_cells,
            table = self.data.ground_truth_table
        )
        # Update cells though
        self.cells = sorted(list(set(self.cells) - set(constrained_cells)))

    def update_margins_from_cells(self,margins, axes = None, constrained_cells = None, table = None):
        constrained_cells = self.constraints['cells'] if constrained_cells is None or len(constrained_cells) <= 0 else constrained_cells
        margins = deepcopy(margins)
        table = self.data.ground_truth_table if table is None else table
        axes = self.constraints['all_axes'] if axes is None else axes
        for cell in constrained_cells:
            cell_arr = np.asarray(cell)
            for ax in axes:
                if len(ax) < ndims(self):
                    axc = self.axes_complement([ax],same_length = True)
                    axc = axc[0]
                    # Subtract non-total margins
                    # print('ax update',ax,cell_arr[axc],table[tuplize(cell)], residual_margins[ax][cell_arr[axc]],residual_margins[ax][cell_arr[axc]]-table[tuplize(cell)])
                    margins[ax][cell_arr[axc]] -= table[tuplize(cell)].to(int32)
                else:
                    # Subtract cell values from grand total (only once)
                    # print('grand total update',ax,table[tuplize(cell)], residual_margins[ax],residual_margins[ax]-table[tuplize(cell)])
                    margins[ax] -= table[tuplize(cell)].to(int32)
        return margins


    def update_cell_constraints_deterministically(self):
        # Get latest constraint table
        constraint_table = self.constraint_table()
        # Get flags for whether cells are constrained or not
        cell_unconstrained = (constraint_table == -1)
        # Repeat process if there any margins with only one unknown
        while any([any(cell_unconstrained.sum(axis = ax,keepdims = True).flatten() == 1) for ax in self.constraints['constrained_axes']]):
            # If cells need to be filled in deterministically the entire table must be available
            try:
                assert self.table_nonnegative(self.data.ground_truth_table)
            except:
                self.logger.error('Table has negative entries')
                raise Exception('Cannot fill in remainder of cells determinstically as table values are not provided for these cells.')
            # Go through each margin
            for ax in self.constraints['constrained_axes']:
                # If there any margins with only one unknown cell
                if any(cell_unconstrained.sum(dim = tuplize(ax),keepdims = True).flatten() == 1):
                    # Get axis complement
                    ax_complement = list(flatten(self.axes_complement(ax, same_length = True)))
                    # Create cell index
                    cell_index = [{
                        "axis": ax,
                        "index": list(flatten(np.where(cell_unconstrained.sum(dim = tuplize(ax)) == 1)))
                    }]
                    # Update last index
                    last_axis_index = cell_index[-1]['index']
                    for subax in ax_complement:
                        # Find indices
                        cell_index.append({
                            "axis": subax,
                            "index": np.where(cell_unconstrained.take(indices = last_axis_index[0], dim = subax))
                        })
                        # Update last index
                        last_axis_index = cell_index[-1]['index']
                    # Construct cell index
                    query_index = np.zeros(ndims(self), dtype='uint16')
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
        if len(self.constraints.get('cells',[])) > 0:
            # Update cell constraints if there are any deterministic solutions
            # based on the cell and margin constraints provided
            self.update_cell_constraints_deterministically()
            # Update residual margins
            self.update_residual_margins_from_cell_constraints(self.constraints['cells'])

    def constrained_margins(self,table = None):
        if table is None:
            return {ax:self.data.margins[ax] for ax in self.constraints['constrained_axes']}
        else:
            return {ax:table.sum(dim = tuplize(ax), keepdims = True, dtype = int32).flatten() for ax in self.constraints['constrained_axes']}

    def unconstrained_margins(self,table = None):
        if table is None:
            return {ax:self.data.margins[ax] for ax in self.constraints['unconstrained_axes']}
        else:
            return {ax:table.sum(dim = tuplize(ax), keepdims = True, dtype = int32).flatten() for ax in self.constraints['unconstrained_axes']}

    def update_unconstrained_margins_from_table(self, table):

        for ax in self.constraints['unconstrained_axes']:
            self.data.margins[tuplize(ax)] = table.sum(
                dim = tuplize(ax), 
                keepdims = True, 
                dtype = int32
            ).flatten().to(device = self.device)
            self.residual_margins[tuplize(ax)] = table.sum(
                dim = tuplize(ax), 
                keepdims = True, 
                dtype = int32
            ).flatten().to(device = self.device)

    def update_constrained_margins_from_table(self, table):
        for ax in self.constraints['constrained_axes']:
            self.data.margins[tuplize(ax)] = table.sum(
                dim = tuplize(ax), 
                keepdims = True, 
                dtype = int32
            ).flatten().to(device = self.device)
            self.residual_margins[tuplize(ax)] = table.sum(
                dim = tuplize(ax), 
                keepdims = True, 
                dtype = int32
            ).flatten().to(device = self.device)

    def update_margins_from_table(self, table):
        for axes in powerset(range(ndims(self))):
            margin_data = table.sum( 
                dim = tuplize(axes), 
                keepdims = True, 
                dtype = int32
            ).flatten().to(dtype = int32,device = self.device)
            # Update margins iff they are non-negative
            if torch.all(margin_data >= 0):
                self.data.margins[tuplize(axes)] = margin_data
                self.residual_margins[tuplize(axes)] = margin_data
    
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
    
    def update_table(self, tab: torch.tensor) -> None:
        try:
            assert np.array_equal(np.shape(tab), np.asarray(list(self.dims)))
        except:
            self.logger(f"Cannot update table of dims {np.asarray(list(self.dims))} to table of dims {np.shape(tab)}")
            raise Exception('Table update failed.')
        # Update table
        if not torch.is_tensor(tab):
            self.data.ground_truth_table = torch.from_numpy(tab).int().to(dtype = int32,device = self.device)
        else:
            self.data.ground_truth_table = tab.int().to(dtype = int32,device = self.device)
        # Update table properties
        self.update_table_properties_from_table()

    def update_table_properties_from_table(self) -> None:
        # Update dimensions
        self.data.dims = dict(zip(self.dim_names,np.asarray(np.shape(self.data.ground_truth_table), dtype=CORE_COORDINATES_DTYPES['origin'])))
        # Update margins
        self.update_margins_from_table(self.data.ground_truth_table)
    
    def export(self, dirpath: str = './synthetic_dummy', overwrite: bool = False) -> None:
        # Make directory if it does not exist
        makedir(dirpath)

        if self.data.ground_truth_table is not None:
            # Get filepath experiment filepath
            filepath = os.path.join(dirpath, 'table.txt')

            # Write experiment summaries to file
            if (not os.path.exists(filepath)) or overwrite:
                write_txt(self.data.ground_truth_table.cpu().detach().numpy(), filepath)

        for axis, margin in self.data.margins.items():
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
        for n in range(1,ndims(self)):
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
    def __init__(self, config: Config = None, **kwargs: dict):
        # Set configuration file
        super().__init__(config, **kwargs)


class ContingencyTableDependenceModel(ContingencyTable):
    def __init__(self, config: Config = None, **kwargs: dict):
        # Set configuration file
        super().__init__(config, **kwargs)


class ContingencyTable2D(ContingencyTableIndependenceModel, ContingencyTableDependenceModel):

    def __init__(self, config: Config = None, **kwargs: dict):
        # Set configuration file
        super().__init__(config, **kwargs)
        # Build
        self.build(config = config, **kwargs)

    def admissibility_debugging(self,sampler_name,table0):
        try:
            assert self.table_admissible(table0)
        except:
            constraint_table = self.constraint_table()
            table0_copy = deepcopy(table0)
            table0_copy[constraint_table>=0] = -1
            self.logger.debug(f'Generated table (free cells)')
            self.logger.debug(table0_copy)
            self.logger.debug('True table')
            self.logger.debug(self.data.ground_truth_table)
            if not self.table_cells_admissible(table0):
                table0_copy = deepcopy(table0)
                table0_copy[constraint_table<0] = -1
                self.logger.debug("Cell constraints equal")
                self.logger.debug(np.array_equal(table0_copy,constraint_table))
                self.logger.debug('Cell constraint difference')
                self.logger.debug(table0_copy-constraint_table)
                self.logger.debug("table0_fixed_cells")
                self.logger.debug(table0_copy)
                # print("constraints")
                # print(constraint_table)
                raise Exception(f'{sampler_name} sampler yielded a cell-inadmissible contingency table')
            elif not self.table_margins_admissible(table0):
                table0_copy = deepcopy(table0)
                table0_copy[constraint_table<0] = 0
                self.logger.debug("total value of constrained cells")
                self.logger.debug(constraint_table[constraint_table>0].sum())
                self.logger.debug("total value of residual margins")
                self.logger.debug([v.sum() for v in self.residual_margins.values()])
                self.logger.debug('table total')
                self.logger.debug(torch.ravel(self.data.ground_truth_table.sum()))
                for ax in sorted(self.constraints['constrained_axes']):
                    if len(ax) < ndims(self):
                        self.logger.debug(f"True summary statistics")
                        self.logger.debug(self.data.margins[tuplize(ax)])
                        self.logger.debug(f"Residual summary statistics")
                        self.logger.debug(self.residual_margins[tuplize(ax)])
                        self.logger.debug(f"Fixed cell summary statistics")
                        self.logger.debug(table0_copy.sum(dim = tuplize(ax),keepdims = True).flatten())
                        self.logger.debug(f'Current summary statistics')
                        self.logger.debug(table0.sum(dim = tuplize(ax),keepdims = True).flatten())
                raise Exception(f'{sampler_name} sampler yielded a margin-inadmissible contingency table')
            else:
                raise Exception('Unrecognized error encountered.')
    
    def margin_import(self, axis, intensity: list = None):

        margin0 = None
        # Read initialised table
        if 'margin0' in list(self.config.settings['mcmc']['contingency_table'].keys()):
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
            elif isinstance(initialisation, (list, torch.tensor)) and \
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
            intensity = np.repeat(
                1./np.prod(list(self.dims)), 
                np.prod(list(self.dims))
            ).reshape(np.asarray(list(self.dims)))
        # Sum probabilities to one in the case of multinomial sampling
        if len(axis) < ndims(self):
            intensity /= intensity.sum()
        
        while (not self.margin_sparse_admissible(margin)) or (firstIteration):
            if len(axis) < ndims(self):
                margin = np.random.multinomial(
                    n = self.data.margins[tuplize(range(ndims(self)))], 
                    pvals = intensity.sum(dim = axis)
                )
            else:
                # This is sampling the grand total
                margin = np.array([np.random.poisson(intensity.sum(dim = axis))])
            # The loop can now end
            firstIteration = False
        return margin

    def table_monte_carlo_sample(self,intensity:list = None, margins: dict = {}, **__) -> Union[torch.tensor, None]:
        # Update margins
        if margins is not None:
            self.update_margins(margins)

        # Define axes
        constrained_axis1 = (1,)
        constrained_axis2 = (0,)
        
        # Initialise table to zero
        table0 = torch.zeros(tuple(list(self.dims)), dtype = float32)

        # Get N (sum or row or column sums)
        N = self.residual_margins[tuplize(range(ndims(self)))].detach().cpu().numpy()[0]

        # Generate permutation of N items
        permutation = torch.randperm(N) + torch.tensor(1,dtype = int32)

        # Loop through first axis
        for i in range(self.data.dims[self.dim_names[constrained_axis2]]):
            # Look at first r_i items
            if i == 0:
                permutation_subset = permutation[0 : (self.residual_margins[constrained_axis1][i])]
            else:
                permutation_subset = permutation[torch.sum(self.residual_margins[constrained_axis1][:(i)]).to(int32):
                                                torch.sum(self.residual_margins[constrained_axis1][:(i+1)]).to(int32)]
                    
            # Loop through columns
            for j in range(self.data.dims[self.dim_names[constrained_axis1]]):
                # Create index appropriately
                query_index = np.zeros(ndims(self), dtype='uint16')
                query_index[constrained_axis2] = i
                query_index[constrained_axis1] = j
                query_index = tuplize(query_index)

                if query_index in self.constraints['cells']:
                    # Initialise cell at constraint
                    table0[query_index] = self.data.ground_truth_table[query_index].to(float32)
                else:
                    # Count number of entries between c_1+...+c_{j-1}+1 and c_1+...+c_j
                    if j == 0:
                        table0[query_index] = (
                            ((1 <= permutation_subset) & \
                             (permutation_subset <= self.residual_margins[constrained_axis2][j])).sum()
                        ).to(float32)
                    else:
                        table0[query_index] = (
                            (((torch.sum(self.residual_margins[constrained_axis2][:j])+1) <= permutation_subset) & \
                            (permutation_subset <= torch.sum(self.residual_margins[constrained_axis2][:(j+1)]))).sum()
                        ).to(float32)
        self.admissibility_debugging('Monte Carlo',table0)
        return table0.to(device = self.device,dtype = float32)
    
    def table_import(self,intensity:list = None, margins: dict = {}, **__) -> Union[torch.tensor, None]:
        # Read initial table
        table0 = None
        if 'table0' in list(self.config.settings['mcmc']['contingency_table'].keys()):
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
            elif isinstance(initialisation, (list, torch.tensor)) and \
                    (np.shape(initialisation) == tuple(list(self.dims))):
                table0 = initialisation
            else:
                self.logger.warning(
                    f"Initial table {initialisation} is neither a list nor a valid filepath.")
        else:
            self.logger.error(f"No table 0 provided in config.")
        
        if table0 is None:
            raise Exception('Table 0 could not be found.')
        else:
            return table0.to(device = self.device,dtype = float32)

    def table_random_sample(self,intensity:list = None, margins: dict = {}, **__) -> Union[torch.tensor, None]:

        try:
            assert len(self.constraints['constrained_axes']) == 0
        except:
            raise Exception ('table_random_sample is only used by tables with no margin constraints')
        
        # Initialise table to zero
        table0 = torch.zeros(tuple(list(self.dims)), dtype = float32)
        min_cell_value,max_cell_value = 0,1
        for cell in self.constraints['cells']:
            table0[cell] = self.data.ground_truth_table[cell].to(float32)
            min_cell_value = min(min_cell_value,table0[cell])
            max_cell_value = max(max_cell_value,table0[cell])
        
        # Define support to sample from
        cell_value_range = [min(min_cell_value,1), max(max_cell_value,100)]
        # Update remaining table cells
        for cell in self.cells:
            # Sample value randomly
            table0[cell] = torch.randint(low = cell_value_range[0],high = cell_value_range[1],size=(1,),dtype = float32)

        self.admissibility_debugging('Random solution',table0)
        return table0.to(device = self.device,dtype = float32)

    def table_direct_sampling(self, intensity:list = None, margins:dict = {}, **kwargs):
        # Get direct sampling proposal
        direct_sampling_proposal = getattr(kwargs['ct_mcmc'],f"direct_sampling_proposal_{ndims(self)}way_table")
        table0 = direct_sampling_proposal(
            table_prev = None,
            log_intensity = torch.log(intensity)
        )
        self.admissibility_debugging('Maximum entropy solution',table0)
        return table0.to(device = self.device,dtype = float32)

    def table_maximum_entropy_solution(self, intensity:list = None, margins: dict = {}, **__) -> Union[torch.tensor, None]:
        '''
        This is the solution of the maximum entropy in the non-integer case
        X_ij = (r_i*c_j) / T
        '''
        # Update margins
        if margins is not None:
            self.update_margins(margins)

        # Ground truth table
        ground_truth_table = deepcopy(self.data.ground_truth_table).to(dtype = float32,device = self.device)
        # Initialise table to zero
        table0 = torch.zeros(tuple(list(self.dims))).to(dtype = float32,device = self.device)
        # Free cells
        free_cells = set(deepcopy(self.cells))
        # Unchosen free cells
        unchosen_free_cells = deepcopy(free_cells)
        # Keep track of zero tensor
        zero_tensor = torch.tensor(0.0,dtype=float32,device=self.device)

        # Apply cell constaints if at least one cell is fixed
        if len(self.constraints['cells']) > 0:
            # Extract indices
            cells = torch.tensor(self.constraints['cells'], dtype=torch.long, device = self.device)
            rows, cols = cells[:, 0], cells[:, 1]
            # Fix table cells
            table0[rows, cols] = ground_truth_table[rows, cols]

        # Keep a copy of summay statitics
        residual_margins = deepcopy(self.residual_margins)
        # Get only constrained residual margins
        residual_margins = {k:v for k,v in residual_margins.items() if k in self.constraints['constrained_axes']}
        
        self.logger.debug(f"INIT RESIDUAL MARGINS {dict({k:v.sum() for k,v in residual_margins.items()})}")

        table_admissible = False
        while not table_admissible:
            # self.logger.iteration(f"{len(unchosen_free_cells)} to choose from")
            random_cell_idx = np.random.randint(len(unchosen_free_cells))
            random_cell = list(unchosen_free_cells)[random_cell_idx]
            # Find each cell's minimum allowable by the margins contribution
            cell_min_contribution = self.minres_given_constraints(random_cell,residual_margins)
            # if no contribution is allowed then this cell must be fixed
            if cell_min_contribution <= 0:
                # self.logger.iteration(f"{random_cell} cell removed")
                free_cells.remove(random_cell)
                unchosen_free_cells.remove(random_cell)
            # if contribution is allowed then add 1
            else:
                # Contribution is 1
                contribution = torch.tensor(1.0,dtype=float32,device=self.device)
                # Add 1 to table
                table0[random_cell] += contribution
                self.logger.iteration(f"{random_cell}: {contribution}. Total residual: {residual_margins[tuplize(range(ndims(self)))]}")
                # Update residual margins
                residual_margins = self.update_margins_from_cells(
                    margins = residual_margins, 
                    axes = list(residual_margins.keys()),
                    constrained_cells = [random_cell],
                    table = {random_cell:contribution}
                )
                unchosen_free_cells.remove(random_cell)
            # If all cells have been chosen at least once, then reset possible cell choices
            if len(unchosen_free_cells) <= 0:
                unchosen_free_cells = deepcopy(free_cells)

            if torch.equal(residual_margins[tuplize(range(ndims(self)))],zero_tensor) or \
                len(free_cells) <= 0:
                table_admissible = self.table_admissible(table0)

        zero_cells = 0
        for c in self.cells:
            if table0[c[0],c[1]] <= 0: zero_cells += 1
        self.logger.progress(f"{zero_cells} zero cells out of {len(self.cells)}")
        self.admissibility_debugging('Maximum entropy solution',table0)
        return table0.to(device = self.device,dtype = float32)
        
    def minres_given_constraints(self,cell,residual_margins):
        # get all contraint axes
        ax_constraints = list(residual_margins.keys())
        
        # If empty return residual 1 for all non-zero entries
        if len(ax_constraints) <= 0:
            return torch.tensor(1,dtype=float32,device=self.device)
        
        # If at least one margin is fixed
        else:
            applicable_margins = []
            for cax,margin in residual_margins.items():
                if cax == (0,1):  # Total
                    applicable_margins.append(margin)
                elif cax == (0,):
                    applicable_margins.append(margin[cell[1]])
                elif cax == (1,):
                    applicable_margins.append(margin[cell[0]])

            return min(applicable_margins)
