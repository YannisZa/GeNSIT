import os
import sys
import torch
import time
import collections
import numpy as np
import pandas as pd

from copy import deepcopy
from torch import int32
from tqdm.auto import tqdm
from typing import Dict,Tuple,List,Set
from joblib import Parallel, delayed

from gensit.utils.misc_utils import f_to_df,df_to_f,f_to_array, makedir, setup_logger,write_compressed_string,read_compressed_string, unpack_dims, ndims, flatten
from gensit.contingency_table import ContingencyTable

def instantiate_markov_basis(ct:ContingencyTable,**kwargs): #-> Union[MarkovBasis,None]:
    if hasattr(sys.modules[__name__], ct.markov_basis_class):
        return getattr(sys.modules[__name__], ct.markov_basis_class)(ct,**kwargs)
    else:
        raise Exception(f"Input class {ct.markov_basis_class} not found")

class MarkovBasis(object):
    def __init__(self, ct:ContingencyTable,**kwargs):
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__,
            console_level = level,
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        
        # Update logger level
        self.logger.setLevels( console_level = level )
        
        # Get contingency table
        self.ct = ct
        
        # Enable/disable tqdm
        self.tqdm_disabled = self.ct.config['contingency_table'].get('disable_tqdm',True)

        # Get number of workers
        self.n_threads = self.ct.config['inputs'].get('n_threads',1)
        
        # Raise implementation error if more constraints are provided than implemented
        if ndims(self.ct) > 2:
            self.logger.error('Markov Bases for multi-way tables has not been implemented yet.')
            raise Exception('Markov Bases cannot be constructed.')
        
        # Get flag for storing markov basis
        self.export = True
        if 'outputs' in list(self.ct.config.settings.keys()) and \
            'export_basis' in list(self.ct.config.settings['outputs'].keys()):
            self.export = self.ct.config.settings['outputs']['export_basis']
        
        # Initialise bases
        # self.basis_functions = None
        self.basis_dictionaries = None

        # Basis function index
        self.i = 0

    def generate(self) -> None:
        pass

    def build(self) -> None:
        # Try to import markov basis
        imported = False 
        # This is too Slowwww....
        # imported = self.import_markov_basis()

        if not imported:
            self.logger.debug('Generating Markov Basis...')
            # Generate bases
            self.generate()

            self.logger.debug('Checking Markov Basis validity...')
            # Check that all basis are admissible
            try:
                # This is too Slowwww...
                # assert self.check_markov_basis_validity_sequentially()
                assert True
            except:
                self.logger.error(f"Invalid Markov bases functions found.")
                raise Exception('Inadmissible Markov basis functions found or wrong number of Markov basis functions generated')

            if self.export:
                self.logger.debug('Exporting Markov basis...')
                # Store bases
                self.export_markov_basis()

    def active_candidates(self,cell:Tuple[int,int],all_cells:List[Tuple[int,int]]):
        # Returns all possible candidates of + sign cells for every remaining inactive cell  in `all_cells`
        # The `cell` provided already has been given a + sign
        # Active cell candidates for all_cells before index have already been examined
        # and are disregarded from the search
        # ASSUMPTION 1: all_cells have been sorted lexicographically (first by row and then by col index) e.g. for 1x3 table we get
        # all_cells = [(0,0),(0,1),(0,2)]

        # The remaining cells that have not been examined yet must have a col index > that the col index of cell
        # This is because all previous cells before `cell` have already been activated (with either a + or a - sign)
        # Also cells that share the row index with `cell` cannot be activated (set to +) as that would violate the table margins
         # Get all valid candidate
        constrained_axes = set(self.ct.constraints['constrained_axes'])
        if (len(constrained_axes.symmetric_difference({(1,),(0,1)}))==0):
            return [c for c in all_cells if c[1] > cell[1] and c[0] == cell[0]]
        elif (len(constrained_axes.symmetric_difference({(0,),(0,1)}))==0):
            return [c for c in all_cells if c[0] > cell[0] and c[1] == cell[1]]
        elif (len(constrained_axes.symmetric_difference({(0,),(1,),(0,1)}))==0):
            return [c for c in all_cells if c[0] != cell[0] and c[1] > cell[1]]
        else:
            raise Exception(f"No active candidates found for constrained axes {constrained_axes}")

    def basis_function_admissible(self,f) -> bool:
        # This is checking condition 1.5a in Diaconis 1992 paper for a single choice of Markov basis
        # checks that the markov basis has at least one non-zero entry
        # and finally checks that that the markov basis does not change any of the fixed cells
        tab = torch.tensor(f_to_array(f),dtype = int32)
        # basis fully expanded to match table dims
        full_tab = torch.tensor(f_to_array(f,shape = unpack_dims(self.ct.data.dims,time_dims = False)),dtype = int32)

        return (not torch.any(self.ct.table_constrained_margins_summary_statistic(tab))) and \
                (torch.any(tab)) and \
                torch.all(self.ct.table_cell_constraint_summary_statistic(full_tab)==0)
        
    def check_markov_basis_validity_sequentially(self) -> bool:
        for i in tqdm(
            range(len(self.basis_dictionaries)),
            disable = False,
            desc = 'Checking Markov Basis validity sequentially',
            leave = False
        ):
            if not self.check_markov_basis_validity(i):
                return False
        return True

    def check_markov_basis_validity_concurrently(self) -> bool:
        return np.all(list(
            Parallel(n_jobs = self.n_threads)(
                delayed(self.check_markov_basis_validity)(i) for i in tqdm(
                    range(len(self.basis_dictionaries)),
                    disable = self.tqdm_disabled,
                    desc = 'Checking Markov Basis validity concurrently',
                    leave = False
                )
            )
        ))

    def check_markov_basis_validity(self,index) -> bool:
        return self.basis_function_admissible(self.basis_dictionaries[index])

    def random_basis_function(self) -> pd.DataFrame:
        return f_to_df(self.basis_dictionaries[np.random.choice(range(len(self.basis_dictionaries)))])


    def __str__(self):
        return '\n'.join([f_to_df(item).to_string() for item in self.basis_dictionaries])

    def __len__(self):
        return len(self.basis_dictionaries)

    def __iter__(self):
        return iter(self.basis_dictionaries)

    def __next__(self):
        if self.i < self.__len__():
            bf = self.basis_dictionaries[self.i]
            self.i += 1
            return bf
        else:
            raise StopIteration

    def generate_one_margin_preserving_markov_basis(self) -> None:
        # Create list of row-column pairs such that no pair share the same column
        basis_cells = []

        # Get all cells in lexicographic order (order first by row and then by col index)
        sorted_cells = sorted(self.ct.cells)
        sorted_cells_set = set(sorted_cells)

        # Loop through each pair combination and keep only ones that don't share a row OR column
        for index,tup1 in tqdm(
            enumerate(sorted_cells),
            total = len(sorted_cells),
            disable = self.tqdm_disabled,
            desc = 'Generating one margin Markov Basis cells',
            leave = False
        ):
            # Get all active candidate cells
            inactive_candidate_cells = []
            if index < len(sorted_cells)-1:
                inactive_candidate_cells = self.active_candidates(tup1,sorted_cells[(index+1):])
            # Loop through inactive candidates
            for tup2 in inactive_candidate_cells:
                # Every cell in the proposed basis should be entirely contained
                # in the list of available (free) cells.
                if set([tup1,tup2]).issubset(sorted_cells_set):
                    basis_cells.append((tup1,tup2))

        # Define set of Markov bases
        # self.basis_functions = []
        self.basis_dictionaries = []
        # Define active cells i.e. cells of a basis function that map to non-zero values
        self.basis_active_cells = []
        for index in tqdm(
            range(len(basis_cells)),
            disable = self.tqdm_disabled,
            desc = 'Generating one margin Markov Basis functions',
            leave = False
        ):
            # This is commented out because it checked in the previous loop
            # # Make sure that no cell in the basis is a constrained cell
            # if np.any([basis_cell in self.ct.constraints['cells'] for basis_cell in  basis_cells[index]]):
            #     continue
            
            # Construct Markov basis function
            def make_f_i(findex):
                def f_i(x):
                    return int(x == basis_cells[findex][0]) - int(x == basis_cells[findex][1])
                return f_i
            # Make function
            my_f_i = make_f_i(index)
            my_f_i_dict = dict(zip(basis_cells[index],list(map(my_f_i, basis_cells[index]))))
            # Update cells that map to non-zero values (i.e. active cells)
            # https://stackoverflow.com/questions/46172705/how-to-omit-keys-with-empty-non-zero-values
            # self.basis_active_cells.append({k: v for k, v in my_f_i_dict.items() if v != 0})
            # Add function to list of Markov bases
            # self.basis_functions.append(my_f_i)
            self.basis_dictionaries.append(my_f_i_dict)
            
    def construct_both_margin_preserving_markov_basis_for_cell(self,index,tup1,sorted_cells,sorted_cells_set):
        # self.logger.hilight(f"Finding active candidates")
        # Get all active candidate cells
        active_candidate_cells = []
        if index < len(sorted_cells)-1:
            active_candidate_cells = self.active_candidates(tup1,sorted_cells[(index+1):])
        # self.logger.hilight(f"Looping over {len(active_candidate_cells)} active candidates")
        # Loop through active candidates
        basis_functions = []
        # for tup2 in active_candidate_cells:
        #     # Every cell in the proposed basis should be entirely contained
        #     # in the list of available (free) cells.
        #     basis_cells = (tup1,tup2,(tup1[0],tup2[1]),(tup2[0],tup1[1]))
            
        #     # if all([bc in sorted_cells for bc in basis_cells]):
        #     if set(basis_cells).issubset(sorted_cells_set):
            
        #         # Construct Markov basis move and store it in a dictionary
        #         basis_functions.append({
        #             basis_cells[0]: np.int8(1),
        #             basis_cells[1]: np.int8(1),
        #             basis_cells[2]: np.int8(-1),
        #             basis_cells[3]: np.int8(-1)
        #         })
        
        if len(active_candidate_cells) > 0:
            # Randomly choose an activate candidate cell
            randm_idx = np.random.choice(list(range(len(active_candidate_cells))))
            
            tup2 = active_candidate_cells[randm_idx]
            # Every cell in the proposed basis should be entirely contained
            # in the list of available (free) cells.
            basis_cells = (tup1,tup2,(tup1[0],tup2[1]),(tup2[0],tup1[1]))
            
            # if all([bc in sorted_cells for bc in basis_cells]):
            if set(basis_cells).issubset(sorted_cells_set):
            
                # Construct Markov basis move and store it in a dictionary
                basis_functions.append({
                    basis_cells[0]: np.int8(1),
                    basis_cells[1]: np.int8(1),
                    basis_cells[2]: np.int8(-1),
                    basis_cells[3]: np.int8(-1)
                })
        
        return basis_functions
    
    def build_basis_dictionaries_in_sequence(self):
        # Get all cells in lexicographic order (order first by row and then by col index)
        sorted_cells = sorted(self.ct.cells)
        sorted_cells_set = set(sorted_cells)
        
        # Define set of Markov bases
        self.basis_dictionaries = []
        # Loop through each pair combination and keep only ones that don't share a row OR column
        for index,tup1 in enumerate(tqdm(
            sorted_cells,
            total = len(sorted_cells),
            disable = self.tqdm_disabled,
            desc = 'Generating both margin Markov Basis cells in sequence',
            leave = False
        )):
            # Add function to list of Markov bases
            self.basis_dictionaries += self.construct_both_margin_preserving_markov_basis_for_cell(
                index,
                tup1,
                sorted_cells = sorted_cells,
                sorted_cells_set = sorted_cells_set
            )
    
    def build_basis_dictionaries_in_parallel(self):
        # Get all cells in lexicographic order (order first by row and then by col index)
        sorted_cells = sorted(self.ct.cells)
        sorted_cells_set = set(sorted_cells)
        
        # Process active flag by tqdm position
        self.basis_dictionaries = list(flatten(
            Parallel(n_jobs = self.n_threads)(
                delayed(self.construct_both_margin_preserving_markov_basis_for_cell)(
                    index = i,
                    tup1 = tup1,
                    sorted_cells = sorted_cells,
                    sorted_cells_set = sorted_cells_set
                ) for i,tup1 in tqdm(
                    enumerate(sorted_cells),
                    total = len(sorted_cells),
                    disable = False,# self.tqdm_disabled,
                    desc = 'Generating both margin Markov Basis cells in parallel',
                    leave = False
                )
            )
        ))

    def generate_both_margin_preserving_grobner_markov_basis(self) -> None:
        # Get all cells in lexicographic order (order first by row and then by col index)
        self.logger.info(f"{len(self.ct.cells)} free cells + {len(self.ct.constraints['cells'])} constrained cells out of {np.prod(list(self.ct.data.dims.values()))} total cells ({len(self.ct.cells)+len(self.ct.constraints['cells'])} = {np.prod(list(self.ct.data.dims.values()))})")

        
        self.graph = collections.defaultdict(set)
        self.origins = set()
        self.destinations = set()
        for (o,d) in self.ct.cells:
            self.graph[f"o{o+1}"].add(f"d{d+1}")
            self.graph[f"d{d+1}"].add(f"o{o+1}")
            self.origins.add(f"o{o+1}")
            self.destinations.add(f"d{d+1}")
        self.unique_cycles = set()

        # Find all cycles in graph
        all_cycles = self.find_all_cycles(max_length=10)
        # Define set of Markov bases
        self.basis_dictionaries = []
        for cycle in tqdm(all_cycles,desc='Construct Markov basis from cycles'):
            mb_move = {}
            counter = 0
            cycle_complete = cycle+[cycle[0]]
            for p1, p2 in zip(cycle_complete, cycle_complete[1:]):
                origin = int(p1.replace('o','') if 'o' in p1 else p2.replace('o',''))-1
                destination = int(p2.replace('d','') if 'd' in p2 else p1.replace('d',''))-1
                # Convert bipartite graph cycle to markov basis move
                mb_move[(origin,destination)] = 1 if counter % 2 == 0 else -1
                counter += 1
            # Add to markov bases
            self.basis_dictionaries.append(mb_move)
                # self.logger.info(f"Cycle ({len(cycle)}): {cycle}.\n MB move: {mb_move}")
                # self.logger.info(f"Cycle ({len(cycle)}).\n MB move: {mb_move}")
        # for mb_move in tqdm(self.basis_dictionaries,total=len(self.basis_dictionaries)):
        #     self.logger.info(f"{mb_move}")

    
    def find_all_cycles(self, max_length: int) -> list[list]:
        """
        Finds all unique simple cycles starting from each origin node, but only up to length `max_length`.

        Args:
            max_length:  The maximum allowed number of nodes in a cycle (i.e. #edges in the cycle).
        
        Returns:
            A list of lists, each inner list is a unique cycle (in its canonical form),
            where len(cycle) <= max_length.
        """
        self.unique_cycles.clear()  # reset from any previous call

        # For each origin, start a DFS.  We track path as a list of nodes, plus a set for O(1) membership checks.
        for origin_node in tqdm(self.origins, total=len(self.origins), desc=f'Find cycles of length {max_length}'):
            # Initialize path = [origin_node].  We mark origin_node as “visited” in path_nodes_set.
            self._dfs_find_cycles(
                start_node=origin_node,
                current_node=origin_node,
                path=[origin_node],
                path_nodes_set={origin_node},
                max_length=max_length
            )

        # Convert each canonical tuple back to a list before returning.
        return [list(cycle_tuple) for cycle_tuple in self.unique_cycles]

    def _dfs_find_cycles(
        self,
        start_node: any,
        current_node: any,
        path: list,
        path_nodes_set: set,
        max_length: int
    ):
        """
        Recursive DFS to discover cycles, but never allow a cycle's node‐count to exceed max_length.

        Args:
            start_node:      The origin where this DFS began.
            current_node:    The node we are currently visiting.
            path:            List of nodes from start_node up to current_node (inclusive).
            path_nodes_set:  A set of the same nodes in 'path', for O(1) membership checks.
            max_length:      The maximum permitted number of nodes in any cycle.

        Behavior:
            - If neighbor == start_node and len(path) >= 3 and len(path) <= max_length:
                we have a valid cycle (origin → … → origin).  (len(path) is #distinct nodes.)
            - If neighbor is already in path_nodes_set (and not equal to start_node), skip it (no revisiting).
            - Otherwise, if adding the neighbor would make path‐length > max_length, skip recursing.
            - Else, add neighbor and recurse.
        """
        for neighbor in self.graph[current_node]:
            # ----- CASE 1: We might close a cycle by returning to start_node -----
            if neighbor == start_node and len(path) >= 3:
                # If we appended start_node, the cycle would have node‐count = len(path) + 1,
                # but we only store the “distinct nodes” (we always strip off the duplicate tail before canonicalizing).
                # In your code, you do: cycle = path + [start_node], then strip off the last in _add_unique_cycle.
                # Here, `len(path)` is exactly the number of distinct nodes.  So check:
                if len(path) <= max_length:
                    cycle = path + [start_node]
                    self._add_unique_cycle(cycle)
                # Whether we recorded it or not, continue exploring other neighbors from current_node:
                continue

            # ----- CASE 2: Already visited (non‐start) ⇒ skip to avoid non‐simple path -----
            if neighbor in path_nodes_set:
                continue

            # ----- CASE 3: Check if adding this neighbor would exceed max_length -----
            # Currently path has `len(path)` distinct nodes.  If we append this neighbor as a new node,
            # we would end up with `len(path) + 1` distinct nodes.  If that is > max_length, skip recursing.
            if len(path) + 1 > max_length:
                continue

            # ----- CASE 4: Safe to go deeper -----
            path.append(neighbor)
            path_nodes_set.add(neighbor)

            # Recurse, flipping to neighbor, but we remain in origin/destination logic if needed.
            # (Your original code didn’t enforce a strict bipartite check here, but presumably
            #  the bipartite property was encoded in self.graph.  We simply keep DFS going.)
            self._dfs_find_cycles(
                start_node=start_node,
                current_node=neighbor,
                path=path,
                path_nodes_set=path_nodes_set,
                max_length=max_length
            )

            # Backtrack:
            path_nodes_set.remove(neighbor)
            path.pop()

    def _add_unique_cycle(self, cycle_path: list):
        """
        Canonicalize a discovered cycle (path includes duplicated start at end) and store it if new.

        Args:
            cycle_path: e.g. ['o1','d2','o2','d3','o1']
        """
        # Strip off the final duplicate of the start:
        nodes_in_cycle = tuple(cycle_path[:-1])
        if not nodes_in_cycle:
            return

        # Find the lexicographically smallest node (so that rotation always begins there).
        min_node = min(nodes_in_cycle)

        # Collect all indices where the cycle touches min_node
        min_indices = [i for i, node in enumerate(nodes_in_cycle) if node == min_node]
        canonical_form = None

        for start_index in min_indices:
            # 1) Forward rotation so that nodes_in_cycle[start_index] is first
            forward = collections.deque(nodes_in_cycle)
            forward.rotate(-start_index)
            forward_tuple = tuple(forward)

            # 2) Reverse the entire cycle, then rotate so that min_node is first in the reversed list
            rev_list = list(nodes_in_cycle)[::-1]
            try:
                rev_start_idx = rev_list.index(min_node)
            except ValueError:
                reverse_tuple = forward_tuple
            else:
                rev_deque = collections.deque(rev_list)
                rev_deque.rotate(-rev_start_idx)
                reverse_tuple = tuple(rev_deque)

            candidate = forward_tuple if forward_tuple <= reverse_tuple else reverse_tuple
            if canonical_form is None or candidate < canonical_form:
                canonical_form = candidate

        if canonical_form:
            self.unique_cycles.add(canonical_form)


    def generate_both_margin_preserving_markov_basis(self) -> None:
        # Get all cells in lexicographic order (order first by row and then by col index)
        self.logger.info(f"{len(self.ct.cells)} free cells + {len(self.ct.constraints['cells'])} constrained cells out of {np.prod(list(self.ct.data.dims.values()))} total cells ({len(self.ct.cells)+len(self.ct.constraints['cells'])} = {np.prod(list(self.ct.data.dims.values()))})")

        self.build_basis_dictionaries_in_sequence()
        # if self.n_threads == 1:
        #     start_time = time.time()
        #     self.build_basis_dictionaries_in_sequence()
        #     print('Sequentially',time.time()-start_time)
        # else:
        #     start_time = time.time()
        #     self.build_basis_dictionaries_in_parallel()
        #     print('In parallel',time.time()-start_time)

        self.logger.hilight(f"{len(self.basis_dictionaries)} basis functions found")

    def import_basis_function(self,filepath:str) -> Dict:
        # Import basis function from csv
        table = pd.read_csv(filepath,index_col = 0,header = 0)
        table.index = table.index.astype('str')
        # Remove zeros
        table_dict = df_to_f(table)
        table_dict = {k: v for k, v in table_dict.items() if v != 0}
        return table_dict

    def import_markov_basis(self) -> None:
        # WARNING: IT IS POSSIBLE TO LOAD A MARKOV BASIS WITH DIFFERENT CELL CONSTRAINTS 
        # THAN THE ONES PROVIDED. THEREFORE BE VERY CAREFUL TO ENSURE THAT YOU LOAD A MARKOV BASIS
        # WITH NO CELL CONSTRAINTS AND THEN DISREGARD MOVES INCOMPATIBLE WITH CONSTRAINTS PROVIDED.

        # Check if output directory is provided
        if not 'outputs' in list(self.ct.config.settings.keys()) or \
            not 'out_directory' in list(self.ct.config.settings['outputs'].keys()) or \
            self.ct.config.settings['outputs']['out_directory'] == '':
            self.logger.warning(f'Output directory not provided. Markov bases cannot be found.')
            # Set export flag to false
            self.export = False
            return False
        # Define filepath
        dirpath = os.path.join(
            self.ct.config['outputs']['out_directory'],
            'markov_basis/'
        )
        
        # Create filepath
        unpacked_dims = unpack_dims(self.ct.data.dims,time_dims = False)
        table_dims = 'x'.join(list(map(str,unpacked_dims)))
        axes = '_'.join(sorted([str(ax).replace(' ','') for ax in self.ct.constraints['constrained_axes']]))
        filepath = os.path.join(
            dirpath,
            f"table_{table_dims}_axes_{axes}_preserved_markov_basis.gzip.json"
        )
        
        # Import markov bases if they exist
        if (not os.path.isfile(filepath)) or (not os.path.exists(dirpath)):
            self.logger.warning(f'Markov bases do not exist in {filepath}')
        else:
            self.logger.hilight('Reading Markov basis functions.')
            # Read markov basis functions
            self.basis_dictionaries = read_compressed_string(filepath)

            # If there are cell constraints update the basis functions
            if len(self.ct.constraints['cells']) > 0:
                self.update_markov_basis()
                
            return True

        return False

    def export_markov_basis(self) -> None:
        # Export markov bases to file
        unpacked_dims = unpack_dims(self.ct.data.dims,time_dims = False)
        table_dims = 'x'.join(list(map(str,unpacked_dims)))

        if not 'outputs' in list(self.ct.config.settings.keys()) or \
            not 'out_directory' in list(self.ct.config.settings['outputs'].keys()) or \
            self.ct.config.settings['outputs']['out_directory'] == '':
            self.logger.warning(f'Output directory not provided. Markov bases cannot be exported.')
            return

        # Define filepath
        dirpath = os.path.join(
            self.ct.config.settings['outputs']['out_directory'],
            'markov_basis/'
        )
        # Create filepath
        axes = '_'.join(sorted([str(ax).replace(' ','') for ax in self.ct.constraints['constrained_axes']]))
        filepath = os.path.join(
            dirpath,
            f"table_{table_dims}_axes_{axes}_preserved_markov_basis.gzip.json"
        )
        # Remove whitespace
        filepath = filepath.replace(" ","")

        # Make directory
        makedir(dirpath)
        # Do not overwrite functions
        if (not os.path.isfile(filepath)) and (os.path.exists(dirpath)):
            write_compressed_string(str(self.basis_dictionaries),filepath)

    def update_markov_basis(self):
        # Exclude the functions changing any fixed cells
        updated_basis_dictionaries = [] 
        for i in tqdm(
            range(len(self.basis_dictionaries)),
            disable = self.tqdm_disabled,
            leave = False
        ):
            if self.basis_function_admissible(self.basis_dictionaries[i]):
                updated_basis_dictionaries.append(self.basis_dictionaries[i])
        self.basis_dictionaries = updated_basis_dictionaries


class MarkovBasis1DTable(MarkovBasis):

    def __init__(self, ct:ContingencyTable,**kwargs):
        # Initialise superclass constructor
        super().__init__(ct,**kwargs)

        # Build object
        self.build()

    def true_markov_basis_length(self):
        unpacked_dims = unpack_dims(self.ct.data.dims,time_dims = False)
        if unpacked_dims[0] == 1:
            return int(unpacked_dims[1]*(unpacked_dims[1]-1)/2)
        elif unpacked_dims[1] == 1:
            return int(unpacked_dims[0]*(unpacked_dims[0]-1)/2)
        else:
            raise Exception(f'Unexpected table size {unpacked_dims} for MarkovBasis1DTable.')

    def generate(self) -> None:
        if len(self.ct.constraints['constrained_axes']) == 1:
            self.generate_one_margin_preserving_markov_basis()
        else:
            raise Exception(f'Unexpected table size {unpack_dims(self.ct.data.dims,time_dims = False)} for MarkovBasis1DTable.')



class MarkovBasis2DTable(MarkovBasis):

    def __init__(self, ct:ContingencyTable,**kwargs):
        # Initialise superclass constructor
        super().__init__(ct,**kwargs)

        # Build object
        self.build()

    def true_markov_basis_length(self):
        unpacked_dims = unpack_dims(self.ct.data.dims,time_dims = False)
        if np.array_equal(self.ct.constraints['constrained_axes'],np.asarray([[1],[0,1]],dtype='int32')):
            return int(unpacked_dims[1]*(unpacked_dims[1]-1)*unpacked_dims[0]/2)
        elif np.array_equal(self.ct.constraints['constrained_axes'],np.asarray([[0],[0,1]],dtype='int32')):
            return int(unpacked_dims[0]*(unpacked_dims[0]-1)*unpacked_dims[1]/2)
        else:
            return int(unpacked_dims[1]*(unpacked_dims[1]-1)*unpacked_dims[0]*(unpacked_dims[0]-1)/4)

    def generate(self) -> None:
        if len(self.ct.constraints['constrained_axes']) == 2:
            self.generate_one_margin_preserving_markov_basis()
        elif len(self.ct.constraints['constrained_axes']) == 3:
            # This is very slow and should be avoided
            # self.generate_both_margin_preserving_grobner_markov_basis()
            self.generate_both_margin_preserving_markov_basis()
        else:
            raise ValueError(f"Cannot generaive markov basis for constraints {self.ct.constraints['constrained_axes']}")
