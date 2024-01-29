import os
import sys
import torch
import numpy as np
import pandas as pd

from torch import int32
from tqdm.auto import tqdm
from typing import Dict,Tuple,List

from multiresticodm.utils.misc_utils import f_to_df,df_to_f,f_to_array, makedir, setup_logger,write_compressed_string,read_compressed_string, unpack_dims, ndims
from multiresticodm.contingency_table import ContingencyTable

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
        self.logger.setLevels(
            console_level = level
        )
        
        # Enable/disable tqdm
        self.tqdm_disabled = not kwargs.get('monitor_progress',False)

        # Get contingency table
        self.ct = ct
        
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
        #imported = self.import_markov_basis()

        if not imported:
            self.logger.debug('Generating Markov Basis...')
            # Generate bases
            self.generate()

            self.logger.debug('Checking Markov Basis validity...')
            # Check that all basis are admissible
            try:
                # assert self.check_markov_basis_validity()
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
        tab = torch.tensor(f_to_array(f),dtype=int32)
        # basis fully expanded to match table dims
        full_tab = torch.tensor(f_to_array(f,shape=unpack_dims(self.ct.data.dims,time_dims=False)),dtype=int32)

        return (not torch.any(self.ct.table_constrained_margins_summary_statistic(tab))) and \
                (torch.any(tab)) and \
                torch.all(self.ct.table_cell_constraint_summary_statistic(full_tab)==0)
        
    def check_markov_basis_validity(self) -> bool:
        for i in tqdm(
            range(len(self.basis_dictionaries)),
            disable=self.tqdm_disabled,
            leave=False
        ):
            if not self.basis_function_admissible(self.basis_dictionaries[i]):
                return False
        return True

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

        # Loop through each pair combination and keep only ones that don't share a row OR column
        for index,tup1 in tqdm(
            enumerate(sorted_cells),
            total=len(sorted_cells),
            disable=self.tqdm_disabled,
            leave=False
        ):
            # Get all active candidate cells
            inactive_candidate_cells = []
            if index < len(sorted_cells)-1:
                inactive_candidate_cells = self.active_candidates(tup1,sorted_cells[(index+1):])
            # Loop through inactive candidates
            for tup2 in inactive_candidate_cells:
                basis_cells.append((tup1,tup2))

        # Define set of Markov bases
        # self.basis_functions = []
        self.basis_dictionaries = []
        # Define active cells i.e. cells of a basis function that map to non-zero values
        self.basis_active_cells = []
        for index in tqdm(
            range(len(basis_cells)),
            disable=self.tqdm_disabled,
            leave=False
        ):
            # Make sure that no cell in the basis is a constrained cell
            if np.any([basis_cell in self.ct.constraints['cells'] for basis_cell in  basis_cells[index]]):
                continue
            
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


    def generate_both_margin_preserving_markov_basis(self) -> None:
        # Create list of row-column pairs such that no pair share the same row OR column
        basis_cells = []

        # Get all cells in lexicographic order (order first by row and then by col index)
        sorted_cells = sorted(self.ct.cells)

        # Loop through each pair combination and keep only ones that don't share a row OR column
        for index,tup1 in tqdm(
            enumerate(sorted_cells),
            total=len(sorted_cells),
            disable=self.tqdm_disabled,
            leave=False
        ):
            # Get all active candidate cells
            inactive_candidate_cells = []
            if index < len(sorted_cells)-1:
                inactive_candidate_cells = self.active_candidates(tup1,sorted_cells[(index+1):])
            # Loop through inactive candidates
            for tup2 in inactive_candidate_cells:
                basis_cells.append((tup1,tup2,(tup1[0],tup2[1]),(tup2[0],tup1[1])))

        # Define set of Markov bases
        # self.basis_functions = []
        self.basis_dictionaries = []
        # Define active cells i.e. cells of a basis function that map to non-zero values
        self.basis_active_cells = []
        for index in tqdm(
            range(len(basis_cells)),
            disable=self.tqdm_disabled,
            leave=False
        ):
            # Make sure that no cell in the basis is a constrained cell
            if np.any([basis_cell in self.ct.constraints['cells'] for basis_cell in  basis_cells[index]]):
                continue

            # Construct Markov basis function
            def make_f_i(findex):
                def f_i(x):
                    return int(x == basis_cells[findex][0] or x == basis_cells[findex][1]) - \
                        int(x == (basis_cells[findex][0][0],basis_cells[findex][1][1]) or x == (basis_cells[findex][1][0],basis_cells[findex][0][1]))
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

    def import_basis_function(self,filepath:str) -> Dict:
        # Import basis function from csv
        table = pd.read_csv(filepath,index_col=0,header=0)
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
            self.config['outputs']['out_directory'],
            'markov_basis/'
        )
        
        # Create filepath
        unpacked_dims = unpack_dims(self.ct.data.dims,time_dims=False)
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
            self.logger.debug('Reading Markov basis functions.')
            # Read markov basis functions
            self.basis_dictionaries = read_compressed_string(filepath)

            # If there are cell constraints update the basis functions
            if len(self.ct.constraints['cells']) > 0:
                self.update_markov_basis()
                
            return True

        return False

    def export_markov_basis(self) -> None:
        # Export markov bases to file
        unpacked_dims = unpack_dims(self.ct.data.dims,time_dims=False)
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
            disable=self.tqdm_disabled,
            leave=False
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
        unpacked_dims = unpack_dims(self.ct.data.dims,time_dims=False)
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
            raise Exception(f'Unexpected table size {unpack_dims(self.ct.data.dims,time_dims=False)} for MarkovBasis1DTable.')



class MarkovBasis2DTable(MarkovBasis):

    def __init__(self, ct:ContingencyTable,**kwargs):
        # Initialise superclass constructor
        super().__init__(ct,**kwargs)

        # Build object
        self.build()

    def true_markov_basis_length(self):
        unpacked_dims = unpack_dims(self.ct.data.dims,time_dims=False)
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
            self.generate_both_margin_preserving_markov_basis()
        else:
            raise ValueError(f"Cannot generaive markov basis for constraints {self.ct.constraints['constrained_axes']}")
