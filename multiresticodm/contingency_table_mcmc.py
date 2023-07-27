import logging
import numpy as np

from os import path
from copy import deepcopy
from argparse import Namespace
from typing import Union, Tuple, Dict, List
from scipy.stats import nchypergeom_fisher


from multiresticodm.utils import  str_in_list, tuplize, flatten, set_numba_torch_threads
from multiresticodm.math_utils import logsumexp, log_factorial
from multiresticodm.probability_utils import uniform_binary_choice, log_odds_cross_ratio
from multiresticodm.markov_basis import instantiate_markov_basis,MarkovBasis
from multiresticodm.contingency_table import ContingencyTable, ContingencyTable2D


class ContingencyTableMarkovChainMonteCarlo(object):

    """
    Work Station for holding and running Monte Carlo methods on table space.
    """

    def __init__(self, ct: ContingencyTable, table_mb:MarkovBasis=None, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.logger.disabled = kwargs.get('disable_logger',False)
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)

        if isinstance(ct, ContingencyTable2D):

            # Get markov basis object
            self.ct = ct

            # Update margin initialisation
            self.update_margin_initialisation_solver()

            # Update table initialisation
            self.update_table_initialisation_solver()
            
            # Update table proposal
            self.update_proposal_type()

        else:
            raise Exception(f'ContingencyTableMarkovChainMonteCarlo did not recognise input of type {ct.__class__.__name__}')

        # Get number of cores/threads to use in (pseudo) parallelisation
        self.n_workers = int(self.ct.config.settings['inputs'].get('n_workers',1))
        self.n_threads = list(self.ct.config.settings['inputs'].get('n_threads',['1','1']))

        # Update numba threads
        set_numba_torch_threads(self.n_threads)

        # Initialise Markov Chain Monte Carlo proposal, acceptance ratio
        self.build()

        # Pseudo initialise table markov basis
        self.table_mb = Namespace(**{'basis_dictionaries': []})

        # Generate/import Markov Basis (if required)
        if (self.proposal_type.lower() == 'degree_one' or \
            self.proposal_type.lower() == 'degree_higher'):
            if table_mb != None:
                # Pass markov basis object
                if table_mb.ct == self.ct:
                    self.table_mb = table_mb
                else:
                    raise Exception('Incompatible Markov Basis object passed')
            else:
                # Instantiate markov basis object
                self.table_mb = instantiate_markov_basis(
                    self.ct, 
                    self.logger.disabled
                )

        # self.logger.info(self.__str__())

    def __str__(self):
        return f"""
            Markov Chain Monte Carlo algorithm
            Target: {self.target_distribution}
            Dataset: {self.ct.config.settings['inputs']['dataset']}
            Dimensions: {"x".join([str(x) for x in self.ct.dims])}
            Total: {self.ct.margins[tuplize(range(self.ct.ndims()))]}
            Sparse margins: {self.ct.sparse_margins}
            Constrained cells: {len(self.ct.constraints['cells'])}
            Constrained axes: {", ".join([str(x) for x in self.ct.constraints['constrained_axes']])}
            Unconstrained axes: {", ".join([str(x) for x in self.ct.constraints['unconstrained_axes']])}
            Scheme: {self.acceptance_type.title()}
            Proposal: {self.proposal_type.replace('_',' ').title()}
            Number of Table Markov bases: {len(self.table_mb.basis_dictionaries)}
            Number of workers: {self.n_workers}
            Number of threads (numpy,numba): {str(self.n_threads)}
            Random number generation seed: {self.ct.seed}
        """

    def __repr__(self):
        if self.acceptance_type == 'Metropolis Hastings':
            return f"MetropolisHastings({self.ct.__class__.__name__})"
        elif self.acceptance_type == 'Gibbs':
            return f"Gibbs({self.ct.__class__.__name__})"
        else:
            return f"MarkovChainMonteCarlo({self.ct.__class__.__name__})"

    def update_proposal_type(self, proposal_type: str = None) -> None:
        # Set to config proposal if none is provided
        if proposal_type is None:
            self.proposal_type = self.ct.config.settings['mcmc']['contingency_table']['proposal']
        else:
            self.proposal_type = proposal_type
        # Change proposal if it is invalid for the constraints provided
        if self.ct.distribution_name == 'poisson':
            # Default to direct sampling - cell constraints can be handled here
            # No need to use markov bases
            self.proposal_type == 'direct_sampling'
        elif str_in_list(self.ct.distribution_name,['multinomial','product_multinomial']) and \
            len(self.ct.constraints['cells']) > 0:
            # Direct sampling for these distributions is not possible when cell constraints exist
            if self.proposal_type == 'direct_sampling':
                # Default to degree higher proposal
                self.proposal_type == 'degree_higher'
        elif (self.ct.distribution_name == 'fishers_hypergeometric') and \
            self.proposal_type == 'direct_sampling' and \
            len(self.ct.constraints['cells']) > 0:
                # Distribution cannot be directly sampled
                # Default to degree higher
                self.proposal_type = 'degree_higher'
        

    def update_table_initialisation_solver(self, solver_name: str = None) -> None:
        
        # Set to config proposal if none is provided
        if solver_name is None:
            self.table_initialisation_solver_name = self.ct.config.settings['mcmc']['contingency_table']['table0']
        else:
            self.table_initialisation_solver_name = solver_name

        if len(self.ct.constraints['constrained_axes']) == 0:
            # This is the unconstrained case
            self.table_initialisation_solver_name = 'table_random_sample'
        else:
            # Read solver name for table intialisation
            self.table_initialisation_solver_name = self.ct.config.settings['mcmc']['contingency_table']['table0']
        
        # Build solver for table and margin
        self.table_solver = self.ct.map_table_solver_name_to_function(
            solver_name=self.table_initialisation_solver_name
        )
        
    def update_margin_initialisation_solver(self, solver_name: str = None) -> None:
        # Set to config proposal if none is provided
        if solver_name is None:
            self.margin_initialisation_solver_name = self.ct.config.settings['mcmc']['contingency_table'].get('margin0','multinomial')
        else:
            self.margin_initialisation_solver_name = solver_name
        # Get margin solver
        self.margin_solver = self.ct.map_margin_solver_name_to_function(
            solver_name=self.margin_initialisation_solver_name
        )

    def initialise_table(self,intensity:list=None,margins:dict={}) -> None:
        # If no intensity provided 
        # Use uniform intensity for every cell
        if intensity is None:
            intensity = np.ones(self.ct.dims,dtype='float32')

        # If table can be sampled in closed form
        # Sample directly from the table distribution
        # if self.proposal_type.lower() == 'direct_sampling':
        #     # Get direct sampling proposal
        #     direct_sampling_proposal = getattr(self,f"direct_sampling_proposal_{self.ct.ndims()}way_table")
        #     table0,_,_,_,_ = direct_sampling_proposal(
        #         table_prev=None,
        #         log_intensity=np.log(intensity)
        #     )
        # Otherwise initialise based on user-defined solver
        # else:
        # Sample uncostrained margins
        self.sample_unconstrained_margins(intensity)
        # Use table solver to get initial table
        table0 = self.table_solver(
            intensity=intensity,
            margins=margins
        )
        
        return table0
    
    
    def sample_margins_2way_table(self, intensity: list = None, axes: list = None) -> None:
        
        self.logger.debug('initialise_margins_2way_tables')
        
        # Sort axes
        sorted_axes = sorted(axes,key=len,reverse=True)

        # Set probabilities to uniform
        if intensity is None:
            intensity = np.ones(self.ct.dims,dtype='float32')
        
        margins = {}
        for ax in sorted_axes:
            if len(ax) == self.ct.ndims():
                # Calculate normalisation of multinomial probabilities (total intensity)
                total_intensity = np.sum(intensity.ravel())
                # If this is the only constraint
                if len(sorted_axes) == 1:
                    # Sample grand total from Poisson
                    margins[ax] = np.array([np.random.poisson(total_intensity)],dtype='int32')
                else:
                    # Get margin grand total data
                    margins[ax] = np.array([np.sum(intensity)],dtype='int32')

            elif len(ax) == 1:
                # Get total
                grand_total = self.ct.margins[tuplize(range(self.ct.ndims()))][0]
                # Compute multinomial probabilities
                table_probabilities = intensity/intensity.sum()
                margin_probabilities = np.sum(
                    table_probabilities,
                    axis=tuplize(ax)
                ).flatten()
                # Sample row or column margin from Multinomial conditioned on grand total
                margins[ax] = np.random.multinomial(grand_total,margin_probabilities).astype('int32')
            else:
                raise Exception(f"margins for axes {ax} could not be initialised")

            # Update margin for given axis
            self.ct.update_margins({ax:margins[ax]})
        
        return margins

    def sample_constrained_margins(self, intensity: list = None) -> None:
        return self.sample_margins(intensity,self.ct.constraints['constrained_axes'])
    
    def sample_unconstrained_margins(self, intensity: list = None) -> None:
        return self.sample_margins(intensity,self.ct.constraints['unconstrained_axes'])

    def initialise_margin(self, axis, intensity: list = None) -> None:
        return self.margin_solver(
                axis=axis,
                intensity=intensity
        )
    
    def initialise_unconstrained_margins(self, intensity: list = None) -> None:
        unconstrained_axes = sorted(self.ct.constraints['unconstrained_axes'],key=len,reverse=True)
        margins = {}
        for ax in unconstrained_axes:
            # Initialise margin
            margins[ax] = self.initialise_margin(axis=ax, intensity=intensity)
            # Update margin
            self.ct.update_margins({ax:margins[ax]})
        return margins

    def initialise_constrained_margins(self, intensity: list = None) -> None:
        constrained_axes = sorted(self.ct.constraints['constrained_axes'],key=len,reverse=True)
        margins = {}
        for ax in constrained_axes:
            margins[ax] = self.initialise_margin(axis=ax, intensity=intensity)
            # Update margin
            self.ct.update_margins({ax:margins[ax]})
        return margins


    def build_table_distribution(self) -> None:

        build_successful = True

        self.target_distribution = self.ct.distribution_name.replace("_", " ").capitalize()
        self.sample_margins = getattr(self, f'sample_margins_{self.ct.ndims()}way_table')
        if self.proposal_type.lower() == 'direct_sampling':
            self.acceptance_type = 'Direct sampling'
            self.proposal = getattr(
                self, f'{self.proposal_type.lower()}_proposal_{self.ct.ndims()}way_table')
            self.log_acceptance_ratio = getattr(
                self, f'direct_sampling_log_acceptance_ratio_{self.ct.ndims()}way_table')
        elif self.proposal_type.lower() == 'degree_one':
            self.acceptance_type = 'Metropolis Hastings'
            self.proposal = getattr(
                self, f'{self.proposal_type.lower()}_proposal_{self.ct.ndims()}way_table')
            self.log_acceptance_ratio = getattr(
                self, f'metropolis_log_acceptance_ratio_{self.ct.ndims()}way_table')
            self.log_target_measure_difference = getattr(
                self, f'log_target_measure_difference_{self.ct.ndims()}way_table')
        elif self.proposal_type.lower() == 'degree_higher':
            self.acceptance_type = 'Gibbs'
            self.proposal = getattr(
                self, f'{self.proposal_type.lower()}_proposal_{self.ct.ndims()}way_table_{self.ct.distribution_name}')
            self.log_acceptance_ratio = getattr(
                self, f'gibbs_log_acceptance_ratio_{self.ct.ndims()}way_table')
            self.log_target_measure_difference = getattr(
                self, f'log_target_measure_difference_{self.ct.ndims()}way_table')
        elif hasattr(self,f"{self.proposal_type.lower()}_proposal_{self.ct.ndims()}way_table"):
            self.acceptance_type = f"{self.proposal_type.lower().replace('_',' ').capitalize()} sampling"
            self.proposal = getattr(self,f"{self.proposal_type.lower()}_proposal_{self.ct.ndims()}way_table")
            self.log_acceptance_ratio = getattr(self, f'direct_sampling_log_acceptance_ratio_{self.ct.ndims()}way_table')
        else:
            build_successful = False

        return build_successful

    def build(self) -> None:

        build_successful = True

        # Number of steps to make for fixed row and column sums when the either one of the two is known 
        # and the other's distribution is only known
        self.table_steps = int(self.ct.config.settings['mcmc']['contingency_table'].get('table_steps',1))

        # Number of steps to skip storing a sample (thinning)
        self.table_thinning = 1
        if str_in_list('thinning', self.ct.config.settings['mcmc']['contingency_table'].keys()):
            self.table_thinning = int(
                self.ct.config.settings['mcmc']['contingency_table']['thinning'])

        if len(self.ct.distribution_name) > 0:
            build_successful = build_successful and self.build_table_distribution()
        else:
            raise Exception(
                f"margin constraints for axes {','.join([str(ax) for ax in self.ct.constraints['constrained_axes']])} cannot be handled for contingency table {self.ct.__class__.__name__} with ndims {self.ct.ndims()}")

        if not build_successful:
            raise Exception(
                f'Proposal type {self.proposal_type.lower()} for type {self.ct.__class__.__name__} not defined')

    def log_intensities_to_multinomial_log_probabilities(self, log_intensity: Union[np.array, np.ndarray], column_indices: List[Tuple] = None):

        self.logger.debug('Log intensities to multinomial log probabilities')

        # Compute cell intensities
        # depending on dimensionality of log cell intensities
        if len(np.shape(log_intensity)) == 1:
            # Slice log cell intensities at column of interest
            if column_indices is not None:
                log_intensity = log_intensity[column_indices]
            # Get dimensions of cell intensities
            nrows,ncols = np.shape(log_intensity)
            # Compute unnormalised log probabilities
            log_intensities_colsums = np.array([logsumexp(log_intensity[j]) for j in range(ncols)])
        elif len(np.shape(log_intensity)) == 2:
            # Slice log cell intensities at column of interest
            if column_indices is not None:
                log_intensity = log_intensity[:, column_indices]
            # Get dimensions of cell intensities
            I, J = np.shape(log_intensity)
            # Compute unnormalised log probabilities
            log_intensities_colsums = np.array(
                [logsumexp(log_intensity[:, j]) for j in range(J)])
        else:
            raise Exception(
                f'Cannot handle log_intensity with dimensions {len(np.shape(log_intensity))}')

        # Get total intensities (normalising factor in log space)
        total_log_intensities = logsumexp(log_intensities_colsums)
        # Get normalised probabilities
        log_probabilities = log_intensities_colsums-total_log_intensities

        return self.ct.margins[tuplize(range(self.ct.ndims()))], log_probabilities, log_intensities_colsums

    def get_table_step_size_support(self, table_prev: np.ndarray, basis_function: np.ndarray, non_zero_cells: List[Tuple]) -> Union[list, np.array]:
        # Get maximum value of current table at cells where markov basis non-negative
        # Get minimum values across two diagonal of the 2x2 subtable
        min_value1 = min([table_prev[non_zero_cells[0]],
                         table_prev[non_zero_cells[3]]])
        min_value2 = min([table_prev[non_zero_cells[1]],
                         table_prev[non_zero_cells[2]]])

        # If markov basis has a positive upper leftmost cell then
        # then that determines the step size support
        if basis_function[non_zero_cells[0]] > 0:
            # The smallest step size is the negative of the minimum across the main diagonal
            # The largest step size is the minimum across the other diagonal
            step_sizes = list(range(-min_value1, min_value2+1))
        else:
            # The smallest step size is the negative of the minimum across the other diagonal
            # The largest step size is the minimum across the main diagonal
            step_sizes = list(range(-min_value2, min_value1+1))

        return step_sizes

    
    # ------------------------------#
    #        Two-way tables        #
    # ------------------------------#
    
    def poisson_sample_2way_table(self,margin_probabilities):
        # Initialise table to zero
        table_new = np.zeros(self.ct.dims, dtype='int32')
        for cell in self.ct.constraints['cells']:
            table_new[cell] = np.int32(self.ct.table[cell])
        # Update remaining table cells
        continue_loop = True
        while continue_loop:
            for cell in self.ct.cells:
                table_new[cell] = np.int32(np.random.poisson(margin_probabilities[cell])) 
            # Continue loop only if table is not sparse admissible
            continue_loop = not self.ct.table_sparse_admissible(table_new)
        return table_new

    def multinomial_sample_2way_table(self,margin_probabilities):
        # This is the case with the grand total fixed
        # Get constrained axes
        axis_constrained = min(self.ct.constraints['constrained_axes'], key=len)
        continue_loop = True
        while continue_loop:
            # Sample from multinomial
            table_new = np.random.multinomial(
                n=self.ct.margins[tuplize(axis_constrained)].item(),
                pvals=margin_probabilities.ravel()).astype('int32')
            # Reshape table to match original dims
            table_new = np.reshape(table_new, tuplize(self.ct.dims))
            # Continue loop only if table is not sparse admissible
            continue_loop = not self.ct.table_sparse_admissible(table_new)
            
        # Make sure sampled table has the right shape
        # if np.shape(table_new) != tuplize(self.dims):
            # table_new = table_new.T
        
        return table_new

    def product_multinomial_sample_2way_table(self,margin_probabilities):
        # Get constrained axes
        axis_constrained = min(self.ct.constraints['constrained_axes'], key=len)
        # Get all unconstrained axes
        axis_uncostrained = deepcopy(self.ct.constraints['unconstrained_axes'])
        # Get plain uncostrained axis (must have length 1)
        axis_uncostrained_flat = next(flatten(axis_uncostrained))
        # This is the case with either margins fixed (but not both)
        # Sample from product multinomials
        continue_loop = True
        while continue_loop:
            table_new = np.array(
                [
                    np.random.multinomial(
                        n=msum, 
                        pvals=margin_probabilities.take(
                            indices=i, 
                            axis=axis_uncostrained_flat
                        )
                    )
                    for i, msum in enumerate(self.ct.margins[tuplize(axis_constrained)])
                ],
                dtype='int32'
            )
            # Continue loop only if table is not sparse admissible
            continue_loop = not self.ct.table_sparse_admissible(table_new)

        # Reshape table to match original dims
        table_new = np.reshape(table_new, tuplize(self.ct.dims))
        # Make sure sampled table has the right shape
        try:
            assert self.ct.table_admissible(table_new) and self.ct.table_sparse_admissible(table_new)
        except:
            print(self.ct.table_admissible(table_new),self.ct.table_sparse_admissible(table_new))
            raise Exception()
        # if np.shape(table_new) != tuplize(self.dims):
            # table_new = table_new.T
        
        return table_new

    def direct_sampling_proposal_2way_table(self, table_prev: np.ndarray, log_intensity: np.ndarray) -> Tuple[Dict, Dict, int, Dict, Dict]:

        self.logger.debug(f'direct_sampling_2way_table for {self.ct.distribution_name}')
        
        # Get smallest in length constrained axis
        if len(self.ct.constraints['constrained_axes']) > 0:
            axis_constrained = min(self.ct.constraints['constrained_axes'], key=len)
        else:
            axis_constrained = np.array([])
        # Get all unconstrained axes
        axis_uncostrained = deepcopy(self.ct.constraints['unconstrained_axes'])
        # Initialise shape of probability normalisation matrix
        new_shape = np.ones(2, dtype='uint8')

        if len(axis_constrained) == self.ct.ndims():
            # Calculate normalisation of multinomial probabilities (total intensity)
            probabilities_normalisation = np.array([logsumexp(log_intensity.ravel())],dtype=np.float64)
            axis_uncostrained_flat = None
        elif len(axis_constrained) > 0 and len(axis_uncostrained) > 0:
            # Get plain uncostrained axis (must have length 1)
            axis_uncostrained_flat = next(flatten(axis_uncostrained))
            # Calculate normalisation of multinomial probabilities for each row or column
            probabilities_normalisation = np.array([
                logsumexp(log_intensity.take(
                            indices=idx,
                            axis=axis_uncostrained_flat
                        )) for idx in range(self.ct.dims[axis_uncostrained_flat])
            ],dtype=np.float64)
            # Update shape of probability normalisation matrix
            new_shape[axis_uncostrained_flat] = self.ct.dims[axis_uncostrained_flat]
        else:
            probabilities_normalisation = np.array([0],dtype=np.float64)
            new_shape = np.array([1,1])
            axis_uncostrained_flat = None

        # Calculate probabilities
        margin_probabilities = np.exp(
            log_intensity-probabilities_normalisation.reshape(tuple(new_shape))
        )

        # There is a casting issue with numpy that forces the following recasting and normalisation
        # margin_sum = margin_probabilities.sum(
        #     axis=tuplize(axis_constrained)
        # ).reshape(new_shape).astype('float64')
        # margin_probabilities /= margin_sum
        
        # Initialise table
        table_new = np.zeros(self.ct.dims,dtype='int32')
        firstIteration = True
        # Resample if margins are not allowed to be sparse but contain zeros
        while (not self.ct.table_sparse_admissible(table_new)) or firstIteration:
            if self.ct.distribution_name == 'poisson':
                # This is the uncostrained (in terms of margins) case
                table_new = self.poisson_sample_2way_table(
                    margin_probabilities
                )
            elif self.ct.distribution_name == 'multinomial':
                # This is the singly constrained (in terms of margins) case
                table_new = self.multinomial_sample_2way_table(
                    margin_probabilities
                )
            elif self.ct.distribution_name == 'product_multinomial':
                # This is the total constrained (in terms of margins) case
                table_new = self.product_multinomial_sample_2way_table(
                    margin_probabilities
                )
            elif self.ct.distribution_name == 'fishers_hypergeometric':
                # This is the doubly constrained (in terms of margins) case
                self.logger.warning(f"Monte carlo sampling of central fishers hypergeometric.")
                table_new = self.ct.table_monte_carlo_sample()
            else:
                raise Exception(f"Proposal mechanism could not be found for {self.ct.distribution_name}")
            firstIteration = False

        try:
            assert self.ct.table_admissible(table_new) and self.ct.table_sparse_admissible(table_new)
        except:
            self.logger.error((self.ct.table_admissible(table_new)),(self.ct.table_sparse_admissible(table_new)))
            raise Exception('FAILED')
        
        return table_new, None, None, None, None
    

    def degree_one_proposal_2way_table(self, table_prev: np.ndarray, log_intensity: np.ndarray) -> Tuple[Dict, Dict, int, Dict]:

        self.logger.debug('2way table degree one move')

        # Sample Markov basis function index uniformly at random
        func_index = np.random.randint(len(self.table_mb))

        # Sample epsilon uniformly at random
        epsilon = uniform_binary_choice()

        # initialise new table
        table_new = np.zeros(shape=self.ct.dims, dtype='int32')

        # Store old table
        table_new[:] = table_prev

        # Construct new table
        for k in self.table_mb.basis_dictionaries[func_index].keys():
            table_new[k] = table_prev[k] + epsilon * \
                self.table_mb.basis_dictionaries[func_index][k]

        return table_new.astype('int32'), \
            int(epsilon), \
            self.table_mb.basis_dictionaries[func_index], \
            func_index, \
            {'support': [-1, 1], 'probs': [0.5, 0.5]}

    def degree_higher_proposal_2way_table_multinomial(self, table_prev: np.ndarray, log_intensity: np.ndarray) -> Tuple[Dict, Dict, int, Dict, Dict]:
        return self.degree_higher_proposal_2way_table_product_multinomial(
            table_prev=table_prev,
            log_intensity=log_intensity
        )

    def degree_higher_proposal_2way_table_product_multinomial(self, table_prev: np.ndarray, log_intensity: np.ndarray) -> Tuple[Dict, Dict, int, Dict, Dict]:

        self.logger.debug('2way table degree higher move product multinomial')

        # Sample Markov basis function index uniformly at random
        func_index = np.random.randint(len(self.table_mb))
        
        # Get non-zero cells of Markov basis and sort them lexicographically
        non_zero_cells = sorted(list(self.table_mb.basis_dictionaries[func_index].keys()))

        # Normalise intensities at basis function cells
        log_total_intensity = logsumexp(np.array([log_intensity[cell] for cell in non_zero_cells]))
        log_probabilities = np.array([log_intensity[cell] - log_total_intensity for cell in non_zero_cells])

        # Copy previous table
        table_new = np.zeros(table_prev.shape)
        table_new[:] = table_prev

        # Get total of previous table evaluated at basis functions
        table_prev_total = int(np.sum([table_prev[cell] for cell in non_zero_cells]))

        # Sample new cells of basis functions
        updated_cells = np.random.multinomial(n=table_prev_total, pvals=np.exp(log_probabilities)).astype('int32')

        # Update cells in new table
        for i,cell in enumerate(non_zero_cells):
            table_new[cell] = updated_cells[i]

        return table_new.astype('int32'), \
            None, \
            self.table_mb.basis_dictionaries[func_index], \
            func_index, \
            None
        

    def degree_higher_proposal_2way_table_fishers_hypergeometric(self, tab_prev: np.ndarray, log_intensity: np.ndarray) -> Tuple[Dict, Dict, int, Dict]:

        self.logger.debug(
            '2way table degree higher move Fishers hypergeometric')

        # Sample Markov basis function index uniformly at random
        func_index = np.random.randint(len(self.table_mb))

        # Get non-zero cells of Markov basis
        # and sort them lexicographically
        non_zero_cells = sorted(list(self.table_mb.basis_dictionaries[func_index].keys()))
        positive_cells = [cell for cell in non_zero_cells if self.table_mb.basis_dictionaries[func_index][cell] > 0]

        # Copy previous table
        tab_new = np.zeros(np.shape(tab_prev))
        tab_new[:] = tab_prev

        # Compute log odds ratio for 2x2 table
        # Note that this form is a simplified version of the ratio 
        # of odds ratios for the four table cells that have been changed
        omega = np.exp(log_odds_cross_ratio(log_intensity, *positive_cells))
        # Convert infinities to value
        if (not np.isfinite(omega)) & (omega > 0):
            omega = 1e6
        elif (omega == 0):
            omega = 1e-6

        # Get row and column sums of 2x2 subtable
        rsum = int(tab_prev[non_zero_cells[0]] + tab_prev[non_zero_cells[1]])
        csum = int(tab_prev[non_zero_cells[0]] + tab_prev[non_zero_cells[2]])
        total = np.sum([int(tab_prev[non_zero_cells[i]]) for i in range(len(non_zero_cells))])

        # Sample upper leftmost entry of 2x2 subtable from non-central hypergeometric distribution
        new_val = nchypergeom_fisher.rvs(M=total, n=rsum, N=csum, odds=omega)

        # Update upper leftmost entry and propagate to rest of cells
        tab_new[non_zero_cells[0]] = int(new_val)
        tab_new[non_zero_cells[1]] = int(rsum - new_val)
        tab_new[non_zero_cells[2]] = int(csum - new_val)
        tab_new[non_zero_cells[3]] = int(total - rsum - csum + new_val)

        return tab_new.astype('int32'), \
            None, \
            self.table_mb.basis_dictionaries[func_index], \
            func_index, \
            None

    def metropolis_log_acceptance_ratio_2way_table(self, table_new: np.ndarray, table_prev: np.ndarray, log_intensity: np.ndarray) -> Union[float, None]:
        '''Acceptance ratio for one degree proposals
        This gives the Metropolis Hastings ratio i.e.
        p(x')p(x|x') / p(x)p(x'|x)
        '''

        self.logger.debug('2way table Metropolis Hastings acceptance')
        # Find cells where the images of the two functions differ at
        cells_of_interest = np.argwhere(table_new != table_prev).tolist()

        # Compute log target measure difference
        log_acc = self.log_target_measure_difference(
            table_new, 
            table_prev, 
            cells_of_interest, 
            log_intensity
        )
        try:
            assert np.isfinite(log_acc)
        except:
            print(cells_of_interest)
            print(log_acc)
            raise Exception(
                f"Metropolis Hastings log acceptance ratio is infinite for table_new = {table_new}, table_prev = {table_prev}")

        if self.ct.table_nonnegative(table_new):
            return log_acc
        else:
            return -np.infty

    def gibbs_log_acceptance_ratio_2way_table(self, table_new: np.ndarray, table_prev: np.ndarray, log_intensity: np.ndarray) -> float:
        ''' Acceptance ratio for higher degree proposals
        This gives the Gibbs acceptance ratio i.e. equal to 1 iff proposal is non-negative'''

        self.logger.debug('2way table degree higher acceptance (NO VALIDITY CHECK - ASSUMES PROPOSAL IS WELL DEFINED)')

        if self.ct.table_nonnegative(table_new):
            return 0
        else:
            self.logger.error(
                f'Proposed table {table_new} is inadmissible or non positive')
            return -np.infty

    def direct_sampling_log_acceptance_ratio_2way_table(self, table_new: np.ndarray, table_prev: np.ndarray, log_intensity: np.ndarray) -> Tuple[Dict, Dict, int, Dict]:
        return 0


    def log_target_measure_difference_2way_table(self, table_new: np.ndarray, table_prev: np.ndarray, cells: Union[np.array, np.ndarray, list], log_intensity: np.ndarray) -> float:
        '''
        Computes log of acceptance ratio.
        log (sigma(table_new)) / (sigma(table_prev)) = log(sigma(table_new)) - log(sigma(table_prev))
        where sigma(f) = prod_{x} (f(x)!)^{-1}. This simplifies to
        sum_{x} log(table_prev(x)!) - log(table_new(x)!) where the sum is only taken
        with respect to the x in X for which table_new(x)neq table_prev(x)
        '''
        log_ratio = 0
        # Loop through each list
        for i in range(len(cells)):
            # Convert to tuple
            cells[i] = tuple(cells[i])

            # First add log odds ratio of intensities times the table entry difference (step size)
            # Add range of values between numerator and denominator to numerator (if table_new > table_prev) in the denominator
            # Note: addition in the denominator happens because the target measure has a power of -1 in the factorial
            # or to denominator (if table_new < table_prev)
            # # Add odds ratio of intensities to numerator
            if log_intensity is not None:
                # Note that this form is a simplified version of the ratio 
                # of odds ratios for the four table cells that have been changed
                log_ratio += (table_new[cells[i]] - table_prev[cells[i]])*log_intensity[cells[i]]
            if table_new[cells[i]] > table_prev[cells[i]]:
                log_ratio -= log_factorial(start=max(table_prev[cells[i]], 1), end=max(table_new[cells[i]], 1))
            elif table_new[cells[i]] < table_prev[cells[i]]:
                log_ratio += log_factorial(start=max(table_new[cells[i]], 1), end=max(table_prev[cells[i]], 1))
            else:
                continue

        return log_ratio
    

    def table_gibbs_step(self, table_prev: np.ndarray, log_intensity: np.ndarray) -> Tuple[Union[list, np.array, np.ndarray], Union[list, np.array, np.ndarray], float]:

        self.logger.debug('Table Gibbs step')

        # Propose new sample
        table_new, \
        _, \
        _, \
        _, \
        _ = self.proposal(
            table_prev, 
            log_intensity
        )

        # Evaluate acceptance ratio
        log_acc = self.log_acceptance_ratio(
                table_new, 
                table_prev, 
                log_intensity
        )
        

        # Accept/reject
        if (log_acc >= 0) or (np.log(np.random.uniform(0, 1)) < log_acc):
            # Update unconstrained margins
            self.ct.update_unconstrained_margins_from_table(table_new)
            return table_new, 1
        else:
            return table_prev, 0