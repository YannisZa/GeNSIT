import torch
import numpy as np
import torch.distributions as distr

from copy import deepcopy
from argparse import Namespace
from torch import int32, float32
from scipy.stats import nchypergeom_fisher
from typing import Union, Tuple, Dict, List

import gensit.utils.probability_utils as ProbabilityUtils

from gensit.utils.math_utils import log_factorial_sum
from gensit.markov_basis import instantiate_markov_basis,MarkovBasis
from gensit.contingency_table import ContingencyTable, ContingencyTable2D
from gensit.utils.probability_utils import uniform_binary_choice, log_odds_cross_ratio
from gensit.utils.misc_utils import  ndims, set_seed, setup_logger, tuplize, flatten, unpack_dims, safe_cast


class ContingencyTableMarkovChainMonteCarlo(object):

    """
    Work Station for holding and running Monte Carlo methods on table space.
    """

    def __init__(
        self, 
        ct: ContingencyTable, 
        table_mb:MarkovBasis = None, 
        **kwargs
    ):
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__,
            console_level = level, 
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevels( console_level = level )
        
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
        self.n_threads = int(self.ct.config.settings['inputs'].get('n_threads','1'))

        # Initialise Markov Chain Monte Carlo proposal, acceptance ratio
        self.build()

        # Pseudo initialise table markov basis
        self.markov_basis = Namespace(**{'basis_dictionaries': []})

        # Generate/import Markov Basis (if required)
        if (self.proposal_type.lower() == 'degree_one' or \
            self.proposal_type.lower() == 'degree_higher'):
            if table_mb != None:
                # Pass markov basis object
                if table_mb.ct == self.ct:
                    self.markov_basis = table_mb
                    self.ct.config['markov_basis_len'] = int(len(table_mb))
                else:
                    raise Exception('Incompatible Markov Basis object passed')
            else:
                # Instantiate markov basis object
                self.markov_basis = instantiate_markov_basis(
                    self.ct, 
                    monitor_progress = False,
                    logger = self.logger
                )
                self.ct.config['markov_basis_len'] = int(len(self.markov_basis))

        self.logger.debug(self.__str__())

    def __str__(self):
        return f"""
            Markov Chain Monte Carlo algorithm
            Target: {self.target_distribution}
            Dataset: {self.ct.config.settings['inputs']['dataset']}
            Dimensions: {"x".join([str(x) for x in self.ct.data.dims])}
            Total: {self.ct.data.margins[tuplize(range(ndims(self.ct)))]}
            Sparse margins: {self.ct.sparse_margins}
            Constrained cells: {len(self.ct.constraints['cells'])}
            Constrained axes: {", ".join([str(x) for x in self.ct.constraints['constrained_axes']])}
            Unconstrained axes: {", ".join([str(x) for x in self.ct.constraints['unconstrained_axes']])}
            Scheme: {self.acceptance_type.title()}
            Proposal: {self.proposal_type.replace('_',' ').title()}
            Number of Table Markov bases: {len(self.markov_basis.basis_dictionaries)}
            Number of workers: {self.n_workers}
            Number of threads: {str(self.n_threads)}
            Random seed: {self.ct.config['inputs'].get('seed',None)}
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
        elif self.ct.distribution_name in ['multinomial','product_multinomial'] and \
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
            solver_name = self.table_initialisation_solver_name
        )
        
    def update_margin_initialisation_solver(self, solver_name: str = None) -> None:
        # Set to config proposal if none is provided
        if solver_name is None:
            self.margin_initialisation_solver_name = self.ct.config.settings['mcmc']['contingency_table'].get('margin0','multinomial')
        else:
            self.margin_initialisation_solver_name = solver_name
        # Get margin solver
        self.margin_solver = self.ct.map_margin_solver_name_to_function(
            solver_name = self.margin_initialisation_solver_name
        )

    def initialise_table(self,intensity:list = None,margins:dict={}) -> None:
        self.logger.debug('Initialise table')
        
        # If no intensity provided 
        # Use uniform intensity for every cell
        if intensity is None:
            intensity = torch.ones(tuplize(list(unpack_dims(self.ct.data.dims))),dtype = float32)

        # Sample uncostrained margins
        self.sample_unconstrained_margins(intensity)
        # Use table solver to get initial table
        table0 = self.table_solver(
            intensity = intensity,
            margins = margins,
            ct_mcmc = self
        )
        self.logger.debug(table0)
        
        return table0
    
    
    def sample_margins_2way_table(self, intensity: list = None, axes: list = None, constrained:bool = False) -> None:

        self.logger.debug(f'initialise_margins_2way_tables {constrained}')
        
        # Sort axes
        sorted_axes = sorted(axes,key = len,reverse = True)

        # Set probabilities to uniform
        if intensity is None:
            intensity = torch.ones(tuple(list(unpack_dims(self.ct.data.dims))),dtype = float32)

        _ = set_seed(self.ct.config['inputs'].get('seed',None))

        margins = {}
        for ax in sorted_axes:
            if len(ax) == ndims(self.ct):
                # Calculate normalisation of multinomial probabilities (total intensity)
                total_intensity = torch.sum(intensity.ravel())
                # If this is the only constraint
                if not constrained:
                    # Sample grand total from Poisson
                    margins[ax] = distr.poisson.Poisson(
                        rate = total_intensity.item(),
                    ).sample().to(device = self.ct.device,dtype = int32)
                else:
                    # Get margin grand total data
                    margins[ax] = total_intensity.to(device = self.ct.device,dtype = int32)
                    
            elif len(ax) == 1:
                # Get total
                grand_total = (self.ct.data.margins[tuplize(range(ndims(self.ct)))]).to(dtype = int32,device = self.ct.device)
                # Compute multinomial probabilities
                table_probabilities = torch.div(intensity,grand_total)
                margin_probabilities = torch.sum(
                    table_probabilities,
                    dim = tuplize(ax),
                    keepdim = False
                ).flatten()
                # Sample row or column margin from Multinomial conditioned on grand total
                margins[ax] = distr.multinomial.Multinomial(
                    total_count = grand_total.item(),
                    probs = margin_probabilities
                ).sample().to(device = self.ct.device,dtype = int32)
            else:
                raise Exception(f"margins for axes {ax} could not be initialised")

        # Update margins
        self.ct.update_margins(margins)
        return margins

    def sample_constrained_margins(self, intensity: list = None) -> None:
        return self.sample_margins(intensity,self.ct.constraints['constrained_axes'],constrained = True)
    
    def sample_unconstrained_margins(self, intensity: list = None) -> None:
        return self.sample_margins(intensity,self.ct.constraints['unconstrained_axes'],constrained = False)

    def initialise_margin(self, axis, intensity: list = None) -> None:
        return self.margin_solver(
                axis = axis,
                intensity = intensity
        )
    
    def initialise_unconstrained_margins(self, intensity: list = None) -> None:
        unconstrained_axes = sorted(self.ct.constraints['unconstrained_axes'],key = len,reverse = True)
        margins = {}
        for ax in unconstrained_axes:
            # Initialise margin
            margins[ax] = self.initialise_margin(axis = ax, intensity = intensity)
            # Update margin
            self.ct.update_margins({ax:margins[ax]})
        return margins

    def initialise_constrained_margins(self, intensity: list = None) -> None:
        constrained_axes = sorted(self.ct.constraints['constrained_axes'],key = len,reverse = True)
        margins = {}
        for ax in constrained_axes:
            margins[ax] = self.initialise_margin(axis = ax, intensity = intensity)
            # Update margin
            self.ct.update_margins({ax:margins[ax]})
        return margins

    @property
    def table_loss(self):
        return getattr(ProbabilityUtils,f"log_{self.ct.distribution_name}_loss")

    def build_table_distribution(self) -> None:

        build_successful = True
        # Define target distribution
        self.target_distribution = self.ct.distribution_name.replace("_", " ").capitalize()
        
        # Define sample margins
        self.sample_margins = getattr(self, f'sample_margins_{ndims(self.ct)}way_table')
        
        # Define table loss function
        table_likelihood = getattr(ProbabilityUtils,f"log_{self.ct.distribution_name}_pmf_unnormalised")
        def table_likelihood_loss(table:torch.tensor,log_intensity:torch.tensor,**kwargs) -> float:
            return (-1) * table_likelihood(table,log_intensity)
        self.table_likelihood_loss = table_likelihood_loss

        if self.proposal_type.lower() == 'direct_sampling':
            self.acceptance_type = 'Direct sampling'
            self.proposal = getattr(
                self, f'{self.proposal_type.lower()}_proposal_{ndims(self.ct)}way_table')
            self.log_acceptance_ratio = getattr(
                self, f'direct_sampling_log_acceptance_ratio_{ndims(self.ct)}way_table')
        elif self.proposal_type.lower() == 'degree_one':
            self.acceptance_type = 'Metropolis Hastings'
            self.proposal = getattr(
                self, f'{self.proposal_type.lower()}_proposal_{ndims(self.ct)}way_table')
            self.log_acceptance_ratio = getattr(
                self, f'metropolis_log_acceptance_ratio_{ndims(self.ct)}way_table')
            self.log_target_measure_difference = getattr(
                self, f'log_target_measure_difference_{ndims(self.ct)}way_table')
        elif self.proposal_type.lower() == 'degree_higher':
            self.acceptance_type = 'Gibbs'
            self.proposal = getattr(
                self, f'{self.proposal_type.lower()}_proposal_{ndims(self.ct)}way_table_{self.ct.distribution_name}')
            self.log_acceptance_ratio = getattr(
                self, f'gibbs_log_acceptance_ratio_{ndims(self.ct)}way_table')
            self.log_target_measure_difference = getattr(
                self, f'log_target_measure_difference_{ndims(self.ct)}way_table')
        elif hasattr(self,f"{self.proposal_type.lower()}_proposal_{ndims(self.ct)}way_table"):
            self.acceptance_type = f"{self.proposal_type.lower().replace('_',' ').capitalize()} sampling"
            self.proposal = getattr(self,f"{self.proposal_type.lower()}_proposal_{ndims(self.ct)}way_table")
            self.log_acceptance_ratio = getattr(self, f'direct_sampling_log_acceptance_ratio_{ndims(self.ct)}way_table')
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
        if 'thinning' in list(self.ct.config.settings['contingency_table'].keys()):
            self.table_thinning = int(self.ct.config.settings['contingency_table']['thinning'])

        if len(self.ct.distribution_name) > 0:
            build_successful = build_successful and self.build_table_distribution()
        else:
            raise Exception(
                f"margin constraints for axes {','.join([str(ax) for ax in self.ct.constraints['constrained_axes']])} cannot be handled for contingency table {self.ct.__class__.__name__} with ndims {ndims(self.ct)}")

        if not build_successful:
            raise Exception(
                f'Proposal type {self.proposal_type.lower()} for type {self.ct.__class__.__name__} not defined')

    def log_intensities_to_multinomial_log_probabilities(self, log_intensity: torch.tensor, column_indices: List[Tuple] = None):

        self.logger.debug('Log intensities to multinomial log probabilities')

        # Compute cell intensities
        # depending on dimensionality of log cell intensities
        if len(list(log_intensity.shape)) <= 2:
            # Slice log cell intensities at column of interest
            if column_indices is not None:
                log_intensity = log_intensity[...,-1:][column_indices]
            # Compute unnormalised log probabilities
            log_intensities_colsums = torch.logsumexp(log_intensity,dim = 0)
        else:
            raise Exception(
                f'Cannot handle log_intensity with dimensions {len(log_intensity.shape)}')

        # Get total intensities (normalising factor in log space)
        total_log_intensities = torch.logsumexp(log_intensities_colsums)
        # Get normalised probabilities
        log_probabilities = log_intensities_colsums-total_log_intensities

        return self.ct.data.margins[tuplize(range(ndims(self.ct)))], log_probabilities, log_intensities_colsums

    def get_table_step_size_support(self, table_prev: torch.tensor, basis_function: torch.tensor, non_zero_cells: List[Tuple]) -> list:
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
        table_new = torch.zeros(tuple(list(unpack_dims(self.ct.data.dims)))).to(dtype = float32,device = self.ct.device)
        # Get fixed cells
        fixed_cells = np.array(self.ct.constraints['cells'])
        # Apply cell constaints if at least one cell is fixed
        if len(fixed_cells) > 0:
            # Extract indices
            fixed_indices = np.ravel_multi_index(fixed_cells.T, self.ct.dims)
            # Fix table cells
            table_new.view(-1)[fixed_indices] = self.ct.data.ground_truth_table.view(-1)[fixed_indices]

        # Non fixed (free) indices
        free_cells = np.array(self.ct.cells)
        free_indices = [ free_cells[:,i] for i in range(ndims(self.ct)) ]

        # Sample from Poisson
        continue_loop = True
        while continue_loop:
            table_new[ free_indices ] = distr.poisson.Poisson(rate = margin_probabilities[ free_indices ]).sample().to(device = self.ct.device,dtype = float32)
            # Continue loop only if table is not sparse admissible
            continue_loop = False#not self.ct.table_sparse_admissible(table_new)
        return table_new

    def multinomial_sample_2way_table(self,margin_probabilities):
        # This is the case with the grand total fixed
        # Initialise table to zero
        # Get fixed cells
        table_new = torch.zeros(tuple(list(unpack_dims(self.ct.data.dims)))).to(dtype = float32,device = self.ct.device)
        fixed_cells = np.array(self.ct.constraints['cells'])
        # Apply cell constaints if at least one cell is fixed
        if len(fixed_cells) > 0:
            # Extract indices
            fixed_indices = np.ravel_multi_index(fixed_cells.T, self.ct.dims)
            # Fix table cells
            table_new.view(-1)[fixed_indices] = self.ct.data.ground_truth_table.view(-1)[fixed_indices]

        # Non fixed (free) indices
        free_cells = np.array(self.ct.cells)
        free_indices = [ free_cells[:,i] for i in range(ndims(self.ct)) ]
        # Get constrained axes
        axis_constrained = min(self.ct.constraints['constrained_axes'], key = len)
        
        # Sample from one multinomial
        continue_loop = True
        while continue_loop:
            # Sample free cells from multinomial
            updated_cells = distr.multinomial.Multinomial(
                total_count = self.ct.residual_margins[tuplize(axis_constrained)].item(),
                probs = margin_probabilities[free_indices].ravel()
            ).sample()
            # Update free cells
            table_new[free_indices] = updated_cells.to(dtype = float32,device = self.ct.device)
            # Reshape table to match original dims
            table_new = torch.reshape(table_new, tuplize(list(unpack_dims(self.ct.data.dims))))
            # Continue loop only if table is not sparse admissible
            continue_loop = False#not self.ct.table_sparse_admissible(table_new)

        return table_new.to(dtype = float32,device = self.ct.device)

    def product_multinomial_sample_2way_table(self,margin_probabilities):
        # This is the case with either margins fixed (but not both)
        # Initialise table to zero
        table_new = torch.zeros(tuple(list(unpack_dims(self.ct.data.dims)))).to(dtype = float32,device = self.ct.device)
        # Get fixed cells
        fixed_cells = np.array(self.ct.constraints['cells'])
        # Apply cell constaints if at least one cell is fixed
        if len(fixed_cells) > 0:
            # Extract indices
            fixed_indices = np.ravel_multi_index(fixed_cells.T, self.ct.dims)
            # Fix table cells
            table_new.view(-1)[fixed_indices] = self.ct.data.ground_truth_table.view(-1)[fixed_indices]

        # Non fixed (free) indices
        free_cells = np.array(self.ct.cells)
        free_indices = [ free_cells[:,i] for i in range(ndims(self.ct)) ]

        # Get constrained axes
        axis_constrained = min(self.ct.constraints['constrained_axes'], key = len)
        # Get all unconstrained axes
        axis_uncostrained = deepcopy(self.ct.constraints['unconstrained_axes'])
        # Get plain uncostrained axis (must have length 1)
        axis_uncostrained_flat = next(flatten(axis_uncostrained))
        
        # Sample from product multinomials
        continue_loop = True
        while continue_loop:
            # Sample free cells from multinomial
            updated_cells = [
                ProbabilityUtils.sample_multinomial_row(
                    i,
                    msum,
                    margin_probabilities,
                    free_cells,
                    axis_uncostrained_flat,
                    device = self.ct.device,
                    ndims = ndims(self.ct)
                ) for i, msum in enumerate(
                    self.ct.residual_margins[tuplize(axis_constrained)]
                )
            ]
            table_new[free_indices] = torch.hstack(updated_cells).to(device = self.ct.device,dtype = float32)
            # Afix non-free cells
            table_new[fixed_cells] = self.ct.data.ground_truth_table[fixed_cells].to(device = self.ct.device,dtype = float32)
            # Reshape table to match original dims
            table_new = torch.reshape(table_new, tuplize(list(unpack_dims(self.ct.data.dims)))).to(device = self.ct.device,dtype = float32)

            # Continue loop only if table is not sparse admissible
            continue_loop = not self.ct.table_admissible(table_new)

        # Make sure sampled table has the right shape
        try:
            assert self.ct.table_admissible(table_new)# and self.ct.table_sparse_admissible(table_new)
        except:
            print(self.ct.table_admissible(table_new),self.ct.table_sparse_admissible(table_new))
            raise Exception()
        return table_new.to(float32)


    def direct_sampling_proposal_2way_table(self, table_prev: torch.tensor, log_intensity: torch.tensor) -> Tuple[Dict, Dict, int, Dict, Dict]:

        self.logger.debug(f'direct_sampling_2way_table for {self.ct.distribution_name}')

        # Get smallest in length constrained axis
        if len(self.ct.constraints['constrained_axes']) > 0:
            axis_constrained = min(self.ct.constraints['constrained_axes'], key = len)
        else:
            axis_constrained = torch.tensor([])
        # Get all unconstrained axes
        axis_uncostrained = deepcopy(self.ct.constraints['unconstrained_axes'])
        # Initialise shape of probability normalisation matrix
        new_shape = torch.ones(ndims(self.ct), dtype = torch.uint8)

        if len(axis_constrained) == ndims(self.ct):
            # Calculate normalisation of multinomial probabilities (total intensity)
            probabilities_normalisation = torch.logsumexp(log_intensity.ravel(),dim = 0)
            axis_uncostrained_flat = None
        elif len(axis_constrained) > 0 and len(axis_uncostrained) > 0:
            # Get plain uncostrained axis (must have length 1)
            axis_uncostrained_flat = list(flatten(axis_uncostrained))[0]
            # Calculate normalisation of multinomial probabilities for each row or column
            probabilities_normalisation = log_intensity.logsumexp(dim = axis_constrained,keepdim = True)
            # Update shape of probability normalisation matrix
            new_shape[axis_uncostrained_flat] = self.ct.data.dims[self.ct.dim_names[axis_uncostrained_flat]]
        else:
            probabilities_normalisation = torch.tensor([0],dtype = float32)
            new_shape = np.array([1]*ndims(self.ct))
            axis_uncostrained_flat = None

        # Reshape and send to device
        probabilities_normalisation = probabilities_normalisation.reshape(tuple(new_shape)).to(dtype = float32,device = self.ct.device)
        # Calculate probabilities
        margin_probabilities = torch.exp(
            log_intensity-probabilities_normalisation
        )

        # Initialise table
        table_new = torch.zeros(tuple(list(unpack_dims(self.ct.data.dims)))).to(dtype = float32,device = self.ct.device)
        firstIteration = True
        # Resample if margins are not allowed to be sparse but contain zeros
        while (not self.ct.table_admissible(table_new)) or firstIteration:
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
                self.logger.info(f"Monte carlo sampling of central fishers hypergeometric.")
                table_new = self.ct.table_monte_carlo_sample()
            else:
                raise Exception(f"Proposal mechanism could not be found for {self.ct.distribution_name}")
            firstIteration = False

        try:
            assert self.ct.table_admissible(table_new)# and self.ct.table_sparse_admissible(table_new)
        except:
            self.logger.error((self.ct.table_admissible(table_new)),(self.ct.table_sparse_admissible(table_new)))
            raise Exception('FAILED')
        
        return table_new.to(float32), None, None, None, None
    

    def degree_one_proposal_2way_table(self, table_prev: torch.tensor, log_intensity: torch.tensor) -> Tuple[Dict, Dict, int, Dict]:

        self.logger.debug('2way table degree one move')

        # Sample Markov basis function index uniformly at random
        func_index = np.random.randint(len(self.markov_basis))

        # Sample epsilon uniformly at random
        epsilon = torch.tensor(uniform_binary_choice(),dtype = float32)

        # initialise new table
        table_new = torch.zeros(tuple(list(unpack_dims(self.ct.data.dims)))).to(dtype = float32,device = self.ct.device)

        # Store old table
        table_new[:] = table_prev

        # Construct new table
        for k in self.markov_basis.basis_dictionaries[func_index].keys():
            table_new[k] = table_prev[k] + epsilon * \
                torch.tensor(self.markov_basis.basis_dictionaries[func_index][k],dtype = float32)
        return table_new.to(device = self.ct.device,dtype = float32), \
            int(epsilon), \
            self.markov_basis.basis_dictionaries[func_index], \
            func_index, \
            {'support': [-1, 1], 'probs': [0.5, 0.5]}

    
    def degree_higher_proposal_2way_table_fishers_hypergeometric(self, tab_prev: torch.tensor, log_intensity: torch.tensor) -> Tuple[Dict, Dict, int, Dict]:

        self.logger.debug('2way table degree higher move Fishers hypergeometric')

        # Sample Markov basis function index uniformly at random
        func_index = np.random.randint(len(self.markov_basis))

        # Get non-zero cells of Markov basis
        # and sort them lexicographically
        non_zero_cells = sorted(list(self.markov_basis.basis_dictionaries[func_index].keys()))
        positive_cells = [cell for cell in non_zero_cells if self.markov_basis.basis_dictionaries[func_index][cell] > 0]

        # Copy previous table
        tab_new = torch.zeros(tab_prev.shape,dtype = float32,device = self.ct.device)
        tab_new[:] = tab_prev

        # Compute log odds ratio for 2x2 table
        # Note that this form is a simplified version of the ratio 
        # of odds ratios for the four table cells that have been changed
        omega = torch.exp(log_odds_cross_ratio(log_intensity, *positive_cells))
        
        # Cast torch to numpy safely within specified range to 
        # prevent over or underflow 
        omega = safe_cast(
            omega,
            minval = np.float32(1e-6),
            maxval = np.float32(1e6)
        )
        
        # Get row and column sums of 2x2 subtable
        rsum = np.int32(tab_prev[non_zero_cells[0]] + tab_prev[non_zero_cells[1]])
        csum = np.int32(tab_prev[non_zero_cells[0]] + tab_prev[non_zero_cells[2]])
        total = np.sum([np.int32(tab_prev[non_zero_cells[i]]) for i in range(len(non_zero_cells))],dtype='int32')
        
        # Sample upper leftmost entry of 2x2 subtable from non-central hypergeometric distribution
        try:
            new_val = nchypergeom_fisher.rvs(M = total, n = rsum, N = csum, odds = omega)
        except:
            new_val = 0
            print('\n')
            print('omega',omega)
            print('total',total)
            print('rsum',rsum)
            print('csum',csum)
            print('non_zero_cells',non_zero_cells)
            print('markov_basis',self.markov_basis.basis_dictionaries[func_index])
            print('tab_prev',[tab_new[c] for c in non_zero_cells])

        # Update upper leftmost entry and propagate to rest of cells
        tab_new[non_zero_cells[0]] = torch.tensor(new_val).to(device = self.ct.device,dtype = float32)
        tab_new[non_zero_cells[1]] = torch.tensor(rsum - new_val,dtype = float32)
        tab_new[non_zero_cells[2]] = torch.tensor(csum - new_val,dtype = float32)
        tab_new[non_zero_cells[3]] = torch.tensor(total - rsum - csum + new_val,dtype = float32)
        
        return tab_new.to(device = self.ct.device,dtype = float32), \
            None, \
            self.markov_basis.basis_dictionaries[func_index], \
            func_index, \
            None

    def metropolis_log_acceptance_ratio_2way_table(self, table_new: torch.tensor, table_prev: torch.tensor, log_intensity: torch.tensor) -> Union[float, None]:
        '''Acceptance ratio for one degree proposals
        This gives the Metropolis Hastings ratio i.e.
        p(x')p(x|x') / p(x)p(x'|x)
        '''

        self.logger.debug('2way table Metropolis Hastings acceptance')
        # Find cells where the images of the two functions differ at
        cells_of_interest = torch.argwhere(table_new != table_prev).detach().numpy().tolist()

        # Compute log target measure difference
        log_acc = self.log_target_measure_difference(
            table_new, 
            table_prev, 
            cells_of_interest, 
            log_intensity
        )
        try:
            assert not torch.isnan(log_acc)
        except:
            print(cells_of_interest)
            print(log_acc)
            raise Exception(f"Metropolis Hastings log acceptance ratio is infinite for table_new = {table_new}, table_prev = {table_prev}")

        if self.ct.table_nonnegative(table_new):
            return log_acc
        else:
            return -torch.tensor([float('inf')])

    def gibbs_log_acceptance_ratio_2way_table(self, table_new: torch.tensor, table_prev: torch.tensor, log_intensity: torch.tensor) -> float:
        ''' Acceptance ratio for higher degree proposals
        This gives the Gibbs acceptance ratio i.e. equal to 1 iff proposal is non-negative'''

        self.logger.debug('2way table degree higher acceptance (NO VALIDITY CHECK - ASSUMES PROPOSAL IS WELL DEFINED)')

        if self.ct.table_nonnegative(table_new):
            return 0
        else:
            self.logger.error(
                f'Proposed table {table_new} is inadmissible or non positive')
            return -torch.tensor([float('inf')])

    def direct_sampling_log_acceptance_ratio_2way_table(self, table_new: torch.tensor, table_prev: torch.tensor, log_intensity: torch.tensor) -> Tuple[Dict, Dict, int, Dict]:
        return 0


    def log_target_measure_difference_2way_table(self, table_new: torch.tensor, table_prev: torch.tensor, cells: torch.tensor, log_intensity: torch.tensor) -> float:
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
            # Add odds ratio of intensities to numerator
            if log_intensity is not None:
                # Note that this form is a simplified version of the ratio 
                # of odds ratios for the four table cells that have been changed
                log_ratio += (table_new[cells[i]] - table_prev[cells[i]])*log_intensity[cells[i]]
            if table_new[cells[i]] > table_prev[cells[i]]:
                print(max(table_prev[cells[i]], 1),max(table_new[cells[i]], 1))
                print(torch.arange(max(table_prev[cells[i]], 1),max(table_new[cells[i]], 1),1))
                log_ratio -= log_factorial_sum(torch.arange(max(table_prev[cells[i]], 1),max(table_new[cells[i]], 1),1))
            elif table_new[cells[i]] < table_prev[cells[i]]:
                print(max(table_new[cells[i]], 1),max(table_prev[cells[i]], 1))
                print(torch.arange(max(table_new[cells[i]], 1),1),max(table_prev[cells[i]], 1))
                log_ratio += log_factorial_sum(torch.arange(max(table_new[cells[i]], 1),max(table_prev[cells[i]], 1),1))
            else:
                continue

        return log_ratio
    

    def table_gibbs_step(self, table_prev: torch.tensor, log_intensity: torch.tensor) -> Tuple[torch.tensor, int]:

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
        if (log_acc >= 0) or (torch.log(torch.rand(1)) < log_acc):
            # Update unconstrained margins
            self.ct.update_unconstrained_margins_from_table(table_new)
            return table_new, 1
        else:
            return table_prev, 0