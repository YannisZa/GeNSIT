import pytest
import logging
import warnings
import itertools
import scipy.stats
import numpy as np
import pandas as pd
from copy import deepcopy
from argparse import Namespace
from gensit.config import Config
from gensit.utils.misc_utils import *
from gensit.contingency_table import instantiate_ct
from gensit.markov_basis import instantiate_markov_basis
from gensit.spatial_interaction_model import ProductionConstrained
from gensit.contingency_table_mcmc import ContingencyTableMarkovChainMonteCarlo

@pytest.fixture
def default_config():
    # Import config
    config = Config("tests/test_configs/test_0_default_config.toml")
    # Check that config is valid
    return config

@pytest.fixture
def dummy_degree_one_proposal_2x3_table():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"dataset":'tests/test_fixtures',"generate_data":False,"contingency_table":{"ct_type":"ContingencyTable2DDependenceModel","import":{"true_table":"degree_one_proposal_2x3_table.csv"}}}}
    # Build a contingency table
    ct = instantiate_ct(config)
    return ct

@pytest.fixture
def dummy_degree_one_proposal_noncentral_2x3_table():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"dataset":'tests/test_fixtures',"generate_data":False,"contingency_table":{"ct_type":"ContingencyTable2DDependenceModel","import":{"true_table":"degree_one_proposal_noncentral_2x3_table.csv"}}}}
    # Build a contingency table
    ct = instantiate_ct(config)
    return ct

@pytest.fixture
def dummy_degree_one_proposal_addition_2x3_table(mb):
    # Cannot import basis functions through contingency tables as they violate non-negativity constraint
    f = mb.import_basis_function("tests/test_fixtures/degree_one_proposal_addition_2x3_table.csv")
    return f

@pytest.fixture
def dummy_degree_higher_proposal_2x3_table():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"dataset":'tests/test_fixtures',"generate_data":False,"contingency_table":{"ct_type":"ContingencyTable2DDependenceModel","import":{"true_table":"degree_higher_proposal_2x3_table.csv"}}}}
    # Build a contingency table
    ct = instantiate_ct(config)
    return ct

@pytest.fixture
def dummy_degree_higher_proposal_addition_2x3_table(mb):
    # Cannot import basis functions through contingency tables as they violate non-negativity constraint
    f = mb.import_basis_function("tests/test_fixtures/degree_higher_proposal_addition_2x3_table.csv")
    return f

@pytest.fixture
def dummy_degree_higher_proposal_noncentral_2x3_table():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"dataset":'tests/test_fixtures',"generate_data":False,"contingency_table":{"ct_type":"ContingencyTable2DDependenceModel","import":{"true_table":"degree_higher_proposal_noncentral_2x3_table.csv"}}}}
    # Build a contingency table
    ct = instantiate_ct(config)
    return ct

@pytest.fixture
def dummy_degree_higher_proposal_noncentral_addition_2x3_table(mb):
    # Cannot import basis functions through contingency tables as they violate non-negativity constraint
    f = mb.import_basis_function("tests/test_fixtures/degree_higher_proposal_noncentral_addition_2x3_table.csv")
    return f


@pytest.fixture
def dummy_degree_higher_proposal_2x2_table():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"dataset":'tests/test_fixtures',"generate_data":False,"contingency_table":{"ct_type":"ContingencyTable2DDependenceModel","import":{"true_table":"tests/test_fixtures/degree_higher_proposal_2x2_table.csv"}}},
                        'mcmc':{'contingency_table':{'proposal':'degree_higher','table0':'monte_carlo_sample','table_steps':1}}}
    # Build a contingency table
    ct = instantiate_ct(config)
    return ct

@pytest.fixture
def dummy_degree_higher_proposal_2x2_table():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"dataset":'tests/test_fixtures',"generate_data":False,"contingency_table":{"ct_type":"ContingencyTable2DDependenceModel","import":{"true_table":"degree_higher_proposal_2x2_table.csv"}}},
                        'mcmc':{'contingency_table':{'proposal':'degree_higher','table0':'monte_carlo_sample','table_steps':1}}}
    # Build a contingency table
    ct = instantiate_ct(config)
    return ct

@pytest.fixture
def dummy_degree_higher_proposal_addition_2x2_table(mb):
    # Cannot import basis functions through contingency tables as they violate non-negativity constraint
    f = mb.import_basis_function("tests/test_fixtures/degree_higher_proposal_addition_2x2_table.csv")
    return f

@pytest.fixture
def ct(default_config):
    # Build a contingency table
    ct = instantiate_ct(default_config)
    return ct

@pytest.fixture
def mb(ct):
    mb = instantiate_markov_basis(ct)
    return mb

@pytest.fixture
def ct_mcmc(ct):
    print(ct.config.settings['mcmc']['contingency_table'])
    print('ct.unconstrained',ct.unconstrained)
    ct_mcmc = ContingencyTableMarkovChainMonteCarlo(ct)
    return ct_mcmc

@pytest.fixture
def sim(default_config):
    # Build a contingency table
    sim = ProductionConstrained(default_config)
    return sim

@pytest.fixture
def ct_2x2():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"dataset":"tests/test_fixtures","generate_data":False,"contingency_table":{"ct_type":"ContingencyTable2DDependenceModel","import":{"true_table":"degree_higher_proposal_2x2_table.csv"}}},
                    'mcmc':{'contingency_table':{'proposal':'degree_higher','table0':'monte_carlo_sample','table_steps':1,'column_sum_proposal':'mode_estimate','column_sum_steps':10}}}
    # Build a contingency table
    ct_2x2 = instantiate_ct(config)
    return ct_2x2

@pytest.fixture
def mb_2x2(ct_2x2):
    mb = instantiate_markov_basis(ct_2x2)
    return mb

def test_build(ct):

    # Initialise Markov Chain Monte Carlo
    mcmc = ContingencyTableMarkovChainMonteCarlo(ct)

    assert mcmc.target_distribution == "Multiple central hypergeometric x Multinomial"
    assert mcmc.proposal_type == 'degree_higher'
    assert mcmc.acceptance_type == 'Gibbs'

    # Change config
    ct_copy = deepcopy(ct)
    ct_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DDependenceModel'
    ct_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_one'
    ct_copy = instantiate_ct(ct_copy.config)

    # Initialise Markov Chain Monte Carlo
    mcmc_copy = ContingencyTableMarkovChainMonteCarlo(ct_copy)

    assert mcmc_copy.target_distribution == "Fisher's multiple non-central hypergeometric x Multinomial"
    assert mcmc_copy.proposal_type == 'degree_one'
    assert mcmc_copy.acceptance_type == 'Metropolis Hastings'

    # Change config
    ct_copy = deepcopy(ct)
    ct_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DDependenceModel'
    ct_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'direct_sampling'
    del ct_copy.config.settings['mcmc']['contingency_table']['column_sum_proposal']
    ct_copy = instantiate_ct(ct_copy.config)

    # Initialise Markov Chain Monte Carlo
    mcmc_copy = ContingencyTableMarkovChainMonteCarlo(ct_copy)

    assert mcmc_copy.target_distribution == "Product Multinomial"
    assert mcmc_copy.proposal_type == 'direct_sampling'
    assert mcmc_copy.acceptance_type == 'Direct sampling'


def test_log_odds_ratio(ct):
    # Create a 2d array of pseudo log intensities
    a = np.array([[1,2,3],[4,5,6]])
    b = np.array([[2,3,5,6],[6,3,2,1],[4,7,8,2]])

    # Initialise Markov Chain Monte Carlo
    mcmc = ContingencyTableMarkovChainMonteCarlo(ct)

    assert mcmc.log_odds_ratio(a,(1,1)) == 0
    assert mcmc.log_odds_ratio(a,(0,1)) == 0
    assert mcmc.log_odds_ratio(b,(1,1)) == -3
    assert mcmc.log_odds_ratio(b,(1,3)) == 0
    assert mcmc.log_odds_ratio(b,(0,2)) == -7

def test_log_odds_ratio(ct):
    # Create a 2d array of pseudo log intensities
    a = np.array([[1,2,5],[4,5,6]])
    # Initialise Markov Chain Monte Carlo
    mcmc = ContingencyTableMarkovChainMonteCarlo(ct)

    assert mcmc.log_odds_cross_ratio(a,(1,1),(0,2)) == 2
    assert mcmc.log_odds_cross_ratio(a,(0,2),(1,1)) == 2

def test_log_product_multinomial_pmf(ct,sim):
    # Initialise Markov Chain Monte Carlo
    mcmc = ContingencyTableMarkovChainMonteCarlo(ct)
    # Log intensities
    log_intensities = sim.log_intensity(sim.log_true_destination_attraction,np.array([0.9,3.0]))

    f_sample = {(0,0):3,(0,1):1,(0,2):2,(1,0):2,(1,1):1,(1,2):1}
    f_sample = f_to_array(f_sample)
    # print(f_sample)
    # print(pd.DataFrame(log_intensities))

    te = log_product_multinomial_pmf(f_sample,log_intensities)
    assert abs(te-(-13.884524045652537)) <= 1e-9

def test_similarity_measure(ct,sim):
    # Log intensities
    log_intensities = sim.log_intensity(sim.log_true_destination_attraction,np.array([0.9,3.0]))
    print(pd.DataFrame(log_intensities))

    f_true = {(0,0):3,(0,1):1,(0,2):2,(1,0):2,(1,1):1,(1,2):2}
    f_true = f_to_array(f_true)
    f_sample = {(0,0):3,(0,1):0,(0,2):3,(1,0):1,(1,1):2,(1,2):2}
    f_sample = f_to_array(f_sample)
    similarity_measure1 = table_similarity_measure(f_true,f_true,log_intensities)
    assert similarity_measure1 == 0
    similarity_measure2 = table_similarity_measure(f_sample,f_true,log_intensities)
    assert abs(similarity_measure2 - 1.2322043403155405) <= 1e-5

    # Add test for colsums similarity measure

# def test_degree_one_proposal_central_hypergeometric(ct,dummy_degree_one_proposal_2x3_table,dummy_degree_one_proposal_addition_2x3_table):

#     # Change config
#     ct_copy = deepcopy(ct)
#     ct_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DIndependenceModel'
#     ct_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_one'
#     ct_copy = instantiate_ct(ct_copy.config)

#     # Initialise Markov Chain Monte Carlo
#     mcmc = ContingencyTableMarkovChainMonteCarlo(ct_copy)

#     # Fix random seed
#     np.random.seed(1234)

#     # Random intialisation
#     fold = ct_copy.table_monte_carlo_sample()

#     # Propose new sample
#     fnew,step_size,basis_func,index,epsilon_distribution = mcmc.proposal(fold,np.array([]))

#     # Proposal should be both admissible and non-negative
#     assert ct.table_admissible(fnew) and ct.table_nonnegative(fnew)
#     # Proposal basis function should match the one expected
#     assert basis_func == dummy_degree_one_proposal_addition_2x3_table
#     # Proposal function should match the one expected
#     assert np.array_equal(fnew,dummy_degree_one_proposal_2x3_table.table)
#     # Epsilon probabilities should be equal
#     assert epsilon_distribution['support'] == [-1,1]
#     assert epsilon_distribution['probs'] == [0.5,0.5]


# def test_degree_higher_proposal_pathological_case_central_hypergeometric(ct,dummy_degree_higher_proposal_2x3_table,dummy_degree_higher_proposal_addition_2x3_table):
#     # This test the acceptance and proposal of a table such that for a given choice of markov basis there is no choice of
#     # step size that yields a non-negative proposal except for the case of stepsize (epsilon) = 0

#     # Change config
#     ct_copy = deepcopy(ct)
#     ct_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DIndependenceModel'
#     ct_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_higher'
#     ct_copy = instantiate_ct(ct_copy.config)

#     # Initialise Markov Chain Monte Carlo
#     mcmc = ContingencyTableMarkovChainMonteCarlo(ct_copy)

#     # Fix random seed
#     np.random.seed(1284)

#     # Random intialisation
#     fold = ct_copy.table_iterative_residual_filling_solution()

#     # Propose new sample
#     fnew,step_size,basis_func,index,epsilon_distribution = mcmc.proposal(fold,np.array([]))

#     print(fold)
#     # print(f_to_df(add))
#     # print(f_to_df(dummy_degree_higher_proposal_addition_2x3_table))
#     print(fnew)
#     # print(dummy_degree_higher_proposal_2x3_table)
#     print(f_to_array(basis_func,shape=(ct.I,ct.J)))
#     print(dummy_degree_higher_proposal_2x3_table)

#     # Proposal should be both admissible and non-negative
#     assert ct.table_admissible(fnew) and ct.table_nonnegative(fnew)
#     # Proposal basis function should match the one expected
#     assert basis_func == dummy_degree_higher_proposal_addition_2x3_table
#     # Proposal function should match the one expected
#     assert np.array_equal(fnew,dummy_degree_higher_proposal_2x3_table.table)
#     # assert all([abs(non_negative_epsilon_probs[j] - true_non_zero_probabilities[j]) <= 1e-9 for j in range(len(true_non_zero_probabilities))])


# def test_degree_one_acceptance_ratio_central_hypergeometric(ct,dummy_degree_one_proposal_2x3_table,dummy_degree_one_proposal_addition_2x3_table):

#     # Change config
#     ct_copy = deepcopy(ct)
#     ct_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DIndependenceModel'
#     ct_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_one'
#     ct_copy.config.settings['logging'] = 'debug'
#     ct_copy = instantiate_ct(ct_copy.config)

#     # Initialise Markov Chain Monte Carlo
#     mcmc = ContingencyTableMarkovChainMonteCarlo(ct_copy)

#     # Fix random seed
#     np.random.seed(1254)

#     # Random intialisation
#     fold = ct.table_monte_carlo_sample()

#     # Propose new sample
#     fnew,step_size,basis_func,index,_ = mcmc.proposal(fold,np.array([]))

#     print(fold)
#     print(f_to_array(basis_func,shape = ct_copy.shape()))
#     print(fnew)

#     # Compute acceptance ratio
#     log_acc = mcmc.log_acceptance_ratio(fnew,fold,np.zeros((ct.I,ct.J)))

#     assert abs(log_acc-(-2.07944154)) <= 1e-5

#     # Tweak proposed sample
#     new_ct = deepcopy(ct)

#     # Update table with proposed sample
#     new_ct.update(fnew)

#     # Make proposal have negative entries
#     f_negative = {(0,0):-30,(1,2):-30,(1,0):30,(0,2):30}
#     # Get current table and update it (DO NOT USE THE update FUNCTIO OF ContingencyTable BECAUSE IT WILL PREVENT NEGATIVE CELL VALUES!!!)
#     # INSTEAD USE THE NATIVE UPDATE FUNCTION OF A DICTIONARY
#     fnew = new_ct.table
#     for k,v in f_negative.items():
#         fnew[k] += v

#     # Compute acceptance ratio
#     log_acc = mcmc.log_acceptance_ratio(fnew,fold,np.zeros((ct.I,ct.J)))

#     assert new_ct.table_admissible(fnew)
#     assert not new_ct.table_nonnegative(fnew)
#     assert log_acc == -np.infty

# def test_degree_higher_acceptance_ratio_central_hypergeometric(ct,mb,dummy_degree_higher_proposal_2x3_table,dummy_degree_higher_proposal_addition_2x3_table):

#     # Change config
#     ct_copy = deepcopy(ct)
#     ct_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DIndependenceModel'
#     ct_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_higher'
#     ct_copy = instantiate_ct(ct_copy.config)

#     # Initialise Markov Chain Monte Carlo
#     mcmc = ContingencyTableMarkovChainMonteCarlo(ct_copy)

#     # Fix random seed
#     np.random.seed(1214)

#     # Random intialisation
#     fold = ct.table_iterative_residual_filling_solution()

#     # Propose new sample
#     fnew,step_size,basis_func,index,_ = mcmc.proposal(fold,np.array([]))
#     # add = {key: int(value*step_size) for key,value in basis_func.items()}

#     print(fold)
#     # print(f_to_df(add))
#     print(fnew)
#     print(f_to_array(basis_func,shape = ct.shape()))

#     # Compute acceptance ratio
#     log_acc = mcmc.log_acceptance_ratio(fnew,fold,np.zeros((ct.I,ct.J)))
#     print(log_acc)

#     assert abs(log_acc-0.0) <= 1e-5

#     # Tweak proposed sample
#     new_ct = deepcopy(ct)

#     # Update table with proposed sample
#     new_ct.update(fnew)

#     # Make proposal have negative entries
#     f_negative = {(0,0):-30,(1,2):-30,(0,2):30,(1,0):30}
#     # Get current table and update it (DO NOT USE THE update FUNCTIO OF ContingencyTable BECAUSE IT WILL PREVENT NEGATIVE CELL VALUES!!!)
#     # INSTEAD USE THE UPDATE NATIVE FUNCTION OF A DICTIONARY
#     fnew = new_ct.table
#     for k,v in f_negative.items():
#         fnew[k] += v

#     # Compute acceptance ratio
#     log_acc = mcmc.log_acceptance_ratio(fnew,fold,np.zeros((ct.I,ct.J)))

#     assert new_ct.table_admissible(fnew)
#     assert not new_ct.table_nonnegative(fnew)
#     # The above should hold in general. This check was removed from gibbs_log_acceptance_ratio_2way_table for computational reasons
#     assert log_acc == -np.infty

# def test_degree_higher_proposal_and_acceptance_central_hypergeometric(ct_2x2,dummy_degree_higher_proposal_2x2_table,dummy_degree_higher_proposal_addition_2x2_table):

#     # Change config
#     ct_2x2_copy = deepcopy(ct_2x2)
#     ct_2x2_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DIndependenceModel'
#     ct_2x2_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_higher'
#     ct_2x2_copy = instantiate_ct(ct_2x2_copy.config)

#     # Initialise Markov Chain Monte Carlo
#     mcmc = ContingencyTableMarkovChainMonteCarlo(ct_2x2_copy)

#     # Fix random seed
#     np.random.seed(1284)

#     # Random intialisation
#     fold = ct_2x2_copy.table

#     # Propose new sample
#     fnew,step_size,basis_func,_,epsilon_distribution = mcmc.proposal(fold,np.array([]))

#     # Get acceptance
#     log_acc = mcmc.log_acceptance_ratio(fnew,fold,np.zeros((ct_2x2_copy.I,ct_2x2_copy.J)))

#     print(fold)
#     print(f_to_array(dummy_degree_higher_proposal_addition_2x2_table,shape = ct_2x2_copy.shape()))
#     print(fnew)
#     print(dummy_degree_higher_proposal_2x2_table)

#     # Proposal should be both admissible and non-negative
#     assert ct_2x2_copy.table_admissible(fnew) and ct_2x2_copy.table_nonnegative(fnew)
#     # Proposal basis function should match the one expected
#     assert basis_func == dummy_degree_higher_proposal_addition_2x2_table
#     # Proposal function should match the one expected
#     assert np.array_equal(fnew,dummy_degree_higher_proposal_2x2_table.table)
#     # Acceptance ratio should always be one
#     assert abs(log_acc-0.0) <= 1e-9


# def test_degree_one_proposal_and_acceptance_non_central_hypergeometric(ct,sim,dummy_degree_one_proposal_noncentral_2x3_table,dummy_degree_one_proposal_addition_2x3_table):

#     # Change config
#     ct_copy = deepcopy(ct)
#     ct_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DDependenceModel'
#     ct_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_one'
#     ct_copy = instantiate_ct(ct_copy.config)

#     # Initialise Markov Chain Monte Carlo
#     mcmc = ContingencyTableMarkovChainMonteCarlo(ct_copy)

#     # Fix random seed
#     np.random.seed(1314)#1294

#     # Random intialisation
#     fold = ct_copy.table

#     auxiliary_parameters = np.array([sim.delta,sim.gamma,sim.kappa,sim.epsilon])
#     # Construct intensities
#     log_intensities = sim.log_intensity(sim.log_true_destination_attraction,np.concatenate([np.array([0.9,0.3*100]),auxiliary_parameters]))

#     # Propose new sample
#     fnew,step_size,basis_func,_,epsilon_distribution = mcmc.proposal(fold,log_intensities)

#     # print(step_size)
#     # print(epsilon_distribution['support'])
#     # print(epsilon_distribution['probs'])
#     # print(log_intensities)
#     print(np.exp(log_intensities))

#     # Get acceptance
#     log_acc = mcmc.log_acceptance_ratio(fnew,fold,log_intensities)

#     # Step sizes for non-negative proposals for basis function 41
#     step_sizes = np.array([-1,1])
#     # Find indices of epsilon distribution corresponding to this support
#     indices_of_interest = [i for i, x in enumerate(epsilon_distribution['support']) if x in step_sizes]
#     # Find all other indices
#     indices_of_interest_complement = set(range(len(epsilon_distribution['support']))) - set(indices_of_interest)
#     indices_of_interest_complement = list(indices_of_interest_complement)

#     # Define the true non-zero probabilities
#     true_non_zero_probabilities = [0.5, 0.5]

#     # Get all steps sizes that result in non-negative proposals
#     non_negative_epsilon_support = [epsilon_distribution['support'][i] for i in indices_of_interest]
#     # Get probabilities corrensponding to all steps sizes that result in non-negative proposals
#     non_negative_epsilon_probs = [epsilon_distribution['probs'][i] for i in indices_of_interest]
#     # Get probabilities corrensponding to all steps sizes that result in negative proposals
#     negative_epsilon_probs = [epsilon_distribution['probs'][i] for i in indices_of_interest_complement]

#     print(fold)
#     print(fnew)
#     print(dummy_degree_one_proposal_noncentral_2x3_table)
#     print(f_to_array(basis_func,shape = ct_copy.shape()))
#     print(log_acc)

#     # Proposal should be both admissible and non-negative
#     assert ct_copy.table_admissible(fnew) and ct_copy.table_nonnegative(fnew)
#     # Proposal basis function should match the one expected
#     assert basis_func == dummy_degree_one_proposal_addition_2x3_table
#     # Proposal function should match the one expected
#     assert np.array_equal(fnew,dummy_degree_one_proposal_noncentral_2x3_table.table)
#     # Epsilon probabilities should be equal
#     assert all(np.equal(epsilon_distribution['support'], list(step_sizes)))
#     # All probabilities corresponding to step sizes that yield negative proposals should be zero
#     assert all(probs==0.0 for probs in negative_epsilon_probs)
#     # All probabilities corresponding to step sizes that yield non-negative proposals should be as specified
#     assert all([abs(non_negative_epsilon_probs[j] - true_non_zero_probabilities[j]) <= 1e-9 for j in range(len(true_non_zero_probabilities))])
#     # Acceptance ratio should always be one
#     assert abs(log_acc-9.71231792754822) <= 1e-9


# def test_degree_higher_proposal_and_acceptance_non_central_hypergeometric(ct,sim,dummy_degree_higher_proposal_noncentral_2x3_table,dummy_degree_higher_proposal_noncentral_addition_2x3_table):

#     # Change config
#     ct_copy = deepcopy(ct)
#     ct_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DDependenceModel'
#     ct_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_higher'
#     ct_copy = instantiate_ct(ct_copy.config)
#     updated_table = np.ones(ct_copy.shape())*(-1)
#     updated_table[(1,2)] = 4
#     ct_copy.update(updated_table)

#     # Initialise Markov Chain Monte Carlo
#     mcmc = ContingencyTableMarkovChainMonteCarlo(ct_copy)

#     # Fix random seed
#     np.random.seed(1294)

#     # Random intialisation
#     fold = ct_copy.table

#     auxiliary_parameters = np.array([sim.delta,sim.gamma,sim.kappa,sim.epsilon])
#     # Construct intensities
#     log_intensities = sim.log_intensity(sim.log_true_destination_attraction,np.concatenate([np.array([0.9,0.3*100]),auxiliary_parameters]))

#     # Propose new sample
#     fnew,step_size,basis_func,_,epsilon_distribution = mcmc.proposal(fold,log_intensities)
#     # add = {key: int(value*step_size) for key,value in basis_func.items()}

#     # Get acceptance
#     log_acc = mcmc.log_acceptance_ratio(fnew,fold,log_intensities)

#     print(fold)
#     print(fnew)
#     print('\n')
#     print(f_to_array(basis_func,shape = ct_copy.shape()))
#     print(f_to_array(dummy_degree_higher_proposal_noncentral_addition_2x3_table,shape = ct_copy.shape()))
#     print(dummy_degree_higher_proposal_noncentral_2x3_table)
#     # print(non_negative_epsilon_probs)


#     # Proposal should be both admissible and non-negative
#     assert ct_copy.table_admissible(fnew) and ct_copy.table_nonnegative(fnew)
#     # Proposal basis function should match the one expected
#     assert basis_func == dummy_degree_higher_proposal_noncentral_addition_2x3_table
#     # Proposal function should match the one expected
#     assert np.array_equal(fnew,dummy_degree_higher_proposal_noncentral_2x3_table.table)
#     # Acceptance ratio should always be one
#     assert abs(log_acc-0) <= 1e-9


# def test_column_sum_direct_sampling(sim,ct_mcmc):

#     # Fix seed
#     np.random.seed(1234)

#     # Create log destination attraction
#     xx = np.array([-0.91629073,-0.91629073,-1.60943791])
#     # Create theta
#     theta = np.array([1.0,0.5*100,0.1,10000,1.1,1.0])

#     log_lambdas = sim.log_intensity(xx,theta)

#     print(log_lambdas)
#     # True sample from multinomial
#     true_column_sum = np.array([6,1,4]).reshape((1,3))

#     # Sample from multinomial distribution
#     column_sum = [0]
#     while (not ct_mcmc.ct.sparse_margins) and (0 in column_sum):
#         column_sum,_,_,_,_ = ct_mcmc.direct_sampling_proposal_1way_table_multinomial(np.empty(sim.dims[1]),log_lambdas)

#     assert len(column_sum) == len(true_column_sum)
#     assert np.array_equal(column_sum, true_column_sum)


# def test_column_sum_mode_estimate(sim,ct_mcmc):

#     # Create log destination attraction
#     xx = np.array([-0.78845736,-1.70474809,-1.01160091])
#     # Create theta
#     theta = np.array([0.9,0.3*10,0.1,10000,1.1,1.0])
#     # Get log intensities
#     log_lambdas = sim.log_intensity(xx,theta)

#     print('lambdas')
#     print(pd.DataFrame(np.exp(log_lambdas)))

#     # Get dimensions
#     I,J = np.shape(log_lambdas)

#     # Previous sample from multinomial
#     prev_column_sum = np.array([[2,1,8]])
#     current_column_sum = np.array([[5,2,4]])

#     # Convert log intensities to log expected colsums
#     _, log_probabilities, _ = ct_mcmc.log_intensities_to_multinomial_log_probabilities(log_lambdas)

#     # print('log_probabilities')
#     # print(np.exp(log_probabilities))

#     # Compute mode of multinomial distribution
#     mode,_,_,_,_ = ct_mcmc.mode_estimate_proposal_1way_table_multinomial(prev_column_sum,log_lambdas)

#     # Find mode through search
#     support = [x for x in itertools.permutations([1,2,3,4,5,6,7,9],J) if sum(x) == sim.total_flow]
#     true_mode = deepcopy(prev_column_sum)
#     max_prob = 0
#     for v in support:
#         if scipy.stats.multinomial.pmf(v,n = sim.total_flow,p = np.exp(log_probabilities)) > max_prob:
#             max_prob = scipy.stats.multinomial.pmf(v,n = sim.total_flow,p = np.exp(log_probabilities))
#             true_mode = np.array(v)

#     # print(true_mode)

#     assert np.all(abs(mode - true_mode)<=1e-9)

#     ''' Passing cell parameters to slice column sum vector'''

#     # Previous sample from multinomial
#     prev_column_sum = np.array([[7,2,2]])
#     current_column_sum = np.array([[5,2,4]])

#     # Define cells
#     column_indices = [0,2]
#     column_cells = [(0,c) for c in column_indices]
#     unused_column_indices = [c for c in range(J) if c not in column_indices]

#     # Define true sampling probabilities
#     sampling_probabilities = np.array([0.558652515421978, 0.44134748457802186])

#     # Sliced total
#     total = np.sum(prev_column_sum[:,column_indices])

#     # Convert log intensities to log expected colsums
#     _, log_probabilities, _ = ct_mcmc.log_intensities_to_multinomial_log_probabilities(log_lambdas[:,column_indices])

#     # Compute mode of multinomial distribution
#     mode,_,_,_,_ = ct_mcmc.mode_estimate_proposal_1way_table_multinomial(prev_column_sum,log_lambdas,column_cells)
#     # Fill in parts of mode that have not been changed
#     for k in unused_column_indices:
#         mode[:,k] = prev_column_sum[:,k]

#     # Find mode through search
#     support = [x for x in itertools.permutations([1,2,3,4,5,6,7,9],2) if sum(x) == total]
#     true_mode = deepcopy(prev_column_sum)
#     max_prob = 0
#     for v in support:
#         if scipy.stats.multinomial.pmf(v,n = total,p = np.exp(log_probabilities)) > max_prob:
#             max_prob = scipy.stats.multinomial.pmf(v,n = total,p = np.exp(log_probabilities))
#             true_mode[:,column_indices] = v

#     print('previous',prev_column_sum)
#     print('mode',mode)
#     print('true_mode',true_mode)
#     print(np.exp(log_probabilities))
#     assert np.all(abs(mode - true_mode)<=1e-5)
#     assert np.all(abs(np.exp(log_probabilities) - sampling_probabilities)<=1e-6)


# def test_2x2_table_mode_estimate(sim,ct_mcmc):

#     # True table
#     a = {(3,13):4,(3,24):25,(8,13):7,(8,24):3}

#     # Use true table to construct intensities
#     log_l = f_to_array(a)
#     log_l[log_l <= 0] = 1
#     log_l = np.log(log_l)

#     # Current table
#     randf = {(3,13):9,(3,24):20,(8,13):2,(8,24):8}
#     # Convert to array
#     randtab = f_to_array(randf,shape=(9,25))
#     # Slice 2D array using list of indices
#     randtab = randtab[tuple(np.array(list(randf.keys())).T)].reshape((2,2))

#     # Compute log odds ratio for 2x2 table
#     omega = np.exp(ct_mcmc.log_odds_cross_ratio(log_l,*[(3,13),(8,24)]))

#     # Get rowsums and total
#     rsums,csums,ttotal = f_to_df(randf).sum(axis = 1).values, f_to_df(randf).sum(axis = 0).values, f_to_df(randf).sum().sum()

#     # Compute mode of multinomial distribution
#     mode = ct_mcmc.mode_estimate_2way_table_non_central_hypergeometric(randtab,omega)

#     # Find mode through search
#     true_mode = -1
#     max_prob = 0
#     for s in range(0,min([rsums[0],csums[0]])+1):
#         if scipy.stats.nchypergeom_fisher.pmf(k = s, M = ttotal, n = csums[0], N = rsums[0], odds = omega) > max_prob:
#             max_prob = scipy.stats.nchypergeom_fisher.pmf(k = s, M = ttotal, n = csums[0], N = rsums[0], odds = omega)
#             true_mode = s

#     assert abs(mode - a[(3,13)]) <= 1e-100
#     assert abs(mode - true_mode) <= 1e-100

#     ''' Another case'''

#     # True table
#     a = {(0,0):2,(0,1):9,(1,0):11,(1,1):7}

#     # Use true table to construct intensities
#     log_l = f_to_array(a)
#     log_l[log_l <= 0] = 1
#     log_l = np.log(log_l)

#     # Current table
#     randf = {(0,0):8,(0,1):3,(1,0):5,(1,1):13}
#     # Convert to array
#     randtab = f_to_array(randf,shape=(2,2))
#     # Slice 2D array using list of indices
#     randtab = randtab[tuple(np.array(list(randf.keys())).T)].reshape((2,2))

#     # Compute log odds ratio for 2x2 table
#     omega = np.exp(ct_mcmc.log_odds_cross_ratio(log_l,*[(0,0),(1,1)]))

#     # Get rowsums and total
#     rsums,csums,ttotal = f_to_df(randf).sum(axis = 1).values, f_to_df(randf).sum(axis = 0).values, f_to_df(randf).sum().sum()

#     # Compute mode of multinomial distribution
#     mode = ct_mcmc.mode_estimate_2way_table_non_central_hypergeometric(randtab,omega)

#     # Find mode through search
#     true_mode = -1
#     max_prob = 0
#     for s in range(0,min([rsums[0],csums[0]])+1):
#         if scipy.stats.nchypergeom_fisher.pmf(k = s, M = ttotal, n = csums[0], N = rsums[0], odds = omega) > max_prob:
#             max_prob = scipy.stats.nchypergeom_fisher.pmf(k = s, M = ttotal, n = csums[0], N = rsums[0], odds = omega)
#             true_mode = s

#     assert abs(mode - a[(0,0)]) <= 1e-100
#     assert abs(mode - true_mode) <= 1e-100

# def test_column_sum_mcmc_degree_one_proposal_and_log_acceptance_ratio(ct_mcmc,sim):

#     # Fix seed
#     np.random.seed(1234)

#     # Create log destination attraction
#     xx = np.array([-0.91629073,-0.91629073,-1.60943791])
#     # Create theta
#     theta = np.array([1.0,0.5*10,0.1,10000,1.1,1.0])
#     # Get log intensities
#     log_lambdas = sim.log_intensity(xx,theta)
#     # Get dimensions
#     I,J = np.shape(log_lambdas)

#     # Convert log intensities to log expected colsums
#     _, log_probabilities, _ = ct_mcmc.log_intensities_to_multinomial_log_probabilities(log_lambdas)

#     # Previous sample from multinomial
#     prev_column_sum = np.array([[2,1,8]])
#     current_column_sum = np.array([[2,2,7]])

#     # Degree one move
#     new_column_sum,_,basis_func,_,_ = ct_mcmc.degree_one_proposal_1way_table_multinomial(prev_column_sum,log_lambdas)

#     # Acceptance
#     # First convert dictionary to array
#     print('log_lambdas')
#     print(log_lambdas)
#     print('new_column_sum')
#     print(new_column_sum)
#     print('move')
#     print(f_to_array(basis_func,shape=(1,J)))
#     print('prev_column_sum')
#     print(prev_column_sum)
#     print('probabilities',np.exp(log_probabilities))

#     assert np.all(abs(current_column_sum - new_column_sum)<=1e-9)

#     log_acc = ct_mcmc.degree_one_1way_table_multinomial_log_acceptance_ratio(new_column_sum,prev_column_sum,log_lambdas)

#     print(log_acc)

#     assert abs(log_acc-1.9987957098215254)<=1e-7

#     ''' Edge case '''
#     # Previous sample from multinomial
#     prev_column_sum = np.array([1,1,9]).reshape((1,3))
#     current_column_sum = np.array([2,0,9]).reshape((1,3))

#     # Degree one move
#     new_column_sum,_,mb_move,_,distr = ct_mcmc.degree_one_proposal_1way_table_multinomial(prev_column_sum,log_lambdas)

#     print(prev_column_sum)
#     print('move')
#     print(f_to_array(mb_move,shape = prev_column_sum.shape))
#     print(new_column_sum)
#     print('probabilities',np.exp(log_probabilities))

#     assert np.all(abs(current_column_sum - new_column_sum)<=1e-9)

#     # Acceptance
#     log_acc = ct_mcmc.degree_one_1way_table_multinomial_log_acceptance_ratio(new_column_sum,prev_column_sum,log_lambdas)

#     assert log_acc == float("-inf")

# def test_column_sum_mcmc_degree_higher_proposal_and_log_acceptance_ratio(ct_mcmc,sim):

#     # Fix seed
#     np.random.seed(1234)

#     # Create log destination attraction
#     xx = np.array([-0.91629073,-0.91629073,-1.60943791])
#     # Create theta
#     theta = np.array([1.0,0.5*10,0.1,10000,1.1,1.0])
#     # Get log intensities
#     log_lambdas = sim.log_intensity(xx,theta)
#     # Get dimensions
#     I,J = np.shape(log_lambdas)

#     # Convert log intensities to log expected colsums
#     _, log_probabilities, _ = ct_mcmc.log_intensities_to_multinomial_log_probabilities(log_lambdas)

#     # Previous sample from multinomial
#     prev_column_sum = np.array([2,1,8]).reshape((1,3))
#     current_column_sum = np.array([2,5,4]).reshape((1,3))

#     # Degree higher move
#     new_column_sum,eps,mb_move,mb_index,distr = ct_mcmc.degree_higher_proposal_1way_table_multinomial(prev_column_sum,log_lambdas)

#     # Acceptance
#     log_acc = ct_mcmc.degree_higher_1way_table_multinomial_log_acceptance_ratio(new_column_sum,prev_column_sum,log_lambdas)

#     print(prev_column_sum)
#     print(f_to_array(mb_move,shape=(1,3)))
#     print(mb_move)
#     print(new_column_sum)
#     print('probabilities',np.exp(log_probabilities))
#     # Proposal should be both admissible and non-negative
#     assert ct_mcmc.column_sum_mb.ct.table_admissible(new_column_sum) and ct_mcmc.column_sum_mb.ct.table_positive(new_column_sum)
#     # Proposal basis function should match the one expected
#     assert mb_move == {(0,1):1,(0,2):-1}
#     # Proposal function should match the one expected
#     assert np.array_equal(current_column_sum,new_column_sum)
#     # Acceptance ratio should always be one
#     assert abs(log_acc-0) <= 1e-9

#     ''' Edge case '''
#     # Previous sample from multinomial
#     prev_column_sum = np.array([1,1,9]).reshape((1,3))
#     current_column_sum = np.array([1,1,9]).reshape((1,3))

#     # Degree one move
#     new_column_sum,eps,mb_move,mb_index,distr = ct_mcmc.degree_higher_proposal_1way_table_multinomial(prev_column_sum,log_lambdas)

#     # Acceptance
#     log_acc = ct_mcmc.degree_higher_1way_table_multinomial_log_acceptance_ratio(new_column_sum,prev_column_sum,log_lambdas)

#     print(new_column_sum)
#     print(prev_column_sum)
#     print(f_to_array(mb_move,shape=(1,3)))
#     print('probabilities',np.exp(log_probabilities))

#     # Proposal should be both admissible and non-negative
#     assert ct_mcmc.column_sum_mb.ct.table_admissible(new_column_sum) and ct_mcmc.column_sum_mb.ct.table_positive(new_column_sum)
#     # Proposal basis function should match the one expected
#     assert mb_move == {(0,0):1,(0,1):-1}
#     # Proposal function should match the one expected
#     assert np.array_equal(current_column_sum, new_column_sum)
#     # Acceptance ratio should always be one
#     assert abs(log_acc-0)<=1e-9


# def test_column_sum_mcmc_degree_higher_proposal_using_stepsize_support_shrinkage(ct_mcmc,sim):

#     # Fix seed
#     np.random.seed(1234)

#     # Create log destination attraction
#     xx = np.array([-0.78845736,-1.70474809,-1.01160091])
#     # Create theta
#     theta = np.array([0.9,0.3*10,0.03333333,10000,1.1,1.0])
#     # Get log intensities
#     log_lambdas = sim.log_intensity(xx,theta)
#     # Get dimensions
#     I,J = np.shape(log_lambdas)

#     print('intensities')
#     # print(pd.DataFrame(np.exp(log_lambdas)))
#     print(np.exp(log_lambdas))

#     # Previous sample from multinomial
#     prev_column_sum = np.array([[5,1,5]])
#     current_column_sum = np.array([[5,3,3]])

#     # Define cells
#     column_indices = [1,2]
#     column_cells = [(0,c) for c in column_indices]

#     # Convert log intensities to log expected colsums
#     _, log_probabilities, _ = ct_mcmc.log_intensities_to_multinomial_log_probabilities(log_lambdas[:,column_indices])
#     probs = np.exp(log_probabilities)

#     # SLiced total
#     total = np.sum(prev_column_sum[:,column_indices])

#     # Degree one move
#     new_column_sum,eps,mb_move,mb_index,distr = ct_mcmc.degree_higher_proposal_1way_table_multinomial(prev_column_sum,log_lambdas)

#     # Acceptance
#     log_acc = ct_mcmc.degree_higher_1way_table_multinomial_log_acceptance_ratio(new_column_sum,prev_column_sum,log_lambdas)

#     # # Find mode through search
#     # support = [x for x in itertools.permutations([1,2,3,4,5,6,7,9],2) if sum(x) == total]
#     # true_mode = deepcopy(prev_column_sum)
#     # max_prob = 0
#     #
#     # for v in support:
#     #     if scipy.stats.multinomial.pmf(v,n = total,p = probs) > max_prob:
#     #         max_prob = scipy.stats.multinomial.pmf(v,n = total,p = probs)
#     #         true_mode[column_indices] = v
#     # print('true_mode',true_mode)

#     # Proposal should be both admissible and non-negative
#     assert ct_mcmc.column_sum_mb.ct.table_admissible(new_column_sum) and ct_mcmc.column_sum_mb.ct.table_positive(new_column_sum)
#     # Proposal basis function should match the one expected
#     assert mb_move == {(0,1):1,(0,2):-1}
#     # Proposal function should match the one expected
#     assert np.array_equal(current_column_sum,new_column_sum)
#     # Acceptance ratio should always be one
#     assert abs(log_acc-0) <= 1e-9

#     ''' More complicated case '''

#     # Fix seed
#     np.random.seed(1224)

#     # Previous sample from multinomial
#     prev_column_sum = np.array([[7,3,1]])
#     current_column_sum = np.array([[5,3,3]])

#     # Define cells
#     column_indices = [0,2]
#     column_cells = [(0,c) for c in column_indices]

#     # Convert log intensities to log expected colsums
#     _, log_probabilities, _ = ct_mcmc.log_intensities_to_multinomial_log_probabilities(log_lambdas[:,column_indices])
#     probs = np.exp(log_probabilities)

#     # SLiced total
#     total = np.sum(prev_column_sum[:,column_indices])

#     # Degree one move
#     new_column_sum,eps,mb_move,mb_index,distr = ct_mcmc.degree_higher_proposal_1way_table_multinomial(prev_column_sum,log_lambdas)

#     # Acceptance
#     log_acc = ct_mcmc.degree_higher_1way_table_multinomial_log_acceptance_ratio(new_column_sum,prev_column_sum,log_lambdas)

#     # # Find mode through search
#     # support = [x for x in itertools.permutations([1,2,3,4,5,6,7,9],2) if sum(x) == total]
#     # true_mode = deepcopy(prev_column_sum)
#     # max_prob = 0
#     #
#     # for v in support:
#     #     if scipy.stats.multinomial.pmf(v,n = total,p = probs) > max_prob:
#     #         max_prob = scipy.stats.multinomial.pmf(v,n = total,p = probs)
#     #         true_mode[column_indices] = v

#     print(prev_column_sum)
#     print(f_to_array(mb_move,shape = prev_column_sum.shape))
#     print(new_column_sum)
#     print('\n')
#     # print('true_mode',true_mode)
#     print('probabilities',np.exp(log_probabilities))

#     # Proposal should be both admissible and non-negative
#     assert ct_mcmc.column_sum_mb.ct.table_admissible(new_column_sum) and ct_mcmc.column_sum_mb.ct.table_positive(new_column_sum)
#     # Proposal basis function should match the one expected
#     assert mb_move == {(0,0):1,(0,2):-1}
#     # Proposal function should match the one expected
#     assert np.array_equal(current_column_sum,new_column_sum)
#     # Acceptance ratio should always be one
#     assert abs(log_acc-0) <= 1e-9

#     print('\n')

#     ''' Edge case '''

#     np.random.seed(1244)

#     # Previous sample from multinomial
#     prev_column_sum = np.array([[1,1,9]])
#     current_column_sum = np.array([[1,1,9]])

#     # Degree higher move
#     new_column_sum,eps,mb_move,mb_index,distr = ct_mcmc.degree_higher_proposal_1way_table_multinomial(prev_column_sum,log_lambdas)

#     # Acceptance
#     log_acc = ct_mcmc.degree_higher_1way_table_multinomial_log_acceptance_ratio(new_column_sum,prev_column_sum,log_lambdas)

#     # Step sizes for non-negative proposals for basis function
#     step_sizes = np.array([0])

#     # Define the true non-zero probabilities
#     true_non_zero_probabilities =  np.array([1.0])

#     print(prev_column_sum)
#     print(f_to_array(mb_move,shape = prev_column_sum.shape))
#     print(new_column_sum)
#     print('probabilities',np.exp(log_probabilities))

#     # Proposal should be both admissible and non-negative
#     assert ct_mcmc.column_sum_mb.ct.table_admissible(new_column_sum) and ct_mcmc.column_sum_mb.ct.table_positive(new_column_sum)
#     # Proposal basis function should match the one expected
#     assert mb_move == {(0,0):1,(0,1):-1}
#     # Proposal function should match the one expected
#     assert np.array_equal(current_column_sum,new_column_sum)
#     # Acceptance ratio should always be one
#     assert abs(log_acc-0)<=1e-9


# def test_degree_higher_proposal_non_central_hypergeometric_using_stepsize_support_shrinkage(ct,ct_2x2,sim,mb_2x2,dummy_degree_higher_proposal_noncentral_2x3_table,dummy_degree_higher_proposal_noncentral_addition_2x3_table):

#     # Change config
#     ct_copy = deepcopy(ct)
#     ct_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DDependenceModel'
#     ct_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_higher'
#     ct_copy = instantiate_ct(ct_copy.config)
#     updated_table = np.ones(ct_copy.shape())*(-1)
#     updated_table[(1,2)] = 4
#     ct_copy.update(updated_table)

#     # Initialise Markov Chain Monte Carlo
#     mcmc = ContingencyTableMarkovChainMonteCarlo(ct_copy)

#     # Fix random seed
#     np.random.seed(1294)

#     # Random intialisation
#     fold = ct_copy.table

#     auxiliary_parameters = np.array([sim.delta,sim.gamma,sim.kappa,sim.epsilon])
#     # Construct intensities
#     log_intensities = sim.log_intensity(sim.log_true_destination_attraction,np.concatenate([np.array([0.9,0.3*100]),auxiliary_parameters]))

#     # Propose new sample
#     fnew,step_size,basis_func,_,epsilon_distribution = mcmc.proposal(fold,log_intensities)
#     # add = {key: int(value*step_size) for key,value in basis_func.items()}

#     # print(step_size)
#     # print(epsilon_distribution['support'])
#     # print(epsilon_distribution['probs'])
#     # print(log_intensities)
#     # print(np.exp(log_intensities))

#     # Get acceptance
#     log_acc = mcmc.log_acceptance_ratio(fnew,fold,log_intensities)

#     # Step sizes for non-negative proposals for basis function 41
#     # step_sizes = np.array([-3,2])

#     # Define the true non-zero probabilities
#     # true_non_zero_probabilities =  np.array([1.92743716e-22, 6.36819425e-17, 4.67562705e-12])

#     # Convert log intensities to log expected colsums
#     _, log_probabilities, _ = mcmc.log_intensities_to_multinomial_log_probabilities(log_intensities)

#     # Find cells for which markov basis is non-zero
#     positive_cells = sorted([k for k in basis_func.keys() if basis_func[k] > 0])

#     # Convert function to table
#     table = fold
#     # Define omega
#     omega = np.exp(mcmc.log_odds_cross_ratio(log_intensities,*positive_cells))
#     # Get row, column and total table sums
#     rsums,csums,ttotal = table.sum(axis = 1), table.sum(axis = 0), table.sum().sum()

#     # Find mode through search
#     # true_mode = np.zeros(len(log_probabilities))
#     # max_prob = 0
#     # true_non_zero_probabilities = []
#     # for s in step_sizes:
#     #     true_non_zero_probabilities.append(scipy.stats.nchypergeom_fisher.pmf(k = fold[positive_cells[0]]+s*basis_func[positive_cells[0]], M = ttotal, n = csums[0], N = rsums[0], odds = omega))
#     #     if scipy.stats.nchypergeom_fisher.pmf(k = fold[positive_cells[0]]+s*basis_func[positive_cells[0]], M = ttotal, n = csums[0], N = rsums[0], odds = omega) > max_prob:
#     #         max_prob = scipy.stats.nchypergeom_fisher.pmf(k = fold[positive_cells[0]]+s*basis_func[positive_cells[0]], M = ttotal, n = csums[0], N = rsums[0], odds = omega)
#     #         true_mode = fold[positive_cells[0]]+s*basis_func[positive_cells[0]]
#     # # Renormalise true probabilities
#     # true_non_zero_probabilities /= np.sum(true_non_zero_probabilities)

#     # Proposal should be both admissible and non-negative
#     assert ct_copy.table_admissible(fnew) and ct_copy.table_nonnegative(fnew)
#     # Proposal basis function should match the one expected
#     assert basis_func == dummy_degree_higher_proposal_noncentral_addition_2x3_table
#     # Proposal function should match the one expected
#     assert np.array_equal(fnew,dummy_degree_higher_proposal_noncentral_2x3_table.table)
#     # Acceptance ratio should always be one
#     assert abs(log_acc-0) <= 1e-9

#     ''' Another case '''

#     # Define true table
#     true_table = {(0,0):3,(0,1):5,(1,0):7,(1,1):2}
#     true_table = f_to_array(true_table)
#     # Define log intensities based on true table
#     log_intensities = np.log(true_table)

#     # Copy table
#     ct_2x2_copy = deepcopy(instantiate_ct(ct_2x2.config))
#     # Update columns
#     ct_2x2_copy.table = true_table
#     ct_2x2_copy.rowsums = [8,9]
#     ct_2x2_copy.colsums = [10,7]
#     ct_2x2_copy.update_table_properties_from_margins(updateCells = True)
#     ct_2x2_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DDependenceModel'
#     ct_2x2_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_higher'

#     np.random.seed(1254)#1294

#     # Define mcmc
#     mcmc = ContingencyTableMarkovChainMonteCarlo(ct_2x2_copy)

#     # Propose new sample
#     fold = {(0,0):1,(0,1):7,(1,0):9,(1,1):0}
#     fold = f_to_array(fold)
#     fnew,step_size,basis_func,_,epsilon_distribution = mcmc.proposal(fold,log_intensities)
#     # add = {key: int(value*step_size) for key,value in basis_func.items()}

#     # Get acceptance
#     log_acc = mcmc.log_acceptance_ratio(fnew,fold,log_intensities)

#     # Step sizes for non-negative proposals for basis function 41
#     step_sizes = np.array([0,1,2,3,4,5,6,7])

#     # Convert log intensities to log expected colsums
#     _, log_probabilities, _ = mcmc.log_intensities_to_multinomial_log_probabilities(log_intensities)

#     # Find cells for which markov basis is non-zero
#     positive_cells = sorted([k for k in basis_func.keys() if basis_func[k] > 0])

#     # Convert function to table
#     table = fold
#     # Define omega
#     omega = np.exp(mcmc.log_odds_cross_ratio(log_intensities,*positive_cells))
#     # Get row, column and total table sums
#     rsums,csums,ttotal = table.sum(axis = 1), table.sum(axis = 0), table.sum().sum()

#     # Find mode through search
#     true_mode = np.zeros(len(log_probabilities))
#     max_prob = 0
#     true_non_zero_probabilities = []
#     for s in step_sizes:
#         true_non_zero_probabilities.append(scipy.stats.nchypergeom_fisher.pmf(k = fold[positive_cells[0]]+s*basis_func[positive_cells[0]], M = ttotal, n = csums[0], N = rsums[0], odds = omega))
#         if scipy.stats.nchypergeom_fisher.pmf(k = fold[positive_cells[0]]+s*basis_func[positive_cells[0]], M = ttotal, n = csums[0], N = rsums[0], odds = omega) > max_prob:
#             max_prob = scipy.stats.nchypergeom_fisher.pmf(k = fold[positive_cells[0]]+s*basis_func[positive_cells[0]], M = ttotal, n = csums[0], N = rsums[0], odds = omega)
#             true_mode = fold[positive_cells[0]]+s*basis_func[positive_cells[0]]
#     # Renormalise true probabilities
#     true_non_zero_probabilities /= np.sum(true_non_zero_probabilities)

#     print('fold')
#     print(fold)
#     print('fnew')
#     print(fnew)
#     print('true_table')
#     print(true_table)
#     print('intensities')
#     print(np.exp(log_intensities))
#     print('\n')
#     print(f_to_array(basis_func,shape = fold.shape))
#     print('true_mode',true_mode)

#     # Proposal should be both admissible and non-negative
#     assert ct_2x2_copy.table_admissible(fnew) and ct_2x2_copy.table_nonnegative(fnew)
#     # Proposal basis function should match the one expected
#     assert basis_func == {(0,0):1,(0,1):-1,(1,0):-1,(1,1):1}
#     assert abs(true_mode-true_table[(0,0)]) <= 1e-9
#     # Proposal function should match the one expected
#     assert np.array_equal(fnew,true_table)

#     # Acceptance ratio should always be one
#     assert abs(log_acc-0) <= 1e-9

# def test_table_monte_carlo_product_of_multinomials(sim,ct_mcmc):

#     # Fix seed
#     np.random.seed(1234)
#     numba_set_seed(1234)

#     # Get true table
#     true_table = ct_mcmc.ct.table

#     # Use true table to obtain intensities
#     # IGNORE WARNING OF TAKING LOGS OF ZEROS - IT DOES NOT MATTER HERE
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         log_lambdas = np.log(true_table)

#     for j in range(5):

#         # Sample table
#         table,_,_,_,_ = ct_mcmc.direct_sampling_2way_table(true_table,log_lambdas)
#         # Assert rowsums are conserved
#         assert all([abs(np.sum(table,axis = 1)[i] - ct_mcmc.ct.rowsums[i]) <= 1e-9 for i in range(ct_mcmc.ct.I)])
#         # Assert colsums are strictly positive
#         assert all([np.sum(table,axis = 0)[k] > 0 for k in range(ct_mcmc.ct.J)])

#     assert np.array_equal(table,np.array([[2,0,4],[3,2,0]]))

# def test_degree_one_proposal_and_acceptance_product_multinomials(ct,ct_2x2,sim):

#     # Change config
#     ct_copy = deepcopy(ct)
#     ct_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DDependenceModel'
#     ct_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_one'
#     del ct_copy.config.settings['mcmc']['contingency_table']['column_sum_proposal']
#     ct_copy = instantiate_ct(ct_copy.config)
#     # Update margin to fix
#     ct_copy.constrained = 'rows'

#     # Initialise Markov Chain Monte Carlo
#     mcmc = ContingencyTableMarkovChainMonteCarlo(ct_copy)

#     auxiliary_parameters = np.array([sim.delta,sim.gamma,sim.kappa,sim.epsilon])
#     # Construct intensities
#     log_intensities = sim.log_intensity(sim.log_true_destination_attraction,np.concatenate([np.array([0.9,0.3*100]),auxiliary_parameters]))

#     # Fix random seed
#     np.random.seed(1294)

#     # Random intialisation
#     fold = ct_copy.table

#     # Propose new sample
#     fnew,step_size,basis_func,_,epsilon_distribution = mcmc.proposal(fold,log_intensities)

#     # print(step_size)
#     # print(epsilon_distribution['support'])
#     # print(epsilon_distribution['probs'])
#     print(log_intensities)
#     print(pd.DataFrame(log_intensities))

#     # Get acceptance
#     log_acc = mcmc.log_acceptance_ratio(fnew,fold,log_intensities)

#     # Step sizes for non-negative proposals for basis function 41
#     step_sizes = np.array([-1,1])
#     # Find indices of epsilon distribution corresponding to this support
#     indices_of_interest = [i for i, x in enumerate(epsilon_distribution['support']) if x in step_sizes]
#     # Find all other indices
#     indices_of_interest_complement = set(range(len(epsilon_distribution['support']))) - set(indices_of_interest)
#     indices_of_interest_complement = list(indices_of_interest_complement)

#     # Define the true non-zero probabilities
#     true_non_zero_probabilities = [0.5, 0.5]

#     # Get all steps sizes that result in non-negative proposals
#     non_negative_epsilon_support = [epsilon_distribution['support'][i] for i in indices_of_interest]
#     # Get probabilities corrensponding to all steps sizes that result in non-negative proposals
#     non_negative_epsilon_probs = [epsilon_distribution['probs'][i] for i in indices_of_interest]
#     # Get probabilities corrensponding to all steps sizes that result in negative proposals
#     negative_epsilon_probs = [epsilon_distribution['probs'][i] for i in indices_of_interest_complement]

#     # Addition
#     addition_f = {(1,1):1,(1,2):-1}
#     # New f
#     new_f = {(0,0):3,(0,1):0,(0,2):3,(1,0):2,(1,1):1,(1,2):2}
#     new_f = f_to_array(new_f)


#     # print('\n')
#     print(fold)
#     # # print(f_to_df(basis_func))
#     # # print(f_to_df(dummy_degree_one_proposal_addition_2x3_table))
#     print(fnew)
#     # # print(dummy_degree_one_proposal_noncentral_2x3_table)

#     # Proposal should be both admissible and non-negative
#     assert ct_copy.table_admissible(fnew) and ct_copy.table_nonnegative(fnew)
#     # Proposal basis function should match the one expected
#     assert basis_func == addition_f
#     # Proposal function should match the one expected
#     assert np.array_equal(fnew,new_f)
#     # Epsilon probabilities should be equal
#     assert all(np.equal(epsilon_distribution['support'], list(step_sizes)))
#     # All probabilities corresponding to step sizes that yield negative proposals should be zero
#     assert all(probs==0.0 for probs in negative_epsilon_probs)
#     # All probabilities corresponding to step sizes that yield non-negative proposals should be as specified
#     assert all([abs(non_negative_epsilon_probs[j] - true_non_zero_probabilities[j]) <= 1e-9 for j in range(len(true_non_zero_probabilities))])
#     # Acceptance ratio should always be one
#     assert abs(log_acc-(3.070423875853188)) <= 1e-8

#     # Fix random seed
#     np.random.seed(1264)

#     # Change config
#     ct_copy = deepcopy(ct_2x2)
#     ct_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DDependenceModel'
#     ct_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_one'
#     del ct_copy.config.settings['mcmc']['contingency_table']['column_sum_proposal']
#     ct_copy = instantiate_ct(ct_copy.config)
#     updated_table = np.ones(ct_copy.shape())*(-1)
#     updated_table[(0,0)] = 1
#     updated_table[(0,1)] = 1
#     ct_copy.update(updated_table)
#     # Update margin to fix
#     ct_copy.constrained = 'rows'

#     # Initialise Markov Chain Monte Carlo
#     mcmc = ContingencyTableMarkovChainMonteCarlo(ct_copy)

#     # Set previous table
#     fold = ct_copy.table

#     # Propose new sample
#     fnew,step_size,basis_func,_,epsilon_distribution = mcmc.proposal(fold,log_intensities)
#     print('step_size',step_size)
#     add = {key: int(value*step_size) for key,value in basis_func.items()}

#     # Get acceptance
#     log_acc = mcmc.log_acceptance_ratio(fnew,fold,log_intensities)

#     if log_acc <= np.log(np.random.uniform(0,1)):
#         fnew = fold

#     # Step sizes for non-negative proposals for basis function 41
#     step_sizes = np.array([-1,1])
#     # Find indices of epsilon distribution corresponding to this support
#     indices_of_interest = [i for i, x in enumerate(epsilon_distribution['support']) if x in step_sizes]
#     # Find all other indices
#     indices_of_interest_complement = set(range(len(epsilon_distribution['support']))) - set(indices_of_interest)
#     indices_of_interest_complement = list(indices_of_interest_complement)

#     # Define the true non-zero probabilities
#     true_non_zero_probabilities = [0.5, 0.5]

#     # Get all steps sizes that result in non-negative proposals
#     non_negative_epsilon_support = [epsilon_distribution['support'][i] for i in indices_of_interest]
#     # Get probabilities corrensponding to all steps sizes that result in non-negative proposals
#     non_negative_epsilon_probs = [epsilon_distribution['probs'][i] for i in indices_of_interest]
#     # Get probabilities corrensponding to all steps sizes that result in negative proposals
#     negative_epsilon_probs = [epsilon_distribution['probs'][i] for i in indices_of_interest_complement]

#     # Addition
#     addition_f = {(0,0):1,(0,1):-1}
#     # New f
#     new_f = {(0,0):1,(0,1):1,(1,0):0,(1,1):3}
#     new_f = f_to_array(new_f)


#     print('\n')
#     print(fold)
#     # print(f_to_df(basis_func))
#     # print(f_to_df(dummy_degree_one_proposal_addition_2x3_table))
#     print(fnew)
#     # print(dummy_degree_one_proposal_noncentral_2x3_table)
#     # print(non_negative_epsilon_probs)

#     # Proposal should be both admissible and non-negative
#     assert ct_copy.table_admissible(fnew) and ct_copy.table_nonnegative(fnew)
#     # Proposal basis function should match the one expected
#     assert basis_func == addition_f
#     # Proposal function should match the one expected
#     assert np.array_equal(fnew,new_f)
#     # Epsilon probabilities should be equal
#     assert all(np.equal(epsilon_distribution['support'], list(step_sizes)))
#     # All probabilities corresponding to step sizes that yield negative proposals should be zero
#     assert all(probs==0.0 for probs in negative_epsilon_probs)
#     # All probabilities corresponding to step sizes that yield non-negative proposals should be as specified
#     assert all([abs(non_negative_epsilon_probs[j] - true_non_zero_probabilities[j]) <= 1e-9 for j in range(len(true_non_zero_probabilities))])
#     # Acceptance ratio should always be one
#     assert not np.isfinite(log_acc)


# def test_degree_higher_proposal_and_acceptance_product_multinomials(ct,ct_2x2,sim):

#     auxiliary_parameters = np.array([sim.delta,sim.gamma,sim.kappa,sim.epsilon])
#     # Construct intensities
#     log_intensities = sim.log_intensity(sim.log_destination_attraction,np.concatenate([np.array([0.9,3.0]),auxiliary_parameters]))

#     # Change config
#     ct_copy = deepcopy(ct)
#     ct_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DDependenceModel'
#     ct_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_higher'
#     del ct_copy.config.settings['mcmc']['contingency_table']['column_sum_proposal']
#     ct_copy = instantiate_ct(ct_copy.config)
#     # Update margin to fix
#     ct_copy.constrained = 'rows'
#     updated_table = np.ones(ct_copy.shape())*(-1)
#     updated_table[(0,0)] = 1
#     updated_table[(1,2)] = 4
#     updated_table[(0,2)] = 5
#     updated_table[(1,0)] = 4
#     ct_copy.update(updated_table)


#     # Initialise Markov Chain Monte Carlo
#     mcmc = ContingencyTableMarkovChainMonteCarlo(ct_copy)

#     # Fix random seed
#     np.random.seed(1274)

#     # Random intialisation
#     fold = ct_copy.table


#     # Propose new sample
#     fnew,step_size,basis_func,_,epsilon_distribution = mcmc.proposal(fold,log_intensities)

#     # Get acceptance
#     log_acc = mcmc.log_acceptance_ratio(fnew,fold,log_intensities)

#     # True markov basis
#     m_basis = {(0,0):1,(0,2):-1}
#     # New table
#     new_f = {(0,0):3,(0,1):0,(0,2):3,(1,0):4,(1,1):2,(1,2):4}
#     new_f = f_to_array(new_f)

#     print('log_intensities')
#     print(pd.DataFrame(log_intensities))
#     print('fold')
#     print(fold)
#     print('fnew')
#     print(fnew)
#     print('\n')
#     print(f_to_df(basis_func))


#     # Proposal should be both admissible and non-negative
#     assert ct_copy.table_admissible(fnew) and ct_copy.table_nonnegative(fnew)
#     # Proposal basis function should match the one expected
#     assert basis_func == m_basis
#     # Proposal function should match the one expected
#     assert np.array_equal(fnew,new_f)
#     assert abs(log_acc-0) <= 1e-9

#     ''' Pathological case leading to zero column sums'''

#     # Fix random seed
#     np.random.seed(1214)

#     # Change config
#     ct_copy = deepcopy(ct_2x2)
#     ct_copy.config.settings['inputs']['contingency_table']['ct_type'] = 'ContingencyTable2DDependenceModel'
#     ct_copy.config.settings['mcmc']['contingency_table']['proposal'] = 'degree_higher'
#     del ct_copy.config.settings['mcmc']['contingency_table']['column_sum_proposal']
#     ct_copy = instantiate_ct(ct_copy.config)
#     updated_table = np.ones(ct_copy.shape())*(-1)
#     updated_table[(0,0)] = 1
#     updated_table[(0,1)] = 0
#     updated_table[(1,1)] = 1
#     ct_copy.update(updated_table)
#     # Update margin to fix
#     ct_copy.constrained = 'rows'

#     # Initialise Markov Chain Monte Carlo
#     mcmc = ContingencyTableMarkovChainMonteCarlo(ct_copy)

#     # Set previous table
#     fold = ct_copy.table

#     # Propose new sample
#     fnew,step_size,basis_func,_,epsilon_distribution = mcmc.proposal(fold,log_intensities)

#     # Get acceptance
#     log_acc = mcmc.log_acceptance_ratio(fnew,fold,log_intensities)

#     # True markov basis
#     m_basis = {(0,0):1,(0,1):-1}
#     # New table
#     new_f = {(0,0):1,(0,1):0,(1,0):0,(1,1):1}
#     new_f = f_to_array(new_f)

#     print('fold')
#     print(fold)
#     print('fnew')
#     print(fnew)
#     print('new_f')
#     print(new_f)
#     print(f_to_array(basis_func,shape = fold.shape))
#     print('\n')

#     # Proposal should be both admissible and non-negative
#     assert ct_copy.table_admissible(fnew) and ct_copy.table_nonnegative(fnew)
#     # Proposal basis function should match the one expected
#     assert basis_func == m_basis
#     # Proposal function should match the one expected
#     assert np.array_equal(fnew,new_f)
#     # Acceptance ratio should always be one
#     assert abs(log_acc-0) <= 1e-9
