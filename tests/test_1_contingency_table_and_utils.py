import os
import pytest
import pandas as pd
import numpy as np


# from pprint import pprint
from copy import deepcopy
from argparse import Namespace
from multiresticodm.config import Config
from multiresticodm.utils import f_to_df,df_to_f,f_to_array,array_to_f
from multiresticodm.math_utils import logsumexp,normalise,normalised_manhattan_distance
from multiresticodm.contingency_table import ContingencyTable2D

@pytest.fixture
def default_config():
    # Import config
    config = Config("tests/test_configs/test_0_default_config.toml")
    return config

# @pytest.fixture
# def dummy_generated_4x6_config():
#     # Dummy config to import table
#     config = Namespace(**{'settings':''})
#     config.settings = {'inputs':{"generate_data":False,"contingency_table":{"dependence_model":False,"import":{"file":"tests/test_fixtures/generated_4x6_table.csv"}}}}
#     return config
#
# @pytest.fixture
# def dummy_updated_4x6_config():
#     # Dummy config to import table
#     config = Namespace(**{'settings':''})
#     config.settings = {'inputs':{"generate_data":False,"contingency_table":{"dependence_model":False,"import":{"file":"tests/test_fixtures/updated_4x6_table.csv"}}}}
#     return config
#
# @pytest.fixture
# def dummy_augmented_5x7_config():
#     # Dummy config to import table
#     config = Namespace(**{'settings':''})
#     config.settings = {'inputs':{"generate_data":False,"contingency_table":{"dependence_model":False,"import":{"file":"tests/test_fixtures/augmented_5x7_table.csv"}}}}
#     return config
#
# @pytest.fixture
# def dummy_monte_carlo_4x6_config():
#     # Dummy config to import table
#     config = Namespace(**{'settings':''})
#     config.settings = {'inputs':{"generate_data":False,"contingency_table":{"dependence_model":False,"import":{"file":"tests/test_fixtures/monte_carlo_4x6_table.csv"}}}}
#     return config
#
# @pytest.fixture
# def dummy_max_entropy_4x6_table_config():
#     # Dummy config to import table
#     config = Namespace(**{'settings':''})
#     config.settings = {'inputs':{"generate_data":False,"contingency_table":{"dependence_model":False,"import":{"file":"tests/test_fixtures/max_entropy_4x6_table.csv"}}}}
#     return config
#
# @pytest.fixture
# def dummy_iterative_res_filling_4x6_table_config():
#     # Dummy config to import table
#     config = Namespace(**{'settings':''})
#     config.settings = {'inputs':{"generate_data":False,"contingency_table":{"dependence_model":False,"import":{"file":"tests/test_fixtures/iterative_res_filling_4x6_table.csv"}}}}
#     return config
#
# @pytest.fixture
# def dummy_iterative_uniform_res_filling_4x6_table_config():
#     # Dummy config to import table
#     config = Namespace(**{'settings':''})
#     config.settings = {'inputs':{"generate_data":False,"contingency_table":{"dependence_model":False,"import":{"file":"tests/test_fixtures/iterative_uniform_res_filling_4x6_table.csv"}}}}
#     return config

@pytest.fixture
def dummy_generated_2x3_config():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"generate_data":False,"directory":'./tests/test_fixtures/',"contingency_table":{"dependence_model":False,"import":{"true_table":"generated_2x3_table.csv"}}}}
    return config

@pytest.fixture
def dummy_updated_2x3_config():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"generate_data":False,"directory":'./tests/test_fixtures/',"contingency_table":{"dependence_model":False,"import":{"true_table":"updated_2x3_table.csv"}}}}
    return config

@pytest.fixture
def dummy_augmented_5x7_config():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"generate_data":False,"directory":'./tests/test_fixtures/',"contingency_table":{"dependence_model":False,"import":{"true_table":"augmented_5x7_table.csv"}}}}
    return config

@pytest.fixture
def dummy_monte_carlo_2x3_config():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"generate_data":False,"directory":'./tests/test_fixtures/',"contingency_table":{"dependence_model":False,"import":{"true_table":"monte_carlo_2x3_table.csv"}}}}
    return config

@pytest.fixture
def dummy_max_entropy_2x3_table_config():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"generate_data":False,"directory":'./tests/test_fixtures/',"contingency_table":{"dependence_model":False,"import":{"true_table":"max_entropy_2x3_table.csv"}}}}
    return config

@pytest.fixture
def dummy_iterative_res_filling_2x3_table_config():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"generate_data":False,"directory":'./tests/test_fixtures/',"contingency_table":{"dependence_model":False,"import":{"true_table":"iterative_res_filling_2x3_table.csv"}}}}
    return config

@pytest.fixture
def dummy_iterative_uniform_res_filling_2x3_table_config():
    # Dummy config to import table
    config = Namespace(**{'settings':''})
    config.settings = {'inputs':{"seed":1234,"generate_data":False,"directory":'./tests/test_fixtures/',"contingency_table":{"dependence_model":False,"import":{"true_table":"iterative_uniform_res_filling_2x3_table.csv"}}}}
    return config

@pytest.fixture
def f_sparse():
    return {(0,0): 1, (0,1): 0, (0,2): 0, (0,3): 0, (0,4): 0, (0,5): 0, (1,0): 0, (1,1): 1, (1,2): 0, (1,3): 0, (1,4): 0, (1,5): 0, \
    (2,0): 0, (2,1): 0, (2,2): 0, (2,3): 0, (2,4): 0, (2,5): 0, (3,0): 0, (3,1): 0, (3,2): 0, (3,3): 0, (3,4): 0, (3,5): 0}

@pytest.fixture
def ct(default_config):
    # Build a contingency table
    ct = ContingencyTable2D(default_config)
    return ct

def test_table_to_df_df_to_table_f_to_array_and_array_to_f(ct):

    # Generate f tables
    f1 = {(0,1):4,(0,0):1,(0,3):2,(0,2):3}
    true_table1 = np.array([1,4,3,2])
    f2 = {(1,0):7,(1,1):5,(1,2):6,(1,3):8,(0,1):4,(0,0):1,(0,3):2,(0,2):3}
    true_table2 = np.array([[1,4,3,2],[7,5,6,8]])
    f3 = {(1,0):7,(1,1):5,(0,1):4,(0,0):1}
    true_table3 = np.array([[1,4,0,0],[7,5,0,0]])

    assert np.all(abs(true_table1 - f_to_df(f1).values[0])<=1e-9)
    assert df_to_f(pd.DataFrame([true_table1])) == f1
    assert np.all(abs(true_table1 - f_to_array(f1))<=1e-9)
    assert array_to_f(true_table1) == f1

    assert np.all(abs(true_table2 - f_to_df(f2).values)<=1e-9)
    assert df_to_f(pd.DataFrame(true_table2)) == f2
    assert np.all(abs(true_table2 - f_to_array(f2))<=1e-9)
    assert array_to_f(true_table2) == f2

    assert array_to_f(f_to_array(f2)) == f2
    assert np.all(abs(f_to_array(array_to_f(true_table2)) - true_table2)<=1e-9)

    assert np.all(abs(f_to_array(f3,shape=(2,4)) - true_table3)<=1e-9)


def test_normalised_manhattan_distance():
    f1,f2,f3 = {(0,0):1,(0,1):2},{(0,0):0,(0,1):2},{(0,0):-1,(0,1):4}
    f1 = f_to_array(f1)
    f2 = f_to_array(f2)
    f3 = f_to_array(f3)

    dis1 = normalised_manhattan_distance(f1,f2)
    dis2 = normalised_manhattan_distance(f1,f3)
    dis3 = normalised_manhattan_distance(f2,f3)

    assert (dis1 - 1./2) <= 1e-9
    assert (dis2 - 3./(2*2)) <= 1e-9
    assert (dis3 - 3./(2*2)) <= 1e-9


def test_2d_contingency_table_generation(ct,dummy_generated_2x3_config):

    # Import true contingency table
    true_ct = ContingencyTable2D(dummy_generated_2x3_config)

    print(ct)
    print(true_ct)

    assert np.array_equal(ct.table,true_ct.table)

# def test_2d_contingency_table_augmentation(ct,dummy_augmented_5x7_config):
#
#     # Augment by 1 row and column
#     ct.resize(addRows=1,addCols=1)
#
#     # Import true contingency table
#     true_ct = ContingencyTable2D(dummy_augmented_5x7_config)
#
#     assert ct.table.equals(true_ct.table)

def test_2d_contingency_table_update_colsums(ct):

    ct.update_colsums([1,2,3])
    assert np.array_equal(ct.colsums, [1,2,3])

    ct.update_colsums([5,4],column_indices=[0,2])
    assert np.array_equal(ct.colsums, [5,2,4])

def test_2d_contingency_table_update(ct,dummy_updated_2x3_config,f_sparse):

    # Import true contingency table
    true_ct = ContingencyTable2D(dummy_updated_2x3_config)

    # Table updates
    tab1 = np.ones(true_ct.shape())*(-1)
    tab1[(0,0)] = 1
    tab1[(1,1)] = 8

    # Perform update
    ct.update(tab1)

    assert np.array_equal(ct.table,true_ct.table)

    # Reset table
    ct.reset()
    ct_copy = np.asarray([[1,0,3],[2,1,1]])

    table_sparse = f_to_array(f_sparse)
    table_sparse[table_sparse == 0] = -1

    # Perform update
    ct.update(table_sparse)

    assert np.array_equal(ct.table,ct_copy)

def test_2d_contingency_table_reset(ct,dummy_generated_2x3_config,f_sparse):

    # Import true contingency table
    true_ct = ContingencyTable2D(dummy_generated_2x3_config)
    ct = ContingencyTable2D(dummy_generated_2x3_config)

    # Table updates
    f1 = {(0,0):1,(1,1):0,(4,0):100,(0,6):100}
    tab1 = f_to_array(f1)
    # Perform update
    ct.update(tab1)

    # Reset
    ct.reset()

    assert np.array_equal(ct.table,true_ct.table)


def test_table_nonnegative(ct):

    # Generate f tables
    f1 = {(0,0):1,(0,1):2,(1,1):0,(1,0):1}
    tab1 = f_to_array(f1)
    f2 = {(0,0):1,(0,1):2,(1,1):0,(1,0):-1}
    tab2 = f_to_array(f2)
    f3 = {(0,0):0,(0,1):0,(1,1):0,(1,0):0}
    tab3 = f_to_array(f3)
    f4 = {(0,0):0,(0,1):0,(1,1):0,(1,0):-1}
    tab4 = f_to_array(f4)

    assert ct.table_nonnegative(tab1)
    assert not ct.table_nonnegative(tab2)
    assert ct.table_nonnegative(tab3)
    assert not ct.table_nonnegative(tab4)

def test_table_positive(ct):

    # Generate f tables
    f1 = {(0,0):1,(0,1):2,(1,1):0,(1,0):1}
    tab1 = f_to_array(f1)
    f2 = {(0,0):1,(0,1):2,(1,1):0,(1,0):-1}
    tab2 = f_to_array(f2)
    f3 = {(0,0):0,(0,1):0,(1,1):0,(1,0):0}
    tab3 = f_to_array(f3)
    f4 = {(0,0):0,(0,1):0,(1,1):0,(1,0):-1}
    tab4 = f_to_array(f4)
    f5 = {(0,0):1,(0,1):2,(1,1):3,(1,0):4}
    tab5 = f_to_array(f5)

    assert not ct.table_positive(tab1)
    assert not ct.table_positive(tab2)
    assert not ct.table_positive(tab3)
    assert not ct.table_positive(tab4)
    assert ct.table_positive(tab5)

def test_2d_contingency_table_summary_statistic(ct):

    assert np.array_equal(ct.table_summary_statistic(ct.table), np.array([6,5,5,2,4]))#[15, 15, 15, 12, 11, 12, 14, 8, 7, 5]


def test_2d_contingency_table_row_summary_statistic(ct):

    assert np.array_equal(ct.table_row_summary_statistic(ct.table),np.array([6,5])) #[15, 15, 15, 12, 11, 12, 14, 8, 7, 5]

def test_2d_contingency_table_column_summary_statistic(ct):

    assert np.array_equal(ct.table_column_summary_statistic(ct.table), np.array([5,2,4]))#[15, 15, 15, 12, 11, 12, 14, 8, 7, 5]


def test_2d_contingency_table_admissible(ct):

    # Convert table into dictionary
    fct = ct.table

    assert ct.table_admissible(fct)

    # Add and subtract 10 so that col and row sums remain the same
    # This would create negative cells
    # admissibility ONLY looks at col/row sums
    # non-negativity ONLY looks at non-negative cells
    tab1 = deepcopy(fct)
    tab1[(0,0)] += 10
    tab1[(1,1)] += 10
    tab1[(0,1)] -= 10
    tab1[(1,0)] -= 10

    assert ct.table_admissible(tab1)

def test_contingency_table_length(ct):

    print(ct)
    assert len(ct) == 6#24

def test_contingency_table_entropy(ct):

    # Generate f tables
    f1 = {(0,0):1,(0,1):2,(1,1):0,(1,0):1}
    tab1 = f_to_array(f1)
    f2 = {(0,0):0,(0,1):0,(1,1):0,(1,0):0}
    tab2 = f_to_array(f2)

    assert abs(ct.entropy(tab1)-1.039720771) <= 1e-5
    assert abs(ct.entropy(tab2)-0.0) <= 1e-5

def test_contingency_table_sparsity(ct):

    # Generate f tables
    f1 = {(0,0):1,(0,1):2,(1,1):1,(1,0):1}
    tab1 = f_to_array(f1)
    f2 = {(0,0):0,(0,1):0,(1,1):0,(1,0):0}
    tab2 = f_to_array(f2)
    f3 = {(0,0):0,(0,1):0,(1,1):0,(1,0):1}
    tab3 = f_to_array(f3)

    assert abs(ct.sparsity(tab1)-0.0) <= 1e-5
    assert abs(ct.sparsity(tab2)-1.0) <= 1e-5
    assert abs(ct.sparsity(tab3)-0.75) <= 1e-5

def test_2d_table_solver_name_to_function_mapping(ct,
                                                dummy_monte_carlo_2x3_config,
                                                dummy_max_entropy_2x3_table_config,
                                                dummy_iterative_res_filling_2x3_table_config,
                                                dummy_iterative_uniform_res_filling_2x3_table_config):

    np.random.seed(ct.seed)

    # Import true contingency table
    true_ct = ContingencyTable2D(dummy_monte_carlo_2x3_config)
    # Get solver
    solver = ct.map_table_solver_name_to_function('table_monte_carlo_sample')
    # Run solver
    res = solver()

    assert np.array_equal(res,true_ct.table)

    # Import true contingency table
    true_ct = ContingencyTable2D(dummy_max_entropy_2x3_table_config)
    # Get solver
    solver = ct.map_table_solver_name_to_function('table_maximum_entropy_solution')
    # Run solver
    res = solver()
    assert np.array_equal(res,true_ct.table)

    # Import true contingency table
    true_ct = ContingencyTable2D(dummy_iterative_res_filling_2x3_table_config)
    # Get solver
    solver = ct.map_table_solver_name_to_function('table_iterative_residual_filling_solution')
    # Run solver
    res = solver()
    assert np.array_equal(res,true_ct.table)

    # Import true contingency table
    true_ct = ContingencyTable2D(dummy_iterative_uniform_res_filling_2x3_table_config)
    # Get solver
    solver = ct.map_table_solver_name_to_function('table_iterative_uniform_residual_filling_solution')
    # Run solver
    res = solver()
    assert np.array_equal(res,true_ct.table)

def test_2d_contingency_table_monte_carlo_estimate(ct,dummy_monte_carlo_2x3_config,f_sparse):

    np.random.seed(ct.seed)

    # Import true contingency table
    true_ct = ContingencyTable2D(dummy_monte_carlo_2x3_config)

    # Monte carlo sample
    mc = ct.table_monte_carlo_sample()

    print(ct)
    print(true_ct.table)

    assert ct.table_admissible(mc) and ct.table_nonnegative(mc)
    assert np.array_equal(mc,true_ct.table)

    # Update table
    ct.update(f_to_array(f_sparse))

    # New monte carlo estimate
    mc = ct.table_monte_carlo_sample()

    assert ct.table_admissible(mc) and ct.table_nonnegative(mc)


def test_2d_contingency_table_maximum_entropy_solution(ct,dummy_max_entropy_2x3_table_config,f_sparse):

    np.random.seed(ct.seed)

    # Import true contingency table
    true_ct = ContingencyTable2D(dummy_max_entropy_2x3_table_config)

    # Monte carlo sample
    me = ct.table_maximum_entropy_solution()

    assert ct.table_admissible(me) and ct.table_nonnegative(me)
    assert np.array_equal(me,true_ct.table)

    # Update table
    ct.update(f_to_array(f_sparse))

    # New monte carlo estimate
    me = ct.table_maximum_entropy_solution()

    assert ct.table_admissible(me) and ct.table_nonnegative(me)


def test_2d_contingency_table_iterative_residual_filling_solution(ct,dummy_iterative_res_filling_2x3_table_config,f_sparse):

    np.random.seed(ct.seed)

    # Import true contingency table
    true_ct = ContingencyTable2D(dummy_iterative_res_filling_2x3_table_config)

    # Monte carlo sample
    irf = ct.table_iterative_residual_filling_solution()

    assert ct.table_admissible(irf) and ct.table_nonnegative(irf)
    assert np.array_equal(irf,true_ct.table)

    # Update table
    ct.update(f_to_array(f_sparse))

    # New monte carlo estimate
    irf = ct.table_iterative_residual_filling_solution()

    assert ct.table_admissible(irf) and ct.table_nonnegative(irf)


def test_2d_contingency_table_iterative_uniform_residual_filling_solution(ct,dummy_iterative_uniform_res_filling_2x3_table_config,f_sparse):

    np.random.seed(ct.seed)

    # Import true contingency table
    true_ct = ContingencyTable2D(dummy_iterative_uniform_res_filling_2x3_table_config)

    # Monte carlo sample
    imrf = ct.table_iterative_uniform_residual_filling_solution()

    assert ct.table_admissible(imrf) and ct.table_nonnegative(imrf)
    assert np.array_equal(imrf,true_ct.table)

    # Update table
    ct.update(f_to_array(f_sparse))

    # New monte carlo estimate
    imrf = ct.table_iterative_uniform_residual_filling_solution()

    assert ct.table_admissible(imrf) and ct.table_nonnegative(imrf)

def test_logsumexp():
    # Compute log lambdas
    log_lambdas = np.array([[ 1.77625623, -2.39041044, -7.25022428],[-6.06175109, -1.89508443, 1.57843506]])

    I,J = np.shape(log_lambdas)

    print(log_lambdas)

    lambdas = np.exp(log_lambdas)
    my_num = np.sum(lambdas,axis=0)
    my_denum = np.sum(lambdas)
    my_lse = my_num/my_denum

    num = np.zeros(J)
    for j in range(J):
        num[j] = logsumexp(np.array(log_lambdas[:,j]))
    den = logsumexp(log_lambdas)
    lse = np.exp(num-den)

    assert abs(np.exp(den)-11)<=1e-8
    assert len(my_lse) == len(lse)
    assert all([abs(a-b)<=1e-9 for a, b in zip(my_lse, lse)])
