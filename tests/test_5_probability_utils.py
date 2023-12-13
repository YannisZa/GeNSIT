# from pprint import pprint
import numpy as np
import os
import json
import torch
import pytest
import pandas as pd
from copy import deepcopy
from argparse import Namespace
from multiresticodm.config import Config
from multiresticodm.contingency_table import ContingencyTable2D
from multiresticodm.spatial_interaction_model import ProductionConstrainedSIM,TotallyConstrainedSIM
from multiresticodm.utils.probability_utils import *
from tests.test_4_spatial_interaction_model import Test4Helpers

class Test5Helpers:

    def pytorch_log_poisson_unnormalised_pmf(intensity_tor,table_tor):
        # Compute log pmf
        return -torch.sum(intensity_tor) + torch.sum(table_tor*torch.log(intensity_tor)) - torch.sum(torch.lgamma(table_tor+1))

    def pytorch_log_poisson_pmf_totally_constrained_sim(xx_tor,theta_tor,costmat_tor,table_tor):
        intensity_tor = Test4Helpers.pytorch_totally_constrained_intensity(xx_tor,theta_tor,costmat_tor,torch.sum(table_tor))
        return Test5Helpers.pytorch_log_poisson_unnormalised_pmf(intensity_tor,table_tor)
    
    def pytorch_log_multinomial_unnormalised_pmf(intensity_tor,table_tor):
        # Compute log pmf
        return torch.sum(
                torch.mul(
                    table_tor,
                    (torch.log(intensity_tor) - \
                    #  torch.log(torch.sum(table_tor)))
                    torch.log(torch.sum(intensity_tor)))
                )
            ) - torch.sum(torch.lgamma(table_tor+1))
    
    def pytorch_log_multinomial_pmf_totally_constrained_sim(xx_tor,theta_tor,costmat_tor,table_tor):
        intensity_tor = Test4Helpers.pytorch_totally_constrained_intensity(xx_tor,theta_tor,costmat_tor,torch.sum(table_tor))
        return Test5Helpers.pytorch_log_multinomial_unnormalised_pmf(intensity_tor,table_tor)
    
    def log_multinomial_pmf_totally_constrained_sim_jacobian(totally_constrained_sim,xx,theta,table):
        log_intensity = totally_constrained_sim.log_intensity(xx,theta,np.sum(table))
        intensity = np.exp(log_intensity)
        # Compute likelihood derivative wrt to log intensity
        my_log_pmf_jacobian = table/intensity - np.sum(table)/np.sum(intensity)
        intensity_grad = Test4Helpers.totally_constrained_intensity_gradient(totally_constrained_sim,xx,theta,table)
        nrows,ncols = np.shape(log_intensity)
        res = np.zeros(ncols)
        
        for i in range(nrows):
            for j in range(ncols):
                for l in range(ncols):
                    res[l] += my_log_pmf_jacobian[i,j] * intensity_grad[i,j,l]
        return res

    def log_multinomial_pmf_totally_constrained_sim_jacobian2(totally_constrained_sim,xx,theta,table):
        log_intensity = totally_constrained_sim.log_intensity(xx,theta,np.sum(table))
        nrows,ncols = np.shape(log_intensity)
        log_table = np.log(table)
        log_alpha = np.log(theta[0])
        log_intensity_rowsums = np.array([logsumexp(log_intensity[:,j]) for j in range(ncols)])
        log_total_intensity = logsumexp(log_intensity.ravel())
        log_total_table = logsumexp(log_table.ravel())
        res = np.zeros(ncols)
        
        for i in range(nrows):
            for j in range(ncols):
                for l in range(ncols):
                    if l != j:
                        res[l] -= np.exp(log_alpha + log_table[i,j] + log_intensity_rowsums[l] - log_total_intensity)
                        res[l] += np.exp(log_alpha + log_intensity[i,j] + log_intensity_rowsums[l] + log_total_table - 2*log_total_intensity)
                    else:
                        res[l] += np.exp(log_alpha + log_table[i,j])
                        res[l] -= np.exp(log_alpha + log_total_table - log_total_intensity + log_intensity[i,j])
                        res[l] -= np.exp(log_alpha + log_table[i,j] + log_intensity_rowsums[j] - log_total_intensity)
                        res[l] += np.exp(log_alpha + log_intensity[i,j] + log_intensity_rowsums[j] + log_total_table - 2*log_total_intensity)
        return res
    
    def pytorch_log_product_multinomial_unnormalised_pmf(intensity_tor,table_tor):
        # Compute log pmf
        log_colsums = torch.unsqueeze(torch.log((torch.sum(intensity_tor,dim=1)+1e-8)), dim=1)
        log_probabilities = torch.log(intensity_tor) - log_colsums
        return torch.sum(torch.mul(table_tor,log_probabilities)) - torch.sum(torch.lgamma(table_tor+1))

    def pytorch_log_product_multinomial_pmf_totally_constrained_sim(xx_tor,theta_tor,costmat_tor,table_tor):
        intensity_tor = Test4Helpers.pytorch_totally_constrained_intensity(xx_tor,theta_tor,costmat_tor,torch.sum(table_tor))
        return Test5Helpers.pytorch_log_product_multinomial_unnormalised_pmf(intensity_tor,table_tor)

    def pytorch_odds_ratio(intensity_tor):
        # Compute margins
        intensity_rowsums = torch.unsqueeze(torch.sum(intensity_tor,dim=1),dim=1)
        intensity_colsums = torch.unsqueeze(torch.sum(intensity_tor,dim=0),dim=0)
        intensity_total = torch.sum(intensity_rowsums)
        # Compute odds ratio and margins
        odds_ratio = torch.div(
                            torch.mul(intensity_tor,intensity_total),
                            torch.mul(intensity_rowsums,intensity_colsums)
                    )
        return odds_ratio
    
    def pytorch_log_fishers_hypergeometric_unnormalised_pmf(intensity_tor,table_tor):
        # Compute log odds ratio
        odds_ratio_tor = Test5Helpers.pytorch_odds_ratio(intensity_tor)
        # Compute log odds ratio margins
        odds_ratio_colsums_tor = torch.unsqueeze(torch.sum(odds_ratio_tor,dim=0),dim=0)
        log_odds_ratio_probabilities = torch.log(odds_ratio_tor) - torch.log(odds_ratio_colsums_tor)
        return torch.sum(torch.mul(table_tor,log_odds_ratio_probabilities)) - torch.sum(torch.lgamma(table_tor+1))

    def pytorch_log_fishers_hypergeometric_pmf_totally_constrained_sim(xx_tor,theta_tor,costmat_tor,table_tor):
        intensity_tor = Test4Helpers.pytorch_totally_constrained_intensity(xx_tor,theta_tor,costmat_tor,torch.sum(table_tor))
        return Test5Helpers.pytorch_log_fishers_hypergeometric_unnormalised_pmf(intensity_tor,table_tor)

@pytest.fixture
def cost_matrix_2x3():
    cm = np.loadtxt("tests/test_fixtures/cost_matrix_2x3.txt")
    return cm

@pytest.fixture
def log_destination_attraction_2x3():
    lda = np.loadtxt("tests/test_fixtures/log_destination_attraction_2x3.txt")
    return lda

@pytest.fixture
def origin_demand_2x3():
    od = np.loadtxt("tests/test_fixtures/origin_demand_2x3.txt")
    return od

@pytest.fixture
def table_2x3_n100():
    tab = np.loadtxt("tests/test_fixtures/table_2x3_n100.txt")
    return tab

@pytest.fixture
def sim_default_config(cost_matrix_2x3,log_destination_attraction_2x3,origin_demand_2x3):
    # Import config
    sim_default_config = {
        "name":"TotallyConstrained",
        "cost_matrix":cost_matrix_2x3,
        "log_destination_attraction":log_destination_attraction_2x3,
        "origin_demand":origin_demand_2x3
    }
    return sim_default_config

@pytest.fixture
def ct_default(table_2x3_n100):
    # Import config
    kwargs = {"constraints":{"axes":[[0,1]]}}
    ct_default = ContingencyTable2D(table=table_2x3_n100,**kwargs)
    return ct_default

@pytest.fixture
def production_constrained_sim(sim_default_config):
    # Build a contingency table
    sim_default_config['table_distribution_name'] = "product_multinomial"
    sim = ProductionConstrainedSIM(sim_default_config)
    return sim

@pytest.fixture
def totally_constrained_sim(sim_default_config):
    # Build a contingency table
    sim_default_config['table_distribution_name'] = "multinomial"
    sim = TotallyConstrainedSIM(sim_default_config)
    return sim

@pytest.fixture
def totally_constrained_log_intensity(totally_constrained_sim):
    theta = np.array([0.8,0.5,0.1,10000,1.1,1.0])
    return totally_constrained_sim.log_intensity(totally_constrained_sim.log_destination_attraction,theta,100)

@pytest.fixture
def test5_helpers():
    return Test5Helpers


def test_odds_ratio(totally_constrained_log_intensity,test5_helpers):

    # Get dimensions
    nrows,ncols = np.shape(totally_constrained_log_intensity)

    # Convert object to torch tensors
    intensity_tor = torch.tensor(np.exp(totally_constrained_log_intensity),requires_grad=True)

    # My gradient
    my_log_or = log_odds_ratio_wrt_intensity(totally_constrained_log_intensity)

    # Torch gradient
    or_tor = test5_helpers.pytorch_odds_ratio(intensity_tor)
    or_tor = or_tor.detach().cpu().numpy()
    log_or_tor = np.log(or_tor)
    
    print('my version')
    print(my_log_or.shape)
    print('\n')
    print('torch')
    print(log_or_tor.shape)

    for i in range(nrows):
        for j in range(ncols):
            assert abs(my_log_or[i,j]-log_or_tor[i,j]) <= 1e-6



def test_poisson_pmf(ct_default,totally_constrained_log_intensity,test5_helpers):
    # Convert object to torch tensors
    intensity_tor = torch.tensor(np.exp(totally_constrained_log_intensity),requires_grad=True)
    table_tor = torch.tensor(ct_default.table,requires_grad=False)

    # Normalised pmf
    # Compute log pmf using my method
    my_log_pmf = log_poisson_pmf_unnormalised(totally_constrained_log_intensity.astype('float32'),ct_default.table.astype('int32'))
    
    # Compute using torch
    ground_truth_log_pmf = test5_helpers.pytorch_log_poisson_unnormalised_pmf(intensity_tor,table_tor).detach().cpu().numpy()

    assert abs(my_log_pmf-ground_truth_log_pmf) <= 1e-4

def test_poisson_pmf_jacobian(ct_default,totally_constrained_log_intensity,test5_helpers):
    # Dimensions
    nrows,ncols = np.shape(ct_default.table)
    # Convert object to torch tensors
    intensity_tor = torch.tensor(np.exp(totally_constrained_log_intensity),requires_grad=True).float()
    table_tor = torch.tensor(ct_default.table,requires_grad=False).float()
    # Compute likelihood derivative wrt to log intensity using my method
    my_log_pmf_jacobian = log_poisson_pmf_jacobian_wrt_intensity(totally_constrained_log_intensity,ct_default.table)

    # Compute likelihood derivative wrt to log intensity using torch
    torch_intensity_grad,_ = torch.autograd.functional.jacobian(test5_helpers.pytorch_log_poisson_unnormalised_pmf, (intensity_tor,table_tor))
    torch_log_pmf_intensity_grad = torch_intensity_grad.detach().cpu().numpy()

    print('my version')
    print(my_log_pmf_jacobian)
    print('torch version')
    print(torch_log_pmf_intensity_grad)

    for i in range(nrows):
        for j in range(ncols):
            assert abs(my_log_pmf_jacobian[i,j]-torch_log_pmf_intensity_grad[i,j]) <= 1e-5

def test_poisson_pmf_jacobian_wrt_xx(test5_helpers,totally_constrained_sim,ct_default):
    # Dimensions
    nrows,ncols = np.shape(ct_default.table)
    theta = np.array([0.8,0.5])
    xx = totally_constrained_sim.log_destination_attraction
    xx_tor = torch.tensor(xx, requires_grad=True).float()
    theta_tor = torch.tensor(theta, requires_grad=False).float()
    costmat_tor = torch.tensor(totally_constrained_sim.cost_matrix,requires_grad=False).float()
    table_tor = torch.tensor(ct_default.table,requires_grad=False).float()

    # Compute the Jacobian of z with respect to x
    true_jacobian_tor,_,_,_ = torch.autograd.functional.jacobian(
                                    test5_helpers.pytorch_log_poisson_pmf_totally_constrained_sim,
                                    (xx_tor,theta_tor,costmat_tor,table_tor)
                            )
    true_jacobian = true_jacobian_tor.detach().cpu().numpy()

    # Compute my gradient
    log_intensity = totally_constrained_sim.log_flow_matrix(
        totally_constrained_sim.log_destination_attraction,
        theta,
        totally_constrained_sim.origin_demand,
        totally_constrained_sim.cost_matrix,
        np.sum(ct_default.table)
    )
    intensity_gradient_wrt_x = totally_constrained_sim.flow_matrix_jacobian(theta,log_intensity)
    likelihood_gradient_wrt_intensity = log_poisson_pmf_jacobian_wrt_intensity(log_intensity,ct_default.table)
    likelihood_gradient_wrt_x = log_table_likelihood_total_derivative_wrt_x(likelihood_gradient_wrt_intensity,intensity_gradient_wrt_x)

    
    print(likelihood_gradient_wrt_x)
    print('\n')
    print(true_jacobian)
    
    for j in range(ncols):
        assert abs(likelihood_gradient_wrt_x[j]-true_jacobian[j]) <= 1e-5


def test_multinomial_pmf(ct_default,totally_constrained_log_intensity,test5_helpers):
    # Dimensions
    nrows,ncols = np.shape(ct_default.table)

    intensity_tor = torch.tensor(np.exp(totally_constrained_log_intensity),requires_grad=True)
    table_tor = torch.tensor(ct_default.table,requires_grad=False)

    # Normalised pmf
    # Compute log pmf using my method
    my_log_pmf = log_multinomial_pmf_unnormalised(totally_constrained_log_intensity.astype('float32'),ct_default.table.astype('int32'))
    
    # Compute using torch
    ground_truth_log_pmf = test5_helpers.pytorch_log_multinomial_unnormalised_pmf(intensity_tor,table_tor).detach().cpu().numpy()

    assert abs(my_log_pmf-ground_truth_log_pmf) <= 1e-4

def test_multinomial_pmf_jacobian(ct_default,totally_constrained_log_intensity,test5_helpers):
    # Dimensions
    nrows,ncols = np.shape(ct_default.table)

    intensity_tor = torch.tensor(np.exp(totally_constrained_log_intensity),requires_grad=True).float()
    table_tor = torch.tensor(ct_default.table,requires_grad=False).float()
    
    # Compute likelihood derivative wrt to log intensity using my method
    my_log_pmf_jacobian = log_multinomial_pmf_jacobian_wrt_intensity(totally_constrained_log_intensity,ct_default.table)

    # Compute likelihood derivative wrt to log intensity using torch
    torch_intensity_grad,_ = torch.autograd.functional.jacobian(test5_helpers.pytorch_log_multinomial_unnormalised_pmf, (intensity_tor,table_tor))
    torch_log_pmf_intensity_grad = torch_intensity_grad.detach().cpu().numpy()

    print('my version')
    print(my_log_pmf_jacobian)
    print('torch version')
    print(torch_log_pmf_intensity_grad)

    for i in range(nrows):
        for j in range(ncols):
            assert abs(my_log_pmf_jacobian[i,j]-torch_log_pmf_intensity_grad[i,j]) <= 1e-6

def test_multinomial_pmf_jacobian_wrt_xx(test5_helpers,totally_constrained_sim,ct_default):
    # Dimensions
    _,ncols = np.shape(ct_default.table)
    theta = np.array([0.8,0.5])
    xx = totally_constrained_sim.log_destination_attraction
    xx_tor = torch.tensor(xx, requires_grad=True).float()
    theta_tor = torch.tensor(theta, requires_grad=False).float()
    costmat_tor = torch.tensor(totally_constrained_sim.cost_matrix,requires_grad=False).float()
    table_tor = torch.tensor(ct_default.table,requires_grad=False).float()

    # Compute the Jacobian of z with respect to x
    true_jacobian_tor,_,_,_ = torch.autograd.functional.jacobian(
                                    test5_helpers.pytorch_log_multinomial_pmf_totally_constrained_sim,
                                    (xx_tor,theta_tor,costmat_tor,table_tor)
                            )
    true_jacobian = true_jacobian_tor.detach().cpu().numpy()

    # Compute jacobian manually by deriving maths
    manual_jacobian = test5_helpers.log_multinomial_pmf_totally_constrained_sim_jacobian(totally_constrained_sim,xx,theta,ct_default.table)
    manual_jacobian2 = test5_helpers.log_multinomial_pmf_totally_constrained_sim_jacobian2(totally_constrained_sim,xx,theta,ct_default.table)

    # Compute my gradient
    log_intensity = totally_constrained_sim.log_flow_matrix(
        totally_constrained_sim.log_destination_attraction,
        theta,
        totally_constrained_sim.origin_demand,
        totally_constrained_sim.cost_matrix,
        np.sum(ct_default.table)
    )
    intensity_gradient_wrt_x = totally_constrained_sim.flow_matrix_jacobian(theta,log_intensity)
    likelihood_gradient_wrt_intensity = log_multinomial_pmf_jacobian_wrt_intensity(log_intensity,ct_default.table)
    likelihood_gradient_wrt_x = log_table_likelihood_total_derivative_wrt_x(likelihood_gradient_wrt_intensity,intensity_gradient_wrt_x)
    
    print(likelihood_gradient_wrt_x)
    print('\n')
    print(manual_jacobian)
    print('\n')
    print(manual_jacobian2)
    print('\n')
    print(true_jacobian)
    
    for j in range(ncols):
        assert abs(likelihood_gradient_wrt_x[j]-true_jacobian[j]) <= 1e-5
        assert abs(manual_jacobian[j]-true_jacobian[j]) <= 1e-5
        assert abs(manual_jacobian2[j]-true_jacobian[j]) <= 1e-5


def test_product_multinomial_pmf(ct_default,totally_constrained_log_intensity,test5_helpers):
    # Convert object to torch tensors
    intensity_tor = torch.tensor(np.exp(totally_constrained_log_intensity),requires_grad=True)
    table_tor = torch.tensor(ct_default.table,requires_grad=False)

    # Normalised pmf
    # Compute log pmf using my method
    my_log_pmf = log_product_multinomial_pmf_unnormalised(totally_constrained_log_intensity.astype('float32'),ct_default.table.astype('int32'))
    
    # Compute using torch
    ground_truth_log_pmf = test5_helpers.pytorch_log_product_multinomial_unnormalised_pmf(intensity_tor,table_tor).detach().cpu().numpy()

    assert abs(my_log_pmf-ground_truth_log_pmf) <= 1e-5

def test_product_multinomial_pmf_jacobian(ct_default,totally_constrained_log_intensity,test5_helpers):
    # Dimensions
    nrows,ncols = np.shape(ct_default.table)
    # Convert object to torch tensors
    intensity_tor = torch.tensor(np.exp(totally_constrained_log_intensity),requires_grad=True).float()
    table_tor = torch.tensor(ct_default.table,requires_grad=False).float()
    
    # Compute likelihood derivative wrt to log intensity using my method
    my_log_pmf_jacobian = log_product_multinomial_pmf_jacobian_wrt_intensity(totally_constrained_log_intensity,ct_default.table)

    # Compute likelihood derivative wrt to log intensity using torch
    torch_intensity_grad,_ = torch.autograd.functional.jacobian(test5_helpers.pytorch_log_product_multinomial_unnormalised_pmf, (intensity_tor,table_tor))
    torch_log_pmf_intensity_grad = torch_intensity_grad.detach().cpu().numpy()

    print('my version')
    print(my_log_pmf_jacobian)
    print('torch version')
    print(torch_log_pmf_intensity_grad)

    for i in range(nrows):
        for j in range(ncols):
            assert abs(my_log_pmf_jacobian[i,j]-torch_log_pmf_intensity_grad[i,j]) <= 1e-6

def test_product_multinomial_pmf_jacobian_wrt_xx(test5_helpers,totally_constrained_sim,ct_default):
    # Dimensions
    nrows,ncols = np.shape(ct_default.table)
    theta = np.array([0.8,0.5])
    xx = totally_constrained_sim.log_destination_attraction
    xx_tor = torch.tensor(xx, requires_grad=True).float()
    theta_tor = torch.tensor(theta, requires_grad=False).float()
    costmat_tor = torch.tensor(totally_constrained_sim.cost_matrix,requires_grad=False).float()
    table_tor = torch.tensor(ct_default.table,requires_grad=False).float()

    # Compute the Jacobian of z with respect to x
    true_jacobian_tor,_,_,_ = torch.autograd.functional.jacobian(
                                    test5_helpers.pytorch_log_product_multinomial_pmf_totally_constrained_sim,
                                    (xx_tor,theta_tor,costmat_tor,table_tor)
                            )
    true_jacobian = true_jacobian_tor.detach().cpu().numpy()

    # Compute my gradient
    log_intensity = totally_constrained_sim.log_flow_matrix(
        totally_constrained_sim.log_destination_attraction,
        theta,
        totally_constrained_sim.origin_demand,
        totally_constrained_sim.cost_matrix,
        np.sum(ct_default.table)
    )
    intensity_gradient_wrt_x = totally_constrained_sim.flow_matrix_jacobian(theta,log_intensity)
    likelihood_gradient_wrt_intensity = log_product_multinomial_pmf_jacobian_wrt_intensity(log_intensity,ct_default.table)
    likelihood_gradient_wrt_x = log_table_likelihood_total_derivative_wrt_x(likelihood_gradient_wrt_intensity,intensity_gradient_wrt_x)

    
    print(likelihood_gradient_wrt_x)
    print('\n')
    print(true_jacobian)
    
    for j in range(ncols):
        assert abs(likelihood_gradient_wrt_x[j]-true_jacobian[j]) <= 1e-5


def test_fishers_hypergeometric_pmf(ct_default,totally_constrained_log_intensity,test5_helpers):
    # Convert object to torch tensors
    intensity_tor = torch.tensor(np.exp(totally_constrained_log_intensity),requires_grad=True)
    table_tor = torch.tensor(ct_default.table,requires_grad=False)

    # Normalised pmf
    # Compute log pmf using my method
    my_log_pmf = log_fishers_hypergeometric_pmf_unnormalised(totally_constrained_log_intensity.astype('float32'),ct_default.table.astype('int32'))
    
    # Compute using torch
    ground_truth_log_pmf = test5_helpers.pytorch_log_fishers_hypergeometric_unnormalised_pmf(intensity_tor,table_tor).detach().cpu().numpy()

    assert abs(my_log_pmf-ground_truth_log_pmf) <= 1e-5

def test_fishers_hypergeometric_pmf_jacobian_wrt_intensity(ct_default,totally_constrained_log_intensity,test5_helpers):
    # Dimensions
    nrows,ncols = np.shape(ct_default.table)
    # Intensity function
    intensity = np.exp(totally_constrained_log_intensity)
    # Convert object to torch tensors
    intensity_tor = torch.tensor(intensity,requires_grad=True).float()
    table_tor = torch.tensor(ct_default.table,requires_grad=False).float()

    # Compute likelihood derivative wrt to log intensity using my method
    my_log_pmf_jacobian = log_fishers_hypergeometric_pmf_jacobian_wrt_intensity(totally_constrained_log_intensity,ct_default.table)

    # Compute likelihood derivative wrt to log intensity using torch
    torch_intensity_grad,_ = torch.autograd.functional.jacobian(test5_helpers.pytorch_log_fishers_hypergeometric_unnormalised_pmf, (intensity_tor,table_tor))
    torch_log_pmf_intensity_grad = torch_intensity_grad.detach().cpu().numpy()

    print('my version')
    print(my_log_pmf_jacobian)
    print('torch version')
    print(torch_log_pmf_intensity_grad)

    for i in range(nrows):
        for j in range(ncols):
            assert abs(my_log_pmf_jacobian[i,j]-torch_log_pmf_intensity_grad[i,j]) <= 1e-4


def test_fishers_hypergeometric_pmf_jacobian_wrt_xx(test5_helpers,totally_constrained_sim,ct_default):
    # Dimensions
    nrows,ncols = np.shape(ct_default.table)
    theta = np.array([0.8,0.5])
    xx = totally_constrained_sim.log_destination_attraction
    xx_tor = torch.tensor(xx, requires_grad=True).float()
    theta_tor = torch.tensor(theta, requires_grad=False).float()
    costmat_tor = torch.tensor(totally_constrained_sim.cost_matrix,requires_grad=False).float()
    table_tor = torch.tensor(ct_default.table,requires_grad=False).float()

    # Compute the Jacobian of z with respect to x
    true_jacobian_tor,_,_,_ = torch.autograd.functional.jacobian(
                                    test5_helpers.pytorch_log_fishers_hypergeometric_pmf_totally_constrained_sim,
                                    (xx_tor,theta_tor,costmat_tor,table_tor)
                            )
    true_jacobian = true_jacobian_tor.detach().cpu().numpy()

    # Compute my gradient
    log_intensity = totally_constrained_sim.log_flow_matrix(
        totally_constrained_sim.log_destination_attraction,
        theta,
        totally_constrained_sim.origin_demand,
        totally_constrained_sim.cost_matrix,
        np.sum(ct_default.table)
    )
    likelihood_gradient_wrt_intensity = log_fishers_hypergeometric_pmf_jacobian_wrt_intensity(log_intensity,ct_default.table)
    intensity_gradient_wrt_x = totally_constrained_sim.flow_matrix_jacobian(theta,log_intensity)
    print('likelihood_gradient_wrt_intensity',likelihood_gradient_wrt_intensity.shape,likelihood_gradient_wrt_intensity.dtype)
    print('intensity_gradient_wrt_x',intensity_gradient_wrt_x.shape,intensity_gradient_wrt_x.dtype)
    likelihood_gradient_wrt_x = log_table_likelihood_total_derivative_wrt_x(
        likelihood_gradient_wrt_intensity.astype('float32'),
        intensity_gradient_wrt_x.astype('float32')
    )

    
    print(likelihood_gradient_wrt_x)
    print('\n')
    print(true_jacobian)
    
    for j in range(ncols):
        assert abs(likelihood_gradient_wrt_x[j]-true_jacobian[j]) <= 1e-6