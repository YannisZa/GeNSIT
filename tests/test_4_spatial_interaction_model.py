import torch
import pytest
import numpy as np
import pandas as pd

# from pprint import pprint

from multiresticodm.config import Config
from multiresticodm.spatial_interaction_model import instantiate_sim

class Test4Helpers:

    @staticmethod
    def pytorch_totally_constrained_intensity(xx_tor,theta_tor,cost_matrix_tor,table_total_tor):
        # Compute log pmf
        wksp = theta_tor[0] * xx_tor - theta_tor[1] * cost_matrix_tor
        log_norm = torch.logsumexp(wksp,dim=tuple(list(range(2))))
        flows = torch.exp(wksp - log_norm) * table_total_tor
        return flows
    
    @staticmethod
    def pytorch_totally_constrained_sde_potential(theta_tor,xx_tor,origin_demand_tor,cost_mat_tor):
        # Get parameters
        alpha = theta_tor[0]
        beta = theta_tor[1]
        delta = theta_tor[2]
        gamma = theta_tor[3]
        kappa = theta_tor[4]
        epsilon = theta_tor[5]
        log_total_tor = torch.log(torch.sum(origin_demand_tor))

        # Compute log unnormalised expected flow
        log_utility = alpha*xx_tor - beta*cost_mat_tor
        # Compute log normalisation factor
        log_normalisation = torch.logsumexp(log_utility,dim=tuple(list(range(2))))
        # Compute potential
        if alpha == 0:
            return -np.infty
        else:
            return gamma*epsilon*(-(1./alpha)*torch.exp(log_total_tor)*log_normalisation + kappa*torch.sum(torch.exp(xx_tor)) - delta*torch.sum(xx_tor))
        
    @staticmethod
    def totally_constrained_intensity_gradient(totally_constrained_sim,xx,theta,table):
        log_intensity = totally_constrained_sim.log_intensity(xx,theta,table.sum())
        intensity = np.exp(log_intensity)
        ncols = len(xx)
        intensity_total = np.sum(intensity)
        intensity_colsum_probs = np.array([np.sum(intensity[:,j])/intensity_total for j in range(ncols)])
        nrows,ncols = np.shape(log_intensity)
        res = np.zeros((nrows,ncols,ncols))
        alpha = theta[0]
        for i in range(nrows):
            for j in range(ncols):
                for l in range(ncols):
                    if l == j:
                        res[i,j,l] += alpha * intensity[i,j] * (1 - intensity_colsum_probs[j])
                    else:
                        res[i,j,l] -= alpha * intensity[i,j] * intensity_colsum_probs[l]
        return res

    @staticmethod
    def pytorch_production_constrained_intensity(xx_tor,theta_tor,origin_demand_tor,cost_matrix_tor,table_total_tor):
        nrows,ncols = cost_matrix_tor.size()
        wksp = theta_tor[0] * xx_tor - theta_tor[1] * cost_matrix_tor
        log_norms = torch.logsumexp(wksp,dim=1,keepdim=True)
        origin_demand_tor = torch.reshape(origin_demand_tor,((nrows,1)))
        flows = torch.mul(torch.mul(origin_demand_tor,torch.exp(wksp - log_norms)),table_total_tor)
        return flows
    
    @staticmethod
    def pytorch_production_constrained_sde_potential(theta_tor,xx_tor,origin_demand_tor,cost_mat_tor):
        # Get parameters
        alpha = theta_tor[0]
        beta = theta_tor[1]
        delta = theta_tor[2]
        gamma = theta_tor[3]
        kappa = theta_tor[4]
        epsilon = theta_tor[5]
        log_total_tor = torch.tensor(0,requires_grad=False)

        # Compute log unnormalised expected flow
        log_utility = alpha*xx_tor - beta*cost_mat_tor
        # Compute log normalisation factor
        log_norms = torch.logsumexp(log_utility,dim=1,keepdim=False)
        # Compute utility potential
        utility_potential = torch.dot(
            origin_demand_tor,
            log_norms
        )
        # Compute potential
        if alpha == 0:
            return -np.infty
        else:
            return gamma*epsilon*(
                (-1./alpha)*utility_potential + \
                kappa*torch.sum(torch.exp(xx_tor)) - \
                delta*torch.sum(xx_tor)
            )
        
    @staticmethod
    def pytorch_y_data_likelihood(xx_tor,yy_tor,noise_var_tor):
        return 0.5*noise_var_tor*torch.dot((xx_tor-yy_tor),(xx_tor-yy_tor))
    
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
        "cost_matrix":cost_matrix_2x3,
        "log_destination_attraction":log_destination_attraction_2x3,
        "origin_demand":origin_demand_2x3
    }
    return sim_default_config

@pytest.fixture
def sim_default_config2():
    # Import config
    sim_default_config2 = Config("tests/test_configs/test_0_default_config3.toml")

    return sim_default_config2

@pytest.fixture
def production_constrained_sim(sim_default_config):
    # Build a contingency table
    sim_default_config['table_distribution_name'] = "product_multinomial"
    sim_default_config['sim_type'] = "ProductionConstrained"
    sim = instantiate_sim(sim_default_config)
    return sim

@pytest.fixture
def totally_constrained_sim(sim_default_config):
    # Build a contingency table
    sim_default_config['table_distribution_name'] = "multinomial"
    sim_default_config['sim_type'] = "TotallyConstrained"
    sim = instantiate_sim(sim_default_config)
    sim.noise_var = (0.03*np.log(sim.dims[1]))**2
    return sim

@pytest.fixture
def test4_helpers():
    return Test4Helpers


def test_totally_constrained_log_expected_flows(totally_constrained_sim):
    # Create log destination attraction
    # Create theta
    print(totally_constrained_sim.cost_matrix)
    theta = np.array([0.8,0.5,0.1,10000,1.1,1.0])
    log_lambdas = totally_constrained_sim.log_intensity(totally_constrained_sim.log_destination_attraction,theta,100)
    print(pd.DataFrame(log_lambdas))
    true_log_lambdas = np.array([[3.36122466, 2.17502496, 2.57815309],[3.32551037, 2.21073925, 2.50672452]])
    print(pd.DataFrame(true_log_lambdas))
    print(totally_constrained_sim.dims)
    nrows,ncols = totally_constrained_sim.dims
    for i in range(nrows):
        for j in range(ncols):
            assert abs(log_lambdas[i,j]-true_log_lambdas[i,j]) <= 1e-5

def test_totally_constrained_log_intensity_jacobian(test4_helpers,totally_constrained_sim,table_2x3_n100):
    # Dimensions
    nrows,ncols = np.shape(totally_constrained_sim.cost_matrix)
    total = table_2x3_n100.sum()
    theta = np.array([0.8,0.5])
    xx = totally_constrained_sim.log_destination_attraction
    xx_tor = torch.tensor(xx, requires_grad=True)
    theta_tor = torch.tensor(theta, requires_grad=False)
    costmat_tor = torch.tensor(totally_constrained_sim.cost_matrix,requires_grad=False)
    table_total_tor = torch.tensor(total,requires_grad=False)
    
    torch_grad,_,_,_ = torch.autograd.functional.jacobian(test4_helpers.pytorch_totally_constrained_intensity, 
                                                    (xx_tor,theta_tor,costmat_tor,table_total_tor))
    torch_intensity_grad = torch_grad.detach().cpu().numpy()

    # Manually computed gradient
    manual_intensity_grad = test4_helpers.totally_constrained_intensity_gradient(totally_constrained_sim,xx,theta,table_2x3_n100)

    # My gradient
    log_intensity = totally_constrained_sim.log_intensity(
        totally_constrained_sim.log_destination_attraction,
        theta,
        total
    )
    my_intensity_grad = totally_constrained_sim.intensity_gradient(theta,log_intensity)
    
    for i in range(nrows):
        for j in range(ncols):
            print('torch',torch_intensity_grad[i])
            print('mine',my_intensity_grad[i])
            for m in range(ncols):
                assert abs(my_intensity_grad[i,j,m]-torch_intensity_grad[i,j,m]) <= 1e-5
                assert abs(manual_intensity_grad[i,j,m]-torch_intensity_grad[i,j,m]) <= 1e-5

def test_totally_constrained_potential_value(test4_helpers,totally_constrained_sim):
    # Create log destination attraction
    # xx = np.array([-0.91629073,-0.51082562])
    # Create theta
    theta = np.array([1.0,0.5*100,0.1,10000,1.1,1.0])
    theta_tor = torch.tensor(theta,requires_grad=False)
    cost_mat_tor = torch.tensor(totally_constrained_sim.cost_matrix,requires_grad=False)
    origin_demand_tor = torch.tensor(totally_constrained_sim.origin_demand,requires_grad=False)
    xx_tor = torch.tensor(totally_constrained_sim.log_destination_attraction,requires_grad=True)

    # Compute gamma*V_{theta}(x)
    pot,gradPot = totally_constrained_sim.sde_potential_and_gradient(totally_constrained_sim.log_destination_attraction,theta)
    pot_torch = test4_helpers.pytorch_totally_constrained_sde_potential(theta_tor,xx_tor,origin_demand_tor,cost_mat_tor)
    true_pot = 52039.74450161611

    assert abs(pot-true_pot) <= 1e-1
    assert abs(pot_torch-true_pot) <= 1e-1

def test_totally_constrained_potential_gradient(test4_helpers,totally_constrained_sim):
    # Create theta
    theta = np.array([1.0,0.5*100,0.1,10000,1.1,1.0])
    theta_tor = torch.tensor(theta,requires_grad=False)
    cost_mat_tor = torch.tensor(totally_constrained_sim.cost_matrix,requires_grad=False)
    origin_demand_tor = torch.tensor(totally_constrained_sim.origin_demand,requires_grad=False)
    xx = totally_constrained_sim.log_destination_attraction
    xx_tor = torch.tensor(xx,requires_grad=True)

    # Compute gamma*V_{theta}(x) using my method
    pot,gradPot = totally_constrained_sim.sde_potential_and_gradient(totally_constrained_sim.log_destination_attraction,theta)

    _,true_grad_tor,_,_ = torch.autograd.functional.jacobian(test4_helpers.pytorch_totally_constrained_sde_potential, 
                                                    (theta_tor,xx_tor,origin_demand_tor,cost_mat_tor))
    print(true_grad_tor)
    true_grad = true_grad_tor.detach().cpu().numpy()

    for j in range(len(gradPot)):
        assert abs(gradPot[j]-true_grad[j]) <= 1e-1


def test_production_constrained_log_expected_flows(production_constrained_sim,test4_helpers,table_2x3_n100):
    # Create log destination attraction
    # Create theta
    theta = np.array([0.8,0.5,0.1,10000,1.1,1.0])
    log_lambdas = production_constrained_sim.log_intensity(
        production_constrained_sim.log_destination_attraction,
        theta,
        100
    )
    # Convert objects to tensors
    theta_tor = torch.tensor(theta,requires_grad=False)
    log_destination_attraction_tor = torch.tensor(production_constrained_sim.log_destination_attraction,requires_grad=True)
    cost_matrix_tor = torch.tensor(production_constrained_sim.cost_matrix,requires_grad=False)
    origin_demand_tor = torch.tensor(production_constrained_sim.origin_demand,requires_grad=False)
    table_total_tor = torch.tensor(table_2x3_n100.sum(),requires_grad=False)

    true_lambdas = test4_helpers.pytorch_production_constrained_intensity(
        log_destination_attraction_tor,
        theta_tor,
        origin_demand_tor,
        cost_matrix_tor,
        table_total_tor
    )
    true_log_lambdas = np.log(true_lambdas.detach().cpu().numpy())

    nrows,ncols = production_constrained_sim.dims
    for i in range(nrows):
        for j in range(ncols):
            assert abs(log_lambdas[i,j]-true_log_lambdas[i,j]) <= 1e-5

def test_production_constrained_log_intensity_jacobian(test4_helpers,production_constrained_sim,table_2x3_n100):
    # Dimensions
    nrows,ncols = np.shape(production_constrained_sim.cost_matrix)
    total = table_2x3_n100.sum()
    theta = np.array([0.8,0.5])
    xx = production_constrained_sim.log_destination_attraction
    xx_tor = torch.tensor(xx, requires_grad=True)
    theta_tor = torch.tensor(theta, requires_grad=False)
    origin_demand_tor = torch.tensor(production_constrained_sim.origin_demand,requires_grad=False)
    costmat_tor = torch.tensor(production_constrained_sim.cost_matrix,requires_grad=False)
    table_total_tor = torch.tensor(total,requires_grad=False)
    
    torch_grad,_,_,_,_ = torch.autograd.functional.jacobian(test4_helpers.pytorch_production_constrained_intensity, 
                                                    (xx_tor,theta_tor,origin_demand_tor,costmat_tor,table_total_tor))
    torch_intensity_grad = torch_grad.detach().cpu().numpy()

    # My gradient
    log_intensity = production_constrained_sim.log_intensity(
        production_constrained_sim.log_destination_attraction,
        theta,
        total
    )
    my_intensity_grad = production_constrained_sim.intensity_gradient(theta,log_intensity)
    
    for i in range(nrows):
        for j in range(ncols):
            print('torch',torch_intensity_grad[i])
            print('mine',my_intensity_grad[i])
            for m in range(ncols):
                assert abs(my_intensity_grad[i,j,m]-torch_intensity_grad[i,j,m]) <= 1e-5

def test_production_constrained_potential_value(test4_helpers,production_constrained_sim):
    # Dimensions
    nrows,ncols = np.shape(production_constrained_sim.cost_matrix)
    theta = np.array([1.0,0.5*100,0.1,10000,1.1,1.0])
    xx = production_constrained_sim.log_destination_attraction
    xx_tor = torch.tensor(xx, requires_grad=True)
    theta_tor = torch.tensor(theta, requires_grad=False)
    origin_demand_tor = torch.tensor(production_constrained_sim.origin_demand,requires_grad=False)
    costmat_tor = torch.tensor(production_constrained_sim.cost_matrix,requires_grad=False)

    # Compute gamma*V_{theta}(x)
    pot,_ = production_constrained_sim.sde_potential_and_gradient(production_constrained_sim.log_destination_attraction,theta)
    pot_torch = test4_helpers.pytorch_production_constrained_sde_potential(theta_tor,xx_tor,origin_demand_tor,costmat_tor)
    pot_torch = pot_torch.detach().cpu().numpy()
    print(pot_torch)

    assert abs(pot-pot_torch) <= 1e-1

def test_production_constrained_potential_gradient(test4_helpers,production_constrained_sim):
    # Dimensions
    nrows,ncols = np.shape(production_constrained_sim.cost_matrix)
    theta = np.array([1.0,0.5*100,0.1,10000,1.1,1.0])
    xx = production_constrained_sim.log_destination_attraction
    xx_tor = torch.tensor(xx, requires_grad=True)
    theta_tor = torch.tensor(theta, requires_grad=False)
    origin_demand_tor = torch.tensor(production_constrained_sim.origin_demand,requires_grad=False)
    costmat_tor = torch.tensor(production_constrained_sim.cost_matrix,requires_grad=False)

    # Compute gamma*V_{theta}(x) using my method
    pot,gradPot = production_constrained_sim.sde_potential_and_gradient(production_constrained_sim.log_destination_attraction,theta)

    _,true_grad_tor,_,_ = torch.autograd.functional.jacobian(test4_helpers.pytorch_production_constrained_sde_potential, 
                                                    (theta_tor,xx_tor,origin_demand_tor,costmat_tor))
    true_grad = true_grad_tor.detach().cpu().numpy()

    for j in range(len(gradPot)):
        assert abs(gradPot[j]-true_grad[j]) <= 1e-1



def test_data_likelihood_and_gradient(test4_helpers,totally_constrained_sim):
    # Create theta
    np.random.seed(1234)
    theta = np.array([1.0,0.5*100,0.1,10000,1.1,1.0])
    yy = totally_constrained_sim.log_destination_attraction
    xx = yy + np.log(np.random.uniform(low=0.1, high=1.0, size=1))
    theta_tor = torch.tensor(theta,requires_grad=False)
    cost_mat_tor = torch.tensor(totally_constrained_sim.cost_matrix,requires_grad=False)
    yy_tor = torch.tensor(yy,requires_grad=False)
    xx_tor = torch.tensor(xx,requires_grad=True)
    noise_var_tor = torch.tensor(totally_constrained_sim.noise_var)
    
    # Compute log(p(y|x,theta))
    negative_log_likelihood, negative_gradLogLikelihood = totally_constrained_sim.negative_destination_attraction_log_likelihood_and_gradient(xx,1./totally_constrained_sim.noise_var)
    
    true_negative_log_likelihood = test4_helpers.pytorch_y_data_likelihood(xx_tor,yy_tor,1./noise_var_tor)

    assert abs(true_negative_log_likelihood - negative_log_likelihood) <= 1e-5
    # assert abs(true_negative_log_likelihood - negative_log_likelihood) <= 1e-5

    true_grad_tor,_,_ = torch.autograd.functional.jacobian(test4_helpers.pytorch_y_data_likelihood, 
                                                    (xx_tor,yy_tor,1./noise_var_tor))
    true_grad = true_grad_tor.detach().cpu().numpy()

    diff = negative_gradLogLikelihood - true_grad
    print(true_grad)
    print(negative_gradLogLikelihood)
    for i in range(len(diff)):
        assert abs(diff[i]) <= 1e-5
