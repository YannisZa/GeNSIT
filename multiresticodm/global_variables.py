import logging
import torch 

from torch import int32, float32, uint8, int8, float64, int64, int16
from torch import bool as tbool

PARAMETER_DEFAULTS = {
    'alpha': 1, 
    'beta': 0, 
    'delta': 0.1,
    'kappa': 2, 
    'epsilon': 1,
    'bmax': 1,
    'noise_percentage': 3,
    'sigma': 3,
}
MCMC_PARAMETERS = ['alpha','beta','delta','gamma','kappa','epsilon']

XARRAY_SCHEMA = {
    'alpha': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["N","1"]
    }, 
    'beta': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["N","1"]
    }, 
    'kappa': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["N","1"]
    }, 
    'delta': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["N","1"]
    }, 
    'sigma': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["N","1"]
    }, 
    'log_destination_attraction': {
        "coords":["time","destination"],
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "new_shape":["N","T","J"]
    },
    'table': {
        "coords":["origin","destination"],
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "new_shape":["N","I","J"]
    },
    'loss': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["N","1"]
    },
    'log_target': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["N","1"]
    },
    'theta_acc': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["N","1"]
    },
    'log_destination_attraction_acc': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["N","1"]
    },
    'table_acc': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["N","1"]
    },
    'computation_time': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["N","1"]
    },
}


NORMS = ['relative_l_0','relative_l_1','relative_l_2','l_0','l_1','l_2']

DISTANCE_FUNCTIONS = [
    'l_p_distance',
    'edit_distance_degree_one',
    'edit_distance_degree_higher',
    'chi_squared_distance',
    'euclidean_distance',
    'chi_squared_column_distance',
    'chi_squared_row_distance'
]

NUMPY_TO_TORCH_DTYPE = {
    'float32':float32,
    'float64':float64,
    'uint8':uint8,
    'int8':int8,
    'int16':int16,
    'int32':int32,
    'int64':int64,
    'bool':tbool,
    'str':str,
    'object':object
}

TORCH_TO_NUMPY_DTYPE = {v:k for k,v in NUMPY_TO_TORCH_DTYPE.items()}

INPUT_SCHEMA = {
    "origin_demand":{"dims":['origin'],"axes":[0],"dtype":"float32", "ndmin":1},
    "destination_demand":{"dims":['destination'],"axes":[1],"dtype":"float32", "ndmin":1},
    "origin_attraction_ts":{"dims":['origin'],"axes":[0],"dtype":"float32", "ndmin":2},
    "destination_attraction_ts":{"dims":['time','destination'],"axes":[0,1],"dtype":"float32", "ndmin":2},
    "log_destination_attraction":{"dims":['destination'],"axes":[0],"dtype":"float32", "ndmin":1},
    "cost_matrix":{"dims":['origin','destination'],"axes":[0,1],"dtype":"float32", "ndmin":2},
    "ground_truth_table":{"dims":['origin','destination'],"axes":[0,1],"dtype":"int32", "ndmin":2},
    "dims":{},
    "grand_total":{}
}

TABLE_INFERENCE_EXPERIMENTS = ['nonjointtablesim_nn','jointtablesim_nn','jointtablesim_mcmc','table_mcmc','table_mcmc','tablesummaries_mcmcconvergence']

INPUT_TYPES = {
    'cost_matrix':torch.float32,
    'origin_demand':torch.float32,
    'destination_demand':torch.float32,
    'origin_attraction_ts':torch.float32,
    'destination_attraction_ts':torch.float32,
    'ground_truth_table':torch.int32
}

TABLE_TYPES = {
    'table':torch.int32,
    'tableerror':torch.float32,
}
    
INTENSITY_TYPES = {
    'intensity':torch.float32,
    'intensityerror':torch.float32,
    'log_destination_attraction':torch.float32,
    'log_origin_attraction':torch.float32,
    'alpha':torch.float32,
    'beta':torch.float32,
    'delta':torch.float32,
    'kappa':torch.float32,
    'sigma':torch.float32,
    'gamma':torch.float32,
}

OUTPUT_TYPES = {
    'loss':torch.float32,
    'log_target':torch.float32,
    'sign':torch.int8,
    **INTENSITY_TYPES,
    **TABLE_TYPES
}

DATA_TYPES = {**INPUT_TYPES,**OUTPUT_TYPES}

AUXILIARY_COORDINATES_DTYPES = {
    'N':torch.int32,
    'dataset':str,
    'covariance':str,
    'step_size':torch.float32
}

CORE_COORDINATES_DTYPES = {
 'iter':torch.int32,
 'seed':torch.int32,
 'time':torch.int32,
 'origin':torch.int32,
 'destination':torch.int32
}

COORDINATES_DTYPES = {**CORE_COORDINATES_DTYPES,**AUXILIARY_COORDINATES_DTYPES,**DATA_TYPES}


NUMPY_TYPE_TO_DAT_TYPE = {
    'float':'%f',
    'int':'start,stop,step',
}

SIM_TYPE_CONSTRAINTS = {
    'TotallyConstrained':'grand_total',
    'ProductionConstrained':'row_margin'
}

TABLE_SOLVERS = ['monte_carlo_sample', 'maximum_entropy_solution',
                 'iterative_residual_filling_solution', 'iterative_uniform_residual_filling_solution']
MARGINAL_SOLVERS = ['multinomial']


METRICS = {
    'SRMSE':{'shape':'(N,1)','loop_over':[],'apply_axis':(0,1),'dtype':'float32'},
    'p_distance':{'shape':'(N,dims)','loop_over':[],'apply_axis':(1,2),'dtype':'float32'},
    'SSI':{'shape':'(N,1)','loop_over':[],'apply_axis':(0,1),'dtype':'float32'},
    'sparsity':{'shape':'(N,1)','loop_over':[],'apply_axis':(0,1),'dtype':'float32'},
    'shannon_entropy':{'shape':'(1,1)','loop_over':[],'apply_axis':(0,1),'dtype':'float32'},
    'von_neumann_entropy':{'shape':'(N,1)','loop_over':[],'apply_axis':(0,1),'dtype':'float32'},
    'coverage_probability':{'shape':'(1,dims)','loop_over':['region_mass'],'apply_axis':(1,2),'dtype':'int32'},
    'edit_degree_higher_error':{'shape':'(N,1)','loop_over':[],'apply_axis':(0,1),'dtype':'float32'},
    'edit_degree_one_error':{'shape':'(N,1)','loop_over':[],'apply_axis':(0,1),'dtype':'float32'},
    'none':{'shape':'(N,dims)','loop_over':[],'apply_axis':(0,1,2),'dtype':''},
}

COLORS = {'monte_carlo_sample_degree_one':'tab:blue',
        'monte_carlo_sample_degree_higher':'tab:blue',
        'iterative_residual_filling_solution_degree_one':'tab:orange',
        'iterative_residual_filling_solution_degree_higher':'tab:orange',
        'maximum_entropy_solution_degree_one':'tab:green',
        'maximum_entropy_solution_degree_higher':'tab:green',
        'iterative_uniform_residual_filling_solution_degree_one':'tab:red',
        'iterative_uniform_residual_filling_solution_degree_higher':'tab:red'}

LINESTYLES = {'monte_carlo_sample_degree_one':'dashed',
            'iterative_residual_filling_solution_degree_one':'dashed',
            'maximum_entropy_solution_degree_one':'dashed',
            'iterative_uniform_residual_filling_solution_degree_one':'dashed',
            'monte_carlo_sample_degree_higher':'solid',
            'iterative_residual_filling_solution_degree_higher':'solid',
            'maximum_entropy_solution_degree_higher':'solid',
            'iterative_uniform_residual_filling_solution_degree_higher':'solid'}


# Type of plots
PLOT_HASHMAP = {
        '00':'table_posterior_mean_convergence_fixed_intensity',
        '01':'colsum_posterior_mean_convergence_fixed_intensity',
        '02':'table_posterior_mean_convergence',
        '10':'table_distribution_low_dimensional_embedding',
        '20':'parameter_mixing',
        '21':'parameter_2d_contours',
        '22':'parameter_histogram',
        '23':'parameter_acf',
        '24':'r2_parameter_grid_plot',
        '25':'log_target_parameter_grid_plot',
        '26':'absolute_error_parameter_grid_plot',
        '30':'destination_attraction_mixing',
        '31':'destination_attraction_predictions',
        '32':'destination_attraction_residuals',
        '40':'origin_destination_table_tabular',
        '41':'origin_destination_table_spatial',
        '42':'origin_destination_table_colorbars'
}

SWEEPABLE_PARAMS = {
    "seed": {
        "is_coord":True        
    },
    "origin_demand": {
        "is_coord":False
    },
    "destination_attraction_ts": {
        "is_coord":False
    },
    "cost_matrix": {
        "is_coord":False
    },
    "name": {
        "is_coord":False
    },
    "alpha": {
        "is_coord":True
    },
    "beta": {
        "is_coord":True
    },
    "bmax": {
        "is_coord":False
    },
    "dt": {
        "is_coord":True
    },
    "delta": {
        "is_coord":True
    },
    "kappa": {
        "is_coord":True
    },
    "sigma": {
        "is_coord":True
    },
    "epsilon": {
        "is_coord":True
    },
    "N": {
        "is_coord":False
    },
    "to_learn": {
        "is_coord":False
    },
    "num_layers": {
        "is_coord":True
    },
    "optimizer": {
        "is_coord":True
    },
    "learning_rate": {
        "is_coord":True
    }
}

INTENSITY_MODELS = ['spatial_interaction_model']

DATE_FORMATS = ['start,stop,step-%m-%Y','start,stop,step_%m_%Y','start,stop,step_%m', 'start,stop,step-%m']

# Caching numba functions
UTILS_CACHED =  True
MATH_UTILS_CACHED =  True
PROBABILITY_UTILS_CACHED = True

# Parallelise numba functions
NUMBA_PARALLELISE = False#True

def sigmoid(beta=torch.tensor(1.0)):
    '''Extends the torch.nn.sigmoid activation function by allowing for a slope parameter.'''
    return lambda x: torch.sigmoid(beta * x)

# Pytorch loss functions
LOSS_FUNCTIONS = {
    'l1loss': torch.nn.L1Loss,
    'mseloss': torch.nn.MSELoss,
    'crossentropyloss': torch.nn.CrossEntropyLoss,
    'ctcloss': torch.nn.CTCLoss,
    'nllloss': torch.nn.NLLLoss,
    'poissonnllloss': torch.nn.PoissonNLLLoss,
    'gaussiannllloss': torch.nn.GaussianNLLLoss,
    'kldivloss': torch.nn.KLDivLoss,
    'bceloss': torch.nn.BCELoss,
    'bcewithlogitsloss': torch.nn.BCEWithLogitsLoss,
    'marginrankingloss': torch.nn.MarginRankingLoss,
    'hingeembeddingloss': torch.nn.HingeEmbeddingLoss,
    'multilabelmarginloss': torch.nn.MultiLabelMarginLoss,
    'huberloss': torch.nn.HuberLoss,
    'smoothl1loss': torch.nn.SmoothL1Loss,
    'softmarginloss': torch.nn.SoftMarginLoss,
    'multilabelsoftmarginloss': torch.nn.MultiLabelSoftMarginLoss,
    'cosineembeddingloss': torch.nn.CosineEmbeddingLoss,
    'multimarginloss': torch.nn.MultiMarginLoss,
    'tripletmarginloss': torch.nn.TripletMarginLoss,
    'tripletmarginwithdistanceloss': torch.nn.TripletMarginWithDistanceLoss,
}

# Pytorch activation functions.
# Pairs of activation functions and whether they are part of the torch.nn module, in which case they must be called
# via func(*args, **kwargs)(x).

ACTIVATION_FUNCS = {
    'abs': [torch.abs, False],
    'celu': [torch.nn.CELU, True],
    'cos': [torch.cos, False],
    'cosine': [torch.cos, False],
    'elu': [torch.nn.ELU, True],
    'gelu': [torch.nn.GELU, True],
    'hardshrink': [torch.nn.Hardshrink, True],
    'hardsigmoid': [torch.nn.Hardsigmoid, True],
    'hardswish': [torch.nn.Hardswish, True],
    'hardtanh': [torch.nn.Hardtanh, True],
    'leakyrelu': [torch.nn.LeakyReLU, True],
    'linear': [None, False],
    'logsigmoid': [torch.nn.LogSigmoid, True],
    'mish': [torch.nn.Mish, True],
    'prelu': [torch.nn.PReLU, True],
    'relu': [torch.nn.ReLU, True],
    'rrelu': [torch.nn.RReLU, True],
    'selu': [torch.nn.SELU, True],
    'sigmoid': [sigmoid, True],
    'silu': [torch.nn.SiLU, True],
    'sin': [torch.sin, False],
    'sine': [torch.sin, False],
    'softplus': [torch.nn.Softplus, True],
    'softshrink': [torch.nn.Softshrink, True],
    'swish': [torch.nn.SiLU, True],
    'tanh': [torch.nn.Tanh, True],
    'tanhshrink': [torch.nn.Tanhshrink, True],
    'threshold': [torch.nn.Threshold, True],
}

OPTIMIZERS = {
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'SparseAdam': torch.optim.SparseAdam,
    'Adamax': torch.optim.Adamax,
    'ASGD': torch.optim.ASGD,
    'LBFGS': torch.optim.LBFGS,
    'NAdam': torch.optim.NAdam,
    'RAdam': torch.optim.RAdam,
    'RMSprop': torch.optim.RMSprop,
    'Rprop': torch.optim.Rprop,
    'SGD': torch.optim.SGD,
}


class Dataset(object):
    pass

