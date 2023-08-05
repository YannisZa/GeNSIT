import logging
import torch 

PARAMETER_DEFAULTS = {
    'alpha': 0, 
    'beta': 1, 
    'kappa': 2, 
    'delta': 0,
    'sigma': 3, 
    'bmax': 1,
    'epsilon': 1,
    'noise_percentage': 3,
}

XARRAY_SCHEMA = {
    'alpha': {
        "coords":["iter"],
        "funcs":[("np",".arange(start,stop,step)")],
        "args_dtype":["int32"],
        "new_shape":["1","N"]
    }, 
    'beta': {
        "coords":["iter"],
        "funcs":[("np",".arange(start,stop,step)")],
        "args_dtype":["int32"],
        "new_shape":["1","N"]
    }, 
    'kappa': {
        "coords":["iter"],
        "funcs":[("np",".arange(start,stop,step)")],
        "args_dtype":["int32"],
        "new_shape":["1","N"]
    }, 
    'delta': {
        "coords":["iter"],
        "funcs":[("np",".arange(start,stop,step)")],
        "args_dtype":["int32"],
        "new_shape":["1","N"]
    }, 
    'sigma': {
        "coords":["iter"],
        "funcs":[("np",".arange(start,stop,step)")],
        "args_dtype":["int32"],
        "new_shape":["1","N"]
    }, 
    'log_destination_attraction': {
        "coords":["destination","iter"],
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "new_shape":["J","N"]
    },
    'loss': {
        "coords":["iter"],
        "funcs":[("np",".arange(start,stop,step)")],
        "args_dtype":["int32"],
        "new_shape":["1","N"]
    },
    'log_target': {
        "coords":["iter"],
        "funcs":[("np",".arange(start,stop,step)")],
        "args_dtype":["int32"],
        "new_shape":["1","N"]
    },
    'computation_time': {
        "coords":["iter"],
        "funcs":[("np",".arange(start,stop,step)")],
        "args_dtype":["int32"],
        "new_shape":["1","N"]
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
    'float32':torch.float32,
    'float64':torch.float64,
    'uint8':torch.uint8,
    'int8':torch.int8,
    'int16':torch.int16,
    'int32':torch.int32,
    'int64':torch.int64,
    'bool':torch.bool
}


INPUT_TYPES = {
    'cost_matrix':'float32',
    'origin_demand':'float32',
    'destination_demand':'float32',
    'origin_attraction_ts':'float32',
    'destination_attraction_ts':'float32',
    'ground_truth_table':'int32'
}

TABLE_TYPES = {
    'table':'int32',
    'tableerror':'float32',
}
    
INTENSITY_TYPES = {
    'intensity':'float32',
    'intensityerror':'float32',
    'log_destination_attraction':'float32',
    'log_origin_attraction':'float32',
    'alpha':'float32',
    'beta':'float32',
    'delta':'float32',
    'kappa':'float32',
    'sigma':'float32',
}

OUTPUT_TYPES = {
    'loss':'float32',
    'log_target':'float32',
    'sign':'int8',
    **INTENSITY_TYPES,
    **TABLE_TYPES
}

DATA_TYPES = {**INPUT_TYPES,**OUTPUT_TYPES}

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
    "sim_type": {
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

