import os
import json
import torch 
import operator

from torch import int32, float32, uint8, int8, float64, int64, int16
from torch import bool as tbool

def deep_walk(indict, pre=None):
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in deep_walk(value, pre + [key]):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    for d in deep_walk(v, pre + [key]):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield pre + [indict]


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

SWEEPABLE_PARAMS = {'iter'}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT,'data/inputs/configs/cfg_parameters.json'), 'r') as f:
    config_params = json.load(f)
    # Find all sweepable parameters
    for key_value_path in deep_walk(config_params):
        if 'sweep' in key_value_path:
            sweep_index = key_value_path.index('sweep')
            if key_value_path[sweep_index-1] != 'file':
                SWEEPABLE_PARAMS.add(key_value_path[sweep_index-1])
            elif key_value_path[sweep_index-2] == 'file':
                SWEEPABLE_PARAMS.add(key_value_path[sweep_index-2])

SWEEPABLE_LENGTH_METADATA = ['seed','iter','origin','destination','time']

SIM_TYPE_CONSTRAINTS = {
    'TotallyConstrained':'grand_total',
    'ProductionConstrained':'row_margin'
}

INTENSITY_INPUTS = {
    'TotallyConstrained':['cost_matrix','origin_demand'],
    'ProductionConstrained':['cost_matrix','origin_demand']
}

INTENSITY_OUTPUTS = {
    'TotallyConstrained':['alpha','beta','log_destination_attraction'],
    'ProductionConstrained':['alpha','beta','log_destination_attraction']
}

TABLE_SOLVERS = ['monte_carlo_sample', 'maximum_entropy_solution',
                 'iterative_residual_filling_solution', 'iterative_uniform_residual_filling_solution']
MARGINAL_SOLVERS = ['multinomial']

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
    'object':object,
    'list':list
}

TORCH_TO_NUMPY_DTYPE = {v:k for k,v in NUMPY_TO_TORCH_DTYPE.items()}


def sigmoid(beta=torch.tensor(1.0)):
    '''Extends the torch.nn.sigmoid activation function by allowing for a slope parameter.'''
    return lambda x: torch.sigmoid(beta * x)


LOSS_DATA_REQUIREMENTS = {
    'dest_attraction_ts_loss': {
        "prediction_data": ['destination_attraction_ts'],
        "validation_data": ['destination_attraction_ts'],
    },
    'table_loss': {
        "prediction_data": ['table'],
        "validation_data": ['log_intensity']
    },
    'total_distance_loss': {
        "prediction_data": ['table'],
        "validation_data": ['cost_matrix','total_cost_by_origin']
    },
    'total_loss': {},
}

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
    'custom':None
}

INPUT_SCHEMA = {
    "origin_demand":{"dims":['origin'],"axes":[0],"dtype":"float32", "ndmin":1},
    "destination_demand":{"dims":['destination'],"axes":[1],"dtype":"float32", "ndmin":1},
    "origin_attraction_ts":{"dims":['origin'],"axes":[0],"dtype":"float32", "ndmin":2},
    "destination_attraction_ts":{"dims":['time','destination'],"axes":[0,1],"dtype":"float32", "ndmin":2},
    "cost_matrix":{"dims":['origin','destination'],"axes":[0,1],"dtype":"float32", "ndmin":2},
    "total_cost_by_origin":{"dims":['origin'],"axes":[0],"dtype":"float32", "ndmin":1},
    "ground_truth_table":{"dims":['origin','destination'],"axes":[0,1],"dtype":"int32", "ndmin":2},
    "dims":{},
    "grand_total":{},
    "to_learn":{},
    "true_parameters":{},
    "dataset":{}
}

TABLE_SCHEMA = {
    'table':{"dims":['origin','destination'],"axes":[0,1],"dtype":"int32", "ndmin":2},
}

INTENSITY_SCHEMA = {
    'intensity':{"dims":['origin','destination'],"axes":[0,1],"dtype":"int32", "ndmin":2},
    'log_destination_attraction':{"dims":['destination','time'],"axes":[0],"dtype":"float32", "ndmin":1},
    'log_origin_attraction':{"dims":['origin','time'],"axes":[0],"dtype":"float32", "ndmin":1},
    'alpha':{"dims":[],"axes":[],"dtype":"float32", "ndmin":0},
    'beta':{"dims":[],"axes":[],"dtype":"float32", "ndmin":0},
    'delta':{"dims":[],"axes":[],"dtype":"float32", "ndmin":0},
    'kappa':{"dims":[],"axes":[],"dtype":"float32", "ndmin":0},
    'sigma':{"dims":[],"axes":[],"dtype":"float32", "ndmin":0},
    'gamma':{"dims":[],"axes":[],"dtype":"float32", "ndmin":0},
    'noise_percentage':{"dims":[],"axes":[],"dtype":"float32", "ndmin":0}
}


OUTPUT_SCHEMA = {
    'loss':{"dims":[],"axes":[],"dtype":"float32", "ndmin":0},
    'total_loss':{"dims":[],"axes":[],"dtype":"float32", "ndmin":0},
    'log_target':{"dims":[],"axes":[],"dtype":"float32", "ndmin":0},
    'sign':{"dims":[],"axes":[],"dtype":"int8", "ndmin":0},
    **INTENSITY_SCHEMA,
    **TABLE_SCHEMA
}
for loss in LOSS_DATA_REQUIREMENTS.keys():
    OUTPUT_SCHEMA[loss] = {"dims":[],"axes":[],"dtype":"float32", "ndmin":0}

DATA_SCHEMA = {**INPUT_SCHEMA,**OUTPUT_SCHEMA}

SAMPLE_DATA_REQUIREMENTS = {
    **dict(zip(list(OUTPUT_SCHEMA.keys()),list(OUTPUT_SCHEMA.keys())))
}
SAMPLE_DATA_REQUIREMENTS['intensity'] = INTENSITY_OUTPUTS

TABLE_INFERENCE_EXPERIMENTS = ['nonjointtablesim_nn','jointtablesim_nn','jointtablesim_mcmc','table_mcmc','table_mcmc','tablesummaries_mcmcconvergence']

EXPERIMENT_OUTPUT_NAMES = {
    "SIM_MCMC": ['log_destination_attraction','theta','sign',
                 'log_target','computation_time',
                 'theta_acc','log_destination_attraction_acc'],
    "JointTableSIM_MCMC": ['log_destination_attraction','theta','sign','table',
                           'log_target','computation_time',
                           'theta_acc','log_destination_attraction_acc','table_acc'],
    "Table_MCMC": ['table','computation_time'],
    "TableSummaries_MCMCConvergence": ['table','computation_time'],
    "SIM_NN": ['log_destination_attraction','theta','loss','computation_time'],
    "NonJointTableSIM_NN": ['log_destination_attraction','theta', 'loss', 'table','computation_time'],
    "JointTableSIM_NN": ['log_destination_attraction','theta','loss', 'table','computation_time']
}

AUXILIARY_COORDINATES_DTYPES = {
    'N':torch.int32,
    'dataset':str,
    'covariance':str,
    'step_size':torch.float32,
    'to_learn':object,
    'alpha':torch.float32,
    'beta':torch.float32,
    'noise_percentage':torch.float32,
    'delta':torch.float32,
    'kappa':torch.float32,
    'sigma':object,
    'title':str,
    'axes':object,
    'cells':str,
    'loss_name':object,
    'loss_function':object,
    'name':str,
    'table_steps': torch.int32
}

CORE_COORDINATES_DTYPES = {
 'iter':torch.int32,
 'time':torch.int32,
 'origin':torch.int16,
 'destination':torch.int16,
 'seed':torch.int32,
#  'table_steps':torch.int32,
#  'theta_steps':torch.int32,
#  'destination_attraction_steps':torch.int32,
}

COORDINATES_DTYPES = {**CORE_COORDINATES_DTYPES,**AUXILIARY_COORDINATES_DTYPES}

NUMPY_TYPE_TO_DAT_TYPE = {
    'float':'%f',
    'int':'start,stop,step',
}


METRICS = {
    'srmse':{'shape':'(N,1)','loop_over':['none'],'apply_axis':(0,1),'dtype':'float32'},
    'p_distance':{'shape':'(N,dims)','loop_over':['none'],'apply_axis':(1,2),'dtype':'float32'},
    'ssi':{'shape':'(N,1)','loop_over':['none'],'apply_axis':(0,1),'dtype':'float32'},
    'sparsity':{'shape':'(N,1)','loop_over':['none'],'apply_axis':(0,1),'dtype':'float32'},
    'shannon_entropy':{'shape':'(1,1)','loop_over':['none'],'apply_axis':(0,1),'dtype':'float32'},
    'von_neumann_entropy':{'shape':'(N,1)','loop_over':['none'],'apply_axis':(0,1),'dtype':'float32'},
    'coverage_probability':{'shape':'(1,dims)','loop_over':['region_mass'],'apply_axis':(1,2),'dtype':'int32'},
    'edit_degree_higher_error':{'shape':'(N,1)','loop_over':['none'],'apply_axis':(0,1),'dtype':'float32'},
    'edit_degree_one_error':{'shape':'(N,1)','loop_over':['none'],'apply_axis':(0,1),'dtype':'float32'},
    'none':{'shape':'(N,dims)','loop_over':['none'],'apply_axis':(0,1,2),'dtype':''},
    '':{'shape':'(N,dims)','loop_over':['none'],'apply_axis':(0,1,2),'dtype':''},
}

OPERATORS = {
    '+' : operator.add,
    '-' : operator.sub,
    '*' : operator.mul,
    '/' : operator.truediv,
    '%' : operator.mod,
    '^' : operator.xor,
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

INTENSITY_MODELS = ['spatial_interaction_model']

DATE_FORMATS = ['start,stop,step-%m-%Y','start,stop,step_%m_%Y','start,stop,step_%m', 'start,stop,step-%m']

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


XARRAY_SCHEMA = {
    'alpha': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    }, 
    'beta': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    }, 
    'kappa': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    }, 
    'delta': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    }, 
    'sigma': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    }, 
    'noise_percentage': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    }, 
    'log_destination_attraction': {
        "coords":["time","destination"],
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "new_shape":["iter","time","destination"]
    },
    'table': {
        "coords":["origin","destination"],
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "new_shape":["iter","origin","destination"]
    },
    'loss': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    },
    'log_target': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    },
    'sign': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    },
    'theta_acc': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    },
    'log_destination_attraction_acc': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    },
    'table_acc': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    },
    'computation_time': {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    },
}
for loss in LOSS_DATA_REQUIREMENTS.keys():
    XARRAY_SCHEMA[loss] = {
        "coords":[],
        "funcs":[],
        "args_dtype":[],
        "new_shape":["iter"]
    }



class Dataset(object):
    pass
