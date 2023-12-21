import os
import json
import torch 
import operator
from copy import deepcopy

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
    "alpha": 1, 
    "beta": 0, 
    "delta": 0.1,
    "kappa": 2, 
    "epsilon": 1,
    "bmax": 1,
    "noise_percentage": 3,
    "sigma": 3,
}

SWEEPABLE_PARAMS = {"iter"}
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT,"data/inputs/configs/schemas/cfg_parameters.json"), "r") as f:
    config_params = json.load(f)
    # Find all sweepable parameters
    for key_value_path in deep_walk(config_params):
        if "sweep" in key_value_path:
            sweep_index = key_value_path.index("sweep")
            if key_value_path[sweep_index-1] != "file":
                SWEEPABLE_PARAMS.add(key_value_path[sweep_index-1])
            elif key_value_path[sweep_index-2] == "file":
                SWEEPABLE_PARAMS.add(key_value_path[sweep_index-2])

SWEEPABLE_LENGTH_METADATA = ["seed","iter","origin","destination","time"]

SIM_TYPE_CONSTRAINTS = {
    "TotallyConstrained":"grand_total",
    "ProductionConstrained":"row_margin"
}

INTENSITY_INPUTS = {
    "TotallyConstrained":["cost_matrix","grand_total"],
    "ProductionConstrained":["cost_matrix","origin_demand","grand_total"]
}

INTENSITY_OUTPUTS = {
    "TotallyConstrained":["alpha","beta","log_destination_attraction"],
    "ProductionConstrained":["alpha","beta","log_destination_attraction"]
}

TABLE_SOLVERS = ["monte_carlo_sample", "maximum_entropy_solution",
                 "iterative_residual_filling_solution", "iterative_uniform_residual_filling_solution"]
MARGINAL_SOLVERS = ["multinomial"]

NORMS = ["relative_l_0","relative_l_1","relative_l_2","l_0","l_1","l_2"]

DISTANCE_FUNCTIONS = [
    "l_p_distance",
    "edit_distance_degree_one",
    "edit_distance_degree_higher",
    "chi_squared_distance",
    "euclidean_distance",
    "chi_squared_column_distance",
    "chi_squared_row_distance"
]

NUMPY_TO_TORCH_DTYPE = {
    "float32":float32,
    "float64":float64,
    "uint8":uint8,
    "int8":int8,
    "int16":int16,
    "int32":int32,
    "int64":int64,
    "bool":tbool,
    "str":str,
    "object":object,
    "list":list
}

TORCH_TO_NUMPY_DTYPE = {v:k for k,v in NUMPY_TO_TORCH_DTYPE.items()}


def sigmoid(beta=torch.tensor(1.0)):
    """Extends the torch.nn.sigmoid activation function by allowing for a slope parameter."""
    return lambda x: torch.sigmoid(beta * x)


RAW_LOSS_DATA_REQUIREMENTS = {
    "dest_attraction_ts_loss": {
        "prediction_data": ["destination_attraction_ts"],
        "validation_data": ["destination_attraction_ts"],
    },
    "table_loss": {
        "prediction_data": ["table"],
        "validation_data": ["log_intensity"],
    },
    "total_distance_loss": {
        "prediction_data": ["table"],
        "validation_data": ["cost_matrix","total_cost_by_origin"]
    },
    "total_loss": {},
    "loss": {}
}

LOSS_DATA_REQUIREMENTS = deepcopy(RAW_LOSS_DATA_REQUIREMENTS)
for k,v in RAW_LOSS_DATA_REQUIREMENTS.items():
    if k in ['total_loss','loss']:
        continue
    # Create a likelihood equivalent loss
    new_k = k.split("_loss")[0] + '_likelihood_loss'
    LOSS_DATA_REQUIREMENTS.update({new_k:v})

LOSS_KWARG_OPERATIONS = {
    "var":  {
        "function": "var*torch.ones(dim).to(device,dtype)",
        "kwargs": {
            "dtype": float32
        }
    }
}

# Pytorch loss functions
LOSS_FUNCTIONS = {
    "l1loss": {
        "function":torch.nn.L1Loss,
        "kwargs_keys":[]
    },
    "mseloss": {
        "function":torch.nn.MSELoss,
        "kwargs_keys":[]
    },
    "crossentropyloss": {
        "function":torch.nn.CrossEntropyLoss,
        "kwargs_keys":[]
    },
    "ctcloss": {
        "function":torch.nn.CTCLoss,
        "kwargs_keys":[]
    },
    "nllloss": {
        "function":torch.nn.NLLLoss,
        "kwargs_keys":[]
    },
    "poissonnllloss": {
        "function":torch.nn.PoissonNLLLoss,
        "kwargs_keys":[]
    },
    "gaussiannllloss": {
        "function":torch.nn.GaussianNLLLoss,
        "kwargs_keys":['var']
    },
    "kldivloss": {
        "function":torch.nn.KLDivLoss,
        "kwargs_keys":[]
    },
    "bceloss": {
        "function":torch.nn.BCELoss,
        "kwargs_keys":[]
    },
    "bcewithlogitsloss": {
        "function":torch.nn.BCEWithLogitsLoss,
        "kwargs_keys":[]
    },
    "marginrankingloss": {
        "function":torch.nn.MarginRankingLoss,
        "kwargs_keys":[]
    },
    "hingeembeddingloss": {
        "function":torch.nn.HingeEmbeddingLoss,
        "kwargs_keys":[]
    },
    "multilabelmarginloss": {
        "function":torch.nn.MultiLabelMarginLoss,
        "kwargs_keys":[]
    },
    "huberloss": {
        "function":torch.nn.HuberLoss,
        "kwargs_keys":[]
    },
    "smoothl1loss": {
        "function":torch.nn.SmoothL1Loss,
        "kwargs_keys":[]
    },
    "softmarginloss": {
        "function":torch.nn.SoftMarginLoss,
        "kwargs_keys":[]
    },
    "multilabelsoftmarginloss": {
        "function":torch.nn.MultiLabelSoftMarginLoss,
        "kwargs_keys":[]
    },
    "cosineembeddingloss": {
        "function":torch.nn.CosineEmbeddingLoss,
        "kwargs_keys":[]
    },
    "multimarginloss": {
        "function":torch.nn.MultiMarginLoss,
        "kwargs_keys":[]
    },
    "tripletmarginloss": {
        "function":torch.nn.TripletMarginLoss,
        "kwargs_keys":[]
    },
    "tripletmarginwithdistanceloss": {
        "function":torch.nn.TripletMarginWithDistanceLoss,
        "kwargs_keys":[]
    },
    "custom":{
        "function":None,
        "kwargs_keys":None
    }
}

INPUT_SCHEMA = {
    "origin_demand":{
        "axes":[0],
        "dtype":"float32", 
        "ndmin":1,
        "funcs":[("np",".arange(start,stop,step)")],
        "args_dtype":["int32"],
        "dims":["origin"],
        "cast_to_xarray":False
    },
    "destination_demand":{
        "axes":[1],
        "dtype":"float32", 
        "ndmin":1,
        "funcs":[("np",".arange(start,stop,step)")],
        "args_dtype":["int32"],
        "dims":["destination"],
        "cast_to_xarray":False
    },
    "origin_attraction_ts":{
        "axes":[0],
        "dtype":"float32", 
        "ndmin":2,
        "funcs":[("np",".arange(start,stop,step)")],
        "args_dtype":["int32"],
        "dims":["origin"],
        "cast_to_xarray":False
    },
    "destination_attraction_ts":{
        "axes":[0,1],
        "dtype":"float32", 
        "ndmin":2,
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "dims":["destination","time"],
        "cast_to_xarray":False
    },
    "cost_matrix":{
        "axes":[0,1],
        "dtype":"float32", 
        "ndmin":2,
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "dims":["origin","destination"],
        "cast_to_xarray":False
    },
    "total_cost_by_origin":{
        "axes":[0],
        "dtype":"float32",
        "ndmin":1,
        "funcs":[("np",".arange(start,stop,step)")],
        "args_dtype":["int32"],
        "dims":["origin"],
        "cast_to_xarray":False
    },
    "ground_truth_table":{
        "axes":[0,1],
        "dtype":"int32", 
        "ndmin":2,
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "dims":["origin","destination"],
        "cast_to_xarray":True
    },
    "grand_total":{
        "axes":[],
        "dtype":"float32",
        "ndmin":0,
        "funcs":[],
        "args_dtype":[],
        "dims":[],

        "cast_to_xarray":False
    },
    "dims":{},
    "to_learn":{},
    "true_parameters":{},
    "dataset":{}
}

TABLE_SCHEMA = {
    "table":{
        "axes":[0,1],
        "dtype":"int32", 
        "ndmin":2,
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "is_iterable": True,
        "dims":["origin","destination"]
    }
}
INTENSITY_SCHEMA = {
    "intensity":{
        "axes":[0,1],
        "dtype":"int32",
        "funcs":[],
        "args_dtype":[],
        "is_iterable": True,
        "dims":["origin","destination"]
    },
    "log_destination_attraction":{
        "axes":[0,1],
        "dtype":"float32",
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "is_iterable": True,
        "dims":["destination","time"]
    },
    "log_origin_attraction":{
        "axes":[0],
        "dtype":"float32", 
    },
    "alpha":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
        "dims":[]
    },
    "beta":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
        "dims":[]
    },
    "delta":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
    },
    "kappa":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
    },
    "sigma":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
    },
    "gamma":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
    },
    "noise_percentage":{
        "axes":[],
        "dtype":"float32",
        "is_iterable": True,
    }
}


OUTPUT_SCHEMA = {
    "loss":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
    },
    "total_loss":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
    },
    "log_target":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
    },
    "theta_acc":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
    },
    "log_destination_attraction_acc":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
    },
    "table_acc":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
    },
    "sign":{
        "axes":[],
        "dtype":"int8",
        "is_iterable": True,
    },
    "r2":{
        "axes":[0,1],
        "dtype":"float32",
        "is_iterable": False,
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "dims":["alpha_range","beta_range"]
    },
    "log_posterior_approximation":{
        "axes":[0,1],
        "dtype":"float32",
        "is_iterable": False,
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "dims":["alpha_range","beta_range"]
    },
    "compute_time":{
        "axes":[],
        "dtype":"float32",
        "is_iterable": True,
    },
    **INTENSITY_SCHEMA,
    **TABLE_SCHEMA
}
for loss in LOSS_DATA_REQUIREMENTS.keys():
    OUTPUT_SCHEMA[loss] = {
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True
    }

DATA_SCHEMA = {**INPUT_SCHEMA,**OUTPUT_SCHEMA}

SAMPLE_DATA_REQUIREMENTS = {
    **dict(zip(list(OUTPUT_SCHEMA.keys()),list(OUTPUT_SCHEMA.keys())))
}
SAMPLE_DATA_REQUIREMENTS["intensity"] = INTENSITY_OUTPUTS

TABLE_INFERENCE_EXPERIMENTS = ["nonjointtablesim_nn","jointtablesim_nn","jointtablesim_mcmc","table_mcmc","table_mcmc","tablesummaries_mcmcconvergence"]

EXPERIMENT_OUTPUT_NAMES = {
    "SIM_MCMC": ["log_destination_attraction","theta","sign",
                 "log_target","compute_time",
                 "theta_acc","log_destination_attraction_acc"],
    "JointTableSIM_MCMC": ["log_destination_attraction","theta","sign","table",
                           "log_target","compute_time",
                           "theta_acc","log_destination_attraction_acc","table_acc"],
    "Table_MCMC": ["table","compute_time"],
    "TableSummaries_MCMCConvergence": ["table","compute_time"],
    "SIM_NN": ["log_destination_attraction","theta","loss","compute_time"],
    "RSquared_Analysis": ["r2"],
    "LogTarget_Analysis": ["log_posterior_approximation"],
    "NonJointTableSIM_NN": ["log_destination_attraction","theta", "loss", "table","compute_time"],
    "JointTableSIM_NN": ["log_destination_attraction","theta","loss", "table","compute_time"]
}

AUXILIARY_COORDINATES_DTYPES = {
    "N":torch.int32,
    "dataset":str,
    "covariance":str,
    "step_size":torch.float32,
    "to_learn":object,
    "alpha":torch.float32,
    "beta":torch.float32,
    "noise_percentage":torch.float32,
    "delta":torch.float32,
    "kappa":torch.float32,
    "sigma":object,
    "title":str,
    "axes":object,
    "cells":str,
    "loss_name":object,
    "loss_function":object,
    "loss_kwargs":object,
    "name":str,
    "table_steps": torch.int32,
    "bmax":torch.float32,
    "cost_matrix":str,
}

CORE_COORDINATES_DTYPES = {
 "iter":torch.int32,
 "time":torch.int32,
 "origin":torch.int16,
 "destination":torch.int16,
 "seed":torch.int32,
 "alpha_range":torch.int32,
 "beta_range":torch.int32
#  "table_steps":torch.int32,
#  "theta_steps":torch.int32,
#  "destination_attraction_steps":torch.int32,
}

COORDINATES_DTYPES = {**CORE_COORDINATES_DTYPES,**AUXILIARY_COORDINATES_DTYPES}

NUMPY_TYPE_TO_DAT_TYPE = {
    "float":"%f",
    "int":"start,stop,step",
}


METRICS = {
    "srmse":{
        "shape":"(N,1)",
        "loop_over":["none"],
        "apply_axis":(0,1),
        "dtype":"float32",
        "ground_truth":".inputs.data.ground_truth_table"
    },
    "p_distance":{
        "shape":"(N,dims)",
        "loop_over":["none"],
        "apply_axis":(1,2),
        "dtype":"float32",
        "ground_truth":".inputs.data.ground_truth_table"
    },
    "ssi":{
        "shape":"(N,1)",
        "loop_over":["none"],
        "apply_axis":(0,1),
        "dtype":"float32",
        "ground_truth":".inputs.data.ground_truth_table"
    },
    "sparsity":{
        "shape":"(N,1)",
        "loop_over":["none"],
        "apply_axis":(0,1),
        "dtype":"float32"
    },
    "shannon_entropy":{
        "shape":"(1,1)",
        "loop_over":["none"],
        "apply_axis":(0,1),
        "dtype":"float32",
        "ground_truth":".inputs.data.ground_truth_table"
    },
    "von_neumann_entropy":{
        "shape":"(N,1)",
        "loop_over":["none"],
        "apply_axis":(0,1),
        "dtype":"float32"
    },
    "coverage_probability":{
        "shape":"(1,dims)",
        "loop_over":["region_mass"],
        "apply_axis":(1,2),
        "dtype":"int32",
        "ground_truth":".inputs.data.ground_truth_table"
    },
    "edit_degree_higher_error":{
        "shape":"(N,1)",
        "loop_over":["none"],
        "apply_axis":(0,1),
        "dtype":"float32",
        "ground_truth":".inputs.data.ground_truth_table"
    },
    "edit_degree_one_error":{
        "shape":"(N,1)",
        "loop_over":["none"],
        "apply_axis":(0,1),
        "dtype":"float32",
        "ground_truth":".inputs.data.ground_truth_table"
    },
    "none":{
        "shape":"(N,dims)",
        "loop_over":["none"],
        "apply_axis":(0,1,2),
        "dtype":"",
        "ground_truth":".inputs.data.ground_truth_table"
    },
    "":{
        "shape":"(N,dims)",
        "loop_over":["none"],
        "apply_axis":(0,1,2),
        "dtype":"",
        "ground_truth":".inputs.data.ground_truth_table"
    },
}

OPERATORS = {
    "+" : operator.add,
    "-" : operator.sub,
    "*" : operator.mul,
    "/" : operator.truediv,
    "%" : operator.mod,
    "^" : operator.xor,
}
SEPARATORS = ['.','[',']','(',')','()']


COLORS = {"monte_carlo_sample_degree_one":"tab:blue",
        "monte_carlo_sample_degree_higher":"tab:blue",
        "iterative_residual_filling_solution_degree_one":"tab:orange",
        "iterative_residual_filling_solution_degree_higher":"tab:orange",
        "maximum_entropy_solution_degree_one":"tab:green",
        "maximum_entropy_solution_degree_higher":"tab:green",
        "iterative_uniform_residual_filling_solution_degree_one":"tab:red",
        "iterative_uniform_residual_filling_solution_degree_higher":"tab:red"}

LINESTYLES = {"monte_carlo_sample_degree_one":"dashed",
            "iterative_residual_filling_solution_degree_one":"dashed",
            "maximum_entropy_solution_degree_one":"dashed",
            "iterative_uniform_residual_filling_solution_degree_one":"dashed",
            "monte_carlo_sample_degree_higher":"solid",
            "iterative_residual_filling_solution_degree_higher":"solid",
            "maximum_entropy_solution_degree_higher":"solid",
            "iterative_uniform_residual_filling_solution_degree_higher":"solid"}

INTENSITY_MODELS = ["spatial_interaction_model"]

DATE_FORMATS = ["start,stop,step-%m-%Y","start,stop,step_%m_%Y","start,stop,step_%m", "start,stop,step-%m"]

# Pytorch activation functions.
# Pairs of activation functions and whether they are part of the torch.nn module, in which case they must be called
# via func(*args, **kwargs)(x).

ACTIVATION_FUNCS = {
    "abs": [torch.abs, False],
    "celu": [torch.nn.CELU, True],
    "cos": [torch.cos, False],
    "cosine": [torch.cos, False],
    "elu": [torch.nn.ELU, True],
    "gelu": [torch.nn.GELU, True],
    "hardshrink": [torch.nn.Hardshrink, True],
    "hardsigmoid": [torch.nn.Hardsigmoid, True],
    "hardswish": [torch.nn.Hardswish, True],
    "hardtanh": [torch.nn.Hardtanh, True],
    "leakyrelu": [torch.nn.LeakyReLU, True],
    "linear": [None, False],
    "logsigmoid": [torch.nn.LogSigmoid, True],
    "mish": [torch.nn.Mish, True],
    "prelu": [torch.nn.PReLU, True],
    "relu": [torch.nn.ReLU, True],
    "rrelu": [torch.nn.RReLU, True],
    "selu": [torch.nn.SELU, True],
    "sigmoid": [sigmoid, True],
    "silu": [torch.nn.SiLU, True],
    "sin": [torch.sin, False],
    "sine": [torch.sin, False],
    "softplus": [torch.nn.Softplus, True],
    "softshrink": [torch.nn.Softshrink, True],
    "swish": [torch.nn.SiLU, True],
    "tanh": [torch.nn.Tanh, True],
    "tanhshrink": [torch.nn.Tanhshrink, True],
    "threshold": [torch.nn.Threshold, True],
}

OPTIMIZERS = {
    "Adagrad": torch.optim.Adagrad,
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SparseAdam": torch.optim.SparseAdam,
    "Adamax": torch.optim.Adamax,
    "ASGD": torch.optim.ASGD,
    "LBFGS": torch.optim.LBFGS,
    "NAdam": torch.optim.NAdam,
    "RAdam": torch.optim.RAdam,
    "RMSprop": torch.optim.RMSprop,
    "Rprop": torch.optim.Rprop,
    "SGD": torch.optim.SGD,
}


class Dataset(object):
    pass
