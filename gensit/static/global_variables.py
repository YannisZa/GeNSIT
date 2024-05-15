import os
import json
import operator
from copy import deepcopy

def deep_walk(indict, pre = None):
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
    "alpha": 0.5, 
    "beta": 0.5, 
    "delta": 0.1,
    "kappa": 1.0, 
    "epsilon": 1.0,
    "bmax": 1,
    "noise_percentage": 3,
    "sigma": 0.0,#0.0141414,
    "dt":0.001
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

ITERATION_PARAMS = ["seed","iter","N"]
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


RAW_LOSS_DATA_REQUIREMENTS = {
    "dest_attraction_ts_loss": {
        "prediction_data": ["destination_attraction_ts"],
        "validation_data": ["destination_attraction_ts"],
    },
    "table_loss": {
        "prediction_data": ["table"],
        "validation_data": ["log_intensity"],
    },
    "total_table_distance_loss": {
        "prediction_data": ["table"],
        "validation_data": ["cost_matrix","total_cost_by_origin"]
    },
    "total_intensity_distance_loss": {
        "prediction_data": ["intensity"],
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
        "dtype": "float32",
        "kwargs": {}
    }
}

VALIDATION_SCHEMA = {
    "test_cells": {
        "dtype":"int32",
        "ndmin":2,
        "cast_to_xarray":False
    },
    "test_cells_mask": {
        "dtype":"int32",
        "ndmin":2,
        "axes":[0,1],
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "dims":["origin","destination"],
        "cast_to_xarray":True
    },
    "test_validation_cells": {
        "dtype":"int32",
        "ndmin":2,
        "cast_to_xarray":False
    },
    "test_validation_cells_mask": {
        "dtype":"int32",
        "ndmin":2,
        "axes":[0,1],
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "dims":["origin","destination"],
        "cast_to_xarray":True
    },
    "validation_cells": {
        "dtype":"int32",
        "ndmin":2,
        "cast_to_xarray":False
    },
    "validation_cells_mask": {
        "dtype":"int32",
        "ndmin":2,
        "axes":[0,1],
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "dims":["origin","destination"],
        "cast_to_xarray":True
    },
    "zero_train_cells": {
        "dtype":"int32",
        "ndmin":2,
        "cast_to_xarray":False
    },
    "zero_train_cells_mask": {
        "dtype":"int32",
        "ndmin":2,
        "axes":[0,1],
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "dims":["origin","destination"],
        "cast_to_xarray":True
    },
    "train_cells": {
        "dtype":"int32",
        "ndmin":2,
        "cast_to_xarray":False
    },
    "train_cells_mask": {
        "dtype":"int32",
        "ndmin":2,
        "axes":[0,1],
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "dims":["origin","destination"],
        "cast_to_xarray":True
    },
}


TRAIN_SCHEMA = {
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
        "cast_to_xarray":True
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
        "dtype":"float32",
        "ndmin":0,
        "cast_to_xarray":False
    },
    "region_features":{
        "dtype":"float32",
        "ndmin":2,
        "cast_to_xarray":False
    },
    "adjacency_matrix":{
        "axes":[0,1],
        "dtype":"bool", 
        "ndmin":2,
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "dims":["origin","destination"],
        "cast_to_xarray":False
    },
    "margins":{},
    "cells_subset":{},
    "dims":{},
    "to_learn":{},
    "true_parameters":{},
    "dataset":{}
}

TABLE_SCHEMA = {
    "table":{
        "axes":[0,1],
        "dtype":"float32", 
        "ndmin":2,
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int16","int16"],
        "dims":["origin","destination"],
        "is_iterable": True
    }
}
INTENSITY_SCHEMA = {
    "intensity":{
        "axes":[0,1],
        "dtype":"float32",
        "funcs":[],
        "args_dtype":[],
        "dims":["origin","destination"],
        "is_iterable": True
        
    },
    "log_destination_attraction":{
        "axes":[0,1],
        "dtype":"float32",
        "funcs":[("np",".arange(start,stop,step)"),("np",".arange(start,stop,step)")],
        "args_dtype":["int32","int32"],
        "dims":["destination","time"],
        "is_iterable": True
        
    },
    "log_origin_attraction":{
        "axes":[0],
        "dtype":"float32"
    },
    "alpha":{
        "axes":[],
        "dtype":"float32", 
        "dims":[],
        "is_iterable": True,
        "default":-1.0
    },
    "beta":{
        "axes":[],
        "dtype":"float32", 
        "dims":[],
        "is_iterable": True,
        "default":-1.0
    },
    "delta":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
        "default":0.0
    },
    "kappa":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
        "default":1.0
    },
    "sigma":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
        "default":"learned"
    },
    "sigma":{
        "axes":[],
        "dtype":"float32", 
        "is_iterable": True,
        "default":"learned"
    },
    "noise_percentage":{
        "axes":[],
        "dtype":"float32",
        "is_iterable": True,
        "default":3.0
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

INPUT_SCHEMA = {**TRAIN_SCHEMA,**VALIDATION_SCHEMA}
DATA_SCHEMA = {**INPUT_SCHEMA,**OUTPUT_SCHEMA}

SAMPLE_DATA_REQUIREMENTS = {
    **dict(zip(list(OUTPUT_SCHEMA.keys()),list(OUTPUT_SCHEMA.keys())))
}
SAMPLE_DATA_REQUIREMENTS["intensity"] = INTENSITY_OUTPUTS

TABLE_INFERENCE_EXPERIMENTS = ["nonjointtablesim_nn","jointtablesim_nn","jointtablesim_mcmc","table_mcmc","table_mcmc","tablesummaries_mcmcconvergence"]

EXPERIMENT_OUTPUT_NAMES = {
    "RSquared_Analysis": ["r2"],
    "LogTarget_Analysis": ["log_posterior_approximation"],
    "SIM_MCMC": ["log_destination_attraction","theta","sign",
                 "log_target","compute_time",
                 "theta_acc","log_destination_attraction_acc"],
    "JointTableSIM_MCMC": ["log_destination_attraction","theta","sign","table",
                           "log_target","compute_time",
                           "theta_acc","log_destination_attraction_acc","table_acc"],
    "Table_MCMC": ["table","compute_time"],
    "SIM_NN": ["log_destination_attraction","theta","loss","compute_time"],
    "NonJointTableSIM_NN": ["log_destination_attraction","theta", "loss", "table","compute_time"],
    "JointTableSIM_NN": ["log_destination_attraction","theta","loss", "table","compute_time"],
    "XGBoost_Comparison": ["intensity","compute_time"],
    "RandomForest_Comparison": ["intensity","compute_time"],
    "GBRT_Comparison": ["intensity","compute_time"],
    "GraphAttentionNetwork_Comparison": ["intensity","compute_time"]
}

AUXILIARY_COORDINATES_DTYPES = {
    "N":"int32",
    "dataset":"str",
    "covariance":"str",
    "step_size":"float32",
    "to_learn":"object",
    "alpha":"float32",
    "beta":"float32",
    "noise_percentage":"float32",
    "delta":"float32",
    "kappa":"float32",
    "sigma":"object",
    "title":"str",
    "axes":"object",
    "cells":"str",
    "loss_name":"object",
    "loss_function":"object",
    "loss_kwargs":"object",
    "name":"str",
    "table_steps": "int32",
    "bmax":"float32",
    "cost_matrix":"str",
    "adjacency_matrix":"str",
    "region_features":"str",
    "destination_attraction_ts":"str",
    "proposal":"str"
}

CORE_COORDINATES_DTYPES = {
 "iter":"int32",
 "time":"int32",
 "origin":"int16",
 "destination":"int16",
 "seed":"int32",
 "alpha_range":"int32",
 "beta_range":"int32"
#  "table_steps":"int32",
#  "theta_steps":"int32",
#  "destination_attraction_steps":"int32",
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
        "dtype":"float32",
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


INTENSITY_MODELS = ["spatial_interaction_model"]

DATE_FORMATS = ["start,stop,step-%m-%Y","start,stop,step_%m_%Y","start,stop,step_%m", "start,stop,step-%m"]

class Dataset(object):
    pass

# Pytorch loss functions
LOSS_FUNCTIONS = {
    "l1loss": {
        "function":"torch.nn.L1Loss",
        "kwargs_keys":[]
    },
    "mseloss": {
        "function":"torch.nn.MSELoss",
        "kwargs_keys":[]
    },
    "crossentropyloss": {
        "function":"torch.nn.CrossEntropyLoss",
        "kwargs_keys":[]
    },
    "ctcloss": {
        "function":"torch.nn.CTCLoss",
        "kwargs_keys":[]
    },
    "nllloss": {
        "function":"torch.nn.NLLLoss",
        "kwargs_keys":[]
    },
    "poissonnllloss": {
        "function":"torch.nn.PoissonNLLLoss",
        "kwargs_keys":[]
    },
    "gaussiannllloss": {
        "function":"torch.nn.GaussianNLLLoss",
        "kwargs_keys":['var']
    },
    "kldivloss": {
        "function":"torch.nn.KLDivLoss",
        "kwargs_keys":[]
    },
    "bceloss": {
        "function":"torch.nn.BCELoss",
        "kwargs_keys":[]
    },
    "bcewithlogitsloss": {
        "function":"torch.nn.BCEWithLogitsLoss",
        "kwargs_keys":[]
    },
    "marginrankingloss": {
        "function":"torch.nn.MarginRankingLoss",
        "kwargs_keys":[]
    },
    "hingeembeddingloss": {
        "function":"torch.nn.HingeEmbeddingLoss",
        "kwargs_keys":[]
    },
    "multilabelmarginloss": {
        "function":"torch.nn.MultiLabelMarginLoss",
        "kwargs_keys":[]
    },
    "huberloss": {
        "function":"torch.nn.HuberLoss",
        "kwargs_keys":[]
    },
    "smoothl1loss": {
        "function":"torch.nn.SmoothL1Loss",
        "kwargs_keys":[]
    },
    "softmarginloss": {
        "function":"torch.nn.SoftMarginLoss",
        "kwargs_keys":[]
    },
    "multilabelsoftmarginloss": {
        "function":"torch.nn.MultiLabelSoftMarginLoss",
        "kwargs_keys":[]
    },
    "cosineembeddingloss": {
        "function":"torch.nn.CosineEmbeddingLoss",
        "kwargs_keys":[]
    },
    "multimarginloss": {
        "function":"torch.nn.MultiMarginLoss",
        "kwargs_keys":[]
    },
    "tripletmarginloss": {
        "function":"torch.nn.TripletMarginLoss",
        "kwargs_keys":[]
    },
    "tripletmarginwithdistanceloss": {
        "function":"torch.nn.TripletMarginWithDistanceLoss",
        "kwargs_keys":[]
    },
    "custom":{
        "function":None,
        "kwargs_keys":None
    }
}

ACTIVATION_FUNCS = {
    "abs": ["torch.abs", False],
    "celu": ["torch.nn.CELU", True],
    "cos": ["torch.cos", False],
    "cosine": ["torch.cos", False],
    "elu": ["torch.nn.ELU", True],
    "gelu": ["torch.nn.GELU", True],
    "hardshrink": ["torch.nn.Hardshrink", True],
    "hardsigmoid": ["torch.nn.Hardsigmoid", True],
    "hardswish": ["torch.nn.Hardswish", True],
    "hardtanh": ["torch.nn.Hardtanh", True],
    "leakyrelu": ["torch.nn.LeakyReLU", True],
    "linear": ["None", False],
    "logsigmoid": ["torch.nn.LogSigmoid", True],
    "mish": ["torch.nn.Mish", True],
    "prelu": ["torch.nn.PReLU", True],
    "relu": ["torch.nn.ReLU", True],
    "rrelu": ["torch.nn.RReLU", True],
    "selu": ["torch.nn.SELU", True],
    "sigmoid": ["lambda x: torch.sigmoid(beta * x)", True],
    "silu": ["torch.nn.SiLU", True],
    "sin": ["torch.sin", False],
    "sine": ["torch.sin", False],
    "softplus": ["torch.nn.Softplus", True],
    "softshrink": ["torch.nn.Softshrink", True],
    "swish": ["torch.nn.SiLU", True],
    "tanh": ["torch.nn.Tanh", True],
    "tanhshrink": ["torch.nn.Tanhshrink", True],
    "threshold": ["torch.nn.Threshold", True],
}

OPTIMIZERS = {
    "Adagrad": "torch.optim.Adagrad",
    "Adam": "torch.optim.Adam",
    "AdamW": "torch.optim.AdamW",
    "SparseAdam": "torch.optim.SparseAdam",
    "Adamax": "torch.optim.Adamax",
    "ASGD": "torch.optim.ASGD",
    "LBFGS": "torch.optim.LBFGS",
    "NAdam": "torch.optim.NAdam",
    "RAdam": "torch.optim.RAdam",
    "RMSprop": "torch.optim.RMSprop",
    "Rprop": "torch.optim.Rprop",
    "SGD": "torch.optim.SGD",
}