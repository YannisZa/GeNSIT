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

SIM_DATA_TYPES = {
    'origin_demand':'float32',
    'destination_demand':'float32',
    'log_origin_attraction':'float32',
    'log_destination_attraction':'float32',
    'cost_matrix':'float32'
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

INPUT_TYPES = {
    'cost_matrix':'float32',
    'origin_demand':'float32'
}

SAMPLE_TYPES = {
    'ground_truth_table':'int32',
    'cost_matrix':'float32',
    'table':'int32',
    'tableerror':'float32',
    'intensity':'float32',
    'intensityerror':'float32',
    'log_destination_attraction':'float32',
    'theta':'float32',
    'sign':'int8'
}

DATA_TYPES = {**INPUT_TYPES,**SAMPLE_TYPES}

NUMPY_TYPE_TO_DAT_TYPE = {
    'float':'%f',
    'int':'%d',
}

SIM_TYPE_CONSTRAINTS = {
    'TotalConstrained':'grand_total',
    'ProductionConstrained':'row_margin'
}

TABLE_SOLVERS = ['monte_carlo_sample', 'maximum_entropy_solution',
                 'iterative_residual_filling_solution', 'iterative_uniform_residual_filling_solution']
MARGINAL_SOLVERS = ['multinomial']


METRICS = {
    'SRMSE':{'shape':'(N,1)','loop_over':[],'apply_axis':(0,1)},
    'p_distance':{'shape':'(N,dims)','loop_over':[],'apply_axis':(1,2)},
    'SSI':{'shape':'(N,1)','loop_over':[],'apply_axis':(0,1)},
    'sparsity':{'shape':'(N,1)','loop_over':[],'apply_axis':(0,1)},
    'shannon_entropy':{'shape':'(1,1)','loop_over':[],'apply_axis':(0,1)},
    'von_neumann_entropy':{'shape':'(N,1)','loop_over':[],'apply_axis':(0,1)},
    'coverage_probability':{'shape':'(1,dims)','loop_over':['region_mass'],'apply_axis':(1,2)},
    'edit_degree_higher_error':{'shape':'(N,1)','loop_over':[],'apply_axis':(0,1)},
    'edit_degree_one_error':{'shape':'(N,1)','loop_over':[],'apply_axis':(0,1)},
    'none':{'shape':'(N,dims)','loop_over':[],'apply_axis':(0,1,2)}
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

DATE_FORMATS = ['%d-%m-%Y','%d_%m_%Y','%d_%m', '%d-%m']

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