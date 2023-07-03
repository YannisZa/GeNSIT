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
    "float32":"%f",
    "float64":"%.18e",
    "int32":"%d",
    "int64":"%d"
}

SIM_TYPE_CONSTRAINTS = {
    "TotalConstrained":"grand_total",
    "ProductionConstrained":"row_margin"
}

TABLE_SOLVERS = ['monte_carlo_sample', 'maximum_entropy_solution',
                 'iterative_residual_filling_solution', 'iterative_uniform_residual_filling_solution']
MARGINAL_SOLVERS = ['multinomial']


METRICS = {
    'SRMSE':{"shape":"(N,1)","loop_over":[],'apply_axis':(0,1)},
    'sparsity':{"shape":"(N,1)","loop_over":[],'apply_axis':(0,1)},
    'shannon_entropy':{"shape":"(1,1)","loop_over":[],'apply_axis':(0,1)},
    'von_neumann_entropy':{"shape":"(N,1)","loop_over":[],'apply_axis':(0,1)},
    'coverage_probability':{"shape":"(1,dims)","loop_over":['region_mass'],'apply_axis':(1,2)},
    'edit_degree_higher_error':{"shape":"(N,1)","loop_over":[],"apply_axis":(0,1)},
    'edit_degree_one_error':{"shape":"(N,1)","loop_over":[],"apply_axis":(0,1)},
    'none':{"shape":"(N,dims)","loop_over":[],'apply_axis':(0,1,2)}
}

# Create colormap
COLORS = {'monte_carlo_sample_degree_one':'tab:blue',
        'monte_carlo_sample_degree_higher':'tab:blue',
        'iterative_residual_filling_solution_degree_one':'tab:orange',
        'iterative_residual_filling_solution_degree_higher':'tab:orange',
        'maximum_entropy_solution_degree_one':'tab:green',
        'maximum_entropy_solution_degree_higher':'tab:green',
        'iterative_uniform_residual_filling_solution_degree_one':'tab:red',
        'iterative_uniform_residual_filling_solution_degree_higher':'tab:red'}#,
            # 'H':'tab:purple',
            # 'I':'tab:brown',
            # 'J':'tab:pink'}
LINESTYLES = {"monte_carlo_sample_degree_one":'dashed',
            "iterative_residual_filling_solution_degree_one":'dashed',
            "maximum_entropy_solution_degree_one":'dashed',
            "iterative_uniform_residual_filling_solution_degree_one":'dashed',
            "monte_carlo_sample_degree_higher":'solid',
            "iterative_residual_filling_solution_degree_higher":'solid',
            "maximum_entropy_solution_degree_higher":'solid',
            "iterative_uniform_residual_filling_solution_degree_higher":'solid'}


# Type of plots
PLOT_HASHMAP = {
        "00":"table_posterior_mean_convergence_fixed_intensity",
        "01":"colsum_posterior_mean_convergence_fixed_intensity",
        "02":"table_posterior_mean_convergence",
        "10":"table_distribution_low_dimensional_embedding",
        "20":"parameter_mixing",
        "21":"parameter_2d_contours",
        "22":"parameter_histogram",
        "23":"parameter_acf",
        "24":"r2_parameter_grid_plot",
        "25":"log_target_parameter_grid_plot",
        "26":"absolute_error_parameter_grid_plot",
        "30":"destination_attraction_mixing",
        "31":"destination_attraction_predictions",
        "32":"destination_attraction_residuals",
        "40":"origin_destination_table_tabular",
        "41":"origin_destination_table_spatial",
        "42":"origin_destination_table_colorbars"
}

DATE_FORMATS = ['%d-%m-%Y','%d_%m_%Y','%d_%m', '%d-%m']

# Caching numba functions
UTILS_CACHED =  True
MATH_UTILS_CACHED =  True
PROBABILITY_UTILS_CACHED = True

# Parallelise numba functions
NUMBA_PARALLELISE = False#True
