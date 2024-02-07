import os
import ast
import sys
import json
import click
import torch
import psutil

from multiresticodm.config import Config
from multiresticodm.utils.logger_class import *
from multiresticodm.utils.click_parsers import *
from multiresticodm.static.global_variables import *
from multiresticodm.utils.misc_utils import setup_logger
from multiresticodm.static.plot_variables import PLOT_VIEWS, PLOT_COORDINATES, PLOT_TYPES, LEGEND_LOCATIONS


def set_threads(n_threads):
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)
    torch.set_num_threads(n_threads)

def update_settings(setts):
    # Convert strings to ints
    setts['n_threads'] = int(setts.get('n_threads',1))
    setts['n_workers'] = setts.get('n_workers',1)

    # Update sample names
    sample_names = []
    if 'metric_args' in setts:
        sample_names += list(setts['metric_args']) 
    if 'evaluation_kwargs' in setts:
        sample_names += list([k for k,_ in setts['evaluation_kwargs']])
    sample_names = set(sample_names).intersection(DATA_SCHEMA.keys())
    setts['sample'] = list(sample_names)
    
    return setts

# Get total number of threads
AVAILABLE_CORES = psutil.cpu_count(logical = True)
AVAILABLE_THREADS = psutil.cpu_count(logical = True)


@click.group('multiresticodm')
def cli():
    """
    Command line tool for running multiresolution origin-destination matrix reconstruction
    """
    pass

_common_options = [
    click.option('--norm', '-no', default='relative_l_1', show_default = True,
            type = click.Choice(NORMS), help = f'Sets norm to use in relevant plots.'),
    click.option('--n_workers','-nw', type = click.IntRange(min = 1,max = AVAILABLE_CORES),
            default = '1', help = 'Overwrites number of independent workers used in multiprocessing'),
    click.option('--n_threads','-nt', type = click.IntRange(min = 1,max = AVAILABLE_THREADS), 
                 default = '1',help = '''Overwrites number of threads (per worker) used in multithreading.
            If many are provided first is set as the numpy threads and the second as the numba threads'''),
    click.option('--logging_mode','-log', type = click.Choice(['debug', 'info', 'warning', 'critical']+LOG_LEVELS), default='info', 
            help = f'Type of logging mode used.'),
    click.option('--n', '-n', type = click.IntRange(min = 1), help = 'Overwrites number of iterations of the selected the algorithm'),
    click.option('--table','-tab', type = click.STRING,default = None, help = 'Overwrites input table filename in config'),
    click.option('--device','-dev', type = click.Choice(['cpu', 'cuda', 'mps']), default='cpu',
            help = f'Type of device used for torch operations.')
]

_create_and_run_options = [
    click.argument('config_path', type = click.Path(exists = True), required = True),
    click.option('--data_generation_seed','-dgseed', type = click.IntRange(min = 0), show_default = True,
               default = None, help = 'Overwrites random number generation seed for synthetic data generation.'),
    click.option('--alpha','-alpha', type = click.FloatRange(min = 0,max = 2), default = None,
            help = 'Overwrites alpha parameter in Spatial Interaction Model.'),
    click.option('--beta','-beta', type = click.FloatRange(min = 0,max = 2), default = None,
            help = 'Overwrites beta parameter in Spatial Interaction Model.'),
    click.option('--bmax','-bmax', type = click.FloatRange(min = 1), default = None,
            help = 'Overwrites bmax parameter in Spatial Interaction Model.'),
    click.option('--delta','-delta', type = click.FloatRange(min = 0), default = None,
            help = 'Overwrites delta parameter in Spatial Interaction Model.'),
    click.option('--kappa','-kappa', type = click.FloatRange(min = 0), default = None,
            help = 'Overwrites kappa parameter in Spatial Interaction Model.'),
    click.option('--epsilon','-epsilon', type = click.FloatRange(min = 1), default = None,
            help = 'Overwrites epsilon parameter in Spatial Interaction Model.'),
    click.option('--sigma','-sigma', type = click.FloatRange(min = 0.0), default = None,
            help = 'Overwrites sigma parameter in arris Wilson Model.'),
    click.option('--dt','-dt', type = click.FloatRange(min = 0.0), default = None,
            help = 'Overwrites dt parameter in Harris Wilson Model.'),
    click.option('--noise_percentage','-np', type = click.FloatRange(min = 0.0), default = None,
            help = 'Overwrites noise_percentage parameter in Harris Wilson Model.')
]

_common_run_options = [
    click.option('--load_experiment','-le', multiple = False, type = click.Path(exists = True), default = None, 
                   help='Defines path to existing experiment output in order to load it and resume experimentation.'),
    click.option('--experiment_type','-et', type = click.Choice(list(EXPERIMENT_OUTPUT_NAMES.keys())), multiple = True, callback = to_list,
               default = None, help = 'Decides which experiment types to run'),
    click.option('--title','-ttl', type = click.STRING,
               default = None, help = 'Title appended to output filename of experiment'),
    click.option('--sweep_mode/--no-sweep_mode', default = None,is_flag = True, show_default = True,
              help = f'Flag for whether parameter sweep mode is activated or not.'),
    click.option('--validate_samples/--no-validate_samples', default = None, is_flag = True, show_default = True,
              help = f'Flag for whether every sample generated should by appropriately validated.'),
    click.option('--overwrite/--no-overwrite', default = None,is_flag = True, show_default = True,
              help = f'Flag for whether parameter sweep mode is activated or not.'),
    click.option('--dataset','-d', type = click.Path(exists = False),
               default = None, help = 'Overwrites dataset name in config'),
    click.option('--in_directory','-id', type = click.Path(exists = True),
                default = None, help = 'Overwrites inputs directory in config'),
    click.option('--to_learn','-tl', type = click.Choice(['alpha','beta','kappa','sigma']), default = None,
           help = 'Overwrites parameters to learn.'),
    click.option('--mcmc_workers','-mcmcnw', type = click.IntRange(min = 1,max = AVAILABLE_CORES), help = 'Overwrites number of MCMC workers'),
    click.option('--name','-nm', type = click.Choice(['TotallyConstrained','ProductionConstrained']),
               default = None, help = 'Overwrites spatial interaction model of choice (intensity function)'),
    click.option('--origin_demand','-od', type = click.STRING,
               default = None, help = 'Overwrites input origin demand filename in config'),
    click.option('--cost_matrix','-cm', type = click.STRING,
           default = None, help = 'Overwrites input cost matrix filename in config'),
    click.option('--table0','-tab0', type = click.Choice(TABLE_SOLVERS), default = None,
           help = 'Overwrites table initialisation method name in MCMC.'),
    click.option('--log_destination_attraction','-lda', type = click.STRING,
            default = None, help = 'Overwrites input log destination attraction filename in config'),
    click.option('--destination_attraction_ts','-dats', type = click.STRING,
            default = None, help = 'Overwrites input destination attraction time series filename in config'),
    click.option('--margins','-ma', type = click.STRING, cls = OptionEatAll,
           default = None, help = 'Overwrites input margin filenames in config'),
    click.option('--margin0','-m0', type = click.Choice(MARGINAL_SOLVERS), default = None,
            help = 'Overwrites margin initialisation method name in MCMC.'),
    click.option('--sparse_margins','-sm', is_flag = True, default = False,
           help = 'Flag for allowing sparsity in margins of contingency table'),
    click.option('--store_progress','-sp', default = 1.0, show_default = True, 
           type = click.FloatRange(min = 0.01,max = 1.0),
           help = 'Sets percentage of total samples that will be exported as a batch'),
    click.option('--axes','-ax', cls = PythonLiteralOption, multiple = True, default = None,
               help = '''Overwrites constrained margin axes (axes over which table is summed) in config.\
               Use the following syntax: -ax '[ENTER AXES SEPARATED BY COMMA HERE]' e.g -ax '[0]' -ax '[0, 1]'
               The unconstrained case is just -ax '[]' '''),
    click.option('--cells','-c', type = click.STRING, default = None,
               help = 'Overwrites constrained cells filename in config. '),
    click.option('--seed','-seed', type = click.IntRange(min = 0), show_default = True,
               default = None, help = 'Overwrites random number generation seed for model runs.'),

]

def common_options(func):
    for option in reversed(_common_options):
        func = option(func)
    return func

def common_run_options(func):
    for option in reversed(_common_run_options):
        func = option(func)
    return func

def create_and_run_options(func):
    for option in reversed(_create_and_run_options):
        func = option(func)
    return func

def exec(logger,settings,config_path,**kwargs):
    # Import all modules
    from multiresticodm.experiments import ExperimentHandler
    from multiresticodm.utils.misc_utils import deep_updates,update_device
    
    # Read config
    config = Config(
        path = config_path,
        settings = None,
        console_level = settings.get('logging_mode','info'),
        logger = logger
    )

    # Update settings with overwritten values
    deep_updates(config.settings,settings,overwrite = True)

    # Set device to run code on
    config.settings['inputs']['device'] = update_device(
        settings.get('device','cpu')
    )
    logger.warning(f"Device used: {config.settings['inputs']['device']}")

    # Update root
    config.path_sets_root()

    # Get list of experiments to run provided through command line
    experiment_types = list(kwargs.get('experiment_type',[]))

    # Maintain a dictionary of available experiments and their list index
    experiment_types = {
        exp.get('type',''):i 
        for i,exp in enumerate(config.settings['experiments']) 
        if len(exp.get('type','')) > 0 and \
            (
            (len(experiment_types) > 0 and exp.get('type','') in experiment_types) or \
            len(experiment_types) <= 0
            )
    }
    config.settings.setdefault('experiment_type',experiment_types)
    
    # Create output folder if it does not exist
    if not os.path.exists(config.out_directory):
        logger.info(f"Creating new output directory {config.out_directory}")
        os.makedirs(config.out_directory)

    logger.info(f"Validating config provided...")
    # Validate config
    config.validate()

    # Intialise experiment handler
    eh = ExperimentHandler(
        config,
        logger = logger
    )
    # Run experiments
    eh.run_and_write_experiments_sequentially()

    logger.success('Done')
    

@cli.command('create')
@common_options
@create_and_run_options
@click.option('--dims','-dims', type=(str, int), multiple = True,
                default=[(None,None)], help = 'Overwrites input dimensions size')
@click.option('--synthesis_method','-smthd', type = click.Choice(['sde_solver','sde_potential']),
                default='sde_solver', help = 'Determines method for synthesing data')
@click.option('--synthesis_n_samples','-sn', type = click.IntRange(min = 1),
                default = None, help = 'Determines number of times sde solver will be run to create synthetic data')
def create(
    norm,
    n_workers,
    n_threads,
    logging_mode,
    n,
    table,
    device,
    config_path,
    data_generation_seed,
    alpha,
    beta,
    bmax,
    delta,
    kappa,
    epsilon,
    sigma,
    dt,
    noise_percentage,
    dims,
    synthesis_method,
    synthesis_n_samples
):
    """
    Create synthetic data for origin-destination matrix reconstruction.
    """

    # Unpack dimensions
    if list(dims) != [(None,None)]:
        dims = {v[0]:v[1] for v in dims}
    else:
        dims = None

    # Gather all arguments in dictionary
    settings = {k:v for k,v in locals().items() if k != 'ctx'}
    # Remove all nulls
    settings = {k: v for k, v in settings.items() if v is not None}
    # Capitalise all single-letter arguments
    settings = {(key if len(key) == 1 else key):value for key, value in settings.items()}

    # Update settings
    settings = update_settings(settings)
    
    # Update number of workers
    set_threads(settings['n_threads'])

    # Import modules
    from multiresticodm.utils.misc_utils import deep_updates
    from multiresticodm.experiments import ExperimentHandler

    # Setup logger
    logger = setup_logger(
        __name__,
        console_level = settings.get('logging_mode','info'),
    )

    # Read config
    config = Config(
        path = config_path,
        settings = None,
        console_level = settings.get('logging_mode','info'),
        logger = logger
    )
    # Update settings with overwritten values
    deep_updates(config.settings,settings,overwrite = True)

    # Maintain a dictionary of available experiments and their list index
    experiment_types = {
        exp.get('type',''):i  
        for i,exp in enumerate(config.settings['experiments']) 
        if exp.get('type','') == 'DataGeneration'
    }
    config.settings.setdefault('experiment_type',experiment_types)

    # Update root
    config.path_sets_root()
    
    logger.info(f"Validating config provided...")
    
    # Validate config
    config.validate(experiment_type=','.join(list(experiment_types.keys())))
    # Get sweep-related data
    config.get_sweep_data()

    # Intialise experiment handler
    eh = ExperimentHandler(
        config,
        logger = logger,
        skip_output_prep = True
    )
    # Run experiments
    eh.run_and_write_experiments_sequentially()

@cli.command('run')
# @click.command(context_settings = dict(help_option_names=['-h', '--help']))
@click.option('--proposal','-p', type = click.Choice(['direct_sampling','degree_higher','degree_one']),
            default = None, help = 'Overwrites contingency table MCMC proposal')
@click.option('--loss_name','-ln', type = click.Choice(list(LOSS_DATA_REQUIREMENTS.keys())), callback = to_list,
            default = None, multiple = True, help = 'Overwrites neural net loss name(s)')
@click.option('--loss_function','-lf', type = click.Choice(list(LOSS_FUNCTIONS.keys())), callback = to_list,
            default = None, multiple = True, help = 'Overwrites neural net loss function(s)')
@click.option('--loss_kwarg_keys','-lkk', type = click.STRING, callback = split_to_list,
            default = None, multiple = True, help = 'Overwrites neural net loss function(s) kwarg parameter keys')
@click.option('--grid_size','-gs', type = click.IntRange(min = 1),
            default = None, help = 'Overwrites size of square grid for R^2 and Log Target analyses.')
@click.option('--theta_steps','-pn', type = click.IntRange(min = 1),
            default = None, help = 'Overwrites number of Spatial Interaction Model MCMC theta steps in joint scheme.')
@click.option('--destination_attraction_steps','-dan', type = click.IntRange(min = 1),
            default = None, help = 'Overwrites number of Spatial Interaction Model MCMC theta steps in joint scheme.')
@click.option('--table_steps','-tn', type = click.IntRange(min = 1),
            default = None, help = 'Overwrites number of Spatial Interaction Model MCMC steps in joint scheme.')
@click.option('--alpha0','-alpha0', type = click.FloatRange(min = 0), default = None,
            help = 'Overwrites initialisation of alpha parameter in MCMC.')
@click.option('--beta0','-beta0', type = click.FloatRange(min = 0), default = None,
            help = 'Overwrites initialisation of beta parameter in MCMC.')
@click.option('--beta_max','-bm', type = click.FloatRange(min = 0), default = None,
            help = 'Overwrites maximum beta in SIM parameters.')
@click.option('--covariance','-cov', type = click.STRING, default = None,
            help = 'Overwrites covariance matrix of parameter Gaussian Randow walk proposal')
@click.option('--step_size','-ss', type = click.FloatRange(min = 0), default = None,
            help = 'Overwrites step size in parameter Gaussian Randow walk proposal')
@click.option('--leapfrog_steps','-ls', type = click.IntRange(min = 1), default = None,
            help = 'Overwrites number of steps in Leapfrog Integrator in HMC')
@click.option('--leapfrog_step_size','-lss', type = click.FloatRange(min = 0), default = None,
            help = 'Overwrites number of step size in Leapfrog Integrator in HMC')
@click.option('--ais_leapfrog_steps','-als', type = click.IntRange(min = 1), default = None,
            help = 'Overwrites number of leapfrog steps in AIS HMC proposal (normalising constant sampling)')
@click.option('--ais_leapfrog_step_size','-alss', type = click.FloatRange(min = 0), default = None,
            help = 'Overwrites size of leapfrog steps in AIS HMC proposal (normalising constant sampling)')
@click.option('--ais_samples','-as', type = click.IntRange(min = 1), default = None,
            help = 'Overwrites number of samples in AIS (normalising constant sampling)')
@click.option('--n_bridging_distributions','-nb', type = click.IntRange(min = 1), default = None,
            help = 'Overwrites number of temperatures in tempered distribution in AIS (normalising constant sampling)')
@common_options
@create_and_run_options
@common_run_options
def run(
        log_destination_attraction,
        destination_attraction_ts,
        proposal,
        loss_name,
        loss_function,
        loss_kwarg_keys,
        grid_size,
        theta_steps,
        destination_attraction_steps,
        table_steps,
        alpha0,
        beta0,
        beta_max,
        covariance,
        step_size,
        leapfrog_steps,
        leapfrog_step_size,
        ais_leapfrog_steps,
        ais_leapfrog_step_size,
        ais_samples,
        n_bridging_distributions,
        load_experiment,
        dataset,
        in_directory,
        to_learn,
        mcmc_workers,
        name,
        origin_demand,
        cost_matrix,
        experiment_type,
        title,
        sweep_mode,
        validate_samples,
        overwrite,
        n,
        table,
        device,
        table0,
        margins,
        margin0,
        axes,
        store_progress,
        cells,
        sparse_margins,
        seed,
        config_path,
        data_generation_seed,
        alpha,
        beta,
        bmax,
        delta,
        kappa,
        epsilon,
        sigma,
        dt,
        noise_percentage,
        logging_mode,
        n_workers,
        n_threads,
        norm
    ):
    """
    Run Multiresolution Table Inference based on Physics-Driven Intensity Models.
    """
    # Gather all arguments in dictionary
    settings = {k:v for k,v in locals().items() if k != 'ctx'}
    # Remove all nulls
    settings = {k: v for k, v in settings.items() if v is not None}
    # Remove empty lists
    settings = {k: v for k, v in settings.items() if not hasattr(v,'__len__') or (hasattr(v,'__len__') and len(v) > 0)}
    # Capitalise all single-letter arguments
    settings = {(key if len(key) == 1 else key):value for key, value in settings.items()}
    
    # Update settings
    settings = update_settings(settings)
    
    # Update number of workers
    set_threads(settings['n_threads'])

    # Import all modules
    from numpy import asarray
    
    # Convert covariance to 2x2 array
    if 'covariance' in list(settings.keys()):
        settings['covariance'] = asarray([float(x) for x in settings['covariance'].split(",")]).reshape((2,2)).tolist()

    # Setup logger
    logger = setup_logger(
        __name__,
        console_level = settings.get('logging_mode','info'),
    )

    exec(
        logger,
        settings = settings,
        config_path = config_path,
        experiment_type = experiment_type
    )


_output_options = [
    # Output experiment data search options 
    click.option('--out_directory', '-o', required = True, type = click.Path(exists = True), default='./data/outputs/'),
    click.option('--dataset_name', '-dn', required = False, multiple = True, type = click.STRING),
    click.option('--directories','-d', multiple = True, required = False, type = click.Path(exists = False)),
    click.option('--experiment_type','-et', multiple = True, type = click.STRING, default = [''], cls = NotRequiredIf, not_required_if='directories'),
    click.option('--title','-en', multiple = True, type = click.STRING, default = [''], cls = NotRequiredIf, not_required_if='directories'),
    click.option('--exclude','-exc', type = click.STRING, default = [], multiple = True, cls = NotRequiredIf, not_required_if='directories'),
    click.option('--dates','-date', type = click.STRING, default = None, multiple = True, required = False),
    click.option('--filename_ending', '-fe', default = '', type = click.STRING),
    # Slicing and grouping outputs options
    click.option('--burnin_thinning_trimming', '-btt', default=[], show_default = True, multiple = True, callback = btt_callback,
                 type=(click.STRING,click.IntRange(min = 0),click.IntRange(min = 1),click.IntRange(min = 1)), 
                 help = f'Sets number of initial samples to discard (burnin), number of samples to skip (thinning) and number of samples to keep (trimming).'),
    click.option('--coordinate_slice','-cs', multiple = True, default = None, type = click.STRING, required = False, callback = list_of_str,
            help='Every argument corresponds to a list of keys, operators and values by which the output sweeped parameters will be sliced.'),
    click.option('--input_slice','-is', multiple = True, default = None, type = click.STRING, required = False, callback = list_of_str,
            help='Every argument corresponds to a list of keys and values by which the input sweeped parameters will be sliced.'),
    click.option('--group_by','-gb', multiple = True, default = None, type = click.Choice(SWEEPABLE_PARAMS),required = False,
            help='Every argument corresponds to a list of sweeped parameters that the outputs will be grouped by.'),
    # Options for applying metrics and evaluating expressions on outputs
    click.option('--statistic','-stat', multiple = True, default = None, 
            type = (click.STRING,click.STRING,click.STRING), required = False, callback = unpack_statistics, 
            help='Every argument corresponds to a list of metrics, statistics and their corresponding axes e.g. passing  ("SRMSE", \"mean|sum\",  \"iter|sweep\") corresponds to applying mean across the iter dimension and then sum across sweep dimension before applying SRMSE metric'),
    click.option('--metric','-m', multiple = True, type = click.Choice(METRICS.keys()), required = False, default = None,
                help = f'Sets list of metrics to compute over samples.'),
    click.option('--metric_args', '-ma', multiple = True, required = False,
            type = click.STRING, default = None, callback = to_list, help = f'''Metric keyword arguments.'''),
    click.option('--evaluate','-e', multiple = True, type=(click.STRING, click.STRING), required = False, default=[], 
                callback = list_of_lists, help = f'''Evaluates expressions for one or more datasets. 
                First argument is the name of the evaluation. Second is the evaluation expression'''),
    click.option('--evaluation_kwargs','-ea', multiple = True, type = click.STRING, required = False, default = None, 
                callback = evaluate_kwargs_callback, help = f'''Expression evaluation keyword arguments.'''),
    click.option('--evaluation_library','-el', multiple = True, type = click.STRING, required = False, default = ["np"], callback = list_of_str, help = f'''Expression evaluation libraries that needs to be loaded before applying evaluation'''),
    # Metric-specific options
    click.option('--region_mass', '-r', default=[0.95], show_default = True, multiple = True,
              type = click.FloatRange(0,1), help = f'Sets high posterior density region mass.'),
    click.option('--epsilon_threshold', '-eps', default = 0.001, show_default = True,
        type = click.FLOAT, help = f'Sets error norm threshold below which convergence is achieved. Used only in convergence plots.'),
    # Data storage and checkpointing options
    click.option('--metadata_keys','-k', multiple = True, type = click.STRING, required = False),
    click.option('--force_reload/--no-force_reload', default = False,is_flag = True, show_default = True,
              help = f'Flag for whether output collections should be re-compiled and re-written to file.'),
]

def output_options(func):
    for option in reversed(_output_options):
        func = option(func)
    return func

_plot_coordinate_options = []
for i,var in enumerate(PLOT_COORDINATES):
    _plot_coordinate_options += [
        # General axis-specific options
        click.option(f'--{var}_group', f'-{var}grp', default = [], show_default = False, callback = to_list, type = click.STRING, multiple = True, help = f'Sets {var} group (# groups corresponds to number of subplots). Each call corresponds to a different group/subplot.'),
        click.option(f'--{var}_discrete/--no-{var}_discrete', default = False, show_default = True, is_flag = True, help = f'Flag for whether {var} is discrete or not.'),
        click.option(f'--{var}_scientific/--no-{var}_scientific', default = False, show_default = True, is_flag = True, help = f'Flag for whether {var} labels should be in scientific notation or not.'),
        click.option(f'--{var}_shade/--no-{var}_shade', f'-{var}sh', show_default = False, default = False, is_flag = True, help = f'Sets flag for whether to shade area between lines and {var}-axis.'),
        click.option(f'--{var}_limit', f'-{var}lim', default=[(None,None)], show_default = False, callback = to_list,
            type=(click.FLOAT, click.FLOAT), multiple = True, help = f'Sets {var} min/max limits for each group only if {var} is numerical. Every call corresponds to a different group.'),
        click.option(f'-{var}_scale', f'-{var}sc', default=None, show_default = False,
            type=click.Choice(["linear", "log", "symlog", "logit"]), help = f'Sets {var} axis scaling'),
        # Axis label-specific options
        click.option(f'--{var}_label', f'-{var}lab', default = None, show_default = False,
            type = click.STRING, help = f'Sets {var} axis label.'),
        click.option(f'--{var}_label_size', f'-{var}ls', default = 13, show_default = True,
              type = click.INT, help = f'Sets {var} axis label font size.'),
        click.option(f'--{var}_label_pad', f'-{var}lp', default = 7, show_default = True,
            type = click.INT, help = f'Sets {var} axis label padding.'),
        click.option(f'--{var}_label_rotation', f'-{var}lr', default = 0, show_default = True,
            type = click.INT, help = f'Sets {var} axis label rotation.'),
        # Axis tick-specific options
        click.option(f'--{var}_tick_locations', f'-{var}tl', callback=lambda ctx, param, value: to_list(ctx,param,value,nargs=2), multiple = True, default=[None], show_default = True,
            type=(click.FloatRange(min=0),click.FloatRange(min=0)), help = f'First call is for the major {var}-axis and second is for minor {var}-axis. Each call has a tuple consisting of a starting point and a step size.'),
        click.option(f'--{var}_tick_size', f'-{var}ts', default = (13,10), show_default = True,
              type = (click.INT,click.INT), help = f'Sets {var} tick label size for minor and major axes (first and second arguments).'),
        click.option(f'--{var}_tick_pad', f'-{var}tp', default=(2,10), show_default = True,
            type=(click.INT,click.INT), help = f'Sets {var} tick label padding for minor and major axes (first and second arguments).'),
        click.option(f'--{var}_tick_rotation', f'-{var}tr', default=(0,45), show_default = True,
            type=(click.INT,click.INT), help = f'Sets {var} tick label rotation for minor and major axes (first and second arguments).')
    ]


def plot_coordinate_options(func):
    for option in reversed(_plot_coordinate_options):
        func = option(func)
    return func

@cli.command(name='plot',context_settings = dict(
    ignore_unknown_options = True,
    allow_extra_args = True,
))
@click.argument('plot_view', type = click.Choice(PLOT_VIEWS.keys()), default = None)
@click.argument('plot_type', type = click.Choice(PLOT_TYPES.keys()), default = None)
@output_options
@common_options
# Plot-specific main arguments and options
@click.option('-x', type = click.STRING, required = False, callback=coordinate_parse, multiple = True,
              default = [None], help='Sets x coordinate(s) in plot. First call is for a &-separated major x-axis and second is for a &-separated minor x-axis.')
@click.option('-y', type = click.STRING, required = False, callback=coordinate_parse, multiple = True,
                default = [None], help='Sets y coordinate(s) in plot. First call is for a &-separated major y-axis and second is for a &-separated minor y-axis.')
@click.option('-z', type = click.STRING, required = False, callback=coordinate_parse, multiple = True,
                default = [None], help='Sets z coordinate(s) in plot. First call is for a &-separated major z-axis and second is for a &-separated minor z-axis.')
@click.option('--plot_data_dir', '-pdd', type = click.Path(exists = True), multiple = True, required = False)
# Data-read figure options
@click.option('--label', '-l', default=[''], show_default = False, multiple = True, callback = to_list,
              type = click.STRING, help = f'Sets metadata key(s) to label figure elements by.')
@click.option('--colour', '-c', default='', show_default = False,
              type = click.STRING, help = f'Sets metadata key(s) to colour figure elements by.')
@click.option('--marker', '-mrkr', default='.', show_default = False,
              type = click.STRING, help = f'Sets metadata key(s) to determine figure element marker type by.')
@click.option('--marker_size', '-msz', default='1.0', show_default = False,
              type = click.STRING, help = f'Sets metadata key(s) to size figure element markers by.')
@click.option('--line_style', '-lst', default='-', show_default = False,
              type = click.STRING, help = f'Sets metadata key(s) to style figure element lines by.')
@click.option('--line_width', '-lw', default='1.0', show_default = False,
              type = click.STRING, help = f'Sets metadata key(s) to size (control width of) figure element lines by.')
@click.option('--opacity', '-op', default='1.0', show_default = False,
              type = click.STRING, help = f'Sets metadata key(s) to determine figure element opacity/visibility/transparency by.')
@click.option('--hatch_opacity', '-hchop', default='1.0', show_default = False,
              type = click.STRING, help = f'Sets metadata key(s) to determine hatch pattern opacity/visibility/transparency by.')
@click.option('--hatch', '-hch', default='', show_default = False,
              type = click.STRING, help = f'Sets metadata key(s) to determine figure element marker texture by.')
@click.option('--zorder', '-or', default=[(None,None)], show_default = False, multiple = True, callback = to_list,
              type=(click.Choice(['asc','desc']),click.STRING),
              help = f'''Sets variable to order plot points/lines by in ascending or descending order. 
              If ascending order is set, smaller values are given priority (go on top) and vice versa''')
@click.option('--annotate', '-an', default='', show_default = False,
              type = click.STRING, help = f'Sets metadata key(s) to annotate figure element text type by.')
# General figure options
@click.option('--figure_size', '-fs', default=(7,5), show_default = True,
              type=(click.FLOAT, click.FLOAT), help = f'Sets figure format.')
@click.option('--legend_location', '-loc', default='best', show_default = True,
              type=click.Choice(LEGEND_LOCATIONS), help = f'Sets the legend locations.')
@click.option('--legend_cols', '-lc', default=1, show_default = True,
              type=click.IntRange(min=1), help = f"Sets the legend's number of columns.")
@click.option('--legend_col_spacing', '-lcs', default=1.0, show_default = True,
              type=click.FloatRange(min=0.0), help = f"Sets the legend's spacing between columns.")
@click.option('--legend_pad', '-lp', default=0.1, show_default = True,
              type=click.FloatRange(min=0.0), help = f"Sets the legend's spacing between handlers and labels.")
@click.option('--bbox_to_anchor', '-bbta', default=None, show_default = True, multiple = True, callback = to_list,
              type=click.FLOAT, help = f'Box that is used to position the legend in conjunction with legend_location.')
@click.option('--legend_axis', '-la', default = None, show_default = True,
              type=(click.IntRange(0,None), click.IntRange(0,None)), help = f'Sets axis inside which to plot legend.')
@click.option('--figure_title', '-ft', default = None, show_default = False,
              type = click.STRING, help = f'Sets figure title.')
@click.option('--figure_title_size', '-fts', default = 16, show_default = True,
              type = click.INT, help = f'Sets subplot (group) title font size.')
@click.option('--group_title_size', '-gts', default = 12, show_default = True,
              type = click.INT, help = f'Sets subplot (group) title font size.')
@click.option('--legend_label_size', '-lls', default = 8, show_default = False,
              type = click.INT, help = f'Sets legend font size.')
@click.option('--annotation_size', '-as', default = 12, show_default = False,
              type = click.INT, help = f'Sets text annotation font size.')
@click.option('--colourmap', '-cm', default=['cblue'],required = False, show_default = True, multiple = True, callback = to_list,
            type = click.STRING, help = f'Sets colourmap(s) (e.g. colourmap corresponding to flows).')
@click.option('--colourbar/--no-colourbar', default = None, is_flag = True, show_default = True,
              help = f'Flag for plotting colourbars or not.')
@click.option('--colourbar_title', '-ct', default = [None], show_default = False, multiple = True, callback = to_list, 
              type = click.STRING, help = f'Sets colobar(s) titles (if colourbar(s) exist).')
@click.option('--colourbar_limit', '-cl', type=(click.FLOAT, click.FLOAT), multiple = True, callback = to_list, default = [(None,None)],
              help = f'Sets main colourbar(s) min,max limits (if colourbar(s) exist).')
@click.option('--colourbar_title_size', '-cts', default = None, show_default = False,
              type = click.INT, help = f'Sets colourbar title font size.')
@click.option('--colourbar_label_size', '-cts', default = None, show_default = False,
              type = click.INT, help = f'Sets colourbar label font size.')
@click.option('--colourbar_label_pad', '-clp', default = 7, show_default = True,
              type = click.INT, help = f'Sets colourbar label padding.')
@click.option('--colourbar_label_rotation', '-clr', default = 0, show_default = True,
              type = click.INT, help = f'Sets colourbar label rotation.')
@click.option('--colourmap_segmentation_limits', '-csl', default=[(0,1)], show_default = False, multiple = True, callback = to_list, 
              type=(click.FloatRange(0,1), click.FloatRange(0,1)), 
              help = f"Sets colourbar(s)' colour segmentation min,max limits.")
@click.option('--by_experiment/--no-by_experiment', '-be', default = False, is_flag = True, show_default = True,
              help = f'Flag for plotting data separately for each experiment or not.')
@click.option('--benchmark/--no-benchmark', '-bm', default = None, is_flag = True, show_default = True,
              help = f'Flag for plotting data along with benchmark/baseline (if provided).')
@click.option('--transpose/--no-transpose', default = None, is_flag = True, show_default = True,
              help = f'Flag for taking switching origins with destinations in plots.')
@click.option('--equal_aspect/--no-equal_aspect', default = False, is_flag = True, show_default = True,
              help = f'Flag for setting aspect ratio to equal or not.')
@click.option('--figure_format', '-ff', default='pdf', show_default = True,
              type = click.Choice(['eps', 'png', 'pdf', 'ps']), help = f'Sets figure format.')#, case_sensitive = False)
@click.option('--data_format', '-df', default='json', show_default = True,
              type = click.Choice(['dat', 'txt', 'json', 'csv']), help = f'Sets figure data format.')#, case_sensitive = False)
@click.option('--data_precision', '-dp', default = 5, show_default = True,
              type = click.INT, help = f'Sets figure data precision.')#, case_sensitive = False)
@click.option('--geometry','-g', multiple = False, type = click.File(), default = None, 
                help='Defines path to geometry geojson for visualising flows on a map.')
@click.option('--origin_geometry_type', '-ogt', default='lsoa', show_default = True,
              type = click.STRING, help = f'Define origin geometry type to plot origin destination data.')
@click.option('--destination_geometry_type', '-dgt', default='msoa', show_default = True,
              type = click.STRING, help = f'Define destination geometry type to plot origin destination data.')
@click.option('--origin_ids', '-oid', default = None, show_default = True, multiple = True,
              type = click.STRING, help = f'Subset of origin ids to use in various plots.')
@click.option('--destination_ids', '-did', default = None, show_default = True, multiple = True,
              type = click.STRING, help = f'Subset of destination ids to use in various plots.')
@click.option('--margin_plot', '-mp', default='destination_demand', show_default = True,
              type = click.Choice(['origin_demand', 'destination_demand']), help = f'Sets margin to plot in table flow plots.')
# Additional axis-specific option
@plot_coordinate_options
@click.pass_context
def plot(
        ctx,
        # Output-specific options
        out_directory,
        dataset_name,
        directories,
        experiment_type,
        title,
        exclude,
        filename_ending,
        burnin_thinning_trimming,
        statistic,
        coordinate_slice,
        input_slice,
        group_by,
        metric,
        metric_args,
        evaluate,
        evaluation_kwargs,
        evaluation_library,
        dates,
        epsilon_threshold,
        norm,
        n_workers,
        n_threads,
        logging_mode,
        n,
        table,
        device,
        metadata_keys,
        region_mass,
        force_reload,
        plot_view,
        plot_type,
        # Plot-specific main arguments and options
        x,
        y,
        z,
        plot_data_dir,
        # Data-read figure options
        label,
        colour,
        marker,
        marker_size,
        line_style,
        line_width,
        opacity,
        hatch_opacity,
        hatch,
        zorder,
        annotate,
        # General figure options
        figure_size,
        legend_location,
        legend_cols,
        legend_col_spacing,
        legend_pad,
        bbox_to_anchor,
        legend_axis,
        figure_title,
        figure_title_size,
        group_title_size,
        legend_label_size,
        annotation_size,
        colourmap,
        colourbar,
        colourbar_title,
        colourbar_limit,
        colourbar_title_size,
        colourbar_label_size,
        colourbar_label_pad,
        colourbar_label_rotation,
        colourmap_segmentation_limits,
        by_experiment,
        benchmark,
        transpose,
        equal_aspect,
        figure_format,
        data_format,
        data_precision,
        geometry,
        origin_geometry_type,
        destination_geometry_type,
        origin_ids,
        destination_ids,
        margin_plot,
        # Additional axis-specific option
        x_group,
        y_group,
        z_group,
        x_discrete,
        y_discrete,
        z_discrete,
        x_scientific,
        y_scientific,
        z_scientific,
        x_shade,
        y_shade,
        z_shade,
        x_limit,
        y_limit,
        z_limit,
        x_scale,
        y_scale,
        z_scale,
        x_label,
        y_label,
        z_label,
        x_label_size,
        y_label_size,
        z_label_size,
        x_label_pad,
        y_label_pad,
        z_label_pad,
        x_label_rotation,
        y_label_rotation,
        z_label_rotation,
        x_tick_locations,
        y_tick_locations,
        z_tick_locations,
        x_tick_size,
        y_tick_size,
        z_tick_size,
        x_tick_pad,
        y_tick_pad,
        z_tick_pad,
        x_tick_rotation,
        y_tick_rotation,
        z_tick_rotation
    ):
    """
    Plot experimental outputs.
    """
    # Gather all options in dictionary
    settings = {k:v for k,v in locals().items() if k != 'ctx'}
    
    # Capitalise all single-letter arguments
    settings = {(key if len(key) == 1 else key):value for key, value in settings.items()}
    
    # Add context arguments
    undefined_settings = {ctx.args[i][2:]: ctx.args[i+1] for i in range(0, len(ctx.args), 2)}

    # Add undefined settings to settings
    settings = {**settings, **undefined_settings}

    # Update settings
    settings = update_settings(settings)
    
    # Update number of workers
    set_threads(settings['n_threads'])
    
    # Import modules
    from multiresticodm.plot import Plot

    # Setup logger
    logger = setup_logger(
        __name__,
        console_level = settings.get('logging_mode','info'),
        file_level = 'DEBUG',
    )
    # Run plot
    Plot(
        plot_view = plot_view,
        outputs_directories = list(directories),
        settings = settings,
        logger = logger
    )

    logger.info('Done')

@cli.command(name='summarise',context_settings = dict(
    ignore_unknown_options = True,
    allow_extra_args = True,
))
@output_options
@common_options
@click.option('--algorithm', '-a', default=['linear'], show_default = True, multiple = True,
              type = click.STRING, help = f'Sets algorithm name for use in.')
@click.option('--sort_by','-sort', multiple = True, type = click.STRING, required = False)
@click.option('--ascending','-asc', default = None, is_flag = True, show_default = True, required = False)
@click.pass_context
def summarise(
        ctx,
        out_directory,
        dataset_name,
        directories,
        experiment_type,
        title,
        exclude,
        filename_ending,
        burnin_thinning_trimming,
        statistic,
        coordinate_slice,
        input_slice,
        group_by,
        metric,
        metric_args,
        evaluate,
        evaluation_kwargs,
        evaluation_library,
        dates,
        epsilon_threshold,
        norm,
        n_workers,
        n_threads,
        logging_mode,
        n,
        table,
        device,
        metadata_keys,
        region_mass,
        force_reload,
        algorithm,
        sort_by,
        ascending,
    ):
    """
    Create tabular summary of metadata and metrics computed for experimental outputs.
    """
    # Gather all options in dictionary
    settings = {k:v for k,v in locals().items() if k != 'ctx'}
    # Capitalise all single-letter arguments
    settings = {(key if len(key) == 1 else key):value for key, value in settings.items()}
    # Add context arguments
    undefined_settings = {ctx.args[i][2:]: ctx.args[i+1] for i in range(0, len(ctx.args), 2)}

    # Add undefined settings to settings
    settings = {**settings, **undefined_settings}

    # Update settings
    settings = update_settings(settings)
    
    # Update number of workers
    set_threads(settings['n_threads'])

    # Import modules
    from multiresticodm.outputs import OutputSummary

    # Setup logger
    logger = setup_logger(
        __name__,
        console_level = settings.get('logging_mode','info'),
    )
    logger.info('Gathering data')

    # Run output handler
    outsum = OutputSummary(
        settings = settings,
        logger = logger
    )
    # Collect
    experiment_metadata = outsum.collect_metadata()
    # Write experiment metadata to file
    outsum.write_metadata_summaries(experiment_metadata)

    logger.info('Done')

if __name__ == '__main__':
    # run()
    plot()
    # summarise()
