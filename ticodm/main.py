import os

import ast
import warnings
warnings.simplefilter("ignore")

import json
import click
import logging
import psutil

from ticodm.config import Config
from ticodm.global_variables import TABLE_SOLVERS,MARGINAL_SOLVERS, DATA_TYPES, METRICS, PLOT_HASHMAP, NORMS, DISTANCE_FUNCTIONS


def update_numpy_threads(n_threads):
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)



# Get total number of threads
AVAILABLE_CORES = psutil.cpu_count(logical=False)
AVAILABLE_THREADS = psutil.cpu_count(logical=True)

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        if (value is None) | (value == '[]'):
            return
        else:
            try:
                res = []
                for item in value:
                    res.append(ast.literal_eval(item))
                return res
            except ValueError:
                raise click.BadParameter(value)
    
class NotRequiredIf(click.Option):
    def __init__(self, *args, **kwargs):
        self.not_required_if = kwargs.pop('not_required_if')
        assert self.not_required_if, "'not_required_if' parameter required"
        kwargs['help'] = (kwargs.get('help', '') +
            ' NOTE: This argument is mutually exclusive with %s' %
            self.not_required_if
        ).strip()
        super(NotRequiredIf, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        we_are_present = self.name in opts
        other_present = self.not_required_if in opts

        if other_present:
            if we_are_present:
                raise click.UsageError(
                    "Illegal usage: `%s` is mutually exclusive with `%s`" % (
                        self.name, self.not_required_if))
            else:
                self.prompt = None

        return super(NotRequiredIf, self).handle_parse_result(
            ctx, opts, args)

class OptionEatAll(click.Option):

    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop('save_other_options', True)
        nargs = kwargs.pop('nargs', -1)
        assert nargs == -1, 'nargs, if set, must be -1 not {}'.format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):

        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


@click.group('ticodm')
def cli():
    """
    Command line tool for running data augmentation on spatial Interaction models.
    """
    pass

_common_options = [
    click.option('--norm', '-no', default='relative_l_1', show_default=True,
            type=click.Choice(NORMS), help=f'Sets norm to use in relevant plots.'),
    click.option('--n_workers','-nw', type=click.Choice([str(i) for i in range(1,AVAILABLE_THREADS+1)]),
            default = '1', help = 'Overwrites number of independent workers used in multiprocessing'),
    click.option('--n_threads','-nt', type=click.Choice([str(i) for i in range(1,AVAILABLE_CORES+1)]), multiple=True, 
                 default = ['1','1'],help = '''Overwrites number of threads (per worker) used in multithreading.
            If many are provided first is set as the numpy threads and the second as the numba threads'''),
    click.option('--logging_mode','-log', type=click.Choice(['debug', 'info', 'warning']), default='info', 
            help=f'Type of logging mode used.'),
    click.option('--n','-n', type=click.IntRange(min=1), help = 'Overwrites number of MCMC samples'),
    click.option('--table','-tab', type=click.STRING,default=None, help = 'Overwrites input table filename in config')
]

def common_options(func):
    for option in reversed(_common_options):
        func = option(func)
    return func

@cli.command('run')
# @click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('config_path', type=click.Path(exists=True), required=True)
@click.option('--load_experiment','-le', multiple=False, type=click.Path(exists=True), default=None, 
                help="Defines path to existing experiment output in order to load it and resume experimentation.")
@click.option('--run_experiments','-re', type=click.STRING, multiple=True,
            default = [], help = 'Decides which experiments to run')
@click.option('--experiment_title','-et', type=click.STRING,
            default = '', help = 'Title appended to output filename of experiment')
@click.option('--dataset','-d', type=click.Path(exists=True),
            default=None, help = 'Overwrites inputs dataset in config')
@click.option('--sim_type','-sim', type=click.Choice(['TotalConstrained','ProductionConstrained']),
            default=None, help = 'Overwrites spatial interaction model of choice (intensity function)')
@click.option('--origin_demand','-od', type=click.STRING,
            default=None, help = 'Overwrites input origin demand filename in config')
@click.option('--log_destination_attraction','-lda', type=click.STRING,
            default=None, help = 'Overwrites input log destination attraction filename in config')
@click.option('--cost_matrix','-cm', type=click.STRING,
            default=None, help = 'Overwrites input cost matrix filename in config')
@click.option('--margins','-ma', type=click.STRING, cls=OptionEatAll,
            default=None, help = 'Overwrites input margin filenames in config')
@click.option('--axes','-ax', cls=PythonLiteralOption, multiple=True, default = [],
             help = '''Overwrites constrained margin axes (axes over which table is summed) in config.\
             Use the following syntax: -ax '[ENTER AXES SEPARATED BY COMMA HERE]' e.g -ax '[0]' -ax '[0, 1]'
             The unconstrained case is just -ax '[]' ''')
@click.option('--cells','-c', type=click.STRING, default = None,
             help = 'Overwrites constrained cells filename in config. ')
@click.option('--seed','-seed', type=click.IntRange(min=0), show_default=True,
            default=None, help = 'Overwrites random number generation seed.')
@click.option('--dims','-dims', type=click.IntRange(min=1), cls=OptionEatAll,
            default=None, help = 'Overwrites input table column size')
@click.option('--proposal','-p', type=click.Choice(['direct_sampling','degree_higher','degree_one']),
            default = None, help = 'Overwrites contingency table MCMC proposal')
@click.option('--sparse_margins','-sm', is_flag=True, default = False,
            help = 'Flag for allowing sparsity in margins of contingency table')
@click.option('--store_progress','-sp', default = 1.0, show_default=True, 
            type=click.FloatRange(min=0.01,max=1.0),
            help = 'Sets percentage of total samples that will be exported as a batch')
@click.option('--grid_size','-gs', type=click.IntRange(min=1),
            default = None, help = 'Overwrites size of square grid for R^2 and Log Target analyses.')
@click.option('--theta_steps','-pn', type=click.IntRange(min=1),
            default = None, help = 'Overwrites number of Spatial Interaction Model MCMC theta steps in joInt scheme.')
@click.option('--log_destination_attraction_steps','-dan', type=click.IntRange(min=1),
            default = None, help = 'Overwrites number of Spatial Interaction Model MCMC theta steps in joInt scheme.')
@click.option('--table_steps','-tn', type=click.IntRange(min=1),
            default = None, help = 'Overwrites number of Spatial Interaction Model MCMC steps in joInt scheme.')
@click.option('--k','-k', type=click.IntRange(min=1), default = None,
            help = 'Overwrites size of ensemble of datasets for MCMC convergence diagnostic')
@click.option('--table0','-tab0', type=click.Choice(TABLE_SOLVERS), default = None,
            help = 'Overwrites table initialisation method name in MCMC.')
@click.option('--marginal0','-m0', type=click.Choice(MARGINAL_SOLVERS), default = None,
            help = 'Overwrites margin initialisation method name in MCMC.')
@click.option('--alpha0','-alpha0', type=click.FloatRange(min=0), default = None,
            help = 'Overwrites initialisation of alpha parameter in MCMC.')
@click.option('--beta0','-beta0', type=click.FloatRange(min=0), default = None,
            help = 'Overwrites initialisation of beta parameter in MCMC.')
@click.option('--beta_max','-bm', type=click.FloatRange(min=0), default = None,
            help = 'Overwrites maximum beta in SIM parameters.')
@click.option('--delta','-delta', type=click.FloatRange(min=0), default = None,
            help = 'Overwrites delta parameter in production constrained Spatial Interaction Model differential equation.')
@click.option('--kappa','-kappa', type=click.FloatRange(min=0), default = None,
            help = 'Overwrites kappa parameter in production constrained Spatial Interaction Model differential equation.')
@click.option('--covariance','-cov', type=click.STRING, default = None, 
            help = 'Overwrites covariance matrix of parameter Gaussian Randow walk proposal')
@click.option('--step_size','-ss', type=click.FloatRange(min=0), default = None,
            help = 'Overwrites step size in parameter Gaussian Randow walk proposal')
@click.option('--leapfrog_steps','-ls', type=click.IntRange(min=1), default = None,
            help = 'Overwrites number of steps in Leapfrog Integrator in HMC')
@click.option('--leapfrog_step_size','-lss', type=click.FloatRange(min=0), default = None,
            help = 'Overwrites number of step size in Leapfrog Integrator in HMC')
@click.option('--ais_leapfrog_steps','-als', type=click.IntRange(min=1), default = None,
            help = 'Overwrites number of leapfrog steps in AIS HMC proposal (normalising constant sampling)')
@click.option('--ais_leapfrog_step_size','-alss', type=click.FloatRange(min=0), default = None,
            help = 'Overwrites size of leapfrog steps in AIS HMC proposal (normalising constant sampling)')
@click.option('--ais_samples','-as', type=click.IntRange(min=1), default = None,
            help = 'Overwrites number of samples in AIS (normalising constant sampling)')
@click.option('--n_bridging_distributions','-nb', type=click.IntRange(min=1), default = None,
            help = 'Overwrites number of temperatures in tempered distribution in AIS (normalising constant sampling)')
@common_options
def run(config_path,
            load_experiment,
            dataset,
            sim_type,
            origin_demand,
            log_destination_attraction,
            cost_matrix,
            table,
            margins,
            axes,
            cells,
            run_experiments,
            experiment_title,
            seed,
            dims,
            proposal,
            store_progress,
            grid_size,
            sparse_margins,
            theta_steps,
            log_destination_attraction_steps,
            table_steps,
            k,
            table0,
            margin0,
            alpha0,
            beta0,
            beta_max,
            delta,
            kappa,
            covariance,
            step_size,
            leapfrog_steps,
            leapfrog_step_size,
            ais_leapfrog_steps,
            ais_leapfrog_step_size,
            ais_samples,
            n_bridging_distributions,
            logging_mode,
            n_workers,
            n_threads,
            norm,
            n
        ):
    """
    Run Data Augementation Spatial Interaction Model Markov Chain Monte Carlo.
    :param config_path: Configuration file path    
    """

    # Gather all arguments in dictionary
    settings = locals()
    # Remove all nulls
    settings = {k: v for k, v in settings.items() if v}
    # Convert strings to ints
    settings['n_threads'] = [int(thread) for thread in settings.get('n_threads',[1,1])]
    settings['n_workers'] = settings.get('n_workers',1)
    # Update number of workers
    update_numpy_threads(settings['n_threads'][0])

    # Import all modules
    from numpy import asarray
    from ticodm.experiments import ExperimentHandler
    from ticodm.utils import str_in_list,deep_updates,update_numba_threads

    # Set number of cores used (numba package)
    update_numba_threads(settings['n_threads'])

    # Convert covariance to 2x2 array
    if str_in_list('covariance',settings.keys()):
        settings['covariance'] = asarray([float(x) for x in settings['covariance'].split(",")]).reshape((2,2)).tolist()
    # Capitalise all single-letter arguments
    settings = {(key.upper() if len(key) == 1 else key):value for key, value in settings.items()}

    # Read config
    config = Config(config_path)

    # Update settings with overwritten values
    deep_updates(config.settings,settings,overwrite=True)

    # Update root
    config.set_paths_root()
    # Keep experiment ids argument
    if len(run_experiments) > 0:
        config.settings['experiments']["run_experiments"] = list(run_experiments)
    else:
        config.settings['experiments']["run_experiments"] = list(config.settings['experiments'].keys())

    logging.basicConfig(
        level=logging.getLevelName(settings.get('logging_mode','info').upper()),
        # format='%(asctime)s %(name)-12s %(levelname)-3s %(message)s',
        # datefmt='%m-%d %H:%M:%S'
        format='%(asctime)s.%(msecs)03d %(levelname)-3s %(message)s',
        datefmt='%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Create output folder if it does not exist
    if not os.path.exists(config.settings['outputs']['output_path']):
        logger.info(f"Creating new output directory {config.settings['outputs']['output_path']}")
        os.makedirs(config.settings['outputs']['output_path'])

    # Update settings
    # config.settings = config.update_recursively(config.settings,{"inputs":{"generate":{"I":2,"J":2}}},overwrite=True)

    # Intialise experiment handler
    eh = ExperimentHandler(config)

    # Run experiments
    eh.run_and_write_experiments_sequentially()

    logger.info('Done')


_output_options = [
    click.option('--directories','-d', multiple=True, required=False, type=click.Path(exists=True)),
    click.option('--output_directory', '-o', type=click.STRING,cls=NotRequiredIf, default='./data/outputs/', not_required_if='directories'),
    click.option('--dataset_name', '-dn', multiple=True, type=click.STRING,cls=NotRequiredIf, not_required_if='directories'),
    click.option('--experiment_type','-e', multiple=True, type=click.STRING ,cls=NotRequiredIf, not_required_if='directories'),
    click.option('--experiment_title','-et', multiple=True, type=click.STRING, default = [''], cls=NotRequiredIf, not_required_if='directories'),
    click.option('--exclude','-exc', type=click.STRING, default = '',cls=NotRequiredIf, not_required_if='directories'),
    click.option('--filename_ending', '-fe', default='', type=click.STRING),
    click.option('--burnin', '-b', default=0, show_default=True,
                type=click.IntRange(min=0), help=f'Sets number of initial samples to discard.'),
    click.option('--thinning', '-t', default=1, show_default=True,
                type=click.IntRange(min=1), help=f'Sets number of samples to skip.'),
    click.option('--sample', '-s', multiple = True, required = False,
                type=click.Choice(DATA_TYPES.keys()), help=f'Sets type of samples to compute metrics over.'),
    click.option('--statistic','-stat', multiple=True, default=[['','']], type = (click.STRING,click.STRING),required=False,
            help='Every argument corresponds to a list of statistics and their corresponding axes e.g. passing  (\"mean,sum\",  \"0,1\") corresponds to applying mean across axis 0 and then sum across axis 1'),
    click.option('--metric','-m', multiple=True, type=click.Choice(METRICS.keys()), required=False, default=['none'],
                help=f'Sets list of metrics to compute over samples.'),
    click.option('--dates','-dt', type=click.STRING, default=[''], multiple=True, required=False),
                #  type=click.DateTime(formats=DATE_FORMATS), multiple=True, required=False),
    click.option('--epsilon_threshold', '-eps', default=0.001, show_default=True,
        type=click.FLOAT, help=f"Sets error norm threshold below which convergence is achieved. Used only in convergence plots.")
]


def output_options(func):
    for option in reversed(_output_options):
        func = option(func)
    return func

@cli.command(name='plot',context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@output_options
@common_options
@click.option('--plots', '-p', type=click.Choice(list(PLOT_HASHMAP.keys())), multiple=True, required=True,
              help=f"""Sets plotting functions
                    \b
                    {json.dumps(PLOT_HASHMAP,indent=4,separators=(',', ':')).replace('{','').replace('}','')}
                    """)
@click.option('--geometry','-g', multiple=False, type=click.File(), default=None, 
                help="Defines path to geometry geojson for visualising flows on a map.")
@click.option('--origin_geometry_type', '-ogt', default='lsoa', show_default=True,
              type=click.STRING, help=f"Define origin geometry type to plot origin destination data.")
@click.option('--destination_geometry_type', '-dgt', default='msoa', show_default=True,
              type=click.STRING, help=f"Define destination geometry type to plot origin destination data.")
@click.option('--origin_ids', '-oid', default=None, show_default=True, multiple=True,
              type=click.STRING, help=f"Subset of origin ids to use in various plots.")
@click.option('--destination_ids', '-did', default=None, show_default=True, multiple=True,
              type=click.STRING, help=f"Subset of destination ids to use in various plots.")
@click.option('--margin_plot', '-mp', default='destination_demand', show_default=True,
              type=click.Choice(['origin_demand', 'destination_demand']), help=f"Sets margin to plot in table flow plots.")
@click.option('--embedding_method', '-emb', default='isomap', show_default=False,
              type=click.Choice(['isomap','tsne']), help=f"Sets method for embedding tabular data Into lower dimensional manifold.")
@click.option('--n_bins', '-nb', default=100, show_default=True,
              type=click.INT, help=f"Sets number of bins to use in histograms or number of lags in ACF or number of levels in countour plots.")
@click.option('--nearest_neighbours', '-nn', default=100, show_default=True,
              type=click.INT, help=f"Sets number of nearest neigbours in table_distribution_low_dimensional_embedding.")
@click.option('--distance_metric', '-dis', default='edit_distance_degree_one', show_default=True,
              type=click.Choice(DISTANCE_FUNCTIONS), 
              help=f"Sets distance metric for lower dimensional embedding method")
@click.option('--figure_format', '-ff', default='pdf', show_default=True,
              type=click.Choice(['eps', 'png', 'pdf', 'tex']), help=f"Sets figure format.")#, case_sensitive=False)
@click.option('--data_format', '-df', default='dat', show_default=True,
              type=click.Choice(['dat', 'txt']), help=f"Sets figure data format.")#, case_sensitive=False)
@click.option('--figure_size', '-fs', default=(7,5), show_default=True,
              type=(float, float), help=f"Sets figure format.")
@click.option('--main_colormap', '-mc', default='cblue',required=False, show_default=True,
              type=click.STRING, help=f"Sets main colormap (e.g. colormap corresponding to flows).")
@click.option('--aux_colormap', '-ac', multiple=True, required=False, default=["Greens","Blues"],
              type=click.STRING, help=f"Sets auxiliary colormap (e.g. colormap corresponding to margins).")
@click.option('--label_by', '-l', default=None, show_default=False, multiple=True,
              type=click.STRING, help=f"Sets metadata key(s) to label figure by.")
@click.option('--x_label', '-x', default=None, show_default=False,
              type=click.STRING, help=f"Sets x axis label.")
@click.option('--y_label', '-y', default=None, show_default=False,
              type=click.STRING, help=f"Sets y axis label.")
@click.option('--x_limit', '-xl', default=(None,None), show_default=False,
              type=(float, float), help=f"Sets x limits.")
@click.option('--y_limit', '-yl', default=(None,None), show_default=False,
              type=(float, float), help=f"Sets y limits.")
@click.option('--color_segmentation_limits', '-csl', default=(0,1), show_default=False,
              type=(click.FloatRange(0,1), click.FloatRange(0,1)), 
              help=f"Sets main colorbar's color segmentation limits.")
@click.option('--main_colorbar_limit', '-mcl', type=(float, float), help=f"Sets main colorbar limits.")
@click.option('--auxiliary_colorbar_limit', '-acl', type=(float, float), multiple=True, help=f"Sets auxiliary colorbar(s) limits.")
@click.option('--linewidth', '-lw', default=10, show_default=False,
              type=click.INT, help=f"Sets line width in plots.")
@click.option('--figure_title', '-ft', default=None, show_default=False,
              type=click.STRING, help=f"Sets figure title.")
@click.option('--colorbar_title', '-ct', default=None, show_default=False,
              type=click.STRING, help=f"Sets colobar title (if colorbar exists).")
@click.option('--opacity', '-op', default=1.0, show_default=True,
              type=click.FloatRange(0,1), help=f"Sets level of transparency in plot.")
@click.option('--axis_font_size', '-afs', default=13, show_default=True,
              type=click.INT, help=f"Sets axis font size.")
@click.option('--tick_font_size', '-tfs', default=13, show_default=True,
              type=click.INT, help=f"Sets axis tick or colorbar font size.")
@click.option('--axis_labelpad', '-alp', default=7, show_default=True,
              type=click.INT, help=f"Sets axis label padding.")
@click.option('--axis_label_rotation', '-alr', default=0, show_default=True,
              type=click.INT, help=f"Sets axis label rotation.")
@click.option('--colorbar_labelpad', '-clp', default=7, show_default=True,
              type=click.INT, help=f"Sets colorbar label padding.")
@click.option('--colorbar_label_rotation', '-clr', default=0, show_default=True,
              type=click.INT, help=f"Sets colorbar label rotation.")
@click.option('--title_label_size', '-tls', default=16, show_default=True,
              type=click.INT, help=f"Sets title font size.")
@click.option('--legend_label_size', '-lls', default=None, show_default=False,
              type=click.INT, help=f"Sets legend font size.")
@click.option('--annotation_label_size', '-anls', default=15, show_default=False,
              type=click.INT, help=f"Sets text annotation font size.")
@click.option('--axis_label_size', '-als', default=15, show_default=False,
              type=click.INT, help=f"Sets plot labels font size.")
@click.option('--x_tick_frequency', '-xfq', default=20, show_default=True,
              type=click.INT, help=f"Sets frequency of x ticks font size.")
@click.option('--marker_frequency','-mf', default = None, show_default = False,
            type=click.INT, help="Plots marker every n-th poInt in dataset")
@click.option('--marker_size','-ms', default = 1, show_default = True,
            type=click.INT, help="Sets marker size in plot")
@click.option('--benchmark/--no-benchmark', '-bm', default=False, is_flag=True, show_default=True,
              help=f"Flag for plotting data along with benchmark/baseline (if provided).")
@click.option('--annotate/--no-annotate', default=False, is_flag=True, show_default=True,
              help=f"Flag for annotating plot with text")
@click.option('--transpose/--no-transpose', default=False, is_flag=True, show_default=True,
              help=f"Flag for taking switching origins with destinations in plots.")
@click.option('--colorbar/--no-colorbar', default=True, is_flag=True, show_default=True,
              help=f"Flag for plotting colorbars or not.")
@click.pass_context
def plot(
        ctx,
        directories,
        output_directory,
        dataset_name,
        experiment_type,
        dates,
        epsilon_threshold,
        filename_ending,
        thinning,
        burnin,
        sample,
        statistic,
        metric,
        table,
        experiment_title,
        exclude,
        logging_mode,
        n_workers,
        n_threads,
        norm,
        n,
        plots,
        geometry,
        origin_geometry_type,
        destination_geometry_type,
        origin_ids,
        destination_ids,
        margin_plot,
        embedding_method,
        n_bins,
        nearest_neighbours,
        distance_metric,
        figure_format,
        data_format,
        figure_size,
        main_colormap,
        aux_colormap,
        label_by,
        x_label,
        y_label,
        x_limit,
        y_limit,
        color_segmentation_limits,
        main_colorbar_limit,
        auxiliary_colorbar_limit,
        linewidth,
        opacity,
        figure_title,
        colorbar_title,
        axis_labelpad,
        colorbar_labelpad,
        axis_label_rotation,
        colorbar_label_rotation,
        title_label_size,
        legend_label_size,
        annotation_label_size,
        axis_label_size,
        axis_font_size,
        tick_font_size,
        marker_size,
        x_tick_frequency,
        marker_frequency,
        benchmark,
        annotate,
        transpose,
        colorbar
    ):
    """
    Postprocess and plot ticodm experimental outputs.
    :param experiment_outputs_path: Path to experimental outputs directory
    """

    # Gather all options in dictionary
    settings = locals()
    # Add context arguments
    undefined_settings = {ctx.args[i][2:]: ctx.args[i+1] for i in range(0, len(ctx.args), 2)}

    # Capitalise all single-letter arguments
    settings = {(key.upper() if len(key) == 1 else key):value for key, value in settings.items()}

    # Convert strings to ints
    settings['n_threads'] = [thread for thread in settings.get('n_threads',[1,1])]
    settings['n_workers'] = settings.get('n_workers',1)
    # Update number of workers
    update_numpy_threads(settings['n_threads'][0])
    
    # Import modules
    from ticodm.plot import Plot
    from ticodm.utils import update_numba_threads,deep_updates

    # Add undefined settings to settings
    settings = {**settings, **undefined_settings}

    # Set number of cores used (numba package)
    update_numba_threads(settings['n_threads'])

    logging.basicConfig(
        level=logging.getLevelName(settings.get('logging_mode','info').upper()),
        # format='%(asctime)s %(name)-12s %(levelname)-3s %(message)s',
        # datefmt='%m-%d %H:%M:%S'
        format='%(asctime)s.%(msecs)03d %(levelname)-3s %(message)s',
        datefmt='%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Validate passed plots
    for c in plots:
        if c not in list(PLOT_HASHMAP.keys()):
            raise click.BadOptionUsage("%s is not an available plot." % c)

    logger.info('Starting')
    # Run plot
    Plot(
        plot_ids=plots,
        outputs_directories=list(directories),
        settings=settings
    )

    logger.info('Done')

@cli.command('summarise')
@output_options
@common_options
@click.option("--metadata_keys","-k", multiple=True, type=click.STRING, required=False)
@click.option('--region_mass', '-r', default=[0.95], show_default=True, multiple=True,
              type=click.FloatRange(0,1), help=f"Sets high posterior density region mass.")
@click.option('--algorithm', '-a', default=['linear'], show_default=True, multiple=True,
              type=click.STRING, help=f"Sets algorihm name for use in.")
@click.option("--sort_by","-sort", multiple=True, type=click.STRING, required=False)
@click.option("--ascending","-asc", default=False, is_flag=True, show_default=True, required=False)
def summarise(
        directories,
        output_directory,
        dataset_name,
        experiment_type,
        epsilon_threshold,
        dates,
        filename_ending,
        thinning,
        burnin,
        sample,
        metric,
        statistic,
        table,
        experiment_title,
        exclude,
        logging_mode,
        n_workers,
        n_threads,
        norm,
        n,
        metadata_keys,
        region_mass,
        algorithm,
        sort_by,
        ascending,
    ):
    """
    
    Compute metrics for ticodm experimental outputs in tabular format.
    :param experiment_outputs_path: Path to experimental outputs directory
    """
    # Gather all options in dictionary
    settings = locals()
    settings['figure_format'] = ''
    # Capitalise all single-letter arguments
    settings = {(key.upper() if len(key) == 1 else key):value for key, value in settings.items()}

    # Convert strings to ints
    settings['n_threads'] = [thread for thread in settings.get('n_threads',[1,1])]
    settings['n_workers'] = settings.get('n_workers',1)
    # Update number of workers
    update_numpy_threads(settings['n_threads'][0])

    # Import modules
    from ticodm.outputs import OutputSummary
    from ticodm.utils import update_numba_threads

    # Set number of cores used (numba package)
    update_numba_threads(settings['n_threads'])

    logging.basicConfig(
        level=logging.getLevelName(settings.get('logging_mode','info').upper()),
        # format='%(asctime)s %(name)-12s %(levelname)-3s %(message)s',
        # datefmt='%m-%d %H:%M:%S'
        format='%(asctime)s.%(msecs)03d %(levelname)-3s %(message)s',
        datefmt='%M:%S'
    )
    logger = logging.getLogger(__name__)

    logger.info('Gathering data')


    # Run output handler
    OutputSummary(
        settings=settings
    )

    logger.info('Done')