import matplotlib.cm as cm

from matplotlib.colors import LinearSegmentedColormap

def cmap_exists(name):
    try:
        cm.get_cmap(name)
    except:
        return False
    return True


PLOT_COORDINATES = ["x","y","z"]
PLOT_AUX_COORDINATES = ["x_group","y_group","z_group"]
PLOT_ALL_COORDINATES = PLOT_COORDINATES+PLOT_AUX_COORDINATES

PLOT_CORE_FEATURES = {
    "marker_size": {"dtype": float},
    "line_width":{"dtype": float},
    "line_style": {"dtype": str},
    "colour": {"dtype": str},
    "opacity": {"dtype": float},
    "hatch_opacity": {"dtype": float},
    "zorder": {"dtype": float},
    "annotate": {"dtype": str},
    "label": {"dtype": str},
    "marker": {"dtype": str},
    "hatch": {"dtype": str},
}

PLOT_DERIVATIVES = ["_id"]

PLOT_COORDINATE_DERIVATIVES = []
for var in PLOT_COORDINATES:
    for derivative in PLOT_DERIVATIVES:
        PLOT_COORDINATE_DERIVATIVES.append(var+derivative)

PLOT_VARIABLES = PLOT_ALL_COORDINATES +\
    list(PLOT_CORE_FEATURES.keys())
    
PLOT_VARIABLES_AND_DERIVATIVES = PLOT_VARIABLES + PLOT_COORDINATE_DERIVATIVES

# Type of plots
PLOT_VIEWS = {
        "2d":"plot_2d",
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

PLOT_TYPES = {
    "empty":"",
    "line": "plot",
    "scatter" : "scatter"
}

PLOT_MARKERS = {
    "sample_name": {
        "else":".",
        "table": "s",
        "intensity":"^",
    },
    "type": {
        "else":".",
        "SIM_MCMC": "o",
        "JointTableSIM_MCMC": "s",
        "SIM_NN": "^",
        "NonJointTableSIM_NN": "P",
        "JointTableSIM_NN": "*",
    },
    "sigma":{
        "else":">",
        "low":"v",
        "high":"^"
    }
}

COLOR_NAMES = {
    "yellow": "#F5DDA9",
    "darkblue": "#2F7194",
    "red": "#ec7070",
    "deepred": "#E20000",
    "skyblue": "#1E88E5",
    "lightblue": "#99dfff",
    "green": "#3cc969",
    "darkgreen": "#48675A",
    "darkmagenta": "#8B008B",
    "seagreen": "#2E8B57",
    "lightbrown": "#C6BFA2",
    "brown": "#A52A2A",
    "orange": "#FFA500",
    "lightgreen": "#AFD8BC",
    "grey": "#3D4244"
}

PLOT_COLOURS = {
    "type": {
        "SIM_MCMC": COLOR_NAMES["lightblue"],
        "SIM_NN": COLOR_NAMES["lightgreen"],
        "JointTableSIM_MCMC":COLOR_NAMES["darkblue"],
        "NonJointTableSIM_NN":COLOR_NAMES["green"],
        "JointTableSIM_NN":COLOR_NAMES["darkgreen"],
    },
    "title": {
        "_unconstrained":COLOR_NAMES["orange"],
        "_total_constrained":COLOR_NAMES["darkmagenta"],
        "_row_constrained":COLOR_NAMES["seagreen"],
        "_doubly_constrained":COLOR_NAMES["red"],
        "_doubly_10%_cell_constrained":COLOR_NAMES["darkgreen"],
        "_doubly_20%_cell_constrained":COLOR_NAMES["darkblue"],
    }
}

PLOT_HATCHES = {
    "sigma": {
        "else": "***",
        "high": "+++",
        "low": "OOO"
    },
    "type": {
        "else":".",
        "SIM_MCMC": "oo",
        "JointTableSIM_MCMC": "..",
        "SIM_NN": "-",
        "NonJointTableSIM_NN": "++",
        "JointTableSIM_NN": "**",
    }
}

PLOT_LINESTYLES = {}

RAW_EXPRESSIONS = {
    "SIM_MCMC": 'S-MCMC',
    "JointTableSIM_MCMC": 'TS-MCMC',
    "SIM_NN": 'S-NN',
    "JointTableSIM_NN": 'Ours (Joint)',
    "NonJointTableSIM_NN": 'Ours (Disjoint)',
    "_unconstrained":'$\\emptyset$',
    "_total_constrained":'$T_{++}$',
    "_total_intensity_row_table_constrained":'$\\mathbf{T}_{+\\cdot}$',
    "_row_constrained":'$\\mathbf{T}_{+\\cdot}$',
    "_doubly_constrained":'$\\mathbf{T}_{+\\cdot},\\mathbf{T}_{\\cdot +}$',
    "_doubly_10%_cell_constrained":'$\\mathbf{T}_{+\\cdot},\\mathbf{T}_{\\cdot +},\\mathbf{T}_{\\mathcal{X}_1}$',
    "_doubly_20%_cell_constrained":'$\\mathbf{T}_{+\\cdot},\\mathbf{T}_{\\cdot +},\\mathbf{T}_{\\mathcal{X}_2}$',
    "TotallyConstrained":'$\\Lambda_{++}$',
    "ProductionConstrained":'$\\boldsymbol{\\Lambda}_{+\\cdot}$',
    "['dest_attraction_ts_likelihood_loss','table_likelihood_loss']":'$L\\left(\\mytable,\\myintensity,\\mathbf{x}; \\mathbf{y}, \\boldsymbol{\\sigma}_d\\right)$',
    "['dest_attraction_ts_likelihood_loss','total_intensity_distance_likelihood_loss','table_likelihood_loss']":'$L\\left(\\mytable,\\myintensity,\\mathbf{x}; \\mathbf{y}, \\mathbf{D}^{\\myintenisty}_{\cdot+}, \\boldsymbol{\\sigma}_d\\right)$',
    "['dest_attraction_ts_likelihood_loss','total_intensity_distance_likelihood_loss']":'$L\\left(\\myintensity,\\mathbf{x}; \\mathbf{y}, \\mathbf{D}^{\\myintenisty}_{\cdot+}, \\boldsymbol{\\sigma}_d\\right)$',
    "['dest_attraction_ts_likelihood_loss','total_table_distance_likelihood_loss','table_likelihood_loss']":'$L\\left(\\mytable,\\myintensity,\\mathbf{x}; \\mathbf{y}, \\mathbf{D}^{\\mytable}_{\cdot+}, \\boldsymbol{\\sigma}_d\\right)$',
    "['dest_attraction_ts_likelihood_loss','total_table_distance_likelihood_loss']":'$L\\left(\\mytable,\\mathbf{x}; \\mathbf{y}, \\mathbf{D}^{\\mytable}_{\cdot+}, \\boldsymbol{\\sigma}_d \\right)$',
    "['table_likelihood_loss']":'$L\\left(\\mytable,\\myintensity\\right)$',
    "['total_intensity_distance_likelihood_loss','table_likelihood_loss']":'$L\\left(\\mytable,\\myintensity; \\mathbf{D}^{\\mytable}_{\cdot+}, \\boldsymbol{\\sigma}_d\\right)$',
    "['total_intensity_distance_likelihood_loss']":'$L\\left(\\myintensity; \\mathbf{D}^{\\myintensity}_{\cdot+}, \\boldsymbol{\\sigma}_d \\right)$',
    "['total_intensity_distance_loss']":'$L\\left(\\myintensity; \\mathbf{D}^{\\myintensity}_{\cdot+}\\right)$',
    "['total_table_distance_likelihood_loss','table_likelihood_loss']":'$L\\left(\\mytable,\\myintensity; \\mathbf{D}^{\\mytable}_{\cdot+}, \\boldsymbol{\\sigma}_d \\right)$',
    "['total_table_distance_likelihood_loss']":'$L\\left(\\mytable; \\mathbf{D}^{\\mytable}_{\cdot+}, \\boldsymbol{\\sigma}_d \\right)$',
    "['total_table_distance_loss']":'$L\\left(\\mytable; \\mathbf{D}^{\\mytable}_{\cdot+} \\right)$'
}
LABEL_EXPRESSIONS = {
    "sigma": '$\\sigma = '
}
LATEX_EXPRESSIONS = {**RAW_EXPRESSIONS,**LABEL_EXPRESSIONS}
LATEX_COORDINATES = ['label','annotate']+PLOT_COORDINATES


# Register colormaps
bluegreen = LinearSegmentedColormap.from_list(
        "bluegreen",
        list(
            zip(
                [0.0,0.2,0.4,0.6,0.8,1.0],
                [
                    (24./255.,8./255.,163./255.,1),
                    (7./255.,69./255.,142./255.,1),
                    (5./255.,105./255.,105./255.,1),
                    (7./255.,137./255.,66./255.,1),
                    (15./255.,168./255.,8./255.,1),
                    (97./255.,193./255.,9./255.,1),
                ]
            )
        ),
        N = 256
)
bluegreen.set_bad((0,0,0,1))
if not cmap_exists("bluegreen"):
    cm.register_cmap(cmap = bluegreen, name="bluegreen")

yellowpurple = LinearSegmentedColormap.from_list(
        "yellowpurple",
        list(
            zip(
                [0.0,0.14,0.29,0.43,0.57,0.71,0.86,1.0],
                [
                    (13./255.,8./255.,135./255.,1),
                    (84./255.,2./255.,163./255.,1),
                    (139./255.,10./255.,165./255.,1),
                    (185./255.,50./255.,137./255.,1),
                    (219./255.,92./255.,104./255.,1),
                    (244./255.,136./255.,73./255.,1),
                    (254./255.,188./255.,43./255.,1),
                    (240./255.,249./255.,33./255.,1),
                ]
            )
        ),
        N = 256
)
yellowpurple.set_bad((0,0,0,1))
if not cmap_exists("yellowpurple"):
    cm.register_cmap(cmap = yellowpurple, name="yellowpurple")



yellowblue = LinearSegmentedColormap.from_list(
        "yellowblue",
        list(
            zip(
                # ["orange", "yellow", "lightyellow", "white", "lightblue", "blue", "darkblue"]
                [0.0,0.1,0.3,0.5,0.6,0.9,1.0],
                [(255./255.,165./255.,0./255.,1),
                 (255./255.,255./255.,224./255.,1),
                 (255./255.,255./255.,255./255.,1),
                 (173./255.,216./255.,230./255.,1),
                 (0./255.,0./255.,255./255.,1),
                 (0./255.,0./255.,139./255.,1)]
            )
        ),
        N = 256
)
if not cmap_exists("yellowblue"):
    cm.register_cmap(cmap = yellowblue, name="yellowblue")


redgreen = LinearSegmentedColormap.from_list(
        "redgreen",
        list(
            zip(
                # ["darkred","red","lightcoral","white","palegreen","green","darkgreen"]
                [0.0,0.1,0.3,0.5,0.6,0.9,1.0],
                [(139./255.,0./255.,0./255.,1),
                 (255./255.,0./255.,0./255.,1),
                 (240./255.,128./255.,128./255.,1),
                 (255./255.,255./255.,255./255.,1),
                 (152./255.,251./255.,152./255.,1),
                 (0./255.,128./255.,0./255.,1),
                 (0./255.,100./255.,0./255.,1)]
            )
        ),
        N = 256
)
if not cmap_exists("redgreen"):
    cm.register_cmap(cmap = redgreen, name="redgreen")

cblue = LinearSegmentedColormap.from_list(
        "cblue", 
        [
            (255./255.,255/255.,255./255.),
            (0./255.,120/255.,255./255.),
        ]
)
if not cmap_exists("cblue"):
    cm.register_cmap(cmap = cblue, name="cblue")

cgreen = LinearSegmentedColormap.from_list(
        "cgreen", 
        [
            (255./255.,255./255.,255./255.),
            (0./255.,255./255.,0./255.)
        ]
)
if not cmap_exists("cgreen"):
    cm.register_cmap(cmap = cgreen, name="cgreen")

cred = LinearSegmentedColormap.from_list(
    "cred", 
    [
        (255./255.,255./255.,255./255.),
        (255./255.,0/255.,0./255.),
    ]
)
if not cmap_exists("cred"):
    cm.register_cmap(cmap = cred, name="cred")

LATEX_PREAMBLE = r'''
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{amscd,dsfont}
\usepackage{commath}
\usepackage{xcolor}
\usepackage{color}

\newcommand{\skyblue}[1]{{\textcolor[HTML]{1E88E5}{#1}}}
\newcommand{\deepred}[1]{{\textcolor[HTML]{E20000}{#1}}}
\newcommand{\myseagreen}[1]{{\textcolor[HTML]{2E8B57}{#1}}}
\newcommand{\myorange}[1]{{\textcolor[HTML]{FFA500}{#1}}}
\newcommand{\mydarkmagenta}[1]{{\textcolor[HTML]{8B008B}{#1}}}
\newcommand{\mybrown}[1]{{\textcolor[HTML]{A52A2A}{#1}}}

\newcommand{\frameworkname}{ConTINePI}
\newcommand{\frameworkfullname}{\textbf{Con}strained \textbf{T}able \textbf{I}nference with \textbf{Ne}ural Calibration of a \textbf{P}hysics-Driven \textbf{I}ntensity}
\newcommand{\frameworkpackagelowercase}{continepi}
\newcommand{\frameworktag}{\textsc{\frameworkname}}
\newcommand{\frameworkpackage}{\texttt{\frameworkname}}
\newcommand{\frameworkfig}{\hyperref[fig:framework]{\textsc{\frameworkname}}}
\newcommand{\mytable}{\skyblue{\mathbf{T}}}
\newcommand{\myintensity}{\deepred{\boldsymbol{\Lambda}}}
\newcommand{\mytableoned}{\skyblue{T}}
\newcommand{\myintensityoned}{\deepred{\Lambda}}
\newcommand{\mytablerowsums}{\myseagreen{\mathbf{T}_{\cdot +}}}
\newcommand{\mytablecolsums}{\myorange{\mathbf{T}_{+\cdot}}}
\newcommand{\mytabletotal}{\mydarkmagenta{T_{++}}}
\newcommand{\mytablecells}{\mybrown{\mathbf{T}_{\mathcal{X}}}}
\newcommand{\myintensityrowsums}{\myseagreen{\boldsymbol{\Lambda}_{\cdot +}}}
\newcommand{\myintensitycolsums}{\myorange{\boldsymbol{\Lambda}_{+\cdot}}}
\newcommand{\myintensitytotal}{\mydarkmagenta{\Lambda_{++}}}
'''

# "pgf.texsystem": "xelatex",
# "pgf.rcfonts": False,
LATEX_RC_PARAMETERS = {
    'font.serif': ['Times New Roman'],
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.preamble': LATEX_PREAMBLE,
}

LEGEND_LOCATIONS = [
    'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 
    'center left', 'center right', 'lower center', 'upper center', 'center'
]