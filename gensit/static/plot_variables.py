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

PLOT_VARIABLES = PLOT_ALL_COORDINATES + list(PLOT_CORE_FEATURES.keys())
PLOT_COORDINATES_AND_CORE_FEATURES = PLOT_ALL_COORDINATES + list(PLOT_CORE_FEATURES.keys())
PLOT_VARIABLES_AND_DERIVATIVES = PLOT_VARIABLES + PLOT_COORDINATE_DERIVATIVES

# Type of plots
PLOT_VIEWS = {
        "simple":"plot_simple",
        "spatial":"plot_spatial",
        "tabular":"plot_tabular"
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
    "silver": "#C0C0C0",
    "deepred": "#E20000",
    "skyblue": "#1E88E5",
    "tab20b_purple": "#5856c4",
    "tab20b_green": "#a6c858",
    "tab20b_orange": "#e0ad41",
    "tab20b_red": "#ca4a58",
    "tab20b_pink": "#c153af",
    "tab20c_blue": "#8ebeda"
}

PLOT_COLOURS = {
    "type": {
        "SIM_MCMC": COLOR_NAMES["tab20b_purple"],
        "SIM_NN": COLOR_NAMES["tab20b_orange"],
        "JointTableSIM_MCMC":COLOR_NAMES["tab20b_green"],
        "NonJointTableSIM_NN":COLOR_NAMES["tab20b_red"],
        "JointTableSIM_NN":COLOR_NAMES["tab20c_blue"],
    },
    "title": {
        "_unconstrained":COLOR_NAMES["silver"],
        "_total_constrained":COLOR_NAMES["tab20b_purple"],
        "_row_constrained":COLOR_NAMES["tab20b_green"],
        "_doubly_constrained":COLOR_NAMES["tab20b_orange"],
        "_doubly_10%_cell_constrained":COLOR_NAMES["tab20b_red"],
        "_doubly_20%_cell_constrained":COLOR_NAMES["tab20c_blue"],
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
    "SIM_MCMC": '\\ellamframeworktag',
    "JointTableSIM_MCMC": '\\zachosframeworktag',
    "SIM_NN": '\\gaskinframeworktag',
    "JointTableSIM_NN": 'Joint (\\frameworktag)',
    "NonJointTableSIM_NN": 'Disjoint (\\frameworktag)',
    "_unconstrained":'$\\emptyset$',
    "_total_constrained":'$T_{++}$',
    "_total_intensity_row_table_constrained":'$\\mathbf{T}_{+\\cdot}$',
    "_row_constrained":'$\\mathbf{T}_{+\\cdot}$',
    "_doubly_constrained":'$\\mathbf{T}_{+\\cdot},\\mathbf{T}_{\\cdot +}$',
    "_doubly_10%_cell_constrained":'$\\mathbf{T}_{+\\cdot},\\mathbf{T}_{\\cdot +},\\mathbf{T}_{\\mathcal{X}_1}$',
    "_doubly_20%_cell_constrained":'$\\mathbf{T}_{+\\cdot},\\mathbf{T}_{\\cdot +},\\mathbf{T}_{\\mathcal{X}_2}$',
    "TotallyConstrained":'$\\Lambda_{++}$',
    "ProductionConstrained":'$\\boldsymbol{\\Lambda}_{+\\cdot}$',
    "dest_attraction_ts_likelihood_loss":"$\\lossoperator\\left(\\mathbf{x} \\; ; \\; \\obsdata, \\boldsymbol{\\nu} \\right)$",
    "dest_attraction_ts_likelihood_loss,table_likelihood_loss":"$\\lossoperator\\left(\\mathbf{x}, \\mytable, \\myintensity \\; ; \\; \\obsdata, \\boldsymbol{\\nu} \\right)$",
    "dest_attraction_ts_likelihood_loss,total_intensity_distance_likelihood_loss,table_likelihood_loss":"$\\lossoperator\\left(\\mathbf{x},\\mytable,\\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \\right)$",
    "dest_attraction_ts_likelihood_loss,total_intensity_distance_likelihood_loss":"$\\lossoperator\\left( \\mathbf{x}, \\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \\right)$",
    "dest_attraction_ts_likelihood_loss,total_table_distance_likelihood_loss,table_likelihood_loss":"$\\lossoperator\\left(\\mathbf{x},\\mytable,\\myintensity \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \\right)$",
    "dest_attraction_ts_likelihood_loss,total_table_distance_likelihood_loss":"$\\lossoperator\\left( \\mathbf{x}, \\mytable \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \\right)$",
    "table_likelihood_loss":"$\\lossoperator\\left(\\mytable, \\myintensity \\right)$",
    "total_intensity_distance_likelihood_loss,table_likelihood_loss":"$\\lossoperator\\left(\\mytable,\\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \\right)$",
    "total_intensity_distance_likelihood_loss":"$\\lossoperator\\left(\\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \\right)$",
    "total_intensity_distance_loss":"$\\lossoperator\\left(\\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \\right)$",
    "total_table_distance_likelihood_loss,table_likelihood_loss":"$\\lossoperator\\left(\\mytable,\\myintensity \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \\right)$",
    "total_table_distance_likelihood_loss":"$\\lossoperator\\left(\\mytable \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \\right)$",
    "total_table_distance_loss":"$\\lossoperator\\left(\\mytable \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \\right)$"
}
LABEL_EXPRESSIONS = {
    "sigma": '$\\sigma = '
}
LATEX_EXPRESSIONS = {**RAW_EXPRESSIONS,**LABEL_EXPRESSIONS}
LATEX_COORDINATES = ['label','annotate']+PLOT_COORDINATES+PLOT_AUX_COORDINATES

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

\newcommand{\zachosframeworkbasename}{SIMT-MCMC}
\newcommand{\gaskinframeworkbasename}{SIM-NN}
\newcommand{\ellamframeworkbasename}{SIM-MCMC}
\newcommand{\zachosframework}{\zachosframeworkbasename \;}
\newcommand{\gaskinframework}{\gaskinframeworkbasename \;}
\newcommand{\ellamframework}{\ellamframeworkbasename \;}
\newcommand{\zachosframeworktag}{\textsc{\zachosframeworkbasename}}
\newcommand{\gaskinframeworktag}{\textsc{\gaskinframeworkbasename}}
\newcommand{\ellamframeworktag}{\textsc{\ellamframeworkbasename}}
\newcommand{\zachosframeworktagspace}{\textsc{\zachosframeworkbasename}\;}
\newcommand{\gaskinframeworktagspace}{\textsc{\gaskinframeworkbasename}\;}
\newcommand{\ellamframeworktagspace}{\textsc{\ellamframeworkbasename}\;}

\newcommand{\frameworkname}{GeNSIT}
\newcommand{\frameworktag}{\textsc{\frameworkname}}
\newcommand{\frameworkpackage}{\texttt{gensit}}
\newcommand{\frameworkfullname}{\textbf{Ge}nerating \textbf{N}eural \textbf{S}patial \textbf{I}nteraction \textbf{T}ables \;}
\newcommand{\frameworknamespace}{\frameworkname\;}
\newcommand{\frameworktagspace}{\textsc{\frameworktag\;}}
\newcommand{\frameworkpackagespace}{\frameworkpackage \;}
\newcommand{\mytable}{\skyblue{\mathbf{T}}}
\newcommand{\myintensity}{\deepred{\boldsymbol{\Lambda}}}
\newcommand{\mytableoned}{\skyblue{T}}
\newcommand{\myintensityoned}{\deepred{\Lambda}}
\newcommand{\mytablerowsums}{\myseagreen{\mathbf{T}_{\cdot +}}}
\newcommand{\mytablecolsums}{\myorange{\mathbf{T}_{+\cdot}}}
\newcommand{\mytablerowsumsoned}{\myseagreen{T_{i+}}}
\newcommand{\mytablecolsumsoned}{\myorange{T_{+j}}}
\newcommand{\mytabletotal}{\mydarkmagenta{T_{++}}}
\newcommand{\mytablecells}{\mybrown{\mathbf{T}_{\mathcal{X}}}}
\newcommand{\myintensityrowsums}{\myseagreen{\boldsymbol{\Lambda}_{\cdot +}}}
\newcommand{\myintensitycolsums}{\myorange{\boldsymbol{\Lambda}_{+\cdot}}}
\newcommand{\myintensityrowsumsoned}{\myseagreen{\Lambda_{i+}}}
\newcommand{\myintensitycolsumsoned}{\myorange{\Lambda_{+j}}}
\newcommand{\myintensitytotal}{\mydarkmagenta{\Lambda_{++}}}
\newcommand{\tableconstraints}{\mathcal{C}_{\mytableoned}}
\newcommand{\intensityconstraints}{\mathcal{C}_{\myintensityoned}}
\newcommand{\allconstraints}{\mathcal{C}}
\newcommand{\lossoperator}{\mathcal{L}}
\newcommand{\obsdata}{\mathcal{D}}
\newcommand{\groundtruthtable}{\mytable^{\obsdata}}
'''

# "pgf.texsystem": "xelatex",
# "pgf.rcfonts": False,
LATEX_RC_PARAMETERS = {
    'font.serif': ['Lucida Sans'],
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.preamble': LATEX_PREAMBLE,
}

LEGEND_LOCATIONS = [
    'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 
    'center left', 'center right', 'lower center', 'upper center', 'center'
]