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
    "scatter" : "scatter",
    "imshow": "imshow"
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
        "_doubly_10#%_cell_constrained":COLOR_NAMES["tab20b_red"],
        "_doubly_20#%_cell_constrained":COLOR_NAMES["tab20c_blue"],
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
    "_total_intensity_row_table_constrained":'$\\mytablerowsums$',
    "_row_constrained":'$\\mytablerowsums$',
    "_doubly_constrained":'$\\mytablerowsums,\\mytablecolsums$',
    "_doubly_10#%_cell_constrained":'$\\mytablerowsums,\\mytablecolsums,\\mytablecells{1}$',
    "_doubly_20#%_cell_constrained":'$\\mytablerowsums,\\mytablecolsums,\\mytablecells{2}$',
    "TotallyConstrained":'$\\myintensitytotal$',
    "ProductionConstrained":'$\myintensityrowsums$',
    "dest_attraction_ts_likelihood_loss":"$\\lossoperator\(\\mylogdestattr \\; ; \\; \\obsdata, \\boldsymbol{\\nu} \)$",
    "dest_attraction_ts_likelihood_loss,table_likelihood_loss":"$\\lossoperator\(\\mylogdestattr, \\mytable, \\myintensity \\; ; \\; \\obsdata, \\boldsymbol{\\nu} \)$",
    "dest_attraction_ts_likelihood_loss,total_intensity_distance_likelihood_loss,table_likelihood_loss":"$\\lossoperator\(\\mylogdestattr,\\mytable,\\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \)$",
    "dest_attraction_ts_likelihood_loss,total_intensity_distance_likelihood_loss":"$\\lossoperator\( \\mylogdestattr, \\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \)$",
    "dest_attraction_ts_likelihood_loss,total_table_distance_likelihood_loss,table_likelihood_loss":"$\\lossoperator\(\\mylogdestattr,\\mytable,\\myintensity \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \)$",
    "dest_attraction_ts_likelihood_loss,total_table_distance_likelihood_loss":"$\\lossoperator\( \\mylogdestattr, \\mytable \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \)$",
    "table_likelihood_loss":"$\\lossoperator\(\\mytable, \\myintensity \)$",
    "total_intensity_distance_likelihood_loss,table_likelihood_loss":"$\\lossoperator\(\\mytable,\\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \)$",
    "total_intensity_distance_likelihood_loss":"$\\lossoperator\(\\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \)$",
    "total_intensity_distance_loss":"$\\lossoperator\(\\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \)$",
    "total_table_distance_likelihood_loss,table_likelihood_loss":"$\\lossoperator\(\\mytable,\\myintensity \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \)$",
    "total_table_distance_likelihood_loss":"$\\lossoperator\(\\mytable \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \)$",
    "total_table_distance_loss":"$\\lossoperator\(\\mytable \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \)$"
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

\newcommand{\mysilver}[1]{{\textcolor[HTML]{909090}{#1}}}
\newcommand{\skyblue}[1]{{\textcolor[HTML]{1E88E5}{#1}}}
\newcommand{\deepred}[1]{{\textcolor[HTML]{E20000}{#1}}}
\newcommand{\myseagreen}[1]{{\textcolor[HTML]{2E8B57}{#1}}}
\newcommand{\myorange}[1]{{\textcolor[HTML]{FFA500}{#1}}}
\newcommand{\mydarkmagenta}[1]{{\textcolor[HTML]{8B008B}{#1}}}
\newcommand{\mybrown}[1]{{\textcolor[HTML]{A52A2A}{#1}}}
\newcommand{\tabcmpurple}[1]{{\textcolor[HTML]{5856c4}{#1}}}
\newcommand{\tabcmgreen}[1]{{\textcolor[HTML]{a6c858}{#1}}}
\newcommand{\tabcmorange}[1]{{\textcolor[HTML]{e0ad41}{#1}}}
\newcommand{\tabcmred}[1]{{\textcolor[HTML]{ca4a58}{#1}}}
\newcommand{\tabcmpink}[1]{{\textcolor[HTML]{c153af}{#1}}}
\newcommand{\tabcmblue}[1]{{\textcolor[HTML]{8ebeda}{#1}}}

\newcommand{\zachosframeworkbasename}{SIT-MCMC}
\newcommand{\gaskinframeworkbasename}{SIM-NN}
\newcommand{\ellamframeworkbasename}{SIM-MCMC}
\newcommand{\zachosframeworkfullname}{\textbf{S}patial \textbf{I}nteraction \textbf{T}able \textbf{M}arkov \textbf{C}hain \textbf{M}onte \textbf{C}arlo \;}
\newcommand{\gaskinframeworkfullname}{\textbf{S}patial \textbf{I}nteraction \textbf{M}odel \textbf{N}eural \textbf{N}etwork\;}
\newcommand{\ellamframeworkfullname}{\textbf{S}patial \textbf{I}nteraction \textbf{M}odel \textbf{M}arkov \textbf{C}hain \textbf{M}onte \textbf{C}arlo \;}
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
\newcommand{\frameworkfig}{\hyperref[fig:framework]{\frameworktag}}
\newcommand{\frameworkfullname}{\textbf{Ge}nerating \textbf{N}eural \textbf{S}patial \textbf{I}nteraction \textbf{T}ables}
\newcommand{\frameworknamespace}{\frameworkname\;}
\newcommand{\frameworktagspace}{\textsc{\frameworktag\;}}
\newcommand{\frameworkpackagespace}{\frameworkpackage \;}
\newcommand{\frameworkfigspace}{\hyperref[fig:framework]{\frameworktagspace}}

\newcommand{\tablecolour}[1]{#1}
\newcommand{\intensitycolour}[1]{#1}
\newcommand{\constraintcolour}[1]{\deepred{#1}}
\newcommand{\rowsumscolour}[1]{\deepred{#1}}
\newcommand{\colsumscolour}[1]{\deepred{#1}}
\newcommand{\totalcolour}[1]{\deepred{#1}}
\newcommand{\cellcolour}[1]{\deepred{#1}}
\newcommand{\uncostrainedcolour}[1]{#1}
\newcommand{\totallyconstrainedcolour}[1]{#1}
\newcommand{\singlyconstrainedcolour}[1]{#1}
\newcommand{\singlyconstrainedgaskincolour}[1]{#1}
\newcommand{\singlyconstrainedellamcolour}[1]{#1}
\newcommand{\doublyconstrainedcolour}[1]{#1}
\newcommand{\doublytencellconstrainedcolour}[1]{#1}
\newcommand{\doublytwentycellconstrainedcolour}[1]{#1}

\newcommand{\mytable}{\tablecolour{\mathbf{T}}}
\newcommand{\myintensity}{\boldsymbol{\Lambda}}
\newcommand{\myprobability}{\intensitycolour{\boldsymbol{\pi}}}
\newcommand{\mytableoned}{\tablecolour{T}}
\newcommand{\myintensityoned}{\intensitycolour{\Lambda}}
\newcommand{\mytablerowsums}{\rowsumscolour{\mathbf{T}_{\cdot+}}}
\newcommand{\mytablecolsums}{\colsumscolour{\mathbf{T}_{+\cdot}}}
\newcommand{\mytablerowsumsoned}[1]{\rowsumscolour{T_{#1+}}}
\newcommand{\mytablecolsumsoned}[1]{\colsumscolour{T_{+1}}}
\newcommand{\mytabletotal}{\totalcolour{T_{++}}}
\newcommand{\mycells}[1]{\mathcal{X}#1}
\newcommand{\mytablecells}[1]{\cellcolour{\mathbf{T}_{\mycells{#1}}}}
\newcommand{\myintensityrowsums}{\rowsumscolour{\boldsymbol{\Lambda}_{\cdot +}}}
\newcommand{\myintensitycolsums}{\colsumscolour{\boldsymbol{\Lambda}_{+\cdot}}}
\newcommand{\myintensityrowsumsoned}[1]{\rowsumscolour{\Lambda_{#1+}}}
\newcommand{\myintensitycolsumsoned}[1]{\colsumscolour{\Lambda_{+1}}}
\newcommand{\myintensitytotal}{\totalcolour{\Lambda_{++}}}
\newcommand{\tableconstraints}{\constraintcolour{\mathcal{C}_{\mytableoned}}}
\newcommand{\intensityconstraints}{\constraintcolour{\mathcal{C}_{\myintensityoned}}}
\newcommand{\allconstraints}{\constraintcolour{\mathcal{C}}}
\newcommand{\unconstrained}[1]{\uncostrainedcolour{#1}}


\newcommand{\totallyconstrained}[1]{\totallyconstrainedcolour{#1}}
\newcommand{\singlyconstrained}[1]{\singlyconstrainedcolour{#1}}
\newcommand{\singlyconstrainedgaskin}[1]{\singlyconstrainedgaskincolour{#1}}
\newcommand{\singlyconstrainedellam}[1]{\singlyconstrainedellamcolour{#1}}
\newcommand{\doublyconstrained}[1]{\doublyconstrainedcolour{#1}}
\newcommand{\doublytencellconstrained}[1]{\doublytencellconstrainedcolour{#1}}
\newcommand{\doublytwentycellconstrained}[1]{\doublytwentycellconstrainedcolour{#1}}
\newcommand{\tablespace}{\mathcal{T}}
\newcommand{\obsdata}{\mathcal{D}}
\newcommand{\obstablespace}{\tablespace_{\obsdata}}
\newcommand{\constrainedtablespace}{\tablespace_{\allconstraints}}
\newcommand{\lossoperator}{\mathcal{L}}
\newcommand{\powerset}{\mathcal{P}}
\newcommand{\groundtruthtable}{\mytable^{*}}
\newcommand{\bigoh}[1]{\mathcal{O}(#1)}
\newcommand{\mytheta}{\boldsymbol{\theta}}
\newcommand{\mylogdestattr}{\mathbf{x}}
\newcommand{\mydestattr}{\mathbf{z}}
\newcommand{\mydestattroned}[1]{z_{#1}}
\newcommand{\mylogdestattrobs}{\mathbf{y}}
\newcommand{\markovbasis}{\mathcal{M}}

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