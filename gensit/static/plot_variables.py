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
    "empty": "",
    "line": "plot",
    "scatter" : "scatter",
    "imshow": "imshow",
    "geoshow": "geoshow"
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
        "_total_intensity_row_table_constrained":COLOR_NAMES["tab20b_green"],
        "_doubly_constrained":COLOR_NAMES["tab20b_orange"],
        "_doubly_10#%_cell_constrained":COLOR_NAMES["tab20b_red"],
        "_doubly_20#%_cell_constrained":COLOR_NAMES["tab20c_blue"],
        "_doubly_10%_cell_constrained":COLOR_NAMES["tab20b_red"],
        "_doubly_20%_cell_constrained":COLOR_NAMES["tab20c_blue"],
    },
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
    "JointTableSIM_NN": '\\frameworktag (Joint)',
    "NonJointTableSIM_NN": '\\frameworktag (Disjoint)',
    "_unconstrained":'$\\emptyset$',
    "_total_constrained":'$\\mytabletotal$',
    "_total_intensity_row_table_constrained":'$\\myintensitytotal,\\mytablerowsums$',
    "_row_constrained":'$\\mytablerowsums$',
    "_doubly_constrained":'$\\mytablerowsums,\\mytablecolsums$',
    "_doubly_10#%_cell_constrained":'$\\mytablerowsums,\\mytablecolsums,\\mytablecells{_1}$',
    "_doubly_20#%_cell_constrained":'$\\mytablerowsums,\\mytablecolsums,\\mytablecells{_2}$',
    "_doubly_10%_cell_constrained":'$\\mytablerowsums,\\mytablecolsums,\\mytablecells{_1}$',
    "_doubly_20%_cell_constrained":'$\\mytablerowsums,\\mytablecolsums,\\mytablecells{_2}$',
    "TotallyConstrained":'$\\myintensitytotal$',
    "ProductionConstrained":'$\myintensityrowsums$',
    "dest_attraction_ts_likelihood_loss":"$\\lossoperator\\left(\\mylogdestattr \\; ; \\; \\obsdata, \\boldsymbol{\\nu} \\right)$",
    "dest_attraction_ts_likelihood_loss,table_likelihood_loss":"$\\lossoperator\\left(\\mylogdestattr, \\mytable, \\myintensity \\; ; \\; \\obsdata, \\boldsymbol{\\nu} \\right)$",
    "dest_attraction_ts_likelihood_loss,total_intensity_distance_likelihood_loss,table_likelihood_loss":"$\\lossoperator\\left(\\mylogdestattr,\\mytable,\\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \\right)$",
    "dest_attraction_ts_likelihood_loss,total_intensity_distance_likelihood_loss":"$\\lossoperator\\left(\\mylogdestattr, \\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \\right)$",
    "dest_attraction_ts_likelihood_loss,total_table_distance_likelihood_loss,table_likelihood_loss":"$\\lossoperator\\left(\\mylogdestattr,\\mytable,\\myintensity \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \\right)$",
    "dest_attraction_ts_likelihood_loss,total_table_distance_likelihood_loss":"$\\lossoperator\\left(\\mylogdestattr, \\mytable \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \\right)$",
    "table_likelihood_loss":"$\\lossoperator\\left(\\mytable, \\myintensity \\right)$",
    "total_intensity_distance_likelihood_loss,table_likelihood_loss":"$\\lossoperator\\left(\\mytable,\\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \\right)$",
    "total_intensity_distance_likelihood_loss":"$\\lossoperator\\left( \\myintensity \\; ; \\; \\obsdata^{\\myintensityoned}, \\boldsymbol{\\nu} \\right)$",
    "total_intensity_distance_loss":"$\\lossoperator\\left( \\myintensity \\; ; \\; \\obsdata^{\\myintensityoned} \\right)$",
    "total_table_distance_likelihood_loss,table_likelihood_loss":"$\\lossoperator\\left(\\mytable,\\myintensity \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \\right)$",
    "total_table_distance_likelihood_loss":"$\\lossoperator\\left(\\mytable \\; ; \\; \\obsdata^{\\mytableoned}, \\boldsymbol{\\nu} \\right)$",
    "total_table_distance_loss":"$\\lossoperator\\left(\\mytable \\; ; \\; \\obsdata^{\\mytableoned} \\right)$"
}
LABEL_EXPRESSIONS = {
    "sigma": '$\\sigma = '
}
LATEX_EXPRESSIONS = {**RAW_EXPRESSIONS,**LABEL_EXPRESSIONS}
LATEX_COORDINATES = ['label','annotate']+PLOT_COORDINATES+PLOT_AUX_COORDINATES


# \usepackage{/home/iz230/GeNSIT/gensit/static/neurips_2024}
# \usepackage{mathptmx} % matches times roman font for math equations
# \usepackage{amssymb}

# \usepackage{/home/iz230/GeNSIT/gensit/static/PhDThesisPSnPDF}
# \usepackage{amscd}
# \usepackage{dsfont}
# \usepackage{commath}
# \usepackage{mathtools}
# \usepackage{amsmath}
# \usepackage{amssymb}
# \usepackage{amsthm}
# \usepackage{newtxtext,newtxmath}

LATEX_PREAMBLE = r'''
\usepackage{/home/iz230/GeNSIT/gensit/static/PhDThesisPSnPDF}
\usepackage{color}
\usepackage[table]{xcolor}  % colors

\usepackage{amscd}
\usepackage{dsfont}
\usepackage{commath}
\usepackage{mathtools}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{newtxtext,newtxmath}

\newcommand{\deepred}[1]{{\textcolor[HTML]{E20000}{#1}}}

\newcommand{\zachosframeworkbasename}{SIT-MCMC}
\newcommand{\gaskinframeworkbasename}{SIM-NN}
\newcommand{\ellamframeworkbasename}{SIM-MCMC}
\newcommand{\liuframeworkbasename}{GMEL}
\newcommand{\zachosframeworkfullname}{\textbf{S}patial \textbf{I}nteraction \textbf{T}able \textbf{M}arkov \textbf{C}hain \textbf{M}onte \textbf{C}arlo \;}
\newcommand{\gaskinframeworkfullname}{\textbf{S}patial \textbf{I}nteraction \textbf{M}odel \textbf{N}eural \textbf{N}etwork\;}
\newcommand{\ellamframeworkfullname}{\textbf{S}patial \textbf{I}nteraction \textbf{M}odel \textbf{M}arkov \textbf{C}hain \textbf{M}onte \textbf{C}arlo \;}
\newcommand{\liuframeworkfullname}{\textbf{G}eo-contextual \textbf{M}ultitask \textbf{E}mbedding \textbf{L}earner \;}
\newcommand{\zachosframework}{\zachosframeworkbasename \;}
\newcommand{\gaskinframework}{\gaskinframeworkbasename \;}
\newcommand{\ellamframework}{\ellamframeworkbasename \;}
\newcommand{\liuframework}{\liuframeworkbasename \;}
\newcommand{\zachosframeworktag}{\textsc{\zachosframeworkbasename}}
\newcommand{\gaskinframeworktag}{\textsc{\gaskinframeworkbasename}}
\newcommand{\ellamframeworktag}{\textsc{\ellamframeworkbasename}}
\newcommand{\liuframeworktag}{\textsc{\liuframeworkbasename}}

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

LATEX_RC_PARAMETERS = {
    # 'mathtext.default': 'regular',
    # 'mathtext.fontset': 'stix',
    # 'mathtext.fallback': 'stix',
    'text.usetex': True,
    "font.family": "serif",
    'text.latex.preamble': LATEX_PREAMBLE,
}

LEGEND_LOCATIONS = [
    'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 
    'center left', 'center right', 'lower center', 'upper center', 'center'
]