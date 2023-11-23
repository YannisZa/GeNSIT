import matplotlib.cm as cm

from matplotlib.colors import LinearSegmentedColormap

def cmap_exists(name):
    try:
        cm.get_cmap(name)
    except:
        return False
    return True


PLOT_COORDINATES = ['x','y','z']
PLOT_CORE_FEATURES = {
    'marker': {'dtype': str},
    'hatch': {'dtype': str},
    'visibility': {'dtype': float},
    'colour': {'dtype': str},
    'size': {'dtype': float},
    'zorder': {'dtype': float},
}
PLOT_AUX_FEATURES = {
    'xticks':{},
    'yticks':{},
    'zticks':{}
}

ALL_PLOT_VARIABLES = PLOT_COORDINATES+list(PLOT_CORE_FEATURES.keys())+list(PLOT_AUX_FEATURES.keys())

# Type of plots
PLOT_HASHMAP = {
        'dss':'data_plot_2d_scatter',
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

PLOT_MARKERS = {
    "sample_name": {
        "table": "s",
        "intensity":"^" 
    }
}

COLOR_NAMES = {
    "yellow": "#F5DDA9",
    "darkblue": "#2F7194",
    "red": "#ec7070",
    "skyblue": "#97c3d0",
    "green": "#3cc969",
    "darkgreen": "#48675A",
    "lightbrown": "#C6BFA2",
    "orange": "#EC9F7E",
    "lightgreen": "#AFD8BC",
    "grey": "#3D4244"
}

PLOT_COLOURS = {
    "type": {
        "SIM_MCMC": COLOR_NAMES["skyblue"],
        "SIM_NN": COLOR_NAMES["lightgreen"],
        "JointTableSIM_MCMC":COLOR_NAMES["darkblue"],
        "NonJointTableSIM_NN":COLOR_NAMES["green"],
        "JointTableSIM_NN":COLOR_NAMES["darkgreen"],
    }
}

PLOT_HATCHES = {
    "sigma": {
        "learned": "***",
        "high": "+++",
        "low": "OOO"
    }
}


# Register colormaps
bluegreen = LinearSegmentedColormap.from_list(
        'bluegreen',
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
        N=256
)
bluegreen.set_bad((0,0,0,1))
if not cmap_exists('bluegreen'):
    cm.register_cmap(cmap=bluegreen, name='bluegreen')

yellowpurple = LinearSegmentedColormap.from_list(
        'yellowpurple',
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
        N=256
)
yellowpurple.set_bad((0,0,0,1))
if not cmap_exists('yellowpurple'):
    cm.register_cmap(cmap=yellowpurple, name='yellowpurple')



yellowblue = LinearSegmentedColormap.from_list(
        'yellowblue',
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
        N=256
)
if not cmap_exists('yellowblue'):
    cm.register_cmap(cmap=yellowblue, name='yellowblue')


redgreen = LinearSegmentedColormap.from_list(
        'redgreen',
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
        N=256
)
if not cmap_exists('redgreen'):
    cm.register_cmap(cmap=redgreen, name='redgreen')

cblue = LinearSegmentedColormap.from_list(
        "cblue", 
        [
            (255./255.,255/255.,255./255.),
            (0./255.,120/255.,255./255.),
        ]
)
if not cmap_exists('cblue'):
    cm.register_cmap(cmap=cblue, name='cblue')

cgreen = LinearSegmentedColormap.from_list(
        "cgreen", 
        [
            (255./255.,255./255.,255./255.),
            (0./255.,255./255.,0./255.)
        ]
)
if not cmap_exists('cgreen'):
    cm.register_cmap(cmap=cgreen, name='cgreen')

cred = LinearSegmentedColormap.from_list(
    "cred", 
    [
        (255./255.,255./255.,255./255.),
        (255./255.,0/255.,0./255.),
    ]
)
if not cmap_exists('cred'):
    cm.register_cmap(cmap=cred, name='cred')