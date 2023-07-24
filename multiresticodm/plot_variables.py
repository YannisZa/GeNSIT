import matplotlib.cm as cm

from  matplotlib.colors import LinearSegmentedColormap

# Register colormaps
yellowblue = LinearSegmentedColormap.from_list(
        'yellowblue',
        list(
            zip(
                [0,.15,.4,.5,0.6,.9,1.],
                ["orange", "yellow", "lightyellow", "white", "lightblue", "blue", "darkblue"]
                # "skyblue"
            )
        ),
        N=256
)
cm.register_cmap(cmap=yellowblue, name='yellowblue')


redgreen = LinearSegmentedColormap.from_list(
        'redgreen',
        list(
            zip(
                [0,.15,.4,.5,0.6,.9,1.],
                ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
            )
        ),
        N=256
)
cm.register_cmap(cmap=redgreen, name='redgreen')

cblue = LinearSegmentedColormap.from_list(
        "cblue", 
        [
            (255./255.,255/255.,255./255.),
            (0./255.,120/255.,255./255.),
        ]
)
cm.register_cmap(cmap=cblue, name='cblue')

cgreen = LinearSegmentedColormap.from_list(
        "cgreen", 
        [
            (255./255.,255./255.,255./255.),
            (0./255.,255./255.,0./255.)
        ]
)
cm.register_cmap(cmap=cgreen, name='cgreen')

cred = LinearSegmentedColormap.from_list(
    "cred", 
    [
        (255./255.,255./255.,255./255.),
        (255./255.,0/255.,0./255.),
    ]
)
cm.register_cmap(cmap=cred, name='cred')