import matplotlib.cm as cm

from  matplotlib.colors import LinearSegmentedColormap

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