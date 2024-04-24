import sys
import json

from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor

from gensit.config import Config
from gensit.utils.exceptions import *
from gensit.utils.misc_utils import setup_logger


class RandomForest_Model(object):
    def __init__(
        self,
        *,
        config: Config,
        **kwargs
    ):
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            console_level = level,
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevels( console_level = level )
        
        # Config file
        self.config = config

        # Type of learning model
        self.model_type = 'random_forest'

        DEFAULT_HYPERPARAMS = {
            'rf_n_estimators': 100,
            'rf_oob_score': True, 
            'rf_max_depth': None,
            'rf_min_samples_split': 10,
            'rf_min_samples_leaf': 3
        }
        
        # Set hyperparams
        self.hyperparams = {}
        for pname in DEFAULT_HYPERPARAMS.keys():
            if getattr(self.config['random_forest']['hyperparameters'],pname,None) is None:
                self.hyperparams[pname] = DEFAULT_HYPERPARAMS[pname]
                # Update config
                self.config['random_forest']['hyperparameters'][pname] = DEFAULT_HYPERPARAMS[pname]
            else:
                self.hyperparams[pname] = self.config['random_forest']['hyperparameters'][pname]
        
        # Initialise Random Forest Regressor
        self.random_forest = RandomForestRegressor(
            **{k.replace('rf_',''):v for k,v in self.hyperparams.items()},
            n_jobs = self.config['inputs'].get('n_threads',1)
        )


    def train(self, train_x, train_y, **kwargs):
        # Train
        return self.random_forest.fit(X = train_x, y = train_y)
    
    def predict(self, test_x, trained_model):
        return trained_model.predict(test_x)
    
    def __repr__(self):
        return "RandomForest()"

    def __str__(self):
        return json.dumps(self.hyperparams,indent=2)