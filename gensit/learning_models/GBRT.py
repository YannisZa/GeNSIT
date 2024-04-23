import sys
import json

from copy import deepcopy
from sklearn.ensemble import GradientBoostingRegressor


from gensit.config import Config
from gensit.utils.exceptions import *
from gensit.utils.misc_utils import setup_logger


class GBRT_Model(object):
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
        self.model_type = 'gradient_boosted_regression_trees'

        DEFAULT_HYPERPARAMS = {
            'gbrt_n_estimators': 3,
            'gbrt_max_depth': None,
            'gbrt_min_samples_split': 10,
            'gbrt_min_samples_leaf': 1
        }
        
        # Set hyperparams
        self.hyperparams = {}
        for pname in DEFAULT_HYPERPARAMS.keys():
            if getattr(self.config['gradient_boosted_regression_trees']['hyperparameters'],pname,None) is None:
                self.hyperparams[pname] = DEFAULT_HYPERPARAMS[pname]
                # Update config
                self.config['gradient_boosted_regression_trees']['hyperparameters'][pname] = DEFAULT_HYPERPARAMS[pname]
            else:
                self.hyperparams[pname] = self.config['gradient_boosted_regression_trees']['hyperparameters'][pname]
        
        # Initialise Gradient Boosted Trees Regressor
        self.gbrt = GradientBoostingRegressor(
            **{k.replace('gbrt_',''):v for k,v in self.hyperparams.items()}
        )


    def train(self, train_x, train_y, **kwargs):
        # Train
        return self.gbrt.fit(X = train_x, y = train_y)

    def predict(self, test_x, trained_model):
        return trained_model.predict(test_x)

    def __repr__(self):
        return "GradientBoostedTrees()"

    def __str__(self):
        return json.dumps(self.hyperparams,indent=2)