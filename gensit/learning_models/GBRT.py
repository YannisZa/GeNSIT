import sys
import json
import optuna

from copy import deepcopy
from sklearn.ensemble import GradientBoostingRegressor


from gensit.config import Config
from gensit.utils.exceptions import *
from gensit.utils.misc_utils import setup_logger

MODEL_TYPE = 'gradient_boosted_regression_trees'
MODEL_PREFIX = 'gbrt_'

DEFAULT_HYPERPARAMS = {
    'n_estimators': 3,
    'max_depth': None,
    'min_samples_split': 10,
    'min_samples_leaf': 1
}


class GBRT_Model(object):
    def __init__(
        self,
        *,
        config: Config,
        trial: optuna.trial,
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
        # Hyperparameter optimisation trial
        self.trial = trial
        # Type of learning model
        self.model_type = MODEL_TYPE
        self.model_prefix = MODEL_PREFIX

        # Update hyperparameters
        self.update_hyperparameters()
        
        # Initialise Gradient Boosted Trees Regressor
        self.gbrt = GradientBoostingRegressor(
            **{k.replace('gbrt_',''):v for k,v in self.hyperparams.items()}
        )

    def update_hyperparameters(self):
        # Set hyperparams
        self.hyperparams = {}
        if self.trial is not None:
            OPTUNA_HYPERPARAMS = {
                'n_estimators': self.trial.suggest_int('n_estimators', 10, 100, step = 50),
                'max_depth': self.trial.suggest_int('max_depth', 1, 12, step = 1),
                'min_samples_split': self.trial.suggest_int('min_samples_split', 2, 100, step = 10),
                'min_samples_leaf': self.trial.suggest_int('min_samples_leaf', 1, 50, step = 5),
            }
        
        for pname in DEFAULT_HYPERPARAMS.keys():
            if self.trial is not None and pname in OPTUNA_HYPERPARAMS:
                self.hyperparams[pname] = OPTUNA_HYPERPARAMS[pname]
            elif self.config is None or getattr(self.config[self.model_type]['hyperparameters'],pname,None) is None:
                self.hyperparams[pname] = DEFAULT_HYPERPARAMS[pname]
            else:
                self.hyperparams[pname] =  self.config[self.model_type]['hyperparameters'][(self.model_prefix+pname)]

            if self.config is not None and getattr(self.config[self.model_type]['hyperparameters'],pname,None) is None:
                # Update object and config hyperparameters
                self.config[self.model_type]['hyperparameters'][(self.model_prefix+pname)] = self.hyperparams[pname]

    def train(self, train_x, train_y, **kwargs):
        # Train
        return self.gbrt.fit(
            X = train_x,
            y = train_y
        )

    def predict(self, test_x):
        return self.gbrt.predict(test_x)
    
    def run_single(self,train_x,train_y,test_x):
        # Train
        self.train(
            train_x = train_x,
            train_y = train_y
        )
        
        # Test (predict)
        intensity = self.predict(
            test_x = test_x
        )
        return intensity

    def __repr__(self):
        return self.gbrt.__repr__()
    
    def __str__(self):
        return json.dumps(self.hyperparams,indent=2)