import sys
import json
import optuna
import numpy as np

from copy import deepcopy
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from gensit.config import Config
from gensit.utils.exceptions import *
from gensit.utils.misc_utils import setup_logger


DEFAULT_HYPERPARAMS = {
    'max_depth': None,
    'min_samples_split': 10,
    'min_samples_leaf': 3,
    'oob_score': True
}

MODEL_TYPE = 'random_forest'
MODEL_PREFIX = 'rf_'

class RF_Model(object):
    def __init__(
        self,
        *,
        config: Config,
        trial: optuna.trial,
        logger,
        **kwargs
    ):
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            console_level = level,
            
        ) if kwargs.get('logger',None) is None else logger
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

        # Initialise Random Forest Regressor
        self.random_forest = RandomForestRegressor(
            **{k.replace('rf_',''):v for k,v in self.hyperparams.items()},
            n_jobs = self.config['inputs'].get('n_threads',1),
            **kwargs
        )

    def update_hyperparameters(self):
        # Set hyperparams
        self.hyperparams = {}
        if self.trial is not None:
            OPTUNA_HYPERPARAMS = {
                # 'n_estimators': self.trial.suggest_int('n_estimators', 10, 100, step = 50),
                'max_depth': self.trial.suggest_int('max_depth', 1, 12, step = 1),
                'min_samples_split': self.trial.suggest_int('min_samples_split', 2, 100, step = 10),
                'min_samples_leaf': self.trial.suggest_int('min_samples_leaf', 1, 50, step = 5),
                'oob_score':self.trial.suggest_categorical('oob_score',[True,False])
            } 
        
        self.hyperparams['n_estimators'] = self.config['training']['N']
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
        self.random_forest.fit(train_x, train_y)
    
    def predict_single(self, test_x, estimator_index):
        return self.random_forest.estimators_[estimator_index].predict(test_x)
        
    def __repr__(self):
        return self.random_forest.__repr__()

    def __str__(self):
        return json.dumps(self.hyperparams,indent=2)
        