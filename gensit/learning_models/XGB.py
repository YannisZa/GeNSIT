import sys
import json
import optuna
import inspect
import xgboost as xgb

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from gensit.config import Config
from gensit.utils.exceptions import *
from gensit.utils.misc_utils import setup_logger, fn_name, eval_dtype
from gensit.static.global_variables import ACTIVATION_FUNCS, OPTIMIZERS, LOSS_FUNCTIONS, LOSS_DATA_REQUIREMENTS, LOSS_KWARG_OPERATIONS


MODEL_TYPE = 'xgboost'
MODEL_PREFIX = 'xg_'

DEFAULT_HYPERPARAMS = {
    'n_estimators': 1,
    'max_depth': 8,
    'learning_rate': 0.5, 
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'max_delta_step': 2,
    'min_child_weight': 4
    # 'alpha': 0.5,
    # 'lambda':5,
    # 'gamma':0.5,
    # 'subsample':0.8,
} # binary:hinge binary:logistic


class XGB_Model(object):
    def __init__(
        self,
        *,
        trial: optuna.trial,
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
        # Hyperparameter optimisation trial
        self.trial = trial
        # Type of learning model
        self.model_type = MODEL_TYPE
        
        # Update hyperparameters
        self.update_hyperparameters()

        self.xgboost_model = xgb.XGBRegressor(
            **self.hyperparams
        )

    def update_hyperparameters(self):
        # Set hyperparams
        self.hyperparams = {}
        if self.trial is not None:
            OPTUNA_HYPERPARAMS = {
                'n_estimators': self.trial.suggest_int('n_estimators', 10, 100, step = 50),
                'max_depth': self.trial.suggest_int('max_depth', 1, 12, step = 1),
                'learning_rate': self.trial.suggest_float('learning_rate', 0.01, 1.0), 
                'tree_method': self.trial.suggest_categorical('tree_method', ["hist", "exact", "approx"]),
                'max_delta_step': self.trial.suggest_float('max_delta_step', 0.1, 3.0),
                'min_child_weight': self.trial.suggest_float('min_child_weight', 1.0, 10.0)
            } 
        
        for pname in DEFAULT_HYPERPARAMS.keys():
            if self.trial is not None and pname in OPTUNA_HYPERPARAMS:
                self.hyperparams[pname] = OPTUNA_HYPERPARAMS[pname]
            elif self.config is None or getattr(self.config[MODEL_TYPE]['hyperparameters'],pname,None) is None:
                self.hyperparams[pname] = DEFAULT_HYPERPARAMS[pname]
            else:
                self.hyperparams[pname] =  self.config[MODEL_TYPE]['hyperparameters'][(MODEL_PREFIX+pname)]

            if self.config is not None and getattr(self.config[MODEL_TYPE]['hyperparameters'],pname,None) is None:
                # Update object and config hyperparameters
                self.config[MODEL_TYPE]['hyperparameters'][(MODEL_PREFIX+pname)] = self.hyperparams[pname]


    def train(self, train_x, train_y, **kwargs):
        self.xgboost_model.fit(
            train_x,
            train_y,
            eval_set = [(train_x, train_y)],
            xgb_model = kwargs.get('trained_model',None),
            verbose = False
        )

    def predict(self, test_x):
        return self.xgboost_model.predict(test_x)

    def run_single(self,train_x,train_y,test_x):
        # Get previously trained model if one has been fitted
        try:
            check_is_fitted(self.xgboost_model)
            trained_model = self.xgboost_model
        except NotFittedError as exc:
            trained_model = None

        # Train
        self.train(
            train_x = train_x,
            train_y = train_y,
            trained_model = trained_model
        )
        
        # Test (predict)
        intensity = self.predict(
            test_x = test_x
        )
        return intensity
        
    def __repr__(self):
        return self.xgboost_model.__repr__()

    def __str__(self):
        return json.dumps(self.hyperparams,indent=2)