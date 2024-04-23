import sys
import json
import torch
import inspect
import numpy as np
import xgboost as xgb

from copy import deepcopy

from gensit.config import Config
from gensit.utils.exceptions import *
from gensit.utils.misc_utils import setup_logger, fn_name, eval_dtype
from gensit.static.global_variables import ACTIVATION_FUNCS, OPTIMIZERS, LOSS_FUNCTIONS, LOSS_DATA_REQUIREMENTS, LOSS_KWARG_OPERATIONS


class XGB_Model(object):
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
        self.model_type = 'xgboost'

        DEFAULT_HYPERPARAMS = {
            'xg_max_depth': 8,
            'xg_eta': 0.5, 
            'xg_objective': 'reg:squarederror',
            'xg_eval_metric': 'rmse',
            'xg_gpu_id': 0,
            'xg_tree_method': 'gpu_hist',
            'xg_max_delta_step': 2,
            'xg_min_child_weight': 4
            # 'xg_alpha': 0.5,
            # 'xg_lambda':5,
            # 'xg_gamma':0.5,
            # 'xg_subsample':0.8,
        } # binary:hinge binary:logistic
        
        # Set hyperparams
        self.hyperparams = {}
        for pname in DEFAULT_HYPERPARAMS.keys():
            if getattr(self.config['xgboost']['hyperparameters'],pname,None) is None:
                self.hyperparams[pname] = DEFAULT_HYPERPARAMS[pname]
                # Update config
                self.config['xgboost']['hyperparameters'][pname] = DEFAULT_HYPERPARAMS[pname]
            else:
                self.hyperparams[pname] = self.config['xgboost']['hyperparameters'][pname]


    def train(self, train_x, train_y, **kwargs):
        # Train
        self.dtrain = xgb.DMatrix(train_x, label = train_y)

        return xgb.train(
            self.hyperparams, 
            self.dtrain,
            kwargs.get('N',1), 
            xgb_model = kwargs.get('trained_model',None),
            evals = [(self.dtrain, 'train')],
            verbose_eval = self.config['xgboost'].get('verbose_eval',0)
        )

    def predict(self, test_x, trained_model):
        dtest = xgb.DMatrix(test_x)
        return trained_model.predict(dtest)

    def __repr__(self):
        return "XGBoost()"

    def __str__(self):
        return json.dumps(self.hyperparams,indent=2)