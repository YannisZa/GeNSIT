import os
import sys
import optuna

from gensit.config import Config
from gensit.utils.exceptions import MissingData
from gensit.physics_models.HarrisWilsonModel import HarrisWilson


def instantiate_physics_model(
    config:Config,
    trial:optuna.trial,
    **kwargs
):
    try: 
        physics_model_name = config['training']['physics_model']
    except:
        raise MissingData(
            'training>physics_model',
            data_names = ','.join(list(config.settings.keys()))
                if config is not None \
                else ','.join(list(kwargs.keys()))
        )
    match physics_model_name:
        case "harris_wilson_model":
            return HarrisWilson(
                config = config,
                trial = trial, 
                **kwargs
            )
        case _:
            raise Exception(
                f"Could not find physics model {physics_model_name}"
            )