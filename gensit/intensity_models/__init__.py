import os
import sys
import optuna

from gensit.config import Config
from gensit.utils.exceptions import MissingData
from gensit.intensity_models.spatial_interaction_models import instantiate_sim


def instantiate_intensity_model(
    config:Config,
    trial:optuna.trial=None,
    **kwargs
):
    try: 
        intensity_model_name = config['training']['intensity_model']
    except:
        raise MissingData(
            'training>intensity_model',
            data_names = ','.join(list(config.settings.keys()))
                if config is not None \
                else ','.join(list(kwargs.keys()))
        )
    match intensity_model_name:
        case "spatial_interaction_model":
            return instantiate_sim(
                config = config,
                trial = trial,
                **kwargs
            )
        case _:
            raise Exception(
                f"Could not find intensity model {intensity_model_name}"
            )