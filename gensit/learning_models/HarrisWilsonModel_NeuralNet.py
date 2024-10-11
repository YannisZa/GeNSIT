#!/usr/bin/env python3

import sys
import torch
import inspect
import optuna
import numpy as np

from copy import deepcopy
from torch import nn, float32
from typing import Any, List, Union

from gensit.config import Config
from gensit.utils.exceptions import *
from gensit.physics_models.HarrisWilsonModel import HarrisWilson
from gensit.utils.misc_utils import setup_logger, fn_name, eval_dtype, pop_variable
from gensit.static.global_variables import ACTIVATION_FUNCS, OPTIMIZERS, LOSS_FUNCTIONS, LOSS_DATA_REQUIREMENTS, LOSS_KWARG_OPERATIONS

# Store hyperparameters
DEFAULT_HYPERPARAMS = {
    "nodes_per_layer": {
        "default": 20,
        "layer_specific": {}
    },
    "num_hidden_layers": 1,
    "optimizer": 'Adam',
    "learning_rate": 0.002,
    "biases": {
        "default":[+0.0, +4.0],
        "layer_specific": {}
    },
    "activation_funcs": {
        "default": "linear",
        "layer_specific": {
            1: "abs"
        }
    }
}

def get_architecture(
    input_size: int, output_size: int, n_layers: int, cfg: dict
) -> List[int]:

    # Apply default to all hidden layers
    _nodes = [cfg.get('default')] * n_layers

    # Update layer-specific settings
    _layer_specific = cfg.get('layer_specific', {})
    for layer_id, layer_size in _layer_specific.items():
        _nodes[layer_id] = layer_size
    return [input_size] + _nodes + [output_size]


def get_activation_funcs(n_layers: int, cfg: dict) -> List[callable]:

    """Extracts the activation functions from the config. The config is a dictionary containing the
    default activation function, and a layer-specific entry detailing exceptions from the default. 'None' entries
    are interpreted as linear layers.

    .. Example:
        activation_funcs:
          default: relu
          layer_specific:
            0: ~
            2: tanh
            3:
              name: HardTanh
              args:
                - -2  # min_value
                - +2  # max_value
    """

    def _single_layer_func(layer_cfg: Union[str, dict]) -> callable:

        """Return the activation function from an entry for a single layer"""

        # Entry is a single string
        if isinstance(layer_cfg, str):
            _fun,_call = ACTIVATION_FUNCS[layer_cfg.lower()]
            _fun = eval(_fun, {"torch":torch})
            if _call:
                return _fun()
            else:
                return _fun

        # Entry is a dictionary containing args and kwargs
        elif isinstance(layer_cfg, dict):
            _fun,_call = ACTIVATION_FUNCS[layer_cfg.lower()]
            _fun = eval(_fun, {"torch":torch})
            if _call:
                return _fun(*layer_cfg.get("args", ()), **layer_cfg.get("kwargs", {}))
            else:
                return _fun

        elif layer_cfg is None:
            _f = eval(
                ACTIVATION_FUNCS["linear"][0],
                {"torch":torch}
            )

        else:
            raise ValueError(f"Unrecognized activation function {cfg}!")

    # Use default activation function on all layers
    _funcs = [_single_layer_func(cfg.get("default"))] * (n_layers + 1)

    # Change activation functions on specified layers
    _layer_specific = cfg.get("layer_specific", {})
    for layer_id, layer_cfg in _layer_specific.items():
        _funcs[int(layer_id)] = _single_layer_func(layer_cfg)

    return _funcs


def get_bias(n_layers: int, cfg: dict) -> List[Any]:

    '''Extracts the bias initialisation settings from the config. The config is a dictionary containing the
    default, and a layer-specific entry detailing exceptions from the default. 'None' entries
    are interpreted as unbiased layers.

    .. Example:
        biases:
          default: ~
          layer_specific:
            0: [-1, 1]
            3: [2, 3]
    '''

    # Use the default value on all layers
    biases = [cfg.get('default')] * (n_layers + 1)

    # Amend bias on specified layers
    _layer_specific = cfg.get('layer_specific', {})
    for layer_id, layer_bias in _layer_specific.items():
        biases[layer_id] = layer_bias

    return biases



# -----------------------------------------------------------------------------
# -- Neural net class ---------------------------------------------------------
# -----------------------------------------------------------------------------


class NeuralNet(nn.Module):

    def __init__(
        self,
        *,
        config:Config,
        trial:optuna.trial,
        input_size: int,
        output_size: int,
        **kwargs
    ):
        '''

        :param input_size: the number of input values
        :param output_size: the number of output values
        :param num_hidden_layers: the number of hidden layers
        :param nodes_per_layer: a dictionary specifying the number of nodes per layer
        :param activation_funcs: a dictionary specifying the activation functions to use
        :param biases: a dictionary containing the initialisation parameters for the bias
        :param optimizer: the name of the optimizer to use. Default is the torch.optim.Adam optimizer.
        :param learning_rate: the learning rate of the optimizer. Default is 1e-3.
        :param __: Additional model parameters (ignored)
        '''
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            console_level = level,
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevels( console_level = level )
        
        super().__init__()
        self.flatten = nn.Flatten()

        # Input/output dimensions
        self.input_dim = input_size
        self.output_dim = output_size

        # Configuration file
        self.config = config
        
        # Optuna hyperparameter optimisation trial
        self.trial = trial

        # Type of learning model
        self.model_type = 'neural_network'
        self.model_prefix = ''
        
        # Update hyperparameters
        self.update_hyperparameters()

        # Get architecture, activation functions, and layer bias
        self.architecture = get_architecture(
            input_size, 
            output_size, 
            self.hyperparams['num_hidden_layers'],
            self.hyperparams['nodes_per_layer']
        )
        self.activation_funcs = get_activation_funcs(
            self.hyperparams['num_hidden_layers'], 
            self.hyperparams['activation_funcs']
        )
        self.bias = get_bias(
            self.hyperparams['num_hidden_layers'],
            self.hyperparams['biases']
        )

        # Add the neural net layers
        self.layers = nn.ModuleList()
        for i in range(len(self.architecture) - 1):
            layer = nn.Linear(
                self.architecture[i],
                self.architecture[i + 1],
                bias = self.bias[i] is not None,
            )

            # Initialise the biases of the layers with a uniform distribution
            if self.bias[i] is not None:
                # Use the pytorch default if indicated
                if self.bias[i] == 'default':
                    torch.nn.init.uniform_(layer.bias)
                # Initialise the bias on explicitly provided intervals
                else:
                    torch.nn.init.uniform_(layer.bias, self.bias[i][0], self.bias[i][1])

            self.layers.append(layer)

        # Get the optimizer
        optimizer_function = eval(
            OPTIMIZERS[self.hyperparams['optimizer']],
            {"torch":torch}
        )
        self.optimizer = optimizer_function(self.parameters(), lr = self.hyperparams['learning_rate'])

    def update_hyperparameters(self):
        # Set hyperparams
        self.hyperparams = {}
        if self.trial is not None:
            OPTUNA_HYPERPARAMS = {
                "num_hidden_layers": self.trial.suggest_int('num_hidden_layers', 1, 12, step = 1),
                "learning_rate": self.trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
                # "optimizer":  self.trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
            } 
            # Layer-specific number of nodes
            layer_specific_nodes = []
            for i in range(OPTUNA_HYPERPARAMS['num_hidden_layers']):
                layer_specific_nodes.append(self.trial.suggest_int(f"nodes_per_layer_{i}", 1, 30, step = 4))
            OPTUNA_HYPERPARAMS['nodes_per_layer'] = {
                "default": layer_specific_nodes[0],
                "layer_specific": {i:layer_specific_nodes[i] for i in range(len(layer_specific_nodes))}
            }
            # Layer-specific biases
            layer_specific_biases = []
            for i in range(OPTUNA_HYPERPARAMS['num_hidden_layers']):
                layer_specific_biases.append([0,self.trial.suggest_float(f"biases_{i}", 0, 4)])
            OPTUNA_HYPERPARAMS['biases'] = {
                "default": layer_specific_biases[0],
                "layer_specific": {i:layer_specific_biases[i] for i in range(len(layer_specific_biases))}
            }
            # Layer-specific activation functions
            layer_specific_activation_funcs = []
            for i in range(OPTUNA_HYPERPARAMS['num_hidden_layers']+1):
                layer_specific_activation_funcs.append(
                    self.trial.suggest_categorical(f"activation_funcs_{i}", ["abs","relu","linear","tanh","hardsigmoid"])
                )
            OPTUNA_HYPERPARAMS['activation_funcs'] = {
                "default": layer_specific_activation_funcs[0],
                "layer_specific": {i:layer_specific_activation_funcs[i] for i in range(len(layer_specific_activation_funcs))}
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
    
    # ... Evaluation functions .........................................................................................

    # The model forward pass
    def forward(self, x):
        self.logger.debug('Solving neural net')
        for i in range(len(self.layers)):
            if self.activation_funcs[i] is None:
                x = self.layers[i](x)
            else:
                x = self.activation_funcs[i](self.layers[i](x))
        return x

# ----------------------------------------------------------------------------------------------------------------------
# -- Model implementation ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class HarrisWilson_NN:

    def __init__(
        self,
        *,
        config: Config,
        neural_net: NeuralNet,
        physics_model: HarrisWilson,
        **kwargs,
    ):
        '''Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        Args:
            rng (np.random.Generator): The shared RNG
            neural_net: The neural net architecture
            loss (dict): the loss function to use
            physics_model: The numerical solver
            config: settings regarding model (hyper)parameters and other settings
        '''
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            console_level = level,
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevels( console_level = level )

        # Type of learning model
        self.model_type = 'harris_wilson_neural_network'
        self.model_prefix = ''

        # The numeric solver
        self.physics_model = physics_model

        # Config file
        self.config = config
        
        # Loss arguments
        loss = kwargs.get('loss',{})

        # Initialise neural net, loss tracker and prediction tracker
        self._neural_net = neural_net
        self._neural_net.optimizer.zero_grad()

        # Copy optuna trial
        self.trial = pop_variable(self._neural_net,'trial',None)

        # Store loss function parameters
        self.loss_functions = {}
        self.loss_kwargs = {}

        # Make sure the these configurations have the same length
        try:
            assert len(set([len(loss[k]) for k in ['loss_name','loss_function']])) == 1
        except:
            raise InvalidDataLength(
                data_name_lens = {
                    k:len(loss[k]) \
                    for k in ['loss_name','loss_function']
                }
            )

        # Parse loss functions
        for name,function in zip(loss['loss_name'],loss['loss_function']):
            # Construct kwargs from key names
            fn_kwargs = {}
            for key,value in loss['loss_kwargs'].items():
                
                # If empty key provided move on
                if len(key) <= 0 or key == 'nokey':
                    continue
                
                # Find path to key
                key_path = list(self.config.path_find(key))
                key_path = key_path[0] if len(key_path) > 0 else []
                # Get value of key
                key_val,key_found = self.config.path_get(
                    key_path = key_path
                )
                
                try:
                    assert key_found or (value is not None)
                except:
                    raise Exception(f"""
                        Could not find {name} keyword argument {key} for {function} in settings or
                        {value} provided is invalid.
                    """)
                # Update value based on config key
                if key_found:
                    fn_kwargs[key] = key_val
                # Set key to value if value is not null
                else:
                    fn_kwargs[key] = value
            
            # Get loss function from global variables (standard torch loss functions)
            loss_func = LOSS_FUNCTIONS.get(function.lower(),{}).get('function',None)
            # if failed get loss function from loss dictionary provided
            loss_func = loss.get(name,loss_func) if loss_func is None else eval(loss_func+"()",{"torch":torch})
            # if failed get loss function from physics model defined functions
            loss_func = getattr(self.physics_model,name,loss_func) if loss_func is None else loss_func
            
            # Add kwargs
            if loss_func is not None:
                self.loss_functions[name] = loss_func
                self.loss_kwargs[name] = fn_kwargs
            else:
                raise Exception(f"Loss {name} is missing function {function}. Loss function is set to {loss_func}.")

        self._loss_sample = torch.tensor(0.0, requires_grad = False)
        self._theta_sample = torch.stack(
            [torch.tensor(0.0, requires_grad = False)] * len(self.physics_model.params_to_learn)
        )

    def update_loss(
            self,
            previous_loss:dict,
            n_processed_steps:dict,
            prediction_data:dict,
            validation_data:dict,
            loss_function_names:list = None,
            aux_inputs:dict={},
            **kwargs
        ):
        self.logger.debug('Loss function update')
        
        # Get subset of loss function names
        loss_function_names = loss_function_names if loss_function_names is not None else list(self.loss_functions.keys())
        loss_function_names = list(set(loss_function_names).intersection(set(list(self.loss_functions.keys()))))
        
        for name in loss_function_names:
            # Make sure you have the necessary data
            prediction_data_sizes = {}
            # prediction_set
            for pred_dataset in LOSS_DATA_REQUIREMENTS[name]['prediction_data']:
                try:
                    assert prediction_data.get(pred_dataset,None) is not None
                    # Monitor number of samples in monte carlo estimation of loss
                    # for each prediction dataset
                    prediction_data_sizes[pred_dataset] = len(prediction_data.get(pred_dataset))
                    # print(name,pred_dataset,[pdata.requires_grad for pdata in prediction_data.get(pred_dataset)])
                except:
                    raise Exception(f"Loss {name} is missing prediction data {pred_dataset}.")
            # Make sure the same amount of prediction samples are provided 
            # for each dataset of a given loss functional
            try:
                assert len(set([prediction_data_sizes[k] for k in prediction_data_sizes.keys()])) == 1
            except:
                raise Exception(f"Prediction data sample sizes differ {prediction_data_sizes}.")
            for validation_dataset in LOSS_DATA_REQUIREMENTS[name]['validation_data']:
                try:
                    validation_data[validation_dataset] = validation_data.get(validation_dataset) \
                        if validation_data.get(validation_dataset,None) is not None \
                        else aux_inputs.get(validation_dataset,None)
                    assert validation_data[validation_dataset] is not None
                    # print(name,validation_dataset,validation_data[validation_dataset].requires_grad)
                except:
                    raise Exception(f"Loss {name} is missing validation data {validation_dataset}.")
            
            # Compute monte carlo estimate of loss
            # Number of MC samples
            N_samples = list(prediction_data_sizes.values())[0]
            total_loss = []

            # Get loss function name
            loss_fn_name = LOSS_FUNCTIONS.get(fn_name(self.loss_functions[name]).lower(),{})
            loss_fn_name_keys = loss_fn_name.get('kwargs_keys',None)
            loss_fn_name_keys = loss_fn_name_keys if loss_fn_name_keys is not None else list(self.loss_kwargs[name].keys())
            
            # print(name,fn_name(self.loss_functions[name]).lower())
            # print(list(self.loss_kwargs[name].keys()),loss_fn_name_keys)

            for n in range(N_samples):
                if name in ['total_table_distance_loss','total_table_distance_likelihood_loss',
                            'total_intensity_distance_loss','total_intensity_distance_likelihood_loss']:
                    # Find prediction data
                    pred_dataset = LOSS_DATA_REQUIREMENTS[name]['prediction_data'][0]

                    # Calculate total cost incurred by travelling from every origin
                    total_cost_predicted = torch.mul(
                        prediction_data[pred_dataset][n].to(dtype = float32),
                        validation_data['cost_matrix']
                    ).sum(dim = 1)
                    # Normalise to 1
                    normalised_total_cost_predicted = total_cost_predicted / total_cost_predicted.sum(dim = 0)
                     
                    # Reshape loss kwargs if needed
                    for key, value in deepcopy( self.loss_kwargs.get(name,{})).items():
                        if len(LOSS_KWARG_OPERATIONS.get(key,'')):
                            self.loss_kwargs[name][key] = eval(
                                LOSS_KWARG_OPERATIONS[key]['function'],
                                {
                                    "dim":np.prod(validation_data['total_cost_by_origin'].shape),
                                    "device":self.physics_model.device,
                                    key:value,
                                    "dtype":eval_dtype(LOSS_KWARG_OPERATIONS[key]['dtype'], numpy_format=False),
                                    **LOSS_KWARG_OPERATIONS[key]['kwargs']
                                },
                                globals()
                            )
                    
                    # Add to total loss
                    res = self.loss_functions[name](
                        normalised_total_cost_predicted,
                        validation_data['total_cost_by_origin'],
                        **{k:v for k,v in self.loss_kwargs[name].items() \
                           if k in loss_fn_name_keys}
                    )
                else:
                    # Add to total loss
                    pred_dataset = LOSS_DATA_REQUIREMENTS[name]['prediction_data'][0]
                    validation_dataset = LOSS_DATA_REQUIREMENTS[name]['validation_data'][0]
                    # Reshape loss kwargs if needed
                    for key, value in deepcopy( self.loss_kwargs.get(name,{})).items():
                        if len(LOSS_KWARG_OPERATIONS.get(key,'')):
                            self.loss_kwargs[name][key] = eval(
                                LOSS_KWARG_OPERATIONS[key]['function'],
                                {
                                    "dim":np.prod(validation_data[validation_dataset].shape),
                                    "device":self.physics_model.device,
                                    key:value,
                                    "dtype":eval_dtype(LOSS_KWARG_OPERATIONS[key]['dtype'], numpy_format=False),
                                    **LOSS_KWARG_OPERATIONS[key]['kwargs']
                                },
                                globals()
                            )

                    # Add to total loss
                    res = self.loss_functions[name](
                        prediction_data[pred_dataset][n].to(dtype = float32),
                        validation_data[validation_dataset].to(dtype = float32),
                        **{k:v for k,v in self.loss_kwargs[name].items() \
                            if k in loss_fn_name_keys}
                    )
                # Gather current Monte Carlo total loss
                total_loss.append(res)
            # Update loss
            previous_loss[name] = previous_loss[name] + sum(total_loss) / N_samples
            # Keep track number of loss samples per loss function
            n_processed_steps[name] = n_processed_steps[name] + 1
        return previous_loss,n_processed_steps


    def __repr__(self):
        return f"{self.physics_model.noise_regime}Noise HarrisWilson NeuralNet( {self.physics_model.intensity_model.name}(SpatialInteraction2D) )"

    def __str__(self):

        return f"""
            {'x'.join([str(d) for d in self.physics_model.intensity_model.dims])} Harris Wilson Neural Network using {self.physics_model.intensity_model.name}
            Learned parameters: {', '.join(self.physics_model.params_to_learn)}
            dt: {self.config['harris_wilson_model'].get('dt',0.001) if hasattr(self,'config') else ''}
            Noise regime: {self.physics_model.noise_regime}
        """