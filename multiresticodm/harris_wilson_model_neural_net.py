#!/usr/bin/env python3
'''
    Code extended from https://github.com/ThGaskin/NeuralABM
'''

from copy import deepcopy
import sys
import torch
import numpy as np

from torch import nn, float32
from typing import Any, List, Union

from multiresticodm.config import Config
from multiresticodm.utils import setup_logger
import multiresticodm.probability_utils as ProbabilityUtils
from multiresticodm.harris_wilson_model import HarrisWilson
from multiresticodm.global_variables import ACTIVATION_FUNCS, OPTIMIZERS, LOSS_FUNCTIONS, LOSS_DATA_REQUIREMENTS


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
            _f = ACTIVATION_FUNCS[layer_cfg.lower()]
            if _f[1]:
                return _f[0]()
            else:
                return _f[0]

        # Entry is a dictionary containing args and kwargs
        elif isinstance(layer_cfg, dict):
            _f = ACTIVATION_FUNCS[layer_cfg.get("name").lower()]
            if _f[1]:
                return _f[0](*layer_cfg.get("args", ()), **layer_cfg.get("kwargs", {}))
            else:
                return _f[0]

        elif layer_cfg is None:
            _f = ACTIVATION_FUNCS["linear"][0]

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
        input_size: int,
        output_size: int,
        num_layers: int,
        nodes_per_layer: dict,
        activation_funcs: dict,
        biases: dict,
        optimizer: str = 'Adam',
        learning_rate: float = 0.001,
        **__,
    ):
        '''

        :param input_size: the number of input values
        :param output_size: the number of output values
        :param num_layers: the number of hidden layers
        :param nodes_per_layer: a dictionary specifying the number of nodes per layer
        :param activation_funcs: a dictionary specifying the activation functions to use
        :param biases: a dictionary containing the initialisation parameters for the bias
        :param optimizer: the name of the optimizer to use. Default is the torch.optim.Adam optimizer.
        :param learning_rate: the learning rate of the optimizer. Default is 1e-3.
        :param __: Additional model parameters (ignored)
        '''

        super().__init__()
        self.flatten = nn.Flatten()

        self.input_dim = input_size
        self.output_dim = output_size
        self.hidden_dim = num_layers

        # Get architecture, activation functions, and layer bias
        self.architecture = get_architecture(
            input_size, 
            output_size, 
            num_layers, 
            nodes_per_layer
        )
        self.activation_funcs = get_activation_funcs(
            num_layers, 
            activation_funcs
        )
        self.bias = get_bias(
            num_layers, 
            biases
        )

        # Add the neural net layers
        self.layers = nn.ModuleList()
        for i in range(len(self.architecture) - 1):
            layer = nn.Linear(
                self.architecture[i],
                self.architecture[i + 1],
                bias=self.bias[i] is not None,
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
        self.optimizer = OPTIMIZERS[optimizer](self.parameters(), lr=learning_rate)

    # ... Evaluation functions .........................................................................................

    # The model forward pass
    def forward(self, x):
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
        rng: np.random.Generator,
        neural_net: NeuralNet,
        loss: dict,
        physics_model: HarrisWilson,
        config: Config = None,
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
        self.logger.setLevels(
            console_level = level
        )
        # Store random number generator
        self._rng = rng

        # The numeric    solver
        self.physics_model = physics_model

        # Config file
        self.config = config

        # Initialise neural net, loss tracker and prediction tracker
        self._neural_net = neural_net
        self._neural_net.optimizer.zero_grad()

        # Store loss function parameters
        self.loss_functions = {}
        # Parse loss functions
        for name,function in zip(loss['loss_name'],loss['loss_function']):
            # Get loss function
            loss_func = LOSS_FUNCTIONS.get(function.lower(),None)
            loss_func = loss.get(name,loss_func) if loss_func is None else loss_func()
            if name == 'table_likelihood_loss':
                loss_func = ProbabilityUtils.log_poisson_pmf_unnormalised
            if loss_func is None:
                raise Exception(f"Loss {name} is missing loss function {function}.")
            else:
                self.loss_functions[name] = loss_func

        self._loss_sample = torch.tensor(0.0, requires_grad=False)
        self._theta_sample = torch.stack(
            [torch.tensor(0.0, requires_grad=False)] * len(self.physics_model.params_to_learn)
        )

    def evaluate_loss(self,prediction_data:dict,validation_data:dict,aux_inputs:dict={}):
        self.logger.debug('Loss function update')
        result = torch.tensor(0.0, requires_grad=True) 
        for name in self.loss_functions.keys():
            # Make sure you have the necessary data
            for pred_dataset in LOSS_DATA_REQUIREMENTS[name]['prediction_data']:
                try:
                    assert prediction_data.get(pred_dataset,None) is not None
                except:
                    raise Exception(f"Loss {name} is missing prediction data {pred_dataset}.")
            for validation_dataset in LOSS_DATA_REQUIREMENTS[name]['validation_data']:
                try:
                    validation_data[validation_dataset] = validation_data.get(validation_dataset) \
                        if validation_data.get(validation_dataset,None) is not None \
                        else aux_inputs.get(validation_dataset,None)
                    assert validation_data[validation_dataset] is not None
                except:
                    raise Exception(f"Loss {name} is missing validation data {validation_dataset}.")

            if name == 'total_distance_travelled_loss':
                # Calculate total cost incurred by travelling from every origin
                total_cost_predicted = torch.tensordot(
                    prediction_data['table'].to(dtype=float32),
                    validation_data['cost_matrix']
                ).sum(dim=0)
                # Normalise to 1
                normalised_total_cost_predicted = total_cost_predicted / total_cost_predicted.sum(dim=0)
                # Add to total loss
                res = self.loss_functions[name](
                    normalised_total_cost_predicted,
                    validation_data['total_cost_by_origin']
                )
            else:
                # Add to total loss
                pred_dataset = LOSS_DATA_REQUIREMENTS[name]['prediction_data'][0]
                validation_dataset = LOSS_DATA_REQUIREMENTS[name]['validation_data'][0]
                res = self.loss_functions[name](
                    prediction_data[pred_dataset].to(dtype=float32),
                    validation_data[validation_dataset].to(dtype=float32)
                )
        result = result + res
        return result

    def epoch(
        self,
        *,
        experiment,
        batch_size: int,
        validation_data: dict,
        prediction_data: dict = {},
        loss_functions: dict = {},
        aux_inputs:dict = {},
        dt: float = None,
        **kwargs,
    ):

        '''Trains the model for a single epoch.

        :param training_data: the training data
        :param batch_size: the number of time series elements to process before conducting a gradient descent
                step
        :param epsilon: (optional) the epsilon value to use during training
        :param dt: (optional) the time differential to use during training
        :param __: other parameters (ignored)
        '''

        # Track the training loss
        loss = torch.tensor(0.0, requires_grad=True)

        # Count the number of batch items processed
        experiment.n_processed_steps = 0

        # Copy validation data to override on the fly
        validation_data_copy = deepcopy(validation_data)

        # Process the training set elementwise, updating the loss after batch_size steps
        for t, data in enumerate(validation_data['destination_attraction_ts']):
            predicted_theta = self._neural_net(torch.flatten(data))
            predicted_dest_attraction = self.physics_model.run_single(
                curr_destination_attractions=data,
                free_parameters=predicted_theta,
                dt=dt,
                requires_grad=True,
            )

            # Add destination attraction data for this time step
            validation_data_copy['destination_attraction_ts'] = data
            # Add prediction of destination attraction to predicted data
            # for evaluating the loss function
            prediction_data['destination_attraction_ts'] = torch.flatten(predicted_dest_attraction)
            # Update loss
            loss = loss + self.evaluate_loss(
                data = validation_data_copy,
                predictions = prediction_data,
                aux_inputs = aux_inputs
            )

            experiment.n_processed_steps += 1

            # Update the model parameters after every batch and clear the loss
            if t % batch_size == 0 or t == len(validation_data) - 1:
                loss.backward(retain_graph=True)
                self._neural_net.optimizer.step()
                self._neural_net.optimizer.zero_grad()
                self._time += 1
                self._loss_sample = (
                    loss.clone().detach().cpu().numpy().item() / experiment.n_processed_steps
                )
                self._theta_sample = predicted_theta.clone().detach().cpu()
                self._log_destination_attraction_sample = torch.log(predicted_dest_attraction).clone().detach().cpu()
                del loss
                loss = torch.tensor(0.0, requires_grad=True)
                experiment.n_processed_steps = 0

        return loss, predicted_theta, torch.log(predicted_dest_attraction.squeeze())


    def epoch_time_step(
        self,
        *,
        experiment,
        validation_data: dict,
        dt: float,
        **__,
    ):

        '''Trains the model for a single epoch and time step.

        :param training_data: the training data
        :param batch_size: the number of time series elements to process before conducting a gradient descent step
        :param epsilon: (optional) the epsilon value to use during training
        :param dt: (optional) the time differential to use during training
        :param __: other parameters (ignored)
        '''
        self.logger.debug('Running neural net')
        predicted_theta = self._neural_net(torch.flatten(validation_data['destination_attraction_ts']))
        self.logger.debug('Forward pass on SDE')
        predicted_dest_attraction = self.physics_model.run_single(
            curr_destination_attractions=validation_data['destination_attraction_ts'],
            free_parameters=predicted_theta,
            dt=dt,
            requires_grad=True,
        )
                
        # Update number of processed steps
        experiment.n_processed_steps += 1
        return predicted_theta, predicted_dest_attraction


    def __repr__(self):
        return f"{self.physics_model.noise_regime}Noise HarrisWilson NeuralNet( {self.physics_model.intensity_model.name}(SpatialInteraction2D) )"

    def __str__(self):

        return f"""
            {'x'.join([str(d) for d in self.physics_model.intensity_model.dims])} Harris Wilson Neural Network using {self.physics_model.intensity_model.name} Constrained Spatial Interaction Model
            Learned parameters: {', '.join(self.physics_model.params_to_learn)}
            dt: {self.config['harris_wilson_model'].get('dt',0.001) if hasattr(self,'config') else ''}
            Noise regime: {self.physics_model.noise_regime}
        """