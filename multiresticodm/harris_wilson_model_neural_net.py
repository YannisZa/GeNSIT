#!/usr/bin/env python3
'''
    Code extended from https://github.com/ThGaskin/NeuralABM
'''

import sys
import time
import torch
import coloredlogs
import h5py as h5
import numpy as np
import torch

from torch import nn
from dantro import logging
from typing import Any, List, Union

from multiresticodm.utils import safe_delete
from multiresticodm.harris_wilson_model import HarrisWilson
from multiresticodm.global_variables import ACTIVATION_FUNCS, OPTIMIZERS, LOSS_FUNCTIONS

log = logging.getLogger(__name__)
coloredlogs.install(fmt='%(levelname)s %(message)s', level='INFO', logger=log)

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
            input_size, output_size, num_layers, nodes_per_layer
        )
        self.activation_funcs = get_activation_funcs(num_layers, activation_funcs)
        self.bias = get_bias(num_layers, biases)

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
        # h5group: h5.Group,
        neural_net: NeuralNet,
        loss_function: dict,
        physics_model: HarrisWilson,
        to_learn: list,
        write_every: int = 1,
        write_start: int = 1,
        **__,
    ):
        '''Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        Args:
            rng (np.random.Generator): The shared RNG
            neural_net: The neural cost_matrix
            loss_function (dict): the loss function to use
            physics_model: The numerical solver
            to_learn: the list of parameter names to learn
            write_every: write every iteration
            write_start: iteration at which to start writing
            num_steps: number of iterations of the physics_model
        '''
        # self._h5group = h5group
        self._rng = rng

        # The numerical solver
        self._physics_model = physics_model

        # Config file
        self.config = physics_model.config
        safe_delete(physics_model.config)

        # Initialise neural net, loss tracker and prediction tracker
        self._neural_net = neural_net
        self._neural_net.optimizer.zero_grad()
        self.loss_function = LOSS_FUNCTIONS[loss_function.get('name').lower()](
            loss_function.get('args', None), **loss_function.get('kwargs', {})
        )
        self._current_loss = torch.tensor(0.0, requires_grad=False)
        self._current_predictions = torch.stack(
            [torch.tensor(0.0, requires_grad=False)] * len(to_learn)
        )

        # Setup chunked dataset to store the state data in
        # self._dset_loss = self._h5group.create_dataset(
        #     'loss',
        #     (0,),
        #     maxshape=(None,),
        #     chunks=True,
        #     compression=3,
        # )
        # self._dset_loss.attrs['dim_names'] = ['time']
        # self._dset_loss.attrs['coords_mode__time'] = 'start_and_step'
        # self._dset_loss.attrs['coords__time'] = [write_start, write_every]

        # self.dset_time = self._h5group.create_dataset(
        #     'computation_time',
        #     (0,),
        #     maxshape=(None,),
        #     chunks=True,
        #     compression=3,
        # )
        # self.dset_time.attrs['dim_names'] = ['epoch']
        # self.dset_time.attrs['coords_mode__epoch'] = 'trivial'

        # dset_predictions = []
        # for p_name in to_learn:
        #     dset = self._h5group.create_dataset(
        #         p_name, (0,), maxshape=(None,), chunks=True, compression=3
        #     )
        #     dset.attrs['dim_names'] = ['time']
        #     dset.attrs['coords_mode__time'] = 'start_and_step'
        #     dset.attrs['coords__time'] = [write_start, write_every]

        #     dset_predictions.append(dset)
        # self._dset_predictions = dset_predictions

        # Count the number of gradient descent steps
        self._time = 0
        self._write_every = write_every
        self._write_start = write_start

    def epoch(
        self,
        *,
        training_data: torch.tensor,
        batch_size: int,
        dt: float = None,
        **__,
    ):

        '''Trains the model for a single epoch.

        :param training_data: the training data
        :param batch_size: the number of time series elements to process before conducting a gradient descent
                step
        :param epsilon: (optional) the epsilon value to use during training
        :param dt: (optional) the time differential to use during training
        :param __: other parameters (ignored)
        '''

        # Track the epoch training time
        start_time = time.time()

        # Track the training loss
        loss = torch.tensor(0.0, requires_grad=True)

        # Count the number of batch items processed
        n_processed_steps = 0

        # Process the training set elementwise, updating the loss after batch_size steps
        for t, sample in enumerate(training_data):

            predicted_parameters = self._neural_net(torch.flatten(sample))
            predicted_data = self._physics_model.run_single(
                curr_destination_attractions=sample,
                free_parameters=predicted_parameters,
                dt=dt,
                requires_grad=True,
            )

            # Update loss
            loss = loss + self.loss_function(predicted_data, sample)

            n_processed_steps += 1

            # Update the model parameters after every batch and clear the loss
            if t % batch_size == 0 or t == len(training_data) - 1:
                loss.backward()
                self._neural_net.optimizer.step()
                self._neural_net.optimizer.zero_grad()
                self._time += 1
                self._current_loss = (
                    loss.clone().detach().cpu().numpy().item() / n_processed_steps
                )
                self._current_predictions = predicted_parameters.clone().detach().cpu()
                # self.write_data()
                del loss
                loss = torch.tensor(0.0, requires_grad=True)
                n_processed_steps = 0

        # Write the epoch training time (wall clock time)
        # self.dset_time.resize(self.dset_time.shape[0] + 1, axis=0)
        # self.dset_time[-1] = time.time() - start_time

    def write_data(self):
        '''Write the current state into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        '''
        if self._time >= self._write_start and (self._time % self._write_every == 0):

            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1] = self._current_loss

            for idx, dset in enumerate(self._dset_predictions):
                dset.resize(dset.shape[0] + 1, axis=0)
                dset[-1] = self._current_predictions[idx]
