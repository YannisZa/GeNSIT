import os
import sys
import logging
import numpy as np
import torch

from typing import Union

from multiresticodm.config import Config
from multiresticodm.sim_models import ProductionConstrained,TotalConstrained
from multiresticodm.global_variables import SIM_DATA_TYPES, PARAMETER_DEFAULTS
from multiresticodm.utils import write_txt,makedir,str_in_list
from multiresticodm.probability_utils import log_odds_ratio_wrt_intensity


def instantiate_sim(
        config:Config,
        origin_demand=None,
        destination_demand=None,
        log_origin_attraction=None,
        log_destination_attraction=None,
        cost_matrix=None,
        dims=None,
        learned_parameters: dict = None,
        auxiliary_parameters: dict = None,
        **kwargs
    ):
    
    sim_type = ''
    if isinstance(config,Config) or hasattr(config,'settings'):
        sim_type = config.settings['spatial_interaction_model'].get('sim_type',None)
        if sim_type is not None and hasattr(sys.modules[__name__], sim_type):
            sim_type += 'SIM'
    elif isinstance(config,dict):
        sim_type = config.get('sim_type',None)
        if sim_type is not None and hasattr(sys.modules[__name__], sim_type):
            sim_type += 'SIM'
    else:
        raise ValueError(f"Input class '{sim_type}' not found")
    
    return getattr(sys.modules[__name__], sim_type)(
        config=config,
        origin_demand=origin_demand,
        destination_demand=destination_demand,
        log_origin_attraction=log_origin_attraction,
        log_destination_attraction=log_destination_attraction,
        cost_matrix=cost_matrix,
        dims=dims,
        learned_parameters = learned_parameters,
        auxiliary_parameters = auxiliary_parameters,
        **kwargs
    )
    
    
class SpatialInteraction():
    def export(self):
        pass

class SpatialInteraction2D():
    def __init__(
            self,
            config:Config,
            origin_demand=None,
            destination_demand=None,
            log_origin_attraction=None,
            log_destination_attraction=None,
            cost_matrix=None,
            dims=None,
            true_parameters: dict = None,
            device: str = None,
            **kwargs
    ):
        '''  Constructor '''

        # Import logger
        self.logger = logging.getLogger(__name__)
        self.logger.disabled = kwargs.get('disable_logger',False)
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)

        # SIM name
        self.sim_name = 'SpatialInteraction2D'
        
        # Configuration settings
        self.config = config

        # Device name
        self.device = device

        # Parameters names
        self.aux_param_names = ['bmax']
        self.free_param_names = ['alpha','beta']
        all_parameter_names = self.free_param_names + self.aux_param_names

        # Grand total
        self.grand_total = torch.tensor(self.config['spatial_interaction_model'].get('grand_total',1.0)).float()

        # Dimensions
        self.dims = None
        if dims is not None:
            self.dims = torch.tensor(dims)

        # True and auxiliary parameters
        for param in all_parameter_names:
            setattr(
                self,
                param,
                torch.tensor(true_parameters.get(param,PARAMETER_DEFAULTS[param])).float().to(self.device)
            )
            self.config.settings['spatial_interaction_model']['parameters'][param] = true_parameters.get(param,PARAMETER_DEFAULTS[param])

        # Origin demand
        self.origin_demand = None
        if origin_demand is not None:
            self.origin_demand = origin_demand

        # Destination demand
        self.destination_demand = None
        if destination_demand is not None:
            self.destination_demand = destination_demand

        # Origin attraction
        self.log_origin_attraction = None
        if log_origin_attraction is not None:
            self.log_origin_attraction = log_origin_attraction

        # Destination attraction
        self.log_destination_attraction = None
        if log_destination_attraction is not None:
            self.log_destination_attraction = log_destination_attraction

        # Cost matrix
        self.cost_matrix = None
        if cost_matrix is not None:
            self.cost_matrix = cost_matrix

        # Update dims
        if self.dims is None and self.cost_matrix is not None:
            self.dims = torch.tensor(self.cost_matrix.size())

        # Determine if true data exists
        if np.all([hasattr(self,attr) and getattr(self,attr) is not None for attr in all_parameter_names]):
            self.ground_truth_known = True
        else:
            self.ground_truth_known = False


    def export(self,dirpath:str='./synthetic_dummy',overwrite:bool=False) -> None:
        
        # Make directory if it does not exist
        makedir(dirpath)

        for attr in SIM_DATA_TYPES.keys():
            if hasattr(self,attr) and getattr(self,attr) is not None:
                # Get filepath experiment filepath
                filepath = os.path.join(dirpath,f'{attr}.txt')

                # Write experiment summaries to file
                if (not os.path.exists(filepath)) or overwrite:
                    write_txt((getattr(self,attr)).astype('int32'),filepath)

        self.logger.info(f'Spatial interaction model data successfully exported to {dirpath}')
    
    def inherit_functions(self,sim_model):
        # List of attributes to be inherited
        attributes = [
            f"_log_flow_matrix",
            f"_flow_matrix_jacobian",
            f"_destination_demand",
            # f"_log_flow_matrix_vectorised",
            # "sde_pot",
            # "sde_pot_jacobian",
            # "sde_pot_hessian",
            # "sde_pot_and_jacobian",
            # "sde_ais_pot_and_jacobian",
            # "annealed_importance_sampling_log_z",
            # "annealed_importance_sampling_log_z_parallel",
            # "destination_attraction_log_likelihood_and_jacobian",
        ]
        for attr in attributes:
            # inherit attribute
            if hasattr(sim_model,attr):
                setattr(self, attr, getattr(sim_model,attr))

    def __str__(self):

        return f"""
            {'x'.join([str(d) for d in self.dims])} {self.sim_type} Constrained Spatial Interaction Model
            Origin demand sum: {np.sum(self.origin_demand) if self.origin_demand is not None else None}
            Destination demand sum: {np.sum(self.destination_demand) if self.destination_demand is not None  else None}
            Origin attraction sum: {np.exp(self.log_origin_attraction).sum() if self.log_origin_attraction is not None  else None}
            Destination attraction sum: {np.exp(self.log_destination_attraction).sum() if self.log_destination_attraction is not None  else None}
            Cost matrix sum: {np.sum(self.cost_matrix.ravel()) if self.cost_matrix is not None else None}
            Alpha: {self.alpha}
            Beta: {self.beta}
            Beta scaling: {self.bmax}
        """
    
    def log_odds_ratio(self,log_intensity:np.ndarray):
        """ Reconstruct log odds ratio of intensity function

        Parameters
        ----------
        xx : np.array
            Log destination attraction.
        theta : np.array
            Fitted parameters.

        Returns
        -------
        np.array
            Expected flow matrix (non-integer).

        """
        return log_odds_ratio_wrt_intensity(
            log_intensity
        )

    def log_intensity(self):
        pass

    def intensity_gradient(self):
        pass

    def log_intensity_and_gradient(self):
        pass



class ProductionConstrainedSIM(SpatialInteraction2D):
    """ Object including flow (O/D) matrix inference routines of singly constrained SIM. """

    def __init__(
        self,
        config:Config,
        origin_demand=None,
        destination_demand=None,
        log_origin_attraction=None,
        log_destination_attraction=None,
        cost_matrix=None,
        dims=None,
        learned_parameters: dict = None,
        auxiliary_parameters: dict = None,
        **kwargs
    ):
        '''  Constructor '''
        # Define type of spatial interaction model
        self.sim_type = "ProductionConstrained"
        # Initialise constructor
        super().__init__(
            config=config,
            origin_demand=origin_demand,
            destination_demand=destination_demand,
            log_origin_attraction=log_origin_attraction,
            log_destination_attraction=log_destination_attraction,
            cost_matrix=cost_matrix,
            dims=dims,
            learned_parameters = learned_parameters,
            auxiliary_parameters = auxiliary_parameters,
            **kwargs
        )
        # Make sure you have the necessary data
        for attr in ['origin_demand','log_destination_attraction','cost_matrix']:
            try:
                assert hasattr(self,attr)
            except:
                raise Exception(f"{self.sim_type} requires {attr} but it is missing!")
        # Inherit numba functions
        self.inherit_functions(ProductionConstrained)

        # self.logger.info(('Building ' + self.__str__()))

    def __repr__(self):
        return "ProductionConstrained(SpatialInteraction2D)"

    def log_intensity(self,xx:torch.tensor,theta:torch.tensor,grand_total:torch.float32):
        """ Reconstruct expected flow matrices (intensity function)

        Parameters
        ----------
        xx : np.array
            Log destination attraction.
        theta : np.array
            Fitted parameters.

        Returns
        -------
        np.array
            Expected flow matrix (non-integer).

        """
        return self.log_flow_matrix(
            xx,
            theta,
            self.origin_demand,
            self.cost_matrix,
            grand_total
        )
    
    def intensity_gradient(self,theta:np.ndarray,log_intensity:np.ndarray):
        """ Reconstruct gradient of intensity with respect to xx

        Parameters
        ----------
        xx : np.array
            Log destination attraction.
        theta : np.array
            Fitted parameters.

        Returns
        -------
        np.array
            Expected flow matrix (non-integer).

        """
        return self.flow_matrix_jacobian(
            theta,
            log_intensity
        )

    

    def log_intensity_and_gradient(self,xx:np.ndarray,theta:np.ndarray,total_flow:float):
        
        # Compute log intensity
        log_intensity = self.log_intensity(xx,theta,total_flow)
        # Pack gradient
        intensity_gradient = self.intensity_gradient(theta,log_intensity)
        return log_intensity, intensity_gradient


class TotalConstrainedSIM(SpatialInteraction2D):
    """ Object including flow (O/D) matrix inference routines of SIM with only total constrained. """

    def __init__(
            self,
            config:Config,
            origin_demand=None,
            destination_demand=None,
            log_origin_attraction=None,
            log_destination_attraction=None,
            cost_matrix=None,
            dims=None,
            learned_parameters: dict = None,
            auxiliary_parameters: dict = None,
            **kwargs
    ):
        '''  Constructor '''
        # Define type of spatial interaction model
        self.sim_type = "TotalConstrained"
        # Initiliase constructor
        super().__init__(
            config=config,
            origin_demand=origin_demand,
            destination_demand=destination_demand,
            log_origin_attraction=log_origin_attraction,
            log_destination_attraction=log_destination_attraction,
            cost_matrix=cost_matrix,
            dims=dims,
            learned_parameters = learned_parameters,
            auxiliary_parameters = auxiliary_parameters,
            **kwargs
        )
        # Make sure you have the necessary data
        for attr in ['log_destination_attraction','cost_matrix']:
            try:
                assert hasattr(self,attr)
            except:
                raise Exception(f"{self.sim_type} requires {attr} but it is missing!")
        # Inherit numba functions
        self.inherit_functions(TotalConstrained)

        # self.logger.info(('Building ' + self.__str__()))

    def __repr__(self):
        return "TotalConstrained(SpatialInteraction2D)"
    
    def log_intensity(self,xx:Union[np.ndarray,torch.tensor],theta:Union[np.ndarray,torch.tensor],grand_total:torch.float32):
        """ Reconstruct expected flow matrices (intensity function)

        Parameters
        ----------
        xx : np.array or torch tensor
            Log destination attraction.
        theta : np.array or torch tensor
            Fitted parameters.

        Returns
        -------
        np.array or torch tensor
            Expected flow matrix (non-integer).

        """

        return self._log_flow_matrix(
                xx,
                theta,
                self.origin_demand,
                self.cost_matrix,
                grand_total
        )

    def intensity_demand(self,W_alpha,C_beta):
        return self._destination_demand(W_alpha,C_beta,self.origin_demand,self.grand_total)
    
    def intensity_gradient(self,theta:np.ndarray,log_intensity:np.ndarray):
        """ Reconstruct gradient of intensity with respect to xx

        Parameters
        ----------
        xx : np.array
            Log destination attraction.
        theta : np.array
            Fitted parameters.

        Returns
        -------
        np.array
            Expected flow matrix (non-integer).

        """
        return self.flow_matrix_jacobian(
            theta,
            log_intensity
        )

    def log_intensity_and_gradient(self,xx:np.ndarray,theta:np.ndarray,total_flow:float):
        
        # Compute log intensity
        log_intensity = self.log_intensity(xx,theta,total_flow)
        # Pack gradient
        intensity_gradient = self.intensity_gradient(theta,log_intensity)
        return log_intensity, intensity_gradient