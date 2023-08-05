import os
import sys
import torch
import logging

from multiresticodm.config import Config
from multiresticodm.sim_models import ProductionConstrained,TotallyConstrained
from multiresticodm.global_variables import INPUT_TYPES, PARAMETER_DEFAULTS, Dataset
from multiresticodm.utils import setup_logger, write_txt,makedir
from multiresticodm.probability_utils import log_odds_ratio_wrt_intensity


def instantiate_sim(
        sim_type:str,
        config:Config=None,
        true_parameters=None,
        device=None,
        **kwargs
    ):
    
    if sim_type is not None and hasattr(sys.modules[__name__], sim_type):
        sim_type += 'SIM'
    else:
        raise ValueError(f"Input class '{sim_type}' not found")
    
    return getattr(sys.modules[__name__], sim_type)(
        config=config,
        true_parameters=true_parameters,
        device=device,
        **kwargs
    )
    
    
class SpatialInteraction():
    def export(self):
        pass

class SpatialInteraction2D():
    REQUIRED_INPUTS = []
    REQUIRED_OUTPUTS = []
    def __init__(
            self,
            config:Config=None,
            true_parameters: dict = None,
            device: str = None,
            **kwargs
    ):
        '''  Constructor '''
        # Setup logger
        self.level = config.level if hasattr(config,'level') else kwargs.get('level','INFO')
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            level=self.level,
            log_to_file=kwargs.get('log_to_file',True),
            log_to_console=kwargs.get('log_to_console',True),
        )

        # SIM name
        self.sim_name = 'SpatialInteraction2D'
        
        # Configuration settings
        if config is not None:
            self.config = config

        # Device name
        self.device = device

        # Instantiate dataset 
        self.data = Dataset()

        # Parameters names
        self.aux_param_names = ['bmax']
        self.free_param_names = ['alpha','beta']
        self.all_parameter_names = self.free_param_names + self.aux_param_names

        # Attribute names
        self.attribute_names = ['dims','grand_total']


        # True and auxiliary parameters
        if true_parameters is not None:
            for param in self.all_parameter_names:
                setattr(
                    self,
                    param,
                    torch.tensor(true_parameters.get(param,PARAMETER_DEFAULTS[param])).float().to(self.device)
                )
                self.config.settings['spatial_interaction_model']['parameters'][param] = true_parameters.get(param,PARAMETER_DEFAULTS[param])

        # Read data passed
        for attr in self.data_names:
            setattr(self.data,attr,None)
            if kwargs.get(attr,None) is not None:
                setattr(self.data,attr,kwargs.get(attr,None))
        
        # Grand total
        self.grand_total = kwargs.get('grand_total',torch.tensor(1).int().to(self.device))

        # Get dimensions
        self.dims = kwargs.get('dims',None)
        if self.dims is None and \
            hasattr(self.data,'cost_matrix') and \
            self.data.cost_matrix is not None:
            
            self.dims = list(self.data.cost_matrix.size())
        # Update config
        if hasattr(self,'config'):
            self.config.settings['inputs']['dims'] = self.dims

        # Determine if true data exists
        if all([hasattr(self,attr) and getattr(self,attr) is not None for attr in self.all_parameter_names]):
            self.ground_truth_known = True
        else:
            self.ground_truth_known = False

    def update(self,**kwargs):
        for k,v in kwargs.items():
            if hasattr(self.data,k) and k in self.data_names:
                setattr(self.data,k,v)

    def export(self,dirpath:str='./synthetic_dummy',overwrite:bool=False) -> None:
        
        # Make directory if it does not exist
        makedir(dirpath)

        for attr in INPUT_TYPES.keys():
            if hasattr(self.data,attr) and getattr(self.data,attr) is not None:
                # Get filepath experiment filepath
                filepath = os.path.join(dirpath,f'{attr}.txt')

                # Write experiment summaries to file
                if (not os.path.exists(filepath)) or overwrite:
                    write_txt((getattr(self.data,attr)).astype('int32'),filepath)

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
            {'x'.join([str(d.cpu().detach().numpy()) for d in self.dims])} {self.sim_type} Constrained Spatial Interaction Model
            Origin demand sum: {torch.sum(self.origin_demand) if self.origin_demand is not None else None}
            Destination demand sum: {torch.sum(self.destination_demand) if self.destination_demand is not None  else None}
            Origin attraction sum: {torch.sum(torch.exp(self.log_origin_attraction)) if self.log_origin_attraction is not None  else None}
            Destination attraction sum: {torch.sum(torch.exp(self.log_destination_attraction)) if self.log_destination_attraction is not None  else None}
            Cost matrix sum: {torch.sum(torch.ravel(self.cost_matrix)) if self.cost_matrix is not None else None}
            Alpha: {self.alpha}
            Beta: {self.beta}
            Beta scaling: {self.bmax}
        """

    def check_sample_availability(self,output_names:list,data:dict):
        available = True
        for sample in output_names:
            try:
                assert sample in list(data.keys())
            except:
                available = False
                self.logger.error(f"Sample {sample} is required but does not exist in {','.join(data.keys())}")
        
        return available

    def get_input_kwargs(self,passed_kwargs):
        
        kwargs = {}

        for key in self.data_names:
            # Try to read data from passed kwargs
            # Then try to read stored data
            # else return None
            kwargs[key] = passed_kwargs.pop(key,getattr(self.data,key,None))
        
        for key in self.attribute_names:
            # Try to read attribute from passed kwargs
            # Then try to read stored attribute
            # else return None
            kwargs[key] = passed_kwargs.pop(key,getattr(self,key,None))
        
        kwargs = {**kwargs,**passed_kwargs}

        return kwargs
    
    def log_odds_ratio(self,log_intensity:torch.tensor):
        """ Reconstruct log odds ratio of intensity function

        Parameters
        ----------
        xx : torch.tensor
            Log destination attraction.
        theta : torch.tensor
            Fitted parameters.

        Returns
        -------
        torch.tensor
            Expected flow matrix (non-integer).

        """
        return log_odds_ratio_wrt_intensity(
            log_intensity
        )
    

    def log_intensity(self,**kwargs):
        """ Reconstruct expected flow matrices (intensity function)

        Parameters
        ----------
        grand_total : torch tensor
            Total intensity (equal to table total).

        Returns
        -------
        torch tensor
            Continuous intensity matrix (non-integer).

        """
        
        # Update input kwargs if required
        updated_kwargs = self.get_input_kwargs(kwargs)

        self.check_sample_availability(self.REQUIRED_OUTPUTS+self.REQUIRED_INPUTS,updated_kwargs)

        return self._log_flow_matrix(
                **updated_kwargs
        )

    def intensity_demand(self,**kwargs):
        
        # Update input kwargs if required
        updated_kwargs = self.get_input_kwargs(kwargs)

        self.check_sample_availability(self.REQUIRED_OUTPUTS+self.REQUIRED_INPUTS,updated_kwargs)

        return self._destination_demand(
                **updated_kwargs
        )
    
    def intensity_gradient(self):
        pass

    def log_intensity_and_gradient(self):
        pass
    



class ProductionConstrainedSIM(SpatialInteraction2D):
    """ Object including flow (O/D) matrix inference routines of singly constrained SIM. """
    REQUIRED_INPUTS = ['cost_matrix','origin_demand']
    REQUIRED_OUTPUTS = ['alpha','beta','log_destination_attraction']
    def __init__(
        self,
        config:Config=None,
        true_parameters=None,
        device=None,
        **kwargs
    ):
        '''  Constructor '''
        # Get data names
        self.data_names = ['origin_demand','destination_demand','cost_matrix']
        
        # Initialise constructor
        super().__init__(
            config=config,
            true_parameters=true_parameters,
            device=device,
            **kwargs
        )
        # Define type of spatial interaction model
        self.sim_type = "ProductionConstrained"
        # Make sure you have the necessary data
        for attr in self.REQUIRED_INPUTS:
            try:
                assert hasattr(self.data,attr)
            except:
                raise Exception(f"{self.sim_type} requires {attr} but it is missing!")
        # Inherit numba functions
        self.inherit_functions(ProductionConstrained)

        # self.logger.info(('Building ' + self.__str__()))

    def __repr__(self):
        return "ProductionConstrained(SpatialInteraction2D)"
    
    def intensity_gradient(self,theta:torch.tensor,log_intensity:torch.tensor):
        """ Reconstruct gradient of intensity with respect to xx

        Parameters
        ----------
        xx : torch.tensor
            Log destination attraction.
        theta : torch.tensor
            Fitted parameters.

        Returns
        -------
        torch.tensor
            Expected flow matrix (non-integer).

        """
        return self.flow_matrix_jacobian(
            theta,
            log_intensity
        )

    

    def log_intensity_and_gradient(self,xx:torch.tensor,theta:torch.tensor,grand_total:float):
        
        # Compute log intensity
        log_intensity = self.log_intensity(xx,theta,grand_total)
        # Pack gradient
        intensity_gradient = self.intensity_gradient(theta,log_intensity)
        return log_intensity, intensity_gradient


class TotallyConstrainedSIM(SpatialInteraction2D):
    """ Object including flow (O/D) matrix inference routines of SIM with only total constrained. """
    REQUIRED_INPUTS = ['cost_matrix','origin_demand']
    REQUIRED_OUTPUTS = ['alpha','beta','log_destination_attraction']
    def __init__(
        self,
        config:Config=None,
        true_parameters=None,
        device=None,
        **kwargs
    ):
        '''  Constructor '''
        # Get data names
        self.data_names = ['origin_demand','destination_demand','cost_matrix']

        # Initialise constructor
        super().__init__(
            config=config,
            true_parameters=true_parameters,
            device=device,
            **kwargs
        )
        # Define type of spatial interaction model
        self.sim_type = "TotallyConstrained"
        # Make sure you have the necessary data
        for attr in self.REQUIRED_INPUTS:
            try:
                assert hasattr(self.data,attr)
            except:
                raise Exception(f"{self.sim_type} requires {attr} but it is missing!")
        # Inherit numba functions
        self.inherit_functions(TotallyConstrained)

        # self.logger.info(('Building ' + self.__str__()))

    def __repr__(self):
        return "TotallyConstrained(SpatialInteraction2D)"

    
    def intensity_gradient(self,theta:torch.tensor,log_intensity:torch.tensor):
        """ Reconstruct gradient of intensity with respect to xx

        Parameters
        ----------
        xx : torch.tensor
            Log destination attraction.
        theta : torch.tensor
            Fitted parameters.

        Returns
        -------
        torch.tensor
            Expected flow matrix (non-integer).

        """
        return self.flow_matrix_jacobian(
            theta,
            log_intensity
        )

    def log_intensity_and_gradient(self,xx:torch.tensor,theta:torch.tensor,grand_total:float):
        
        # Compute log intensity
        log_intensity = self.log_intensity(xx,theta,grand_total)
        # Pack gradient
        intensity_gradient = self.intensity_gradient(theta,log_intensity)
        return log_intensity, intensity_gradient