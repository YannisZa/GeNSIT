import os
import sys
import torch

from numpy import array,ndarray

from gensit.config import Config
from gensit.utils.misc_utils import setup_logger, to_json_format
from gensit.static.global_variables import PARAMETER_DEFAULTS, Dataset, INTENSITY_INPUTS, INTENSITY_OUTPUTS
from gensit.utils.probability_utils import log_odds_ratio_wrt_intensity
from gensit.intensity_models.spatial_interaction_models import ProductionConstrained,TotallyConstrained

def instantiate_sim(
        config:Config,
        name:str = None,
        **kwargs
    ):
    if name is None:
        name = config.settings['spatial_interaction_model']['name']
    if hasattr(sys.modules[__name__], name):
        name += 'SIM'
    else:
        raise ValueError(f"Input class '{name}' not found")
    
    return getattr(sys.modules[__name__], name)(
        config = config,
        **kwargs
    )
    
    
class SpatialInteraction():
    def __init__(
            self,
            **kwargs
    ):
        '''  Constructor '''
        # Setup logger
        level = kwargs.get('level',None)
        self.logger = setup_logger(
            __name__+kwargs.get('instance',''),
            console_level = level,
            
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevels( console_level = level )
        
    def export(self):
        pass

class SpatialInteraction2D(SpatialInteraction):
    REQUIRED_INPUTS = []
    REQUIRED_OUTPUTS = []
    def __init__(
            self,
            config: Config = None,
            **kwargs
    ):
        '''  Constructor '''
        super().__init__(**kwargs)
        # SIM name
        self.dims_names = ['origin','destination']
        
        # Config
        self.config = config
        # Configuration settings
        if self.config is not None:
            # True parameters
            true_parameters = self.config['spatial_interaction_model']['parameters']
            # Device name
            self.device = self.config['inputs']['device']
        else:
            # True parameters
            true_parameters = kwargs.get('true_parameters',{})
            # Device name
            self.device = kwargs.get('device','cpu')

        # Instantiate data and parameter dataset 
        self.data = Dataset()
        self.params = Dataset()

        # Parameters names
        self.param_names = ['alpha','beta']

        # Attribute names
        self.attribute_names = ['dims']

        # Auxiliary parameters
        for param in self.param_names:    
            setattr(
                self.params,
                param,
                true_parameters.get(param,PARAMETER_DEFAULTS[param])
            )
            if self.config is not None and param not in self.config.settings['spatial_interaction_model']['parameters']:
                self.config.settings['spatial_interaction_model']['parameters'][param] = to_json_format(true_parameters.get(param,PARAMETER_DEFAULTS[param]))

        # Read data passed
        for attr in self.REQUIRED_INPUTS:
            setattr(self.data,attr,None)
            if kwargs.get(attr,None) is not None:
                setattr(self.data,attr,kwargs.get(attr,None))
        # Grand total
        self.grand_total = kwargs.get('grand_total',torch.tensor(1).float().to(self.device))

        # Get dimensions
        self.dims = kwargs.get('dims',None)
        if self.dims is None and \
            hasattr(self.data,'cost_matrix') and \
            self.data.cost_matrix is not None:
            if torch.is_tensor(self.data.cost_matrix):
                dims = array(list(self.data.cost_matrix.size()))
            elif isinstance(self.data.cost_matrix,ndarray):
                dims = list(self.data.cost_matrix.shape)
            else:
                dims = [self.data.cost_matrix.sizes[d] for d in self.dims_names]
            self.dims = dict(zip(self.dims_names,dims))
        # Update config
        self.config.settings['inputs']['dims'] = self.dims

        # Determine if true data exists
        if all([hasattr(self,attr) and getattr(self,attr) is not None for attr in self.param_names]):
            self.ground_truth_known = True
        else:
            self.ground_truth_known = False

    def update(self,**kwargs):
        for k,v in kwargs.items():
            if hasattr(self.data,k) and k in self.REQUIRED_INPUTS:
                setattr(self.data,k,v)

    
    def inherit_functions(self,sim_model):
        # List of attributes to be inherited
        attributes = [
            "log_flow_matrix",
            "flow_matrix_jacobian",
            "destination_demand",
            "sde_pot",
            "sde_pot_jacobian",
            "sde_pot_hessian",
            "sde_pot_and_jacobian",
            "sde_ais_pot_and_jacobian",
            "annealed_importance_sampling_log_z",
            "destination_attraction_log_likelihood_and_jacobian",
        ]
        for attr in attributes:
            # inherit attribute
            if hasattr(sim_model,attr):
                setattr(self, attr, getattr(sim_model,attr))

    def __str__(self):

        return f"""
            {'x'.join([str(d) for d in self.dims.values()])} {self.name} Spatial Interaction Model
            Origin demand sum: {torch.sum(self.data.origin_demand) if getattr(self.data,'origin_demand',None) is not None else None}
            Destination demand sum: {torch.sum(self.data.destination_demand) if getattr(self.data,'destination_demand',None) is not None  else None}
            Origin attraction sum: {torch.sum(torch.exp(self.data.log_origin_attraction)) if getattr(self.data,'log_origin_attraction',None) is not None  else None}
            Destination attraction sum: {torch.sum(self.data.destination_attraction_ts) if getattr(self.data,'destination_attraction_ts',None) is not None else None}
            Cost matrix sum: {torch.sum(torch.ravel(self.data.cost_matrix)) if getattr(self.data,'cost_matrix',None) is not None else None}
            Alpha: {self.params.alpha if getattr(self.params,'alpha',None) is not None else None}
            Beta: {self.params.beta if getattr(self.params,'beta',None) is not None else None}
            Beta scaling: {self.params.bmax if getattr(self.params,'bmax',None) is not None else None}
        """

    def get_input_kwargs(self,passed_kwargs):
        kwargs = {}
        for key in self.REQUIRED_INPUTS:
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

    def check_sample_availability(self,output_names:list,data:dict):
        available = True
        for sample in output_names:
            try:
                assert sample in list(data.keys())
            except:
                available = False
                self.logger.error(f"Sample {sample} is required but does not exist in {','.join(data.keys())}")
        
        return available
    
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
        self.logger.debug('Computing intensity')
        
        # Update input kwargs if required
        updated_kwargs = self.get_input_kwargs(kwargs)

        self.check_sample_availability(self.REQUIRED_OUTPUTS+self.REQUIRED_INPUTS,updated_kwargs)

        return self.log_flow_matrix(**updated_kwargs)
    

    def intensity_demand(self,**kwargs):
        
        self.logger.debug('Computing intensity demand')

        # Compute log flow
        log_flow = self.log_intensity(**kwargs)
        # Squeeze output
        log_flow = torch.squeeze(log_flow)
        # Compute destination demand
        log_destination_demand = torch.logsumexp(log_flow,dim = 0)
        
        return torch.exp(log_destination_demand)
    

    def intensity_gradient(self,**kwargs):
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

        # Update input kwargs if required
        updated_kwargs = self.get_input_kwargs(kwargs)

        self.check_sample_availability(self.REQUIRED_OUTPUTS+self.REQUIRED_INPUTS,updated_kwargs)

        return self.flow_matrix_jacobian(**updated_kwargs)

    def log_intensity_and_gradient(self,**kwargs):
        # Compute log intensity
        log_intensity = self.log_intensity(**kwargs)
        # Pack gradient
        intensity_gradient = self.intensity_gradient(**kwargs)
        return log_intensity, intensity_gradient
    


class ProductionConstrainedSIM(SpatialInteraction2D):
    """ Object including flow (O/D) matrix inference routines of singly constrained SIM. """
    REQUIRED_INPUTS = INTENSITY_INPUTS['ProductionConstrained']
    REQUIRED_OUTPUTS = INTENSITY_OUTPUTS['ProductionConstrained']
    def __init__(
        self,
        config:Config = None,
        true_parameters:dict = {},
        device:str = None,
        **kwargs
    ):
        '''  Constructor '''
        # Initialise constructor
        super().__init__(
            config = config,
            true_parameters = true_parameters,
            device = device,
            **kwargs
        )
        # Define type of spatial interaction model
        self.name = "ProductionConstrained"
        # Make sure you have the necessary data
        for attr in self.REQUIRED_INPUTS:
            try:
                assert hasattr(self.data,attr)
            except:
                raise Exception(f"{self.name} requires {attr} but it is missing!")
        # Inherit sim-specific functions
        self.inherit_functions(ProductionConstrained)

        self.logger.progress(('Building ' + self.__str__()))

    def __repr__(self):
        return "ProductionConstrained(SpatialInteraction2D)"
    
    
class TotallyConstrainedSIM(SpatialInteraction2D):
    """ Object including flow (O/D) matrix inference routines of SIM with only total constrained. """
    REQUIRED_INPUTS = INTENSITY_INPUTS['TotallyConstrained']
    REQUIRED_OUTPUTS = INTENSITY_OUTPUTS['TotallyConstrained']
    def __init__(
        self,
        config:Config = None,
        true_parameters = {},
        device:str = None,
        **kwargs
    ):
        '''  Constructor '''
        # Initialise constructor
        super().__init__(
            config = config,
            true_parameters = true_parameters,
            device = device,
            **kwargs
        )
        # Define type of spatial interaction model
        self.name = "TotallyConstrained"
        # Make sure you have the necessary data
        for attr in self.REQUIRED_INPUTS:
            try:
                assert hasattr(self.data,attr)
            except:
                raise Exception(f"{self.name} requires {attr} but it is missing!")
        
        # Inherit sim-specific functions
        self.inherit_functions(TotallyConstrained)

        self.logger.progress(('Building ' + self.__str__()))

    def __repr__(self):
        return "TotallyConstrained(SpatialInteraction2D)"
