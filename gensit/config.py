import json
import sys
import math
import os.path
import traceback

from toml import load
from copy import deepcopy
from itertools import product
from collections.abc import Iterable

from gensit import ROOT
from gensit.utils.exceptions import *
from gensit.static.global_variables import deep_walk, CORE_COORDINATES_DTYPES
from gensit.utils.config_data_structures import instantiate_data_type
from gensit.utils.misc_utils import deep_apply, flatten, setup_logger, read_json, expand_tuple, unique, sigma_to_noise_regime, stringify, string_to_numeric, print_json

class Config:

    def __init__(self, path:str = None, settings:dict = None, **kwargs):
        """
        Config object constructor.
        :param path: Path to configuration TOML file
        """
        # Import logger
        self.level = kwargs.get('level','INFO')
        self.logger = setup_logger(
            __name__,
            console_level = self.level,
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        
        # Initialise global vars
        self.reset()

        # Load config
        if bool(path) and os.path.isfile(path) and os.path.exists(path):

            self.logger.debug(f' Loading config from {path}')

            if path.endswith('.toml'):
                self.settings = load(path, _dict = dict)
            elif path.endswith('.json'):
                self.settings = read_json(path)

        elif settings:
            self.settings = settings
        else:
            self.settings = None
            raise Exception(f'Config file (ending in .json or .toml) not found in {path}')

        # Load schema
        self.load_schemas()
        # Load parameter positions
        self.load_parameters()
        # Update root
        self.path_sets_root(**kwargs)

    def update(self,new_settings:dict):
        for key,value in new_settings.items():
            # Find path
            all_paths = self.path_find(
                key = key,
                settings = self.settings,
                current_key_path = [],
                all_key_paths = []
            )
            try:
                assert len(all_paths) > 0
            except:
                raise PathNotExistException(
                    message = f"Cannot find path to {key}",
                    key_path = [key],
                    data = self.settings
                )
            # Update key path
            try:
                assert self.path_set(
                    settings = self.settings,
                    value = value,
                    key_path = all_paths[0]
                )
            except:
                raise PathNotExistException(
                    message = f"Cannot update {all_paths[0]} as it cannot be found in settings",
                    key_path = all_paths[0],
                    data = self.settings
                )
    
    def reset(self):
        # Sweep mode activated is set to false
        self.sweep_active = False
        # Gather all isolated sweeps
        self.isolated_sweep_paths = {}
        # Gather all coupled sweeps
        self.coupled_sweep_paths = {}
        # Keep track of the target name for each sweeped variable
        self.target_names_by_sweep_var = {}

    def __str__(self,settings = None):
        if settings is not None:
            try:
                res_str = json.dumps(settings,indent = 2)
                return res_str
            except Exception as exc:
                self.logger.error(
                    json.dumps(
                        deep_apply(deepcopy(settings),type),
                        indent = 2
                    )
                )
                raise exc
        else:
            try:
                res_str = json.dumps(self.settings,indent = 2)
                return res_str
            except Exception as exc:
                self.logger.error(
                    json.dumps(
                        deep_apply(deepcopy(self.settings),type),
                        indent = 2
                    )
                )
                raise exc

    def load_schemas(self):
        # Load base schema
        with open(
            os.path.join(
                ROOT,
                'data/inputs/configs/schemas/cfg_schema.json'
            ), 
            'r'
        ) as f:
            self.schema = json.load(f)
        # Load experiment-specifc schema
        with open(
            os.path.join(
                ROOT,
                'data/inputs/configs/schemas/cfg_schema_by_experiment.json'
            ), 
            'r'
        ) as f:
            self.experiment_schemas = json.load(f)

    def load_parameters(self):
        # Load all parameter positions
        with open(
            os.path.join(
                ROOT,
                'data/inputs/configs/schemas/cfg_parameters.json'
            ), 
            'r'
        ) as f:
            self.parameters = json.load(f)

    def keys(self):
        return self.settings.keys()

    def get(self,key,default):
        all_paths = self.path_find(
            key = key,
            settings = self.settings,
            current_key_path = [],
            all_key_paths = []
        )
        if len(all_paths) <= 0: return default
            
        if self.has_sweep(all_paths[0],self.settings):
            # Get sweep configuration
            value,found = self.path_get(
                key_path = all_paths[0]+["sweep","range"],
                settings = self.settings
            )
            if not found: return default
            # Parse values
            return self.parse_data(value,(all_paths[0]+["sweep","range"]))
        else:
            # Get sweep configuration
            value,found = self.path_get(
                key_path = all_paths[0],
                settings = self.settings
            )
            if not found: return default
            # Parse values
            return value
        

    def __delitem__(self, key):
        del self.settings[key]

    def __getitem__(self, key):
        return self.settings[key]

    def __setitem__(self, key, value):
        self.settings[key] = value


    def path_sets_root(self,**kwargs)  -> None:
        """
        Add root path to all configured paths (inputs, output directories).
        """
        # Store absolute paths
        in_dir = kwargs.get('input_dir','')
        out_dir = kwargs.get('output_dir','')
        
        if len(in_dir) > 0:
            self.in_directory = in_dir
        else:
            self.in_directory = os.path.join(
                ROOT,
                self.settings['inputs']['in_directory']
            )

        if len(out_dir) > 0:
            self.out_directory = out_dir
        else:
            self.out_directory = os.path.join(
                ROOT,
                self.settings['outputs']['out_directory']
            )


    def deep_apply(self,func,**kwargs):
        return deep_apply(self.settings, func, **kwargs)
        
    def print_types(self):
        settings_copy = deepcopy(self.settings)
        settings_copy = deep_apply(settings_copy, type)
        try:
            print(self.__str__(settings_copy))
        except:
            print(settings_copy)

    def parse_data(self,settings,key_path):
        # Get schema given the path
        schema,_ = self.path_get(
            key_path = key_path,
            settings = self.schema
        )
        # Instantiate data type
        try:
            data = instantiate_data_type(settings,schema,key_path)
        except:
            print('key_path',key_path)
            print('settings',settings)
            print('schema',schema)
            raise
        # return parsed data
        return data.value()

    def has_sweep(self,key_path,settings = None):
        if settings is None:
            settings = self.settings
        # Return True if sweep is a key in the path
        if 'sweep' in list(key_path):
            return True
        else:
            value,_ = self.path_get(
                key_path = key_path, 
                settings = settings
            )
            # Return whether sweep is later down the path
            return isinstance(value,dict) and value.get('sweep',None) is not None
    
    def path_exists(self, key, value, found:bool = False):
        for k, v in (value.items() if isinstance(value, dict) else
            enumerate(value) if isinstance(value, list) else []):
            if k == key:
                found = True
            elif isinstance(v, (list,dict)):
                found = self.path_exists(key,v,found)
        
        return found

    def path_find(self, key, settings:dict = None, current_key_path:list=[], all_key_paths:list=[]):
        if settings is None:
            settings = self.settings
        settings_copy = deepcopy(settings)
        
        for k, v in (settings_copy.items() if isinstance(settings_copy, dict) else
            enumerate(settings_copy) if isinstance(settings_copy, list) else []):
            
            current_key_path.append(k)
            if k == key:
                all_key_paths.append(deepcopy(current_key_path))
            if isinstance(v, (list,dict)):
                all_key_paths = self.path_find(
                    key = key,
                    settings = v,
                    current_key_path = current_key_path,
                    all_key_paths = all_key_paths
                )
            current_key_path.remove(k)

        return all_key_paths


    def path_get(self,key_path=[],settings = None):
        if len(key_path) <= 0:
            return None,False
        if settings is None:
            settings = self.settings
        settings_copy = deepcopy(settings)

        for key in key_path:
            if key == 'sweep':
                if isinstance(settings_copy,dict):
                    if settings_copy.get(key,'not-found') == 'not-found':
                        return None,False
                    else:
                        settings_copy = settings_copy.get(key,'not-found')
                else:
                    return None,False
            else:
                try:
                    settings_copy = settings_copy[key]
                except:
                    return None,False
                    
        return settings_copy,(settings_copy!='not-found')

    def path_delete(self,settings,key_path:list,deleted:bool = False):
        if len(key_path) <= 0:
            return deleted,settings
        deleted = False
        if len(key_path) == 1:
            if (isinstance(settings,dict) and key_path[0] in settings) or \
                isinstance(settings,list) and key_path[0] < len(settings):
                del settings[key_path[0]]
                deleted = True
        else:
            # Convert key to index if it is numeric
            # otherwise keep it as string
            current_key = key_path[0]
            current_key = string_to_numeric(current_key) \
                if isinstance(current_key,str) and current_key.isnumeric() \
                else current_key
            try:
                deleted,settings = self.path_delete(settings[current_key],key_path[1:],deleted)
            except:
                self.logger.warning(f"Could not delete path {key_path}. It might not exist.")
        return deleted,settings

    def path_set(self,settings,value,key_path=[],overwrite:bool = False):
        if len(key_path) <= 0:
            return False
        value_set = False
        if len(key_path) == 1:
            settings[key_path[0]] = value
            value_set = True
        else:
            if overwrite:
                settings[key_path[0]] = {}
                value_set = self.path_set(
                    settings[key_path[0]],
                    value,
                    key_path[1:],
                    overwrite
                )
            else: 
                value_set = self.path_set(
                    settings.get(key_path[0],{}),
                    value,
                    key_path[1:],
                    overwrite
                )
        
        return value_set
    
    def path_modify(self,settings,value,key_path):
        if len(key_path) <= 0:
            return False
        elif len(key_path) == 1:
            if key_path[0] in settings:
                settings[key_path[0]] = value
                return True
            else:
                return False
        else:
            key = key_path[0]
            if key in settings:
                self.path_modify(settings[key], value, key_path[1:])
            else:
                return False
        return True

    def experiment_validate(self):
        # Perform this check only when one experiment is given
        try:
            assert len(self.settings['experiments']) == 1
        except:
            raise Exception('More than one experiments provided. Cannot perform experiment-specific config validation.')
        # Get experiment type
        experiment_type = self.settings['experiments'][0]['type']
        # Get experiment-specific schema
        experiment_schema = self.experiment_schemas.get(experiment_type,None)

        if experiment_schema is not None:
            # Update base schema based on experiment-specific requirements
            for key_path_str in list(experiment_schema.keys()):
                # Convert keys to key path lists
                # Make sure that list indices are of type int and 
                # the rest of the keys are of type str
                key_path = [string_to_numeric(kp) if kp.isnumeric() else kp for kp in key_path_str.split('>')]
                
                # Do not update dtypes!
                if "dtype" in key_path:
                    continue 

                # Delete excluded key paths
                if key_path[-1] == 'exclude' and experiment_schema[key_path_str]:
                    self.logger.warning(f"Excluding key path {' > '.join(key_path_str.split('>')[:-1])}")
                    # Update settings
                    _,_ = self.path_delete(self.settings,key_path[:-1])
                    # Update base schema
                    _,_ = self.path_delete(self.schema,key_path[:-1])
                    # Update base parameters
                    _,_ = self.path_delete(self.parameters,key_path[:-1])
                # Modify the base schema only
                else:
                    self.logger.debug(f"Modifying key path {' > '.join(key_path_str.split('>')[:-1])}")
                    path_modified = self.path_modify(
                        self.schema,
                        experiment_schema[key_path_str],
                        key_path
                    )
                    if not path_modified:
                        raise Exception(f"Could not update base schema at {'>'.join(key_path_str.split('>'))}")
                    
                    # Make sure path can be found in schema
                    new_value,path_found = self.path_get(
                        key_path,
                        self.schema
                    )
                    if not path_found:
                        self.logger.error(f"Value = {new_value}")
                        raise Exception(f"Could not find base schema key path {'>'.join(key_path_str.split('>'))}")
        
        # Validate config
        self.validate(
            parameters = self.parameters,
            settings = self.settings,
            base_schema = self.schema,
            key_path = [],
            experiment_type = experiment_type
        )

    def sweep_mode(self,settings:dict = None):
        # Check if sweep mode is active
        if settings is None:
            settings = self.settings
        
        if not settings.get('sweep_mode',False):
            return False
        
        # Try to find all sweep configs
        variable_key_paths = list(self.path_find(
            key = 'sweep',
            settings = settings,
            current_key_path = [],
            all_key_paths = []
        ))

        # If no sweep key paths found return False
        if len(variable_key_paths) <= 0:
            return False
        else:
            # If there is at least one valid 
            # (non-zero length) sweep configuration return True
            return any([len(key_path) > 0 for key_path in variable_key_paths])

        
    def is_sweepable(self,variable):
        # Find first instance target name
        variable_key_paths = list(self.path_find(
            key = variable,
            settings = self.schema,
            current_key_path = [],
            all_key_paths = []
        ))
        # If no path found raise exception
        if len(variable_key_paths) <= 0 or len(variable_key_paths[0]) <= 0:
            raise Exception(f"{variable} not found in config schema.")
        # Return sweepable flag for first instance of variable
        value,_ = self.path_get(
            key_path = variable_key_paths[0],
            settings = self.schema
        )
        return value.get('sweepable',False) or value.get('file',{}).get('sweepable',False)

    def get_sweep_id(self,sweep:dict={}):
        sweep_id = ''
        if len(sweep) > 0 and isinstance(sweep,dict):
            # Create sweep id by grouping coupled sweep vars together
            # and isolated sweep vars separately
            sweep_id = []
            for v in set(list(self.target_names_by_sweep_var.values())).difference(set(['dataset'])):
                # Map sigma to noise regime
                if str(v) == 'sigma':
                    value = sigma_to_noise_regime(sweep[v])
                # Else use passed sweep value
                else:
                    value = sweep[v]
                # Add to key-value pair to unique sweep id
                sweep_id.append(f"{str(v)}_{stringify(value,preffix='[',suffix=']',scientific=False)}")
            # Join all grouped sweep vars into one sweep id 
            # which will be used to create an output folder
            if len(sweep_id) > 0:
                sweep_id = os.path.join(*sorted(sweep_id,key = lambda x: x.split('_')[0]))
            else:
                sweep_id = ''
        
        return sweep_id
    
    def find_sweep_key_paths(self,settings:dict=None):
        if settings is None:
            settings = self.settings
        for key_val_path in deep_walk(settings):

            if "sweep" in key_val_path and "range" in key_val_path:
                # Index location of sweep in key value path
                sweep_indx = key_val_path.index("sweep")
                # Get sweep settings
                sweep_settings,_ = self.path_get(
                    key_path = key_val_path[:(sweep_indx+1)],
                    settings = settings
                )
                # If the sweep setting is coupled add to coupled sweep paths
                if sweep_settings.get('coupled',False) and len(sweep_settings.get('target_name','')) > 0:
                    # Get target name
                    target_name = sweep_settings.get('target_name','')
                    # Get coupled name and value
                    coupled_name = key_val_path[sweep_indx-1] if key_val_path[sweep_indx-1] != 'file' else key_val_path[sweep_indx-2]
                    coupled_value = key_val_path[:sweep_indx]
                    if target_name in list(self.coupled_sweep_paths.keys()):
                        # Add path to coupled sweeps
                        self.coupled_sweep_paths[target_name][coupled_name] = deepcopy(coupled_value)
                    else:
                        # Find first instance of target name
                        target_name_paths = list(self.path_find(
                            key = target_name,
                            settings = settings,
                            current_key_path = [],
                            all_key_paths = []
                        ))
                        if len(target_name_paths) > 0 and len(target_name_paths[0]):
                            self.coupled_sweep_paths[target_name] = {    
                                target_name : target_name_paths[0],
                                coupled_name : deepcopy(coupled_value)
                            }
                # Else add to isolated sweep paths
                else:
                    target_name = key_val_path[sweep_indx-1] if key_val_path[sweep_indx-1] != 'file' else key_val_path[sweep_indx-2]
                    self.isolated_sweep_paths[target_name] = deepcopy(key_val_path[:sweep_indx])
        
        # Find common keys between isolated and coupled paths
        common_keys = set(list(self.isolated_sweep_paths.keys())).intersection(set(list(self.coupled_sweep_paths.keys())))
        # Remove them from isolated sweeps
        for k in list(common_keys):
            del self.isolated_sweep_paths[k]
        # Get sweep parameter names
        sweep_param_names = list(self.isolated_sweep_paths.keys())
        sweep_param_names += [
            item
            for nested_dict in self.coupled_sweep_paths.values() 
            for item in nested_dict.keys()
        ]
        self.sweep_param_names = list(flatten(sweep_param_names))

    def get_group_id(self,group_by:list=[]):
        
        # Gather sweep dimension names
        sweep_dims = list(self.sweep_params['isolated'].keys())
        sweep_dims += list(self.sweep_params['coupled'].keys())

        # Get list of all dimensions whose values will differ by group (collection id)
        nonshared_coord_dims = set([])
        # Get list of all dimensions whose values will be shared amongst groups 
        # (they will be merged during grouping)
        shared_coord_dims = set([])
        # Get all non-core sweep dims
        non_core_sweep_dims = [k for k in sweep_dims if k not in CORE_COORDINATES_DTYPES]

        for gb in sorted(list(group_by)+non_core_sweep_dims):
            # If this is an isolated sweep parameter
            # add it to the group by
            if gb in self.sweep_params['isolated']:
                nonshared_coord_dims.add(gb)
            # If it is a coupled sweep parameter
            # add the coupled parameters too
            if gb in self.sweep_params['coupled']:
                nonshared_coord_dims.add(gb)
                # If this parameter is the target name
                # add its coupled parameters to the group by
                for coupled_param in self.sweep_params['coupled'].get(gb,[]):
                    nonshared_coord_dims.add(coupled_param['var'])
            # If this parameter is the coupled parameter
            # of a target name add the target name and 
            # the rest of the coupled parameters to the group by
            target_name = self.target_names_by_sweep_var.get(gb,'none')
            if target_name != 'none':
                for coupled_param in self.sweep_params['coupled'].get(target_name,[]):
                    # Add target name
                    nonshared_coord_dims.add(target_name)
                    # Add rest of coupled params
                    nonshared_coord_dims.add(coupled_param['var'])
        
        for dim in self.sweep_params['isolated'].keys():
            if dim not in nonshared_coord_dims:
                shared_coord_dims.add(dim)
        for vals in self.sweep_params['coupled'].values():
            for coupled_dims in vals:
                if coupled_dims['var'] not in nonshared_coord_dims:
                    shared_coord_dims.add(coupled_dims['var'])

        # print(sorted(list(nonshared_coord_dims))) 
        # print(sorted(list(shared_coord_dims)))
        return sorted(list(nonshared_coord_dims)), sorted(list(shared_coord_dims))
        
    def validate(self,parameters = None,settings = None,base_schema = None,key_path=[],**kwargs):
        # Pass defaults if no meaningful arguments are provided
        if parameters is None:
            parameters = self.parameters
        if settings is None:
            settings = self.settings
        if base_schema is None:
            base_schema = self.schema
        
        for k, v in parameters.items() if isinstance(parameters,dict) else enumerate(parameters):
            # Experiment settings are provided in list of dictionaries
            # Special treatment is required
            
            # Append key to path
            key_path.append(k)
            # print('>'.join(list(map(str,key_path))))
            
            if isinstance(v,list):

                for idx, subvalue in enumerate(v):
                    # Append key to path
                    key_path.append(idx)

                    self.validate(
                        subvalue,
                        settings,
                        base_schema,
                        key_path,
                        **kwargs
                    )
                    # Remove it from path
                    key_path.remove(idx)

            # If settings are a dictionary, validate settings 
            # against each key path
            elif k != 'sweep' and isinstance(v,dict):
                # Apply function recursively
                self.validate(
                    v,
                    settings,
                    base_schema,
                    key_path,
                    **kwargs
                )
            
            # If settings are any other (primitive of non-promitive) value,
            # validate settings for given key path
            # Special treament is required depending on:
            # - whether parameter needs to be excluded or not (See 0:)
            # - whether parameter is optional or not (See 1:)
            # - whether sweep parameters are passed or not (See 2:)
            # - whether sweep parameters contain default configurations (See 3:)
            # - whether sweep parameters contain range configurations (See 4:)
            # - whether coupled sweep parameters are passed or not (See 5:)
            # - whether sweep mode is activated (See 6:)
            # - whether sweep is deactivated and only sweep configuration is provided (See 7:)
            # - all data type-specific checks (See 8:)
            else:

                # Check if key was found in settings and schema
                settings_val, settings_found = self.path_get(
                    key_path = key_path, 
                    settings = settings
                )
                schema_val, schema_found = self.path_get(
                    key_path = key_path, 
                    settings = base_schema
                )
                
                # key must exist in schema
                try: 
                    assert schema_found
                except:
                    raise Exception(f"Key {'>'.join(list(map(str,key_path)))} not found in schema.")

                # 1: Check if argument is optional in case it is not included in settings
                # and it is not part of a sweep
                if not 'sweep' in key_path and not settings_found:
                    self.logger.trace('check 1')
                    if isinstance(schema_val,dict) and not schema_val['optional']:
                        raise Exception(f"""
                            Key {'>'.join(list(map(str,key_path)))} is compulsory but not included
                        """)

                if key_path[-1] == 'sweep' and settings_found:
                    self.logger.trace('check sweep')
                    # 2: Check if argument is not sweepable but contains 
                    # a sweep parameter in settings
                    # Get schema of parameter configured for a sweep
                    schema_parent_val, _ = self.path_get(
                        key_path = key_path[:-1], 
                        settings = base_schema
                    )
                    if not schema_parent_val["sweepable"]:
                        self.logger.trace('check 2')
                        raise Exception(f"""
                            Key {'>'.join(list(map(str,key_path)))} is not sweepable 
                            but contains a sweep configuration
                        """)
                    # 3: Check that argument that is sweepable containts 
                    # a default and a range configuration
                    try:
                        assert isinstance(settings_val,dict) \
                            and "default" in list(settings_val.keys())
                    except:
                        raise Exception(f"""
                            Key {'>'.join(list(map(str,key_path)))} is not a dictionary 
                            or does not have 'default' configuration 
                            for a sweep
                        """)
                    # 4: Check whether a range configuration is provided
                    # otherwise replace sweep key-value pair with sweep default
                    if "range" not in list(settings_val.keys()):
                        self.logger.trace('check 4')
                        # Get child key settings
                        settings_child_val, settings_child_found = self.path_get(
                            key_path = (key_path+['default']), 
                            settings = settings
                        )
                        # Update settings with sweep default
                        try:
                            assert self.path_set(
                                settings,
                                settings_child_val,
                                key_path[:-1]
                            )
                        except:
                            raise Exception(f"""
                                Key {'>'.join(list(map(str,key_path)))} could not be updated \
                                with sweep default
                            """)
                        # Flag for whether this parameter should be treated as 
                        # a sweep parameter is set to False due to lack of range configuration
                        is_sweep = False
                    else:
                        is_sweep = True

                    # 5: Check that argument that is coupled sweepable containts 
                    # a target name and that target name exists as a key
                    if settings_val.get('coupled',False):
                        self.logger.trace('check 5')
                        try:
                            assert "target_name" in list(settings_val.keys()) and \
                                self.path_exists(settings_val['target_name'],self.settings)
                        except:
                            raise Exception(f"""
                                Key {'>'.join(list(map(str,key_path)))} does not have a
                                'target_name' configuration for a sweep or
                                target_name does not exist
                            """)
                        target_name = settings_val['target_name']
                        if target_name in list(self.coupled_sweep_paths.keys()):
                            # Add path to coupled sweeps
                            self.coupled_sweep_paths[target_name][key_path[-2]] = deepcopy(key_path[:-1])
                        else:
                            # Create a sublist of paths based on target name
                            # Try to read sweep data from isolated sweeps
                            if target_name in self.isolated_sweep_paths:
                                self.coupled_sweep_paths[target_name] = {    
                                    target_name : deepcopy(self.isolated_sweep_paths[target_name]),
                                    key_path[-2] : deepcopy(key_path[:-1])
                                }
                            # Otherwise find the target name in settings
                            else:
                                target_name_paths = list(self.path_find(
                                    key = target_name,
                                    settings = self.settings,
                                    current_key_path = [],
                                    all_key_paths = []
                                ))
                                if len(target_name_paths) > 0:
                                    self.coupled_sweep_paths[target_name] = {    
                                        target_name : target_name_paths[0],
                                        key_path[-2] : deepcopy(key_path[:-1])
                                    }
                                else:
                                    raise Exception(f"""Target name {target_name} not found in settings.""")
                    else:
                        # Add to isolated sweep paths
                        # iff this parameter should be treated a sweep parameter
                        if is_sweep:
                            target_name = key_path[-2] if key_path[-2] != 'file' else key_path[-3]
                            self.isolated_sweep_paths[target_name] = deepcopy(key_path[:-1])

                # 6: Check that if parameter sweep is activated
                if key_path[-1] == 'sweep_mode':
                    self.logger.trace('check 6')
                    self.sweep_active = settings_val
                # 7: If sweep is deactivated and
                # A: a sweep configuration is provided or the values are provided
                # then read the default sweep configuration
                # B: a sweep configuration is NOT provided but the values are provided
                # then read the values provided
                if key_path[-1] == 'sweep' and (not self.sweep_active or not settings_found):
                    self.logger.trace('check 7')
                    # If a sweep configuration has been provided
                    # read the default values
                    if settings_found:
                        # Get child key settings
                        settings_child_val, settings_child_found = self.path_get(
                            key_path = (key_path+['default']),
                            settings = settings
                        )
                        if not settings_child_found:
                            # Try to read without sweep
                            settings_child_val, settings_child_found = self.path_get(
                                key_path = (key_path[:-1]),
                                settings = settings
                            )

                        # Make sure you can read the default value or the non-sweep value
                        try:
                            assert settings_child_found
                        except:
                            raise Exception(f"""
                                    Key {'>'.join(list(map(str,key_path)))} could not be found \
                                    with sweep default or without sweep
                                """)
                        # Update settings with sweep default
                        try:
                            assert self.path_set(
                                settings,
                                settings_child_val,
                                key_path[:-1]
                            )
                        except:
                            raise Exception(f"""
                                Key {'>'.join(list(map(str,key_path)))} could not be updated \
                                with sweep default
                            """)

                    # Else maybe value have been provided
                    # without specifying a sweep configuration
                    else:
                        settings_child_val, settings_child_found = self.path_get(
                            key_path = key_path[:-1],
                            settings = settings
                        )

                        # If still no values have been provided
                        # Check if this setting is even compulsory
                        parent_schema_val, _ = self.path_get(
                            key_path = key_path[:-1],
                            settings = base_schema
                        )
                        if not settings_child_found and isinstance(parent_schema_val,dict) and not parent_schema_val['optional']:
                            raise Exception(f"""
                                Key {'>'.join(key_path[:-1])} is compulsory but not included
                            """)
                    
                    # If this is the case we have modified the config
                    # Parse settings value into appropriate data structure
                    if settings_child_found:
                        try:
                            entry = instantiate_data_type(
                                data = settings_child_val,
                                schema = schema_val['default'],
                                key_path = key_path[:-1]
                            )
                        except:
                            print(settings_child_val)
                            print(schema_val)
                            print(key_path)
                            traceback.print_exc()
                            self.logger.error(f"Config for experiment(s) {kwargs.get('experiment_type','UNKNOWN')} failed.")
                            sys.exit()

                        # Check that entry is valid
                        try:
                            entry.check()
                        except:
                            traceback.print_exc()
                            self.logger.error(f"Config for experiment(s) {kwargs.get('experiment_type','UNKNOWN')} failed.")
                            sys.exit()
                
                # 8: Check all data type-specific checks
                # according to the schema
                if settings_found:
                    self.logger.trace(f"check 8, has sweep: {self.has_sweep(key_path)}")
                    # If parameter is sweepable but settings 
                    # does not contain a sweep configuration
                    if self.has_sweep(key_path) and not isinstance(settings_val,dict):
                        # Read schema
                        schema_val, _ = self.path_get(
                            key_path = key_path,
                            settings = base_schema
                        )
                        # Parse settings value into approapriate data structure
                        entry = instantiate_data_type(
                            data = settings_val,
                            schema = schema_val,
                            key_path = key_path
                        )
                        # Check that entry is valid
                        try:
                            entry.check()
                        except:
                            traceback.print_exc() 
                            self.logger.error(f"Config for experiment(s) {kwargs.get('experiment_type','experiment_type')} failed.")
                            sys.exit()
                    elif self.has_sweep(key_path) and isinstance(v,dict):
                        # Apply function recursively
                        self.validate(
                            v,
                            settings,
                            base_schema,
                            key_path,
                            **kwargs
                        )
                    else:
                        # Parse settings value into appropriate data structure
                        entry = instantiate_data_type(
                            data = settings_val,
                            schema = schema_val,
                            key_path = key_path
                        )
                        # Check that entry is valid
                        try:
                            entry.check()
                        except:
                            traceback.print_exc() 
                            self.logger.error(f"Config for experiment(s) {kwargs.get('experiment_type','experiment_type')} failed.")
                            sys.exit()
                
            # Remove it from path
            key_path.remove(k)
        
    def prepare_sweep_configurations(self,sweep_params):

        # Compute all combinations of sweep parameters
        # Isolated sweep parameters are cross multiplied
        if len(sweep_params['isolated'].values()) > 0:
            sweep_configurations = [val['values'] for val in sweep_params['isolated'].values()]
        else:
            sweep_configurations = []

        # Group all coupled sweep parameters by target name
        if len(sweep_params['coupled'].values()) > 0:
            # Coupled sweep parameters are merged and then cross multiplied
            coupled_sweep_configurations = []
            coupled_sweep_sizes = {}
            for target_name, target_name_vals in sweep_params['coupled'].items():
                # Gather all coupled sweeps for this target name
                target_sweep_configurations = [sweep_vals['values'] for sweep_vals in target_name_vals]
                # Gather all unique sweep group values by hashing the data to a string
                # Hash data to string
                target_sweep_configurations_str = list(map(lambda x: ' === '.join(list(map(str,x))),zip(*target_sweep_configurations)))
                # Find indices of unique elements
                unique_elem_indices = [target_sweep_configurations_str.index(x) for x in set(target_sweep_configurations_str)]
                # Use these indices to extract the unique sweep group values
                new_coupled_sweeps = [item for idx,item in enumerate(zip(*target_sweep_configurations)) if idx in unique_elem_indices]
                # Add them to list of all coupled sweeps
                coupled_sweep_configurations += [new_coupled_sweeps]
                # Monitor the sweep size for this target name
                coupled_sweep_sizes[target_name] = {
                    "length":len(new_coupled_sweeps),
                    "vars":[sweep_val['var'] for sweep_val in target_name_vals]
                }
            # Cross multiple both sweep configurations
            sweep_configurations = list(product(*(sweep_configurations+coupled_sweep_configurations)))
        elif len(sweep_configurations) > 0:
            # Cross multiple isolated sweep configurations
            sweep_configurations = list(product(*(sweep_configurations)))
        
        # Expand tuples within tuples
        sweep_configurations = [expand_tuple(item) for item in sweep_configurations]

        # Check if configurations are unique
        sweep_configurations_copy = deepcopy(sweep_configurations)
        sweep_configurations_copy = list(map(lambda x: ' === '.join(list(map(str,x))),sweep_configurations_copy))
        try:
            assert len(sweep_configurations_copy) == len(set(sweep_configurations_copy))
        except:
            raise DuplicateData(
                message = 'Sweep configurations contain duplicates!',
                len_data = len(sweep_configurations_copy),
                len_unique_data = len(set(sweep_configurations_copy))
            )
        
        # Print parameter space size
        param_sizes = [f"{v['var']} ({len(v['values'])})" for v in sweep_params['isolated'].values()]
        total_sizes = [len(v['values']) for v in sweep_params['isolated'].values()]
        if len(sweep_params['coupled']) > 0:
            for k,v in coupled_sweep_sizes.items():
                param_sizes += [f"{k}: "+','.join([f"{v['vars']}" ])+f" ({v['length']})"]
                total_sizes += [v['length']]
        
        param_sizes_str = "\n --- ".join(['']+param_sizes)
        total_size_str = len(sweep_configurations)

        return sweep_configurations, param_sizes_str, total_size_str

    def parse_sweep_params(self):
        sweep_params = {"coupled":{},"isolated":{}}
        # Find common keys between isolated and coupled paths
        common_keys = set(list(self.isolated_sweep_paths.keys())).intersection(set(list(self.coupled_sweep_paths.keys())))
        # Remove them from isolated sweeps
        for k in list(common_keys):
            del self.isolated_sweep_paths[k]
        # Keep track of the target name for each sweeped variable
        for key,key_path in self.isolated_sweep_paths.items():
            # Get sweep configuration
            sweep_input,_ = self.path_get(
                key_path = (key_path+["sweep","range"]),
                settings = self.settings
            )
            # Parse values
            # print(key,'isolated',key_path)
            sweep_vals = self.parse_data(sweep_input,(key_path+["sweep","range"]))

            sweep_params['isolated'][key] = {
                "var":key,
                "path": key_path,
                "values": sweep_vals
            }

            # Isolated sweeps have themselves as the target name
            self.target_names_by_sweep_var[key] = key
        coupled_val_lens = {}
        for target_name,coupled_paths in self.coupled_sweep_paths.items():
            # Monitor length of sweep values by target name
            coupled_val_lens[target_name] = {}
            sweep_params['coupled'][target_name] = []
            for key_path in coupled_paths.values():
                # print('coupled',key_path)
                # Get sweep configuration
                sweep_input,_ = self.path_get(
                    key_path = (key_path+["sweep","range"]),
                    settings = self.settings
                )
                # Parse values
                sweep_vals = self.parse_data(sweep_input,(key_path+["sweep","range"]))
                sweep_params['coupled'][target_name].append({
                    "var":key_path[-1] if key_path[-1] != 'file' else key_path[-2],
                    "path": key_path,
                    "values": sweep_vals
                })
                # Target variables should be unique 
                # as they are used for naming output folders
                if key_path[-1] == target_name:
                    try:
                        assert len(sweep_vals) == len(unique(sweep_vals))
                    except:
                        print(sweep_vals)
                        raise Exception(f"""
                            Coupled sweep values for target name {target_name} are not unique.
                            {len(sweep_vals)} values are provided of which {len(unique(sweep_vals))} are unique.
                        """)
                # Add key path length to dict
                coupled_val_lens[target_name][key_path[-1]] = len(sweep_vals)
                # Coupled sweeps have a common target name
                self.target_names_by_sweep_var[target_name] = target_name
        # Make sure all coupled parameters have the same length
        for target_name,keypaths in coupled_val_lens.items():
            try:
                assert len(set(list(keypaths.values()))) == 1
            except:
                raise Exception(f"""Coupled sweep parameters for target name {target_name} 
                                do not all have the same length
                                {json.dumps(keypaths)}""")
        return sweep_params


    def prepare_experiment_config(self,sweep_configuration,cast_to_str:bool = False):
        # Create new config
        new_config = deepcopy(self)
        # Deactivate sweep             
        new_config.settings["sweep_mode"] = False
        # Activate sample exports
        new_config.settings['export_samples'] = True
        # Create sweep dictionary
        sweep = {}
        # Update config
        i = 0
        for value in self.sweep_params['isolated'].values():
            self.logger.debug(f"{value['path']}: {sweep_configuration[i]}")
            new_config.path_set(
                new_config.settings,
                sweep_configuration[i],
                value['path']
            )
            # Update current sweep
            sweep[value['var']] = sweep_configuration[i] \
                if not cast_to_str \
                else str(sweep_configuration[i])
            i += 1
        for sweep_group in self.sweep_params['coupled'].values():
            for value in sweep_group:
                self.logger.debug(f"{value['path']}: {sweep_configuration[i]}")
                new_config.path_set(
                    new_config.settings,
                    sweep_configuration[i],
                    value['path'],
                )
                # Update current sweep
                sweep[value['var']] = sweep_configuration[i] \
                if not cast_to_str \
                else str(sweep_configuration[i])
                i += 1
        # Return config and sweep params
        return new_config,sweep


    def get_sweep_data(self):
        
        if self.sweep_mode():
            # Find config paths to sweeped parameters
            self.find_sweep_key_paths()

            # Parse sweep configurations
            self.sweep_params = self.parse_sweep_params()

            # Get all sweep configurations
            self.sweep_configurations, \
            self.param_sizes_str, \
            self.total_size_str = self.prepare_sweep_configurations(self.sweep_params)
            # Get output folder

            self.base_dir = self.out_directory.split('samples/')[0]
            if len(self.sweep_configurations) > 0:
                self.logger.info("----------------------------------------------------------------------------------")
                self.logger.info(f"Parameter space size: {self.param_sizes_str}")
                self.logger.info(f"Total = {self.total_size_str}.")
                self.logger.info("----------------------------------------------------------------------------------")

    def slice_sweep_configurations(self,sweep:dict,group_by:list=[]):
        # Loop through each configuration
        sliced_sweep_configurations = []
        grouped_sweep_indices = [i for i,d in enumerate(self.sweep_param_names) if d not in group_by]
        grouped_sweep_param_names = [self.sweep_param_names[i] for i in grouped_sweep_indices]
        for sweep_configuration in self.sweep_configurations:
            grouped_sweep_configuration = [sweep_configuration[i] for i in grouped_sweep_indices]
            # All the criteria in this loop must be met in order to slice
            match = True
            # print(dict(zip(grouped_sweep_param_names,grouped_sweep_configuration)))
            for dim in grouped_sweep_param_names:
                val = sweep[dim]
                # Find index of dimension in sweep configuration
                dim_index = grouped_sweep_param_names.index(dim)
                
                # Check if there is a coordinate match
                if val is None or grouped_sweep_configuration[dim_index] is None:
                    match = match and grouped_sweep_configuration[dim_index] == val
                elif isinstance(val,Iterable):
                    match = match and grouped_sweep_configuration[dim_index] == val
                else: 
                    match = match and math.isclose(grouped_sweep_configuration[dim_index], val, rel_tol = 1e-1) 

                if not match:
                    break

            if match:
                sliced_sweep_configurations.append(sweep_configuration)
        
        return sliced_sweep_configurations

    def trim_sweep_configurations(self):
        # Loop through each sweep configuration
        for sweep_conf in self.sweep_configurations:
            # Extract sweep params for this configuration
            _,sweep = self.prepare_experiment_config(
                self.sweep_params,
                sweep_conf
            )
            # Get sweep id
            sweep_id = self.get_sweep_id(sweep = sweep)
            # Check if 'metadata.json' or 'config.json' and 'data.h5' and 'outputs.log'
            # exist in output directory
            file_exists = {}
            for file in ['metadata.json','data.h5','outputs.log']:
                # Create filepath
                filepath = os.path.join(
                    self.outputs_path,
                    'samples',
                    sweep_id,
                    f"outputs.log"
                )
                # Check if path exists and filepath corresponds to a file
                file_exists[file] = os.path.exists(filepath) and os.path.isfile(filepath)
            # Check if necessary data exists 
            data_exists = file_exists['metadata.json'] and \
                file_exists['data.h5'] and \
                file_exists['outputs.log']
            # If necessary data does not exist
            # Add sweep configurations that need to be run
            if not data_exists:
                yield sweep_conf

    def convert_sweep(self,sweep):
        if isinstance(sweep,dict):
            return list(sweep.values())
        else:
            return dict(zip(
                self.sweep_param_names,
                sweep
            ))