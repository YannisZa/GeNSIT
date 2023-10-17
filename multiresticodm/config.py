import json
import toml
import os.path

from operator import mul
from copy import deepcopy
from functools import reduce
from itertools import product

from multiresticodm import ROOT
from multiresticodm.config_data_structures import instantiate_data_type
from multiresticodm.utils import deep_apply, setup_logger, str_in_list, read_json, expand_tuple

class Config:

    def __init__(self, path:str=None, settings:dict=None, **kwargs):
        """
        Config object constructor.
        :param path: Path to configuration TOML file
        """
        # Import logger
        self.level = kwargs.get('level','INFO').upper()
        self.logger = setup_logger(
            __name__,
            level = self.level,
            log_to_console = kwargs.get('log_to_console',False),
            log_to_file = kwargs.get('log_to_file',False),
        ) if kwargs.get('logger',None) is None else kwargs['logger']
        # Update logger level
        self.logger.setLevel(self.level)
        
        # Sweep mode activated is set to false
        self.sweep_active = False

        # Gather all isolated sweeps
        self.isolated_sweep_paths = {}
        # Gather all coupled sweeps
        self.coupled_sweep_paths = {}

        # Load config
        if path:

            self.logger.debug(f' Loading config from {path}')

            if path.endswith('.toml'):
                self.settings = toml.load(path, _dict=dict)
            elif path.endswith('.json'):
                self.settings = read_json(path)

            # Load schema
            self.load_schema()
            # Load parameter positions
            self.load_parameters()

        elif settings:
            self.settings = settings
        else:
            self.settings = None
            raise Exception(f'Config not found in {path}')


    def __str__(self,settings=None):
        if settings is not None:
            return json.dumps(settings,indent=2)
        else:
            return json.dumps(self.settings,indent=2)

    def load_schema(self):
        # Load schema
        with open(
            os.path.join(
                ROOT,
                'data/inputs/configs/cfg_schema.json'
            ), 
            'r'
        ) as f:
            self.schema = json.load(f)

    def load_parameters(self):
        # Load all parameter positions
        with open(
            os.path.join(
                ROOT,
                'data/inputs/configs/cfg_parameters.json'
            ), 
            'r'
        ) as f:
            self.parameters = json.load(f)

    def keys(self):
        return self.settings.keys()

    def get(self,key,default):
        return self.settings.get(key,default)

    def __delitem__(self, key):
        del self.settings[key]

    def __getitem__(self, key):
        return self.settings[key]

    def __setitem__(self, key, value):
        self.settings[key] = value


    def path_sets_root(self)  -> None:
        """
        Add root path to all configured paths (inputs, output directories).
        """
        # Store absolute paths
        self.in_directory = os.path.join(ROOT,self.settings['inputs']['in_directory'].replace('./',''))
        self.out_directory = os.path.join(ROOT,self.settings['outputs']['out_directory'].replace('./',''))


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
        schema,_ = self.path_get(settings=self.schema,path=key_path)
        # Instantiate data type
        data = instantiate_data_type(settings,schema,key_path)
        # return parsed data
        return data.value()

    def has_sweep(self,key_path):
        return str_in_list('sweep',list(key_path))
    
    def path_exists(self, key, value, found:bool=False):
        for k, v in (value.items() if isinstance(value, dict) else
            enumerate(value) if isinstance(value, list) else []):
            if k == key:
                found = True
            elif isinstance(v, (list,dict)):
                found = self.path_exists(key,v,found)
        
        return found


    def path_get(self,settings=None,path=[]):
        if len(path) <= 0:
            return None,False
        if settings is None:
            settings = self.settings
        
        settings_copy = deepcopy(settings)
        for i,key in enumerate(path):
            if key == 'sweep':
                if isinstance(settings_copy,dict):
                    if settings_copy.get(key,'not-found') == 'not-found':
                        return None,False
                    else:
                        settings_copy = settings_copy.get(key,'not-found')
                else:
                    return None,False
            else:
                if isinstance(settings_copy,dict):
                    settings_copy = settings_copy.get(key,'not-found')
                else:
                    return settings_copy,(settings_copy!='not-found')
        
        return settings_copy,(settings_copy!='not-found')

    def path_set(self,settings,value,path=[],overwrite:bool=False):
        if len(path) <= 0:
            return False
        value_set = False
        if len(path) == 1:
            settings[path[0]] = value
            value_set = True
        else:
            if overwrite:
                settings[path[0]] = {}
                value_set = self.path_set(settings[path[0]],value,path[1:],overwrite)
            else: 
                value_set = self.path_set(settings.get(path[0],{}),value,path[1:],overwrite)
        
        return value_set

    def validate_config(self,parameters=None,settings=None,key_path=[]):
        # Pass defaults if no meaningful arguments are provided
        if parameters is None:
            parameters = self.parameters
        if settings is None:
            settings = self.settings
        for k, v in parameters.items():
            # Experiment settings are provided in list of dictionaries
            # Special treatment is required
            if isinstance(v,list): #and k in ['experiments','margins']:
                # Append key to path
                key_path.append(k)
                # print('>'.join(key_path))
                # Get schema for key path
                schema,_ = self.path_get(self.schema,key_path)
                # Create dummy schema so that it can be found recursively
                dummy_schema = {}
                self.path_set(dummy_schema,schema,[k],overwrite=True)
                # Check if key was found in settings and schema
                settings_val, settings_found = self.path_get(settings,key_path)
                if settings_found:
                    # For every entry in settings, 
                    # validate using the same schema
                    # Remove it from path
                    key_path.remove(k)
                    for entry in settings_val:
                        # Create a dummy setting to validate dummy schema
                        dummy_settings = {}
                        self.path_set(dummy_settings,entry,[k],overwrite=True)
                        # Validate config
                        self.validate_config(dummy_schema,dummy_settings,key_path)
                else:
                    # Remove it from path
                    key_path.remove(k)

            # If settings are a dictionary, validate settings 
            # against each key path
            elif k != 'sweep' and isinstance(v,dict):
                # Append key to path
                key_path.append(k)
                # print('>'.join(key_path))
                # Apply function recursively
                self.validate_config(v,settings,key_path)
                # Remove it from path
                key_path.remove(k)
            # If settings are any other (primitive of non-promitive) value,
            # validate settings for given key path
            # Special treament is required depending on:
            # - whether parameter is optional or not (See 1:)
            # - whether sweep parameters are passed or not (See 2:)
            # - whether sweep parameters contain default and range configurations (See 3:)
            # - whether coupled sweep parameters are passed or not (See 4:)
            # - whether sweep mode is activated (See 5:)
            # - whether sweep is deactivated and only sweep configuration is provided (See 6:)
            # - all data type-specific checks (See 7:)
            else:
                if v:
                    # Append key to path
                    key_path.append(k)
                    # print('>'.join(key_path))
                    
                    # Check if key was found in settings and schema
                    settings_val, settings_found = self.path_get(settings,key_path)
                    schema_val, schema_found = self.path_get(self.schema,key_path)

                    # key must exist in schema
                    try: 
                        assert schema_found
                    except:
                        raise Exception(f"Key {'>'.join(key_path)} not found in schema.")
                    
                    # 1: Check if argument is optional in case it is not included in settings
                    # and it is not part of a sweep
                    if not str_in_list("sweep",key_path) and not settings_found:
                        if isinstance(schema_val,dict) and not schema_val['optional']:
                            raise Exception(f"""
                                Key {'>'.join(key_path)} is compulsory but not included
                            """)
                    if key_path[-1] == 'sweep' and settings_found:
                        # 2: Check if argument is not sweepable but contains 
                        # a sweep parameter in settings

                        # Get schema of parameter configured for a sweep
                        schema_parent_val, _ = self.path_get(self.schema,key_path[:-1])
                        if not schema_parent_val["sweepable"]:
                            raise Exception(f"""
                                Key {'>'.join(key_path)} is not sweepable 
                                but contains a sweep configuration
                            """)
                        # 3: Check that argument that is sweepable containts 
                        # a default and a range configuration
                        try:
                            assert isinstance(settings_val,dict) \
                                and str_in_list("default",settings_val.keys()) \
                                and str_in_list("range",settings_val.keys())
                        except:
                            raise Exception(f"""
                                Key {'>'.join(key_path)} is not a dictionary 
                                or does not have 'default' and 'range' configurations 
                                for a sweep
                            """)
                        # 4: Check that argument that is coupled sweepable containts 
                        # a target name and that target name exists as a key
                        if settings_val.get('coupled',False):
                            try:
                                assert str_in_list("target_name",settings_val.keys()) and \
                                    self.path_exists(settings_val['target_name'],self.settings)
                            except:
                                raise Exception(f"""
                                    Key {'>'.join(key_path)} does not have a
                                    'target_name' configuration for a sweep
                                """)
                            target_name = settings_val['target_name']
                            if str_in_list(target_name,self.coupled_sweep_paths.keys()):
                                # Add path to coupled sweeps
                                self.coupled_sweep_paths[target_name][key_path[-2]] = deepcopy(key_path[:-1])
                            else:
                                # Create a sublist of paths based on target name
                                self.coupled_sweep_paths[target_name] = {
                                    target_name : deepcopy(self.isolated_sweep_paths[target_name]),
                                    key_path[-2] : deepcopy(key_path[:-1])
                                }
                        else:
                            # Add to isolated sweep paths
                            self.isolated_sweep_paths[key_path[-2]] = deepcopy(key_path[:-1])


                    # 5: Check that if parameter sweep is activated
                    if key_path[-1] == 'sweep_mode':
                        self.sweep_active = settings_val
                    
                    # 6: If sweep is deactivated and only sweep configuration
                    # is provided then read the default sweep configuration
                    if key_path[-1] == 'sweep' and not self.sweep_active and settings_found:
                        # Get child key settings
                        settings_child_val, _ = self.path_get(settings,(key_path+['default']))
                        # Update settings with sweep default
                        try:
                            assert self.path_set(settings,settings_child_val,key_path[:-1])
                        except:
                            raise Exception(f"""
                                Key {'>'.join(key_path)} could not be updated \
                                with sweep default
                            """)
                        # If this is the case we have modified the config
                        # Parse settings value into approapriate data structure
                        entry = instantiate_data_type(
                            data = settings_val['default'],
                            schema = schema_val['default'],
                            key_path = key_path[:-1]
                        )
                        # Check that entry is valid
                        entry.check()
                    
                    # 7: Check all data type-specific checks
                    # according to the schema
                    if settings_found:
                        # If parameter is sweepable but settings 
                        # does not contain a sweep configuration
                        if self.has_sweep(key_path) and not isinstance(settings_val,dict):
                            # Read schema
                            schema_val, _ = self.path_get(self.schema,key_path)
                            # Parse settings value into approapriate data structure
                            entry = instantiate_data_type(
                                data=settings_val,
                                schema=schema_val,
                                key_path=key_path
                            )
                            # Check that entry is valid
                            entry.check()
                        elif self.has_sweep(key_path) and isinstance(settings_val,dict):
                            # Do nothing
                            None
                        else:
                            # Parse settings value into approapriate data structure
                            entry = instantiate_data_type(
                                data=settings_val,
                                schema=schema_val,
                                key_path=key_path
                            )
                            # Check that entry is valid
                            entry.check()

                    # Remove it from path
                    key_path.remove(k)
    

    def prepare_sweep_configurations(self,sweep_params):

        # Compute all combinations of sweep parameters
        # Isolated sweep parameters are cross multiplied
        isolated_sweep_configurations = [val['values'] for val in sweep_params['isolated'].values()]

        # Keep track of the target name for each sweeped variable
        self.target_names_by_sweep_var = dict(zip(list(sweep_params['isolated'].keys()),list(sweep_params['isolated'].keys())))
        # Group all coupled sweep parameters by target name
        if len(sweep_params['coupled'].values()) > 0:
            # Coupled sweep parameters are merged and then cross multiplied
            coupled_sweep_configurations = []
            coupled_sweep_sizes = {}
            for target_name, target_name_vals in sweep_params['coupled'].items():
                # Gather all coupled sweeps for this target name
                target_sweep_configurations = [sweep_vals['values'] for sweep_vals in target_name_vals.values()]
                # Add them to list of all coupled sweeps
                coupled_sweep_configurations += [[item for item in zip(*target_sweep_configurations)]]
                # Monitor the sweep size for this target name
                coupled_sweep_sizes[target_name] = {
                    "length":len(target_sweep_configurations[0]),
                    "vars":[sweep_val['var'] for sweep_val in target_name_vals.values()]
                }
                # Add coupled variable's target name to dict
                sweeped_vars = [sweep_val['var'] for sweep_val in target_name_vals.values()]
                self.target_names_by_sweep_var.update(
                    dict(
                        zip(
                            sweeped_vars,
                            [target_name]*len(sweeped_vars)
                        )
                    )
                )
            # Cross multiple both sweep configurations
            sweep_configurations = list(product(*(isolated_sweep_configurations+coupled_sweep_configurations)))
        else:
            # Cross multiple isolated sweep configurations
            sweep_configurations = list(product(*(isolated_sweep_configurations)))
        # Expand tuples within tuples
        sweep_configurations = [expand_tuple(item) for item in sweep_configurations]
        
        # Print parameter space size
        param_sizes = [f"{v['var']} ({len(v['values'])})" for v in sweep_params['isolated'].values()]
        total_sizes = [len(v['values']) for v in sweep_params['isolated'].values()]
        if len(sweep_params['coupled'].values()) > 0:
            for k,v in coupled_sweep_sizes.items():
                param_sizes += [f"{k}: "+','.join([f"{v['vars']}" ])+f" ({v['length']})"]
                total_sizes += [v['length']]
        
        param_sizes_str = " x ".join(param_sizes)
        total_size_str = reduce(mul,total_sizes)

        return sweep_configurations, param_sizes_str, total_size_str

    def parse_sweep_params(self):

        # Find common keys between isolated and coupled paths
        common_keys = set(list(self.isolated_sweep_paths.keys())).intersection(set(list(self.coupled_sweep_paths.keys())))
        # Remove them from isolated sweeps
        for k in list(common_keys):
            del self.isolated_sweep_paths[k]
        sweep_params = {"coupled":{},"isolated":{}}
        for key_path in self.isolated_sweep_paths.values():
            # print('isolated',key_path)
            # Get sweep configuration
            sweep_input,_ = self.path_get(
                settings=self.settings,
                path=(key_path+["sweep","range"])
            )
            # Parse values
            sweep_vals = self.parse_data(sweep_input,(key_path+["sweep","range"]))

            sweep_params['isolated'][">".join(key_path)] = {
                "var":key_path[-1],
                "path": key_path,
                "values": sweep_vals
            }
        coupled_val_lens = {}
        for target_name,coupled_paths in self.coupled_sweep_paths.items():
            # Monitor length of sweep values by target name
            coupled_val_lens[target_name] = {}
            sweep_params['coupled'][target_name] = {}
            for key_path in coupled_paths.values():
                # Get sweep configuration
                sweep_input,_ = self.path_get(
                    settings=self.settings,
                    path=(key_path+["sweep","range"])
                )
                # Parse values
                sweep_vals = self.parse_data(sweep_input,(key_path+["sweep","range"]))
                sweep_params['coupled'][target_name][">".join(key_path)] = {
                    "var":key_path[-1],
                    "path": key_path,
                    "values": sweep_vals
                }
                # Add key path length to dict
                coupled_val_lens[target_name][key_path[-1]] = len(sweep_vals)
        # Make sure all coupled parameters have the same length
        for target_name,keypaths in coupled_val_lens.items():
            try:
                assert len(set(list(keypaths.values()))) == 1
            except:
                raise Exception(f"""Coupled sweep parameters for target name {target_name} 
                                do not all have the same length
                                {json.dumps(keypaths)}""")

        return sweep_params