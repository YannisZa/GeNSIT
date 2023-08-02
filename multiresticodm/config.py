import json
import os.path
import toml
import logging

from copy import deepcopy
from multiresticodm.utils import deep_apply, setup_logger, str_in_list, read_json
from multiresticodm.config_data_structures import instantiate_data_type

class Config:

    def __init__(self, path:str=None, settings:dict=None, **kwargs):
        """
        Config object constructor.
        :param path: Path to configuration TOML file
        """
        # Setup logger
        self.level = kwargs.get('level','info').upper()
        self.logger = setup_logger(
            __name__,
            level=self.level,
            log_to_file=True,
            log_to_console=True
        )

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
        with open('./data/inputs/configs/cfg_schema.json', 'r') as f:
            self.schema = json.load(f)

    def load_parameters(self):
        # Load all parameter positions
        with open('./data/inputs/configs/cfg_parameters.json', 'r') as f:
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
        :param root: root path
        :param dump_log: (bool) optionally dump overriden config to disk
        """
        # logger.warning(f"""
           # All input paths, output path and data paths being 'rooted' with {}
            # This requires that all configured paths are relative.
            # """)
        dataset = os.path.abspath(self.settings['inputs']['dataset'])
        self.settings['inputs']['input_path'] = os.path.dirname(dataset)

        # Remove backslash if path starts with it
        directory = os.path.abspath(self.settings['outputs']['directory'])
        self.settings['outputs']['output_path'] = directory

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
        data = instantiate_data_type(settings,schema)
        # return parsed data
        return data.value()

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
                        return settings_copy,False
                    else:
                        settings_copy = settings_copy.get(key,'not-found')
                else:
                    return settings_copy,(settings_copy!='not-found')
            else:
                if isinstance(settings_copy,dict):
                    settings_copy = settings_copy.get(key,'not-found')
                else:
                    return settings_copy,(settings_copy!='not-found')
        
        return settings_copy,(settings_copy!='not-found')

    def path_set(self,settings,value,path=[]):
        
        if len(path) <= 0:
            return False
        
        value_set = False
        if len(path) == 1:
            if str_in_list(path[0],list(settings.keys())):
                settings[path[0]] = value
            value_set = True
        else:
            value_set = self.path_set(settings.get(path[0],'not-found'),value,path[1:])
        
        return value_set

    def validate_config(self,parameters=None,settings=None,key_path=[]):
        # Pass defaults if no meaningful arguments are provided
        if parameters is None:
            parameters = self.parameters
        if settings is None:
            settings = self.settings
        
        for k, v in parameters.items():
            try:
                # Experiment settings are provided in list of dictionaries
                # Special treatment is required
                if isinstance(v,list) and k == 'experiment':
                    # Append key to path
                    key_path.append(k)
                    # Get schema for key path
                    schema,_ = self.path_get(self.schema,key_path)
                    # Create dummy schema so that it can be found recursively
                    dummy_schema = {}
                    self.path_set(dummy_schema,{},key_path[:-1])
                    self.path_set(dummy_schema,schema,[k])
                    # Check if key was found in settings and schema
                    settings_val, settings_found = self.path_get(settings,key_path)
                    if settings_found:
                        # For every entry in settings, 
                        # validate using the same schema
                        for entry in settings_val:
                            # Create a dummy setting to validate dummy schema
                            dummy_settings = {}
                            self.path_set(dummy_settings,{},key_path[:-1])
                            self.path_set(dummy_settings,entry,[k])
                            # Print path to experiment keys
                            # Remove it from path
                            key_path.remove(k)
                            # Validate config
                            self.validate_config(dummy_schema,dummy_settings,key_path)
                    else:
                        # Remove it from path
                        key_path.remove(k)
                # If settings are a dictionary, validate settings 
                # against each key path
                elif isinstance(v,dict):
                    # Append key to path
                    key_path.append(k)
                    # Apply function recursively
                    self.validate_config(v,settings,key_path)
                    # Remove it from path
                    key_path.remove(k)
                # If settings are any other (primitive of non-promitive) value,
                # validate settings for given key path
                # Special treament is required depending on:
                # - whether parameter is optional or not (See 1:)
                # - whether sweep parameters are passed or not (See 2:)
                # - whether sweep mode is activated (See 3:)
                # - all data type-specific checks (See 4:)
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
                        
                        if key_path[-1] == "sweep" and settings_found:
                            # 2: Check if argument is not sweepable but containts 
                            # a sweep parameter in settings

                            # Get schema of parameter configured for a sweep
                            schema_parent_val, _ = self.path_get(self.schema,key_path[:-1])
                            if not schema_parent_val["sweepable"]:
                                raise Exception(f"""
                                    Key {'>'.join(key_path)} is not sweepable \
                                    but contains sweep configuration
                                """)

                            # 2: Check that argument that is sweepable containts 
                            # a default and a range configuration
                            try:
                                assert isinstance(settings_val,dict) \
                                    and str_in_list("default",settings_val.keys()) \
                                    and str_in_list("range",settings_val.keys())
                            except:
                                raise Exception(f"""
                                    Key {'>'.join(key_path)} is not a dictionary \
                                    or does not have 'default' and 'range' configurations \
                                    for a sweep
                                """)

                        # 3: Check that if parameter sweep is activated
                        sweep_active = False
                        if key_path[-1] == 'sweep_mode':
                            sweep_active = settings_val
                        
                        # 3: If sweep is deactivated and only sweep configuration
                        # is provided then read the default sweep configuration
                        if key_path[-1] == "sweep" and not sweep_active and settings_found:
                            # Get parent key settings
                            settings_parent_val, _ = self.path_get(settings,(key_path+['default']))
                            # Update settings with sweep default
                            try:
                                assert self.path_set(settings,settings_parent_val,key_path[:-1])
                            except:
                                raise Exception(f"""
                                    Key {'>'.join(key_path)} could not be updated \
                                    with sweep default
                                """)
                            # If this is the case we have modified the config
                            # Parse settings value into approapriate data structure
                            entry = instantiate_data_type(
                                data=settings_val,
                                schema=schema_val
                            )
                            # Check that entry is valid
                            entry.check(key_path=key_path)
                        
                        # 4: Check all data type-specific checks
                        # according to the schema
                        if settings_found:
                            # If parameter is sweepable but settings 
                            # do not contain a sweep configuration
                            if str_in_list("sweep",key_path) and \
                                (
                                    not isinstance(settings_val,dict) or \
                                    ( 
                                        isinstance(settings_val,dict) and \
                                        not str_in_list("sweep",settings_val.keys())
                                    ) 
                                ):
                                # Read parent schema
                                schema_val, _ = self.path_get(self.schema,key_path[:-1])
                                # Extract schema from sweep's 'default' key 
                                if key_path[-1] == 'range':
                                    # Remove it from path
                                    key_path.remove(k)
                                    continue
                                else:
                                    schema_val = schema_val['default']
                                # Parse settings value into approapriate data structure
                                entry = instantiate_data_type(
                                    data=settings_val,
                                    schema=schema_val
                                )
                            else:
                                try:
                                    # Parse settings value into approapriate data structure
                                    entry = instantiate_data_type(
                                        data=settings_val,
                                        schema=schema_val
                                    )
                                except:
                                    raise Exception
                            # Check that entry is valid
                            entry.check(key_path=key_path)

                        # Remove it from path
                        key_path.remove(k)
            except:

                raise Exception('Exception found in '+'>'.join(key_path))