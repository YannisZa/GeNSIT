import sys
import json
import os.path
import toml
import logging

from pathlib import Path
from copy import deepcopy
from multiresticodm.utils import deep_apply, str_in_list
from multiresticodm.config_data_structures import instantiate_data_type

class Config:

    def __init__(self, path:str=None, settings:dict=None):
        """
        Config object constructor.
        :param path: Path to configuration TOML file
        """
        self.logger = logging.getLogger(__name__)

        # Load config
        if path:
            self.logger.debug(f' Loading config from {path}')
            self.settings = toml.load(path, _dict=dict)

            # Load schema
            with open('./data/inputs/configs/cfg_schema.json', 'r') as f:
                self.schema = json.load(f)

            # Load all parameter positions
            with open('./data/inputs/configs/cfg_parameters.json', 'r') as f:
                self.parameters = json.load(f)

            # Validate config against schema
            self.validate_config(self.parameters,key_path=[])

            sys.exit()

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

    def set_paths_root(self)  -> None:
        """
        Add root path to all configured paths (inputs, output directories).
        :param root: root path
        :param dump_log: (bool) optionally dump overriden config to disk
        """
        # self.logger.warning(f"""
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

    def path_get(self,settings,path=[]):
        if len(path) <= 0:
            return None,False
        settings_copy = deepcopy(settings)
        for i,key in enumerate(path):
            # print(key,settings_copy.keys())
            # print('\n')
            if key == 'sweep':
                if isinstance(settings_copy,dict):
                    if settings_copy.get(key,'not-found') == 'not-found':
                        return settings_copy,False
                    else:
                        settings_copy = settings_copy.get(key,'not-found')
                else:
                    return settings_copy,(settings_copy!='not-found')
            else:
                # print(settings_copy)
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
            settings.setdefault(path[0],value)
            value_set = True
        else:
            value_set = self.path_set(settings.get(path[0],'not-found'),value,path[1:])
        
        return value_set

    def validate_config(self,parameters,key_path=[]):
        for k, v in parameters.items():
            if isinstance(v,dict):
                # Append key to path
                key_path.append(k)
                # Apply function recursively
                self.validate_config(v,key_path)
                # Remove it from path
                key_path.remove(k)
            else:
                if v:
                    # Append key to path
                    key_path.append(k)
                    # key_path = ['neural_network','hyperparameters','biases','layer_specific','type']
                    print('>'.join(key_path))
                    # print('\n')
                    
                    # Check if key was found in settings and schema
                    settings_val, settings_found = self.path_get(self.settings,key_path)
                    schema_val, schema_found = self.path_get(self.schema,key_path)

                    # key must exist in schema
                    try: 
                        assert schema_found
                    except:
                        raise Exception(f"Key {'>'.join(key_path)} not found in schema.")
                    
                    # Check if argument is optional in case it is not included in settings
                    # and it is not part of a sweep
                    if not str_in_list("sweep",key_path) and not settings_found:
                        if isinstance(schema_val,dict) and not schema_val['optional']:
                            raise Exception(f"""
                                Key {'>'.join(key_path)} is compulsory but not included
                            """)
                    
                    if key_path[-1] == "sweep" and settings_found:
                        # Check if argument is not sweepable but containts 
                        # a sweep parameter in settings

                        # Get schema of parameter configured for a sweep
                        schema_parent_val, _ = self.path_get(self.schema,key_path[:-1])
                        if not schema_parent_val["sweepable"]:
                            raise Exception(f"""
                                Key {'>'.join(key_path)} is not sweepable \
                                but contains sweep configuration
                            """)

                        # Check that argument that is sweepable containts 
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

                    # Check that if parameter sweep is activated
                    sweep_active = False
                    if key_path[-1] == 'sweep_parameters':
                        sweep_active = settings_val
                    
                    # If sweep is deactivated and only sweep configuration
                    # is provided then read the default sweep configuration
                    if key_path[-1] == "sweep" and not sweep_active and settings_found:
                        # Get parent key settings
                        settings_parent_val, _ = self.path_get(self.settings,(key_path+['default']))
                        # Update settings with sweep default
                        try:
                            assert self.path_set(self.settings,settings_parent_val,key_path[:-1])
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
                    
                    # Check that data type and data range is valid
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
                            # print(settings_val)
                            # print(schema_val)
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
                            # Parse settings value into approapriate data structure
                            entry = instantiate_data_type(
                                data=settings_val,
                                schema=schema_val
                            )
                        # Check that entry is valid
                        entry.check(key_path=key_path)

                    # Remove it from path
                    key_path.remove(k)
