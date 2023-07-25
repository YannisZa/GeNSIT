import json
import os.path
import toml
import logging

from pathlib import Path
from copy import deepcopy
from multiresticodm.utils import deep_apply

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
            self.validate_config(self.settings,key_path=[])

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

    def validate_config(self,settings,key_path=[]):
        for k, v in self.parameters.items():
            if type(v) == dict:
                # Append key to path
                key_path.append(k)
                # Apply function recursively
                self.validate_config(v,key_path)
            else:
                if v:
                    print('>'.join(key_path),v)
                    print(k)
                    key_path.remove(k)
eyu9uA