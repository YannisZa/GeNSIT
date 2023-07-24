import json
import os.path
import toml
import logging

class Config:

    def __init__(self, path:str=None, settings:dict=None):
        """
        Config object constructor.
        :param path: Path to configuration TOML file
        """
        self.logger = logging.getLogger(__name__)

        if path:
            self.logger.debug(f' Loading config from {path}')
            self.settings = toml.load(path, _dict=dict)
        elif settings:
            self.settings = settings
        else:
            self.settings = None
            raise Exception(f'Config not found in {path}')

    def __str__(self):
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
