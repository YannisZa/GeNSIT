import os
import re
import sys
import dgl
import ast
import zlib
import gzip
import json
import h5py
import torch
import random
import numexpr
import logging
import numbers
import decimal
import operator
import traceback
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import deepcopy
from difflib import SequenceMatcher
from itertools import chain, product, count
from typing import Dict, List, Union, Tuple
from collections.abc import Iterable,MutableMapping,Mapping,Sequence

from gensit.utils.exceptions import *
from gensit.utils.logger_class import *
from gensit.static.global_variables import NUMPY_TYPE_TO_DAT_TYPE,OPERATORS, DATA_SCHEMA
from gensit.static.plot_variables import LABEL_EXPRESSIONS, RAW_EXPRESSIONS

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def tuple_contained(small_tuple, large_tuple):
    return all(elem in large_tuple for elem in small_tuple)


def depth(seq):
    seq = iter(seq)
    try:
        for level in count():
            seq = chain([next(seq)], seq)
            seq = chain.from_iterable(s for s in seq if isinstance(s, Sequence))
    except StopIteration:
        return level

def write_csv(data:pd.DataFrame,filepath:str,**kwargs:Dict) -> None:
    # Write pandas to csv
    if not filepath.endswith('.csv'):
        filepath += '.csv'
    
    data.to_csv(filepath,**kwargs)


def write_npy(data:np.ndarray,filepath:str,**kwargs:Dict) -> None:
    # Write array to npy format
    np.save(file = filepath, arr = data)


def read_xr_data(dirpath:str,sample_gid:Iterable) -> dict:
    sam_name, collection_id = sample_gid
    group_id = f'{sam_name}>{collection_id}.nc'
    
    # Create group-specific filepath
    filepath = os.path.join(dirpath,group_id)
    try:
        with xr.open_dataarray(filepath) as ds:
            data = ds.load()    
    except Exception as exc:
        raise H5DataReadingFailed(message = str(exc))
    
    return {sam_name:data}

def write_xr_data(data:xr.DataArray,dirpath:str,**kwargs:Dict) -> None:
    try: 
        # Group id of interest
        group_id = kwargs.pop('group','')
        if isinstance(group_id,Iterable) and not isinstance(group_id,str):
            group_id = '>'.join([str(grid) for grid in group_id])
        
        # Create group-specific filepath
        filepath = os.path.join(dirpath,group_id+'.nc')

        # Writing the DataArray to a NetCDF file inside a with context
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            data.to_netcdf(
                path = filepath,
                mode = 'w',
                **kwargs
            )
            # close file
            data.close()
        else:                
            data.to_netcdf(
                path = filepath,
                mode = 'a',
                **kwargs
            )
            # close file
            data.close()
    except Exception as exc:
        raise H5DataWritingFailed(message = str(exc))




def write_compressed_npy(data:np.ndarray,filepath:str,**kwargs:Dict) -> None:
    # Write array to npy format
    with gzip.GzipFile(filepath, "wb", compresslevel = 6) as f:
        np.save(f, data)


def write_figure(figure,filepath,**settings):
    settings = settings.copy()
    filename_ending = settings.pop('filename_ending','pdf')
    filepath += '.'+filename_ending

    figure.savefig(
        filepath,
        format=filename_ending,
        **settings
    )
    
    plt.close(figure)


def write_figure_data(plot_data:Union[dict,pd.DataFrame],filepath:str,keys:list=[],figure_settings:dict={}):
    # Create output directory
    makedir(os.path.dirname(filepath))

    for plot_datum in plot_data:
        
        # Keys must be included in figure data
        assert set(list(plot_datum.keys())).issubset(set(keys))

        if figure_settings.get('data_format','dat') == 'dat':
            dtypes = {k:DATA_SCHEMA.get(k,'float') for k in keys},
            # Write dat file
            write_tex_data(
                key_type = dtypes,
                data = list(zip(*[np.asarray(plot_datum[k],dtype = dtypes[k]) for k in keys])),
                filepath = filepath+'_data.dat',
                precision = figure_settings.get('data_precision',19)
            )
        elif figure_settings.get('data_format','dat') == 'json':
            write_json(
                {k: plot_datum.get(k,None)
                    if k != 'outputs'
                    else plot_datum.get(k,None).config.settings
                for k in keys},
                filepath+'_data.json'
            )
        elif figure_settings.get('data_format','dat') == 'csv':
            write_csv(
                pd.DataFrame.from_dict(
                    {k:(
                        plot_datum.get(k,np.array([])).tolist() 
                        if isinstance(plot_datum.get(k),np.ndarray)
                        else plot_datum.get(k,None)
                    )
                    for k in keys}
                ),
                filepath+'_data.csv'
            )
        # Write plot settings to file
        write_json(
            figure_settings,
            filepath+'_settings.json'
        )

def read_file(filepath:str,**kwargs) -> np.ndarray:
    if filepath.endswith('.npy'):
        return read_npy(filepath = filepath,**kwargs)
    elif filepath.endswith('.txt'):
        return np.loadtxt(fname = filepath,**kwargs)
    elif filepath.endswith('.json'):
        return read_json(filepath = filepath)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath,**kwargs)
    elif filepath.endswith('.geojson'):
        return gpd.read_file(filepath,**kwargs)
    else:
        raise Exception(f'Cannot read file ending with {filepath}')


def read_npy(filepath:str,**kwargs:Dict) -> np.ndarray:
    # Write array to npy format
    data = np.load(filepath,allow_pickle=True)
    return data

def read_netcdf_group_ids(nc_data:str,key_path=[]):
    for k in nc_data.groups.keys():
        yield key_path+[k]
    for key,value in nc_data.groups.items():
        yield from read_netcdf_group_ids(value,key_path+[key])

def read_xr_group_ids(nc_data,list_format:bool = True):
    # Get all key paths
    key_paths = list(read_netcdf_group_ids(
        nc_data = nc_data,
        key_path = []
    ))
    # If not paths found return empty list
    if len(key_paths) <= 0:
        return key_paths
    
    # Return the longest paths
    max_len = max([len(kp) for kp in key_paths])
    longest_key_paths = [kp for kp in key_paths if len(kp) == max_len]
    if list_format:
        return ['/'+'/'.join(kp) for kp in longest_key_paths]
    else:
        longest_key_paths_dict = {}
        for keypath in longest_key_paths:
            if keypath[0] in longest_key_paths_dict:
                # Update key paths
                longest_key_paths_dict[keypath[0]].append(
                    ','.join(keypath[1:])
                )
            else:
                longest_key_paths_dict[keypath[0]] = [','.join(keypath[1:])]

        return longest_key_paths_dict
    
    

def read_compressed_npy(filepath:str,**kwargs:Dict) -> np.ndarray:
    # Write array to npy format
    with gzip.GzipFile(filepath, "rb") as f: 
        data = np.load(f)
    return data


def write_tex_data(data:Union[np.array,np.ndarray],key_type:Union[list,np.array,np.ndarray],filepath:str,**kwargs:Dict) -> None:
    if not filepath.endswith('.dat'):
        filepath += '.dat'
    with open(filepath, 'w') as f:
        np.savetxt(
            f, 
            data,
            header=' '.join(key_type.keys()),
            comments='',
            fmt=[(NUMPY_TYPE_TO_DAT_TYPE[key_type[x]]+str(kwargs.get('precision',19))) if "d" in key_type[x] else NUMPY_TYPE_TO_DAT_TYPE[key_type[x]] for x in key_type.keys()]
        )


def write_txt(data:Union[np.array,np.ndarray],filepath:str,**kwargs:Dict) -> None:
    # Write numpy to txt
    if not filepath.endswith('.txt'):
        filepath += '.txt'
    np.savetxt(filepath,data,**kwargs)

def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')


def write_json(data:Dict,filepath:str,**kwargs:Dict) -> None:
    if not filepath.endswith('.json'):
        filepath += '.json'
    with open(filepath, 'w') as f:
        json.dump(data,f,default=json_default,**kwargs)


def print_json(data:Dict,**kwargs:Dict):
    if kwargs.get('newline',False):
        for k in data.keys():
            print(f"{k}: {data[k]}",sep='')
    else:
        print(json.dumps(data,cls = NumpyEncoder, default = json_default, **kwargs))

def write_compressed_string(data:str,filepath:str) -> None:
    with gzip.GzipFile(filename = filepath, mode="w") as f:
        f.write(zlib.compress(data.encode()))


def read_compressed_string(filepath:str) -> None:
    with gzip.GzipFile(filename = filepath, mode="r") as f:
        data = f.read()
        f.close()
    data = ast.literal_eval(zlib.decompress(data).decode())
    return data


def write_compressed_json(table:List[Dict],filepath:str) -> None:
    # https://stackoverflow.com/questions/39450065/python-3-read-write-compressed-json-objects-from-to-gzip-file
    json_str = json.dumps(table) + "\n"               # 2. string (i.e. JSON)
    json_bytes = json_str.encode('utf-8')            # 3. bytes (i.e. UTF-8)

    with gzip.open(filepath, 'w') as fout:       # 4. fewer bytes (i.e. gzip)
        fout.write(json_bytes)


def read_json(filepath:str) -> Dict:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def read_compressed_json(filepath:str) -> List[Dict]:
    with gzip.open(filepath, 'r') as fin:
        f = json.loads(fin.read().decode('utf-8'))
    return f

def get_dims(data):
    try:
        res = list(np.shape(data))
    except:
        res = list(data.dims)
    return res

def parse(value,default:str='none',ndigits:int = 5):
    if value is None:
        return default
    elif isinstance(value,str):
        if len(value) <= 0:
            return default
        try:
            value = string_to_numeric(value)
        except:
            # value = unstringify(value)
            pass
    elif hasattr(value,'__len__'):
        return np.array2string(np.array(value))
    elif isinstance(value,float):
        return np.float32(np.round(value,ndigits))
    
    return value

def makedir(directory:str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def tuple_to_str(t:Tuple[int,int]) -> str:
    return str(t).replace(' ','')


def str_to_tuple(s:str) -> Tuple[int,int]:
    return (int(s.split('(')[1].split(',')[0]),int(s.split(',')[1].split(')')[0]))


def ndims(__self__,time_dims:bool = True, dtype:str = 'uint16'):
    return np.sum([1 for dim in unpack_dims(__self__.data.dims,time_dims = time_dims) if dim > 1],dtype=dtype)

def deep_merge(dict1,dict2):
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], list) and isinstance(value, list):
                result[key].extend(value)
            else:
                result[key] = [result[key]] + [value]
        else:
            result[key] = value
    return result

def deep_get(key, value):
    for k, v in (value.items() if isinstance(value, dict) else
       enumerate(value) if isinstance(value, list) else []):
        if k == key:
            yield v
        elif isinstance(v, (dict, list)):
            for result in deep_get(key, v):
                yield result


def deep_call(input:object,expressions:str,defaults:object,**kwargs):
    characters = ['.','[',']','(',')','()']
    # Convert nested expression into successive get attr
    if isinstance(expressions,str):
        attrs_and_separators = re.split(r"(\(.*\)|\[.*\]|\W+)",expressions)
        attrs_and_separators = [s for s in attrs_and_separators if len(s) > 0]
        value = input
        latest_separator = ''
        for item in attrs_and_separators:
            if np.any([char in item for char in characters]):
                # Character separator, function or dictionary call
                if item == '()':
                    value = value()
                elif item.startswith('(') and item.endswith(')'):
                    # Get function arguments
                    function_arguments = item.replace('(','').replace(')','')
                    function_arguments = function_arguments.strip(' ').split(',')
                    # Call function with arguments
                    value = value(*[kwargs.get(arg,defaults) for arg in function_arguments])
                elif item.startswith('[') and item.endswith(']'):
                    # Get dictionary argument
                    dict_argument = item.replace('[','').replace(']','')
                    # dict_argument = dict_argument.strip(' ').split(',')
                    value = value.get(kwargs.get(dict_argument,defaults),defaults)
                else:
                    # Store separator
                    latest_separator = item.strip()
            else:
                if latest_separator == '.':
                    # Get object attribute
                    value = getattr(value,item.strip(),defaults)
                elif len(latest_separator) <= 0:
                    continue 
                else:
                    raise ValueError(f'Separator character {latest_separator} not recognized')
    elif isinstance(expressions,Iterable):
        value = []
        for i,expr in enumerate(expressions):
            value.append(
                deep_call(
                    input = input,
                    expressions = expr,
                    defaults = defaults[i],
                    kwargs = kwargs
                )
            )
    else:
        value = defaults
    
    return value

def operate(input:object,operations:str,**kwargs):
    operators_and_args = re.split(r"(\(.*\)|\[.*\]|\W+)",operations)
    operators_and_args = [s for s in operators_and_args if len(s) > 0]
    # Assert that the list of operators and arguments must by divisible by 2
    # that is every operator is accopanied by an argument
    try:
        assert len(operators_and_args) % 2 == 0
    except:
        raise Exception(f"{operators_and_args} has length {len(operators_and_args)} which is not divisible by 2.")
    value = input
    for operator_arg in [operators_and_args[x:x+2] for x in range(0, len(operators_and_args), 2)]:
        operator = operator_arg[0]
        attr = operator_arg[1]
        if attr.isnumeric():
            value = OPERATORS[operator](value,string_to_numeric(attr))
        else:
            value = OPERATORS[operator](value,kwargs[attr])
    return value
    

# https://stackoverflow.com/questions/27265939/comparing-python-dictionaries-and-nested-dictionaries
def findDiff(d1,d2,path:str="") -> None:
    for k in d1:
        if k in d2:
            if type(d1[k]) is dict:
                findDiff(d1[k],d2[k], "%s -> %s" % (path, k) if path else k)
            if d1[k] != d2[k]:
                result = [ "%s: " % path, " - %s : %s" % (k, d1[k]) , " + %s : %s" % (k, d2[k])]
        else:
            print("%s%s missing, default %s \n" % ("%s: " % path if path else "", k, d1[k]))

def find_common_substring(names):
    substring_counts={}

    for i in range(0, len(names)):
        for j in range(i+1,len(names)):
            string1 = names[i]
            string2 = names[j]
            match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
            matching_substring = string1[match.a:match.a+match.size]
            if(matching_substring not in substring_counts):
                substring_counts[matching_substring] = 1
            else:
                substring_counts[matching_substring]+=1

    return substring_counts

def get_all_subdirectories(out_path,stop_at:str='config.json',level:int = 2):
    directories = []
    print('out_path',out_path)
    for root, dirs, files in walklevel(out_path,level = level):
        # If this is a dir that matches the stopping condition
        # or has a file that matches the stopping condition
        if any([stop_at in file for file in files]):
            # entry_path = Path(root)
            # directories.append(str(entry_path.absolute()))
            directories.append(root)

    return directories

def walklevel(some_dir, level = 1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def find_dataset_directory(dataset_name):
    # If only dataset is provide - save to that directory
    if len(dataset_name) == 1:
        dataset = dataset_name[0]
    else:
        # Find common substring from datasets
        dataset = find_longest_common_substring(dataset_name)
        if len(dataset) == 0:
            dataset = 'multiple_datasets'
        else:
            dataset += 'datasets'
    return dataset 

def find_longest_common_substring(names):
    # Find common substrings
    substring_counts = find_common_substring(names)
    # Pick maximum occuring one
    return max(substring_counts.items(), key = operator.itemgetter(1))[0]


# https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def update_recursively(d:Dict, u:Dict, overwrite:bool = False) -> Dict:
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update_recursively(d.get(k, {}), v, overwrite)
        else:
            if overwrite:
                # Overwrite even if key exists
                d[k] = v
            else:
                # Do not overwrite if key exists
                if k not in d.keys():
                    d[k] = v
    return d


def deep_flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        # new_key = parent_key + sep + k if parent_key else k
        new_key = k
        if isinstance(v, MutableMapping):
            items.extend(deep_flatten(v, new_key, sep = sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def update_deep_dict(original, new_data, **kwargs):
    for key, value in new_data.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            update_deep_dict(original[key], value)
        else:
            if kwargs.get('overwrite',True):
                original[key] = value


def deep_update(d,key,val,**kwargs):
    for dict_key,dict_val in d.items():
        if (dict_key == key) and kwargs.get('overwrite',True):
            d[key] = val
        elif isinstance(dict_val, dict):
            deep_update(dict_val,key,val,**kwargs)
        elif isinstance(dict_val, list):
            for v in dict_val:
                if isinstance(v,dict):
                    deep_update(v,key,val,**kwargs)
        

def deep_updates(main_dict,update_dict,**kwargs):
    for k,v in update_dict.items():
        deep_update(main_dict,k,v,**kwargs)
    return main_dict
            

def deep_delete(dictionary, keys):
    keys_set = set(keys)  # Just an optimization for the "if key in keys" lookup.

    modified_dict = {}
    for key, value in dictionary.items():
        if key not in keys_set:
            if isinstance(value, MutableMapping):
                modified_dict[key] = deep_delete(value, keys_set)
            else:
                modified_dict[key] = value  # or copy.deepcopy(value) if a copy is desired for non-dicts.
    return modified_dict

def deep_apply(ob, func, **kwargs):
    for k, v in (ob.items() if isinstance(ob,dict) else enumerate(ob)):
        if isinstance(v, Mapping):
            deep_apply(v, func, **kwargs)
        else:
            ob[k] = func(v,**kwargs)
    return ob

def pop_variable(_self,var,default=None):
    if hasattr(_self,var):
        res = getattr(_self,var)
        delattr(_self,var)
        return res
    else:
        return default

def extract_config_parameters(conf,fields = dict):
    trimmed_conf = {}
    if not isinstance(conf,dict):
        return conf
    if isinstance(fields,str):
        trimmed_conf[fields] = conf[fields]
        return trimmed_conf[fields]

    for f,v in fields.items():
        if v == "":
            if f in conf:
                trimmed_conf[f] = conf[f]
        else:
            if f in conf:
                trimmed_conf[f] = extract_config_parameters(conf[f],v)
    return trimmed_conf

def safe_delete(variable,instance = None):
    try:
        del variable
    except:
        pass



def parse_slice_by(slice_by:list):
    if len(slice_by) <= 0:
        return {}
    
    slices = {}
    for key_val in slice_by:
        key,val = key_val

        # Convert value to appropriate data_type
        if isinstance(val,str):
            if "_" in val:
                val = val.split("_")
            # Parse data
            val = list(map(parse,val))
        
        if key in list(slices.keys()):
            if isinstance(val,list):
                slices[key].union(set(val))
            else:
                slices[key].add(val)

        else:
            if isinstance(val,list):
                slices[key] = set(val)
            else:
                slices[key] = {val}
    
    # Map set values to list
    slices = {k:list(v) for k,v in slices.items()}
    return slices

def stringify_coordinate(d):
    if d is None:
        return 'none'
    elif isinstance(d,float) or ((hasattr(d,'dtype') and 'float' in str(d.dtype))) and not np.isfinite(d):
        return str(d)
    else:
        return d

def stringify(data,**kwargs):
    if isinstance(data, dict) and len(data) > 0:
        try:
            res = json.dumps(data)
        except:
            res = str(data)
        return res
    elif isinstance(data,Iterable) and not isinstance(data,str) and len(data) > 0:
        return kwargs.get('preffix','')+ \
            ','.join([stringify(v,**kwargs) for v in data])+ \
            kwargs.get('suffix','')
    elif isinstance(data,numbers.Real) and kwargs.get('scientific',True):
        try:
            assert np.isfinite(data)
        except:
            return "nan"
        if isinstance(data,decimal.Decimal):
            return "{:.2e}".format(data)
        elif data > 100:
            return "{:.0e}".format(data)
        else:
            return str(data)
    elif not data and data != 0:
        return "none"
    else:
        return str(data)
        

def unstringify(data):
    try:
        decoded_data = eval(data)
    except:
        decoded_data = data
    return decoded_data

def stringify_statistic(statistic):
    # Unpack statistic pair
    statistic_name,statistic_dims = statistic
    # Initialise text
    text = ''
    if statistic_name != '':
        text += statistic_name
    if statistic_dims is not None:
        if hasattr(statistic_dims,'__len__'):
            text += '_'+','.join(statistic_dims)
        else:
            text += '_'+str(statistic_dims)
    return text


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        return rng
    else:
        return np.random.default_rng(None)

def update_device(device):
    if device is None or len(device) == 0:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    return device

def set_device_id(device_id):
    torch.cuda.set_device(device_id)


def tuplize(tup):
    # Convert strings
    if isinstance(tup,str):
        tup = int(tup)
    if hasattr(tup,'__len__'):
        return tuple(tup)
    else:
        return tuple([tup])


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'bytes'


def convert_string_to_torch_function(s:str=''):
    if 'mean' == s.lower():
        return torch.mean
    elif 'var' == s.lower() or 'variance' == s.lower():
        return torch.var
    elif 'std' == s.lower():
        return torch.std
    elif 'sum' == s.lower():
        return torch.sum
    elif 'max' == s.lower():
        return torch.max
    elif 'min' == s.lower():
        return torch.min
    elif 'median' == s.lower():
        return torch.median
    elif 'sumnorm' == s.lower():
        def sum_normalise(x,axis = None):
            if axis != (None,):
                return x/torch.sum(x,axis)
            else:
                return x/torch.sum(x)
        return sum_normalise
    elif 'X' in s:
        raise NotImplementedError('Evaluating expressions has not been implemented for torch')
        # Remove dangerous key words
        # Prepare evaluation of mathematical expression
        s = s.replace("^","**").replace("\\","").strip()
        # Define the corresponding function
        def evaluate_expression(data,axis = None):
            if axis is not None:
                return numexpr.evaluate(s.replace("X", "data").replace("axis",axis))
            else:
                return numexpr.evaluate(s.replace("X", "data").replace("axis",""))
        return np.vectorize(evaluate_expression)
    else:
        raise Exception(f"Function name {s} not match to torch function")

def valid_experiment_type(experiment_id,experiment_types):
    valid_experiment = False
    for etype in experiment_types:
        if etype.lower() in experiment_id.lower():
            valid_experiment = True
    return valid_experiment 


def convert_string_to_numpy_shape(s:str,**kwargs):
    # Remove whitespace
    s = s.replace(" ","").replace("(","").replace(")","")
    shape = []
    for dim in s.split(','):
        if dim in kwargs.keys():
            if hasattr(kwargs[dim],'__len__'):
                for _dim in kwargs[dim]:
                    shape.append(_dim)
            else:
                shape.append(kwargs[dim])
        elif np.char.isnumeric(dim):
            shape.append(int(dim))
        else:
            shape.append(None)
    return shape


# Convert dictionary into dataframe
def f_to_df(f:Dict,type:str='int32')-> pd.DataFrame:

    # Map string tuples to integer tuples with values included
    f_mapped = [(k[0],k[1],v) for k, v in f.items()]
    # Create dataframe from list of tuples
    my_table = pd.DataFrame.from_records(f_mapped,columns=['row','column','value'])
    # Pivot table to convert values into row and column indices
    my_table = my_table.pivot(index='row', columns='column',values='value')
    # Remove index names
    my_table.columns.name = None
    my_table.index.name = None

    # Map column and row index to string
    # my_table.index = my_table.index.map(str)
    # my_table.columns = my_table.columns.map(str)

    # Make sure values are integers
    if type == 'int32':
        my_table = my_table.astype('int32')
    else:
        my_table = my_table.astype(type)

    return my_table


def f_to_array(f:Dict,shape:tuple = None)-> Union[np.ndarray,np.generic]:

    # Get tuple indices
    rows,cols = zip(*f.keys())
    # Update entries based on dimensionality of array
    if max(rows) == 0:
        if shape is None:
            arr = np.zeros((1,max(cols)+1))
        else:
            arr = np.zeros(shape)
        arr[0,list(cols)] = list(f.values())
    elif max(cols) == 0:
        if shape is None:
            arr = np.zeros((max(rows)+1,1))
        else:
            arr = np.zeros(shape)
        arr[list(rows),0] = list(f.values())
    else:
        if shape is None:
            arr = np.zeros((max(rows)+1,max(cols)+1))
        else:
            arr = np.zeros(shape)
        arr[list(rows),list(cols)] = list(f.values())
    return arr.astype('int32')
    # return np.array([v for k,v in sorted(f.items(),key = lambda x:str_to_tuple(x[0]))])


def array_to_f(a:Union[np.ndarray,np.generic],axis:int = 0) -> Dict:
    if len(np.shape(a)) == 1 and axis == 0:
        return dict(((0,index),int(value)) for index, value in enumerate(a))
    elif len(np.shape(a)) == 1 and axis == 1:
        return dict(((index,0),int(value)) for index, value in enumerate(a))
    else:
        return dict((index,int(value)) for index,value in np.ndenumerate(a))


def f_to_str(f:dict) -> str:
    return ','.join([str(v) for k,v in sorted(f.items(),key = lambda x:str_to_tuple(x[0]))])


def table_to_str(tab:np.ndarray) -> str:
    return ','.join([str(num) for num in tab.flatten()])

def str_to_table(s:str,dims:list) -> pd.DataFrame:
    data = list(map(int,s.split(',')))
    assert len(data) == np.prod(dims)
    rows = [data[i:i + dims[1]] for i in range(0, len(data), dims[1])]
    return np.asarray(rows)


# Convert table to dictionary of values (f function)
def df_to_f(tbl:pd.DataFrame)-> Dict:

    my_f = {}
    for index, row in tbl.iterrows():
        for colname in row.index:
            my_f[(int(index),int(colname))] = int(row[colname])
    return my_f


def str_to_array(s:str,dims:Tuple):
    # If array should be 1D
    if not isinstance(dims,(list,Tuple,np.ndarray,np.generic)) or len(dims) == 1:
        return np.array(s.split(',')).astype('int32')
    # If array should be higher dimensional
    else:
        return np.reshape(np.asarray(s.split(','),dtype = int),newshape = dims)
    

def create_dynamic_data_label(__self__,data,**kwargs):
    # Read label(s) from settings
    label_by_key,label_by_value = [],[]
    for k in list(__self__.settings['label_by']):
        v = list(deep_get(key = k,value = data))[0]
        if k == "dims":
            v = 'x'.join(list(map(str,list(unpack_dims(v,kwargs.get('time_dims',False))))))
        # If label not included in metadata ignore it
        if len(v) > 0 and v[0] is not None:
            label_by_key.append(''.join(list(map(str,k))))
            label_by_value.append(''.join(list(map(str,v))))

    # Create label for each plot
    x_label = ''
    for i in range(len(label_by_key)):
        if (i < len(label_by_key) - 1):
            x_label += f"{label_by_key[i].replace('table_','').replace('_',' ')}:"+ \
                        f"{str(label_by_value[i]).replace('_',' ')},"
        else:
            x_label += f"{label_by_key[i].replace('table_','').replace('_',' ')}:" + \
                        f"{str(label_by_value[i]).replace('_',' ')}"
    # Add label to error data
    return x_label,label_by_key,label_by_value

def in_range(v,limits:list,allow_nan:bool = False,inclusive:bool = False):
    within_range = True
    if v is None or not np.isfinite(v):
        return allow_nan
    if limits[0] is not None:
        within_range = within_range and (v >= limits[0] if inclusive else v > limits[0])
    if limits[1] is not None:
        within_range = within_range and (v <= limits[1] if inclusive else v < limits[0])
    return within_range

def is_null(v):
    if isinstance(v,str):
        return v == ''
    elif isinstance(v,Iterable):
        return any([is_null(val) for val in v])
    else:
        
        return v is None or pd.isna(v)

def dict_inverse(d:dict):
    return {v:k for k,v in d.items()}

def get_value(d:dict,k:str,default:object = None,apply_latex:bool=False):
    try:
        value = d[k].item()
    except:
        try:
            value = d[k]
        except:
            value = default
    
    if k == 'sigma':
        value = sigma_to_noise_regime(value)
    # print(k,value,type(value))
    if isinstance(value,str) and apply_latex:
        return latex_it(
            key = k,
            value = value,
            default = default
        )
    elif isinstance(value,str):
        return unstringify(value)
    else:
        return value

def hash_major_minor_var(hashmap:dict,data:list,**kwargs):
    # Join major and minor ticks
    major = stringify(data[0],**kwargs)
    minor = stringify(data[1],**kwargs)
    if minor and minor != 'none':
        # Apply hashmap
        return hashmap.get(stringify(
            [major,minor],
            **kwargs
        ))
    else:
        # Apply hashmap
        return hashmap.get(stringify(
            major,
            **kwargs
        ))

def get_keys_in_path(d, target_key, path=[], paths_found = []):
    for key, value in d.items():
        current_path = path + [key]
        if key == target_key:
            # Add it to paths found
            paths_found.append(current_path[:-1])
        if isinstance(value, dict):
            _ = get_keys_in_path(value, target_key, current_path, paths_found)
    return paths_found

def get_value_from_path(d, path=[]):
    if len(path) <= 0:
        return 'not-found'
    elif len(path) == 1:
        return d.get(path[0],'not-found')

    for key in path:
        if isinstance(d.get(key,'not-found'),dict):
            return get_value_from_path(d.get(key,'not-found'),path[1:])
        else:
            return 'not-found'

def string_to_numeric(s):
    f = np.float32(s)
    i = np.int32(f)
    return i if i == f else np.round(f,5)

def sigma_to_noise_regime(sigma = None):
    if sigma:
        if sigma == 'none':
            return 'learned'
        elif isinstance(sigma,str):
            return sigma
        elif (2/float(sigma)**2) >= 10000:
            return 'low'
        elif (2/float(sigma)**2) < 10000 and sigma > 0:
            return 'high'
        else:
            return 'learned'
    else:
        return 'learned'

def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))

def h5_deep_get(name,group,prefix=''):
    for key, val in group.items() if isinstance(group,h5py.Group) \
        else enumerate(group) if isinstance(group,h5py.Dataset) else []:
            path = f'{prefix}/{key}'
            print(path)
            if name == key:
                yield path
            elif isinstance(val, h5py.Group):
                gpath = h5_deep_get(key, val, path)
                if gpath:
                    yield gpath
            elif isinstance(val, h5py.Dataset):
                if name == key:
                    yield path

def broadcast2d(arr,shape,expanded_axis):
    # Squeeze array
    arr = np.squeeze(arr)
    # Expand dims
    arr_reshaped = np.expand_dims(arr,axis = expanded_axis)
    # Repeat along axis
    arr_reshaped = np.repeat(arr_reshaped,shape[expanded_axis],axis = expanded_axis)
    return arr_reshaped.reshape(shape)

def expand_tuple(t):
    result = []
    for item in t:
        if isinstance(item, tuple):
            result.extend(expand_tuple(item))
        else:
            result.append(item)
    return tuple(result)

def unpack_dims(self,time_dims:bool = True):
    try:
        dims = tuple([v for k,v in self.dims.items() if (k != 'time' or time_dims)])
    except:
        try:
            dims = tuple([v for k,v in self["dims"].items() if (k != 'time' or time_dims)])
        except:
            try:
                dims = tuple([v for k,v in self.items() if (k != 'time' or time_dims)])
            except:
                raise Exception(f"Cannot unpack dimensions from {self}")
    return dims

def to_json_format(x):
    if torch.is_tensor(x):
        if x.size == 1:
            x = x.cpu().detach().item()
        else:
            x = x.cpu().detach().numpy()
    if isinstance(x,np.ndarray):
        x = x.tolist()
    if isinstance(x,np.generic):
        x = x.item()
    return x

def tuple_dim(x,dims=()):
    if isinstance(x,tuple):
        dims = (*dims,len(x))
        return tuple_dim(x[0],dims = dims)
    else:
        return dims
    
def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def unique(data):
    if hasattr(data,'__len__') and not isinstance(data,str):
        # Convert nested lists to string
        hashable_data = ['_'.join(str(datum)) for datum in data]
        # Return unique values
        return list(set(hashable_data))
    else:
        return data
    
def invert_dict(data):
    inverse = {}
    for k,v in data.items():
        for x in v:
            inverse.setdefault(x, []).append(k)
    return inverse

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

# def dummy_callback(f):
#     temp = f.result()
#     print(f'callback {f}')
    
def unstack_dims(data:xr.DataArray,dims:list):
    for d in dims:
        if d in data.dims:
            data = data.unstack(d)
    return data

def is_sorted(x):
    return np.all(x[:-1] <= x[1:])

def position_index(active_positions):
    # Find first inactive position
    for i,active in enumerate(active_positions):
        # If position is not active return it
        if not active:
            return i

    return len(active_positions)


def fn_name(fn):
    try:
        return re.sub(r'[,.()\[\]]', '', str(fn.__name__))
    except:
        return re.sub(r'[,.()\[\]]', '', str(fn))
    
def tidy_label(label:str):
    # Replace _ with space
    label = label.replace('_',' ')
    # Replace , with _
    label = label.replace(',','_')
    # Replace double space with single space
    label = label.replace('  ',' ')
    return label

def strip_special_characters(string:str):
    return ''.join(e for e in string if e.isalnum() and e != ',')

def latex_it(key:str, value, default:str='learned',ndigits:int=3):
    # If key has a math expression replace it with math expression
    if str(value) in RAW_EXPRESSIONS:
        return RAW_EXPRESSIONS[str(value)]
    elif key in LABEL_EXPRESSIONS:
        return (LABEL_EXPRESSIONS[key]+str(parse(value, default = default, ndigits = ndigits))+'$')
    elif isinstance(value,str):
        return tidy_label(str())
    else:
        return value


def lexicographic_sort(arr):
    # Get the shape of the array
    shape = arr.shape
    
    # Create an array of indices for each dimension
    indices = [np.arange(dim) for dim in shape]
    
    # Create a meshgrid from the indices
    meshgrid = np.meshgrid(*indices, indexing='ij')
    
    # Reshape the meshgrid to match the shape of the original array
    reshaped_meshgrid = [grid.reshape(-1) for grid in meshgrid]
    
    # Create an array of tuples representing the indices along each dimension
    tuples = np.vstack(reshaped_meshgrid).T
    
    # Use lexsort to get the sorting order based on each dimension
    sorting_order = np.lexsort(tuple(arr[tuple(t)] for t in tuples.T))
    
    # Apply the sorting order to the original array
    sorted_array = arr.reshape(-1, arr.shape[-1])[sorting_order].reshape(arr.shape)
    
    return sorted_array


def xr_expand_multiindex_dims(data,expanded_coords:dict,multiindex_dim:str):
                            
    # Expand dimensions
    data = data.expand_dims(expanded_coords)

    # Expand coordinates
    data = data.assign_coords(expanded_coords)
    
    return xr_restack(
        data,
        multiindex_dim = multiindex_dim,
        added_dims = list(expanded_coords.keys())
    )

def xr_restack(data,multiindex_dim:str,added_dims:list=[],new_multiindex_dim:str=None):
    # Decide on name of new multiindex dim
    new_multiindex_dim = new_multiindex_dim if new_multiindex_dim else multiindex_dim

    # Get all names in multi-index
    multiindex_dim_names = list(data.get_index(multiindex_dim).names)

    # Unstack multi-index
    data = data.unstack(multiindex_dim)
    
    # Re stack updated multi-index
    data = data.stack({new_multiindex_dim: (multiindex_dim_names+added_dims)})

    return data


def xr_islice(data,dim:str='',start:int=0,step:int=1,end:int=None,**kwargs):
    # Get sizes for the data's dim
    try:
        assert dim in data.dims
    except:
        raise MissingData(
            missing_data_name = dim,
            data_names = list(data.dims),
            location = 'xr_islice'
        )

    # Get all possible data sizes
    size_slice = slice(
        start if start is not None else 0,
        end if end is not None else min(data.sizes[dim],end),
        step if step is not None else 1
    )

    return data.isel(**{dim:size_slice})


def xr_apply_by_group(function,data,group_dims:list,sweep_dim:str='sweep',fixed_kwargs:dict={},sweeped_kwargs:dict={},**kwargs):
    # Get data coordinates for every existing dim and find all possible combinations
    sweep_dims = list(data.get_index(sweep_dim).names)
    sweep_dimensions = [d for d in group_dims if d in sweep_dims]
    nonsweep_dimensions = [d for d in group_dims if d in data.dims and d not in sweep_dimensions]
    dimensions = sweep_dimensions + nonsweep_dimensions
    # Gather all result data into list
    results = []
    print(sweep_dimensions,nonsweep_dimensions)
    # Loop every possible coordinate
    for group_label,group_data in xr_restack(
        data,
        multiindex_dim=sweep_dim,
        added_dims=nonsweep_dimensions,
        new_multiindex_dim=sweep_dim
    ).groupby(sweep_dim):
        print(group_label)
        # Get part of group label you need 
        group_sublabel = [
            (list(group_label)[sweep_dims.index(d)] 
            if d in sweep_dims
            else list(group_label)[len(sweep_dims)-1+nonsweep_dimensions.index(d)])
            for d in dimensions
        ]
        print(group_sublabel)
        # Convert tuple to string 
        group_sublabel_str = stringify(list(group_sublabel))
        print(group_sublabel_str)
        try:
            assert group_sublabel_str in sweeped_kwargs
        except:
            raise MissingData(
                missing_data_name = group_sublabel_str,
                data_names = list(sweeped_kwargs.keys()),
                location = 'xr_apply_by_group'
            )
        # Get sweeped function kwargs
        sweeped_kwargs = sweeped_kwargs[group_sublabel_str]
        # Add fixed function kwargs to get all function kwargs
        fn_kwargs = {**sweeped_kwargs,**fixed_kwargs}
        # Add to results
        if kwargs.get('apply_ufunc',False):
            results.append(
                function(
                    group_data,
                    **fn_kwargs
                )
            )
        else:
            results.append(
                xr.apply_ufunc(
                    function,
                    group_data,
                    kwargs=fn_kwargs,
                    exclude_dims=set(['id']),
                    input_core_dims=[['id']],
                    output_core_dims=[['id']]
                )
            )
    
    res = xr.merge(results)
    print(res)
    return res

def xr_apply_and_combine_wrapper(
    data,
    functions:list=[],
    fixed_kwargs:dict={},
    isolated_sweeped_kwargs:dict={},
    coupled_sweeped_kwargs:dict={},
    existing_dim:str = 'sweep'
):
    # Find all combinations of sweep function arguments
    sweep_keys = list(isolated_sweeped_kwargs.keys())+list(coupled_sweeped_kwargs.keys())
    
    isolated_sweep_vals = [vals for vals in isolated_sweeped_kwargs.values()]
    coupled_sweep_vals = []
    for target_name_keyvals in coupled_sweeped_kwargs.values():
        coupled_sweep_vals += [list(zip(*list(target_name_keyvals.values)))]

    if not coupled_sweeped_kwargs and isolated_sweeped_kwargs:
        sweep_vals = list(product(*isolated_sweep_vals))
    elif coupled_sweeped_kwargs and not isolated_sweeped_kwargs:
        sweep_vals = list(product(*coupled_sweep_vals))
    elif coupled_sweeped_kwargs and not isolated_sweeped_kwargs:
        sweep_vals = list(product(*(isolated_sweep_vals+coupled_sweep_vals)))
    else:
        sweep_vals = []

    # Find interesection of fixed and sweeped function arguments.
    # This intersection should be empty - assert it
    keyword_key_intersection = set(
        sweep_keys
    ).intersection(
        set(
            list(fixed_kwargs.keys())
        )
    )
    try:
        assert len(keyword_key_intersection) == 0
    except:
        raise DataConflict(
            property = 'function keyword keys',
            data = list(keyword_key_intersection),
            problem = 'Sweeped and fixed keyword arguments should not have any common keys'
        )        

    # Keep list of function composition outputs
    function_composition_outputs = []
    # For every sweeped function argument
    for function_sweeped_vals in tqdm(
        sweep_vals, 
        leave = False,
        disable = True,
        desc = f"Applying function(s) {', '.join([list(fn.keys())[0] for fn in functions])} over multiple inputs"
    ):
        # Apply each function composition in order 
        # specified by functions argument
        function_output = deepcopy(data)
        for function_composition in functions:
            # Elicit fixed argument and function callable
            function_name = list(function_composition.keys())[0]
            try:
                # Elicit important function settings
                function_settings = list(function_composition.values())[0]
                # such as the function callable
                function_callable = function_settings['callable']
                # Flag for whether to apply function over ufunc or not
                function_apply_ufunc = function_settings.get('apply_ufunc',False)
                # Gather all function keyword arguments
                function_sweeped_kwargs = dict(zip(sweep_keys, function_sweeped_vals))
                function_kwargs = {**fixed_kwargs.get(function_name,{}), **function_sweeped_kwargs}

                # Apply function either through xr.apply_ufunc or directly
                if function_apply_ufunc:
                    function_output = xr.apply_ufunc(
                        function_callable,
                        function_output,
                        kwargs = function_kwargs
                    )
                else:
                    function_output = function_callable(
                        function_output,
                        **function_kwargs
                    )
            except Exception:
                traceback.print_exc()
                raise FunctionFailed(
                    name = function_name,
                    keys = (
                        list(fixed_kwargs.get(function_name,{}).keys()) + \
                        list(isolated_sweeped_kwargs.keys()) + \
                        list(coupled_sweeped_kwargs.keys())
                    )
                )

        # Convert sweep values to iterables
        function_sweeped_kwargs = {
            k:(v if isinstance(v,Iterable) \
                and not isinstance(v,str) \
               else [v]) 
            for k,v in function_sweeped_kwargs.items()
        }
        
        # Expand dimensions
        function_output = function_output.expand_dims(function_sweeped_kwargs)

        # Expand coordinates
        function_output = function_output.assign_coords(function_sweeped_kwargs)

        # Append to function composition outputs list
        function_composition_outputs.append(function_output)

    # Combine all function composition xarrays
    combined_xarray = xr.combine_by_coords(
        function_composition_outputs,
        compat = "no_conflicts"
    )

    # Add sweep dimensions to combined xarray's sweep dimension
    return xr_restack(
        combined_xarray,
        multiindex_dim = existing_dim,
        added_dims = sweep_keys
    )
    
def safe_list_get(l, idx, default):
  try:
    return l[idx]
  except IndexError:
    return default

def unpack_data(data,index):
    if data and isinstance(data,Iterable):
        if len(data) == 1:
            return data[0]
        else:
            return data[index]
    else:
        return data

def flip(items, ncol):
    return chain(*[items[i::ncol] for i in range(ncol)])


def eval_dtype(dtype:str='',numpy_format:bool=False):
    match dtype:
        case "str":
            return str
        case "object":
            return object
        case "list":
            return list
        case "float16":
            return np.float16 if numpy_format else torch.float16
        case "float32":
            return np.float32 if numpy_format else torch.float32
        case "float64":
            return np.float64 if numpy_format else torch.float64
        case "uint8":
            return np.uint8 if numpy_format else torch.uint8
        case "int8":
            return np.int8 if numpy_format else torch.int8
        case "int16":
            return np.int16 if numpy_format else torch.int16
        case "int32":
            return np.int32 if numpy_format else torch.int32
        case "int64":
            return np.int64 if numpy_format else torch.int64
        case "bool":
            return bool if numpy_format else torch.bool
        case _:
            raise Exception(f"Cannot find {'numpy' if numpy_format else 'torch'} type {dtype}.")

def cmap_exists(name):
    try:
        cm.get_cmap(name)
    except:
        return False
    return True

def safe_cast(x,minval:float=np.float32(1e-6),maxval:float=np.float32(1e6)):
    # Convert infinities to value
    try:
        x = x.cpu().detach().numpy()
    except:
        pass
    # Make sure omega is not too large to cause numerical overflow
    x = x if x > minval else minval
    x = x if x < maxval else maxval
    
    return x

def build_graph_from_matrix(weigthed_adjacency_matrix, region_features, device='cpu'):
    '''
    Build graph using DGL library from adjacency matrix.

    Inputs:
    -----------------------------------
    weigthed_adjacency_matrix: graph adjacency matrix of which entries are either 0 or 1.
    node_feats: node features

    Returns:
    -----------------------------------
    g: DGL graph object
    '''
    # get edge nodes' tuples [(src, dst)]
    nonzerocells = weigthed_adjacency_matrix.nonzero()
    # dst, src = nonzerocells.T
    src, dst = nonzerocells.T
    # get edge weights
    # d = weigthed_adjacency_matrix[nonzerocells[:,0],nonzerocells[:,1]]
    d = weigthed_adjacency_matrix[src,dst]
     # create a graph
    g = dgl.DGLGraph()
    # add nodes
    g.add_nodes(weigthed_adjacency_matrix.shape[0])
    # add edges and edge weights
    g.add_edges(src, dst, {'d': torch.tensor(d).float().view(-1, 1)})
    # add node attribute, i.e. the geographical features of census tract
    g.ndata['attr'] = region_features.to(device)
    
    # compute the degree norm
    norm = comp_deg_norm(g)
    # add nodes norm
    g.ndata['norm'] = torch.from_numpy(norm).view(-1,1).to(device) # column vector
    # return
    return g


def comp_deg_norm(g):
    '''
    compute the degree normalization factor which is 1/in_degree
    '''
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

def create_mask(shape,index):
    # Mask all such cells
    mask = np.zeros(shape,dtype=bool)
    for cell in index:
        mask[tuple(cell)] = True
    return mask

def populate_array(shape,index,res):
    # Create an empty output array
    arr = np.zeros(shape,dtype='float32')
    
    # Flatten the indices and values arrays
    flat_indices = np.ravel_multi_index(index, shape)
    
    # Use np.unravel_index to convert flattened indices back to multi-dimensional indices
    multi_indices = np.unravel_index(flat_indices, shape)
    
    # Assign values to the output array using tuple indexing
    arr[multi_indices] = res.flatten()

    return arr

# Obtained from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap = newcmap)

    return newcmap