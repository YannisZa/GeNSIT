import os
import re
import sys
import zlib
import gzip
import json
import h5py
import torch
import random
import numexpr
import logging
import operator
import tikzplotlib
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt


from pathlib import Path
from itertools import chain, count
from difflib import SequenceMatcher
from typing import Dict, List, Union, Tuple
from collections.abc import Iterable,MutableMapping,Mapping,Sequence


from multiresticodm.exceptions import *
from multiresticodm.logger_class import *
from multiresticodm.global_variables import NUMPY_TYPE_TO_DAT_TYPE,OPERATORS

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
    np.save(file=filepath, arr=data)


def write_xr_data(data:xr.DataArray,filepath:str,**kwargs:Dict) -> None:
    try: 
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
            # Group id of interest
            group_id = kwargs.pop('group','')
            # Find all existing data groups
            # with nc.Dataset(filepath, mode='r') as nc_file:
            #     # Get groups
            #     data_group_ids = read_xr_group_ids(nc_file,list_format=True)
            #     # Read existing data
            #     if group_id in data_group_ids:
            #         del nc_file[group_id]
            #     else:
            #         nc_file.createGroup(group_id)
            # existing_data = read_xr_data(filepath=filepath)
                
            data.to_netcdf(
                path = filepath,
                group = group_id,
                mode = 'a',
                **kwargs
            )
    except Exception as exc:
        raise H5DataWritingFailed(message=str(exc))




def write_compressed_npy(data:np.ndarray,filepath:str,**kwargs:Dict) -> None:
    # Write array to npy format
    with gzip.GzipFile(filepath, "wb", compresslevel=6) as f:
        np.save(f, data)


def write_figure(figure,filepath,**settings):
    
    if settings.get('filename_ending','') == '':
        filepath += '.'+settings['figure_format']
    else:
        filepath += '_'+str(settings.get('filename_ending',''))+'.'+settings['figure_format']
        
    if settings['figure_format'] == 'tex':
        # tikzplotlib.clean_figure(fig=figure)
        tikzplotlib.save(filepath,figure=figure)
    else:
        figure.savefig(
            filepath,
            format=settings['figure_format'],
            bbox_inches='tight'
        )
    
    plt.close(figure)


def write_figure_data(plot_settings:Union[dict,pd.DataFrame],filepath:str,key_type:dict={},aux_keys:list=[],**settings):
    
    for plot_sett in plot_settings:
        # Keys must be included in figure data
        assert set(key_type.keys()).issubset(set(list(plot_sett.keys())))

        print_json({k:np.shape(plot_sett.get(k)) for k in (list(key_type.keys())+aux_keys) if k != 'outputs'},newline=True)
        # print('\n\n')
        # print_json({k:(
        #     plot_sett.get(k,np.array([])).tolist() 
        #     if isinstance(plot_sett.get(k),np.ndarray)
        #     else plot_sett.get(k,None)
        # ) for k in (list(key_type.keys())+aux_keys) if k != 'outputs'})
        
        if settings.get('data_format','dat') == 'dat':
            # Write dat file
            write_tex_data(
                key_type=key_type,
                data=list(zip(*[np.asarray(plot_sett[k],dtype=key_type[k]) for k in key_type.keys()])),
                filepath=filepath,
                precision=settings.get('data_precision',19)
            )
        elif settings.get('data_format','dat') == 'json':
            write_json(
                {k: plot_sett.get(k,np.array([])).tolist() 
                    if isinstance(plot_sett.get(k),np.ndarray)
                    else plot_sett.get(k,None)
                    if k != 'outputs'
                    else plot_sett.get(k,None).config.settings
                for k in list(key_type.keys())+aux_keys},
                filepath
            )
        elif settings.get('data_format','dat') == 'csv':
            write_csv(
                pd.DataFrame.from_dict(
                    {k:(
                        plot_sett.get(k,np.array([])).tolist() 
                        if isinstance(plot_sett.get(k),np.ndarray)
                        else plot_sett.get(k,None)
                    )
                    for k in list(key_type.keys())+aux_keys}
                ),
                filepath
            )
        # Write plot settings to file
        write_json(
            settings,
            filepath+'_settings.json'
        )

def read_file(filepath:str,**kwargs) -> np.ndarray:
    if filepath.endswith('.npy'):
        return read_npy(filepath=filepath,**kwargs)
    elif filepath.endswith('.txt'):
        return np.loadtxt(fname=filepath,**kwargs)
    elif filepath.endswith('.json'):
        return read_json(filepath=filepath)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath,**kwargs)
    else:
        raise Exception(f'Cannot read file ending with {filepath}')


def read_npy(filepath:str,**kwargs:Dict) -> np.ndarray:
    # Write array to npy format
    data = np.load(filepath).astype('float32')
    return data

def read_xr_data(filepath:str,**kwargs:Dict) -> xr.DataArray:
    if len(kwargs.get('group','')) > 0:
        return xr.open_dataarray(
            filepath,
            group = kwargs['group']
        )
    else:
        return xr.open_dataset(
            filepath,
            engine = 'h5netcdf'
        )

def read_netcdf_group_ids(nc_data:str,key_path=[]):
    for k in nc_data.groups.keys():
        yield key_path+[k]
    for key,value in nc_data.groups.items():
        yield from read_netcdf_group_ids(value,key_path+[key])

def read_xr_group_ids(nc_data,list_format:bool=True):
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


def write_json(data:Dict,filepath:str,**kwargs:Dict) -> None:
    if not filepath.endswith('.json'):
        filepath += '.json'
    with open(filepath, 'w') as f:
        json.dump(data,f,**kwargs)


def print_json(data:Dict,**kwargs:Dict):
    if kwargs.get('newline',False):
        for k in data.keys():
            print(f"{k}: {data[k]}",sep='')
    else:
        print(json.dumps(data,cls=NumpyEncoder,**kwargs))

def write_compressed_string(data:str,filepath:str) -> None:
    with gzip.GzipFile(filename=filepath, mode="w") as f:
        f.write(zlib.compress(data.encode()))


def read_compressed_string(filepath:str) -> None:
    with gzip.GzipFile(filename=filepath, mode="r") as f:
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

def parse(value,default=None):
    if value is None:
        return 'none'
    elif isinstance(value,str):
        if len(value) <= 0:
            return default
        try:
            value = string_to_numeric(value)
        except:
            pass
    elif hasattr(value,'__len__'):
        return np.array2string(np.array(value))
    elif isinstance(value,float):
        return np.float32(np.round(value,5))
    
    return value

def makedir(directory:str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def tuple_to_str(t:Tuple[int,int]) -> str:
    return str(t).replace(' ','')


def str_to_tuple(s:str) -> Tuple[int,int]:
    return (int(s.split('(')[1].split(',')[0]),int(s.split(',')[1].split(')')[0]))


def ndims(__self__,time_dims:bool=True):
    return np.sum([1 for dim in unpack_dims(__self__.data.dims,time_dims=time_dims) if dim > 1],dtype='uint8')

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
    elif hasattr(expressions,'__len__'):
        value = []
        for i,expr in enumerate(expressions):
            value.append(
                deep_call(
                    input=input,
                    expressions=expr,
                    defaults=defaults[i],
                    kwargs=kwargs
                )
            )
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
            matching_substring=string1[match.a:match.a+match.size]
            if(matching_substring not in substring_counts):
                substring_counts[matching_substring]=1
            else:
                substring_counts[matching_substring]+=1

    return substring_counts

def get_all_subdirectories(out_path,stop_at:str='config.json',level:int=2):
    directories = []
    for root, dirs, files in walklevel(out_path,level=level):
        # If this is a dir that matches the stopping condition
        # or has a file that matches the stopping condition
        if any([stop_at in file for file in files]):
            # entry_path = Path(root)
            # directories.append(str(entry_path.absolute()))
            directories.append(root)

    return directories

def walklevel(some_dir, level=1):
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
    return max(substring_counts.items(), key=operator.itemgetter(1))[0]


# https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def update_recursively(d:Dict, u:Dict, overwrite:bool=False) -> Dict:
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
            items.extend(deep_flatten(v, new_key, sep=sep).items())
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
    for k, v in ob.items():
        if isinstance(v, Mapping):
            deep_apply(v, func, **kwargs)
        else:
            ob[k] = func(v,**kwargs)
    return ob

def pop_variable(_self,var):
    if hasattr(_self,var):
        res = getattr(_self,var)
        delattr(_self,var)
        return res
    else:
        return None

def extract_config_parameters(conf,fields=dict):
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

def safe_delete(variable,instance=None):
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

def stringify(data):
    if isinstance(data,Iterable) and not isinstance(data,str) and len(data) > 0:
        return f"({','.join([stringify(v) for v in data])})"
    elif data == '' or data is None:
        return "none"
    elif hasattr(data,'__len__') and len(data) == 0:
        return "[]"
    else:
        return f"{str(data).replace(' ','')}"

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
    torch.cuda.set_device(device)


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
        def sum_normalise(x,axis=None):
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
        def evaluate_expression(data,axis=None):
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


def f_to_array(f:Dict,shape:tuple=None)-> Union[np.ndarray,np.generic]:

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
    # return np.array([v for k,v in sorted(f.items(),key=lambda x:str_to_tuple(x[0]))])


def array_to_f(a:Union[np.ndarray,np.generic],axis:int=0) -> Dict:
    if len(np.shape(a)) == 1 and axis == 0:
        return dict(((0,index),int(value)) for index, value in enumerate(a))
    elif len(np.shape(a)) == 1 and axis == 1:
        return dict(((index,0),int(value)) for index, value in enumerate(a))
    else:
        return dict((index,int(value)) for index,value in np.ndenumerate(a))


def f_to_str(f:dict) -> str:
    return ','.join([str(v) for k,v in sorted(f.items(),key=lambda x:str_to_tuple(x[0]))])


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
        return np.reshape(np.asarray(s.split(','),dtype = int),newshape=dims)
    

def create_dynamic_data_label(__self__,data,**kwargs):
    # Read label(s) from settings
    label_by_key,label_by_value = [],[]
    for k in list(__self__.settings['label_by']):
        v = list(deep_get(key=k,value=data))[0]
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

def in_range(v,limits:list,allow_nan:bool=False,inclusive:bool=False):
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
        return v is None or np.isnan(v)

def dict_inverse(d:dict):
    return {v:k for k,v in d.items()}

def get_value(d:dict,k:str,default:object = None):
    try:
        value = d[k].item()
    except:
        try:
            value = d[k]
        except:
            value = default
    if k == 'sigma':
        return sigma_to_noise_regime(value)
    else:
        return value

def hash_vars(d:dict,v:list):
    if len(v) == 1:
        return d.get(v[0],None) 
    else:
        key = '('
        for k in v:
            if isinstance(k,str):
                for subk in k.strip('()').split(", "):
                    key += (subk+', ')
            else:
                key += (str(k)+', ')
        # Remove last comma and space
        key = key[:-2]
        # Close parenthesis
        key += ')'

        return d.get(key,None)

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

def setup_logger(
        name,
        console_level:str=None,
        file_level:str=None,
    ):
    print('setting up new logger',name)

    # Silence warnings from other packages
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)
    
    # Get logger
    logger = DualLogger(
        name = name,
        level = console_level
    )

    logger.setLevels(
        console_level = console_level,
        file_level = file_level
    )

    return logger

def sigma_to_noise_regime(sigma=None):
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

def broadcast(arr,shape):
    # Squeeze array
    arr = np.squeeze(arr)
    # Figure out repetitions
    axes = [i for i in range(len(shape)) if shape[i] not in list(arr.shape)]
    # Expand dims
    arr_reshaped = np.expand_dims(arr,axis=axes)
    for ax in axes:
        # Repeat along axis
        arr_reshaped = np.repeat(arr_reshaped,shape[ax],axis=ax)

    return arr_reshaped

def expand_tuple(t):
    result = []
    for item in t:
        if isinstance(item, tuple):
            result.extend(expand_tuple(item))
        else:
            result.append(item)
    return tuple(result)

def unpack_dims(self,time_dims:bool=True):
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
    return x

def tuple_dim(x,dims=()):
    if isinstance(x,tuple):
        dims = (*dims,len(x))
        return tuple_dim(x[0],dims=dims)
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