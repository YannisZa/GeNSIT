import os
import re
import gc
import ast
import zlib
import gzip
import json
import numexpr
import operator
import tikzplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from copy import deepcopy
from itertools import chain, count
from difflib import SequenceMatcher
from numba import njit, set_num_threads
from typing import Dict, List, Union, Tuple
from collections.abc import Iterable,MutableMapping,Mapping,Sequence


from ticodm.global_variables import NUMPY_TYPE_TO_DAT_TYPE#,UTILS_CACHED

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

def str_in_list(text:str,l:Iterable) -> bool:
    # Exact match of string in any of list elements
    return any([text == s for s in list(l)])


def write_csv(data:pd.DataFrame,filepath:str,**kwargs:Dict) -> None:
    # Write pandas to csv
    if not filepath.endswith('.csv'):
        filepath += '.csv'
    
    data.to_csv(filepath,**kwargs)


def write_npy(data:np.ndarray,filepath:str,**kwargs:Dict) -> None:
    # Write array to npy format
    np.save(file=filepath, arr=data)


def write_compressed_npy(data:np.ndarray,filepath:str,**kwargs:Dict) -> None:
    # Write array to npy format
    with gzip.GzipFile(filepath, "wb", compresslevel=6) as f:
        np.save(f, data)


def write_figure(figure,filepath,**settings):
    
    filepath += f"{settings.get('filename_ending','') if len(settings.get('filename_ending','')) == 0 else '_'+settings.get('filename_ending','')}.{settings['figure_format']}"
        
    if settings['figure_format'] == 'tex':
        # tikzplotlib.clean_figure(fig=figure)
        tikzplotlib.save(filepath,figure=figure)
    else:
        figure.savefig(filepath,format=settings['figure_format'])
    
    plt.close(figure)


def write_figure_data(figure_data:Union[dict,pd.DataFrame],dirpath:str,groupby:list=[],key_type:dict={},**settings):
    if isinstance(figure_data,dict):
        for fgid in figure_data.keys():


            # Keys must be included in figure data
            assert set(key_type.keys()).issubset(set(list(figure_data[fgid].keys())))

            # Get filename
            filename = figure_data[fgid]['label'].replace('%','percent').replace(' ','_').replace(':','_').replace(',','_')
            # Add file extension
            filename += '.'+settings.get('data_format','dat')
            # Create filepath
            filepath = os.path.join(dirpath,filename)

            # Write dat file
            write_tex_data(
                key_type=key_type,
                data=list(zip(*[np.asarray(figure_data[fgid][k],dtype=key_type[k]) for k in sorted(key_type.keys())])),
                filepath=filepath
            )
    elif isinstance(figure_data,pd.DataFrame):
        for group_id,group in figure_data.groupby(groupby):
            # Get filename
            filename = group_id.replace('%','percent')
            # Add file extension
            filename += '.'+settings.get('data_format','dat')
            # Create filepath
            filepath = os.path.join(dirpath,filename)

            # Write dat file
            write_tex_data(
                key_type=key_type,
                data=group.loc[:,key_type.keys()].values,
                filepath=filepath
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


def read_compressed_npy(filepath:str,**kwargs:Dict) -> np.ndarray:
    # Write array to npy format
    with gzip.GzipFile(filepath, "rb") as f: 
        data = np.load(f)
    return data


def write_tex_data(data:Union[np.array,np.ndarray],key_type:Union[list,np.array,np.ndarray],filepath:str,**kwargs:Dict) -> None:
    with open(filepath, 'w') as f:
        np.savetxt(
            f, 
            data,
            header=' '.join(sorted(key_type.keys())),
            comments='',
            fmt=[NUMPY_TYPE_TO_DAT_TYPE[key_type[x]] for x in sorted(key_type.keys())]
        )


def write_txt(data:Union[np.array,np.ndarray],filepath:str,**kwargs:Dict) -> None:
    # Write numpy to txt
    if not filepath.endswith('.txt'):
        filepath += '.txt'
    np.savetxt(filepath,data,**kwargs)


def write_json(data:Dict,filepath:str,**kwargs:Dict) -> None:
    with open(filepath, 'w') as f:
        json.dump(data,f,**kwargs)


def print_json(data:Dict,**kwargs:Dict):
    print(json.dumps(data,cls=NumpyEncoder,**kwargs))


def json_print_newline(data:Dict):
    for k in data.keys():
        print(f"{k}: {data[k]}",sep='')


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


def makedir(directory:str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def tuple_to_str(t:Tuple[int,int]) -> str:
    return str(t).replace(' ','')


def str_to_tuple(s:str) -> Tuple[int,int]:
    return (int(s.split('(')[1].split(',')[0]),int(s.split(',')[1].split(')')[0]))


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

def get_directories_and_immediate_subdirectories(path):
    directories = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir():
                directories.append(entry.name)
                with os.scandir(entry) as child_entries:
                    for child_entry in child_entries:
                        if child_entry.is_dir():
                            directories.append(child_entry.path)
    return directories

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


def deep_update(d,key,val,overwrite:bool=True):
    for dict_key,dict_val in d.items():
        if (dict_key == key) and overwrite:
            d[key] = val
        elif isinstance(dict_val, dict):
            deep_update(dict_val,key,val,overwrite)
        

def deep_updates(main_dict,update_dict,overwrite:bool=True):
    for k,v in update_dict.items():
        deep_update(main_dict,k,v,overwrite)
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

def extract_config_parameters(conf,fields=dict):
    trimmed_conf = {}
    if not isinstance(conf,dict):
        return conf
    if isinstance(fields,str):
        trimmed_conf[fields] = conf[fields]
        return trimmed_conf[fields]

    for f,v in fields.items():
        if v == "":
            if str_in_list(f,conf):
                trimmed_conf[f] = conf[f]
        else:
            if str_in_list(f,conf):
                trimmed_conf[f] = extract_config_parameters(conf[f],v)
    return trimmed_conf

def safe_delete(variable):
    try:
        del variable
    except:
        None
    
def safe_delete_and_clean(variable):
    print('Safe delete and clean')
    if not hasattr(variable,'__len__'):
        variable = [variable]
    for var in variable:
        try:
            del variable
        except:
            None
    gc.collect()


def unpack_statistics(settings):
    # Unpack statistics if they exist
    if str_in_list('statistic',settings) and len(settings['statistic']) > 0:
        statistics = []
        for stat,ax in settings['statistic']:
            # print('stat',stat)
            # print('ax',ax)
            if isinstance(stat,str):
                if '|' in stat:
                    stat_unpacked = stat.split('|')
                else:
                    stat_unpacked = [stat]
            elif hasattr(stat,'__len__'):
                stat_unpacked = deepcopy(stat)
            else:
                raise Exception(f'Statistic name {stat} of type {type(stat)} not recognized')
            
            if isinstance(ax,str): 
                if '|' in ax:
                    ax_unpacked = [_ax.split("_") if len(_ax) > 0 else '' for _ax in ax.split('|')]
                else:
                    ax_unpacked = [ax.split("_")]
            elif hasattr(ax,'__len__'):
                ax_unpacked = deepcopy(ax)
            else:
                raise Exception(f'Statistic axes {ax_unpacked} of type {type(ax_unpacked)} not recognized')
            # print('stat_unpacked',stat_unpacked)
            # print('ax_unpacked',ax_unpacked)
            ax_unpacked = [tuple([int(_subax) for _subax in _ax]) \
                            if (_ax is not None and len(_ax) > 0) \
                            else None
                            for _ax in ax_unpacked]
            # Add statistic name and axes pair to list
            statistics.append(list(zip(stat_unpacked,ax_unpacked)))
        return {'statistic': statistics}


def stringify_statistic(statistic):
    # Unpack statistic pair
    statistic_name,statistic_axes = statistic
    # Initialise text
    text = ''
    if statistic_name != '':
        text += statistic_name
    if statistic_axes is not None:
        text += '_'+str(statistic_axes)
    return text


@njit
def numba_set_seed(value):
    if value is not None:
        np.random.seed(value)


def update_numba_threads(n_threads):
    # Update threads used by numba 
    if len(n_threads) == 1:
        set_num_threads(int(n_threads[0]))
    elif len(n_threads) > 1:
        set_num_threads(int(n_threads[1]))
    else:
        raise ValueError(f"Invalid number of threads '{str(n_threads)}' provided ")

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


def convert_string_to_numpy_function(s:str=''):
    if 'mean' == s.lower():
        return np.mean
    elif 'var' == s.lower():
        return np.var
    elif 'std' == s.lower():
        return np.std
    elif 'sum' == s.lower():
        return np.sum
    elif 'max' == s.lower():
        return np.max
    elif 'min' == s.lower():
        return np.min
    elif 'median' == s.lower():
        return np.median
    elif 'sumnorm' == s.lower():
        def sum_normalise(x,axis=None):
            if axis != (None,):
                return x/np.sum(x,axis)
            else:
                return x/np.sum(x)
        return sum_normalise
    elif 'X' in s:
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
        raise Exception(f"Function name {s} not match to numpy function")


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


@njit
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
    

def create_dynamic_data_label(__self__,data):
    # Read label(s) from settings
    label_by_key,label_by_value = [],[]
    for k in list(__self__.settings['label_by']):
        v = list(deep_get(key=k,value=data))
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