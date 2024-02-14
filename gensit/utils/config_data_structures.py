import os
import sys
import traceback

from typing import Any
from collections.abc import Iterable
from numpy import arange, isfinite, isnan, shape, array, repeat

from gensit import ROOT
from gensit.utils.exceptions import *
from gensit.utils.misc_utils import in_range,string_to_numeric,is_null,flatten

def instantiate_data_type(
    data,
    schema:dict,
    key_path:list,
):  
    try:
        data_type = schema["dtype"]
    except:
        print('faulty schema')
        print(schema)
        raise Exception('>'.join(list(map(str,key_path))))
    try:
        data_type = data_type[0].capitalize() + data_type[1:]
    except:
        print('schema',schema)
        print('data_type',data_type)
        raise Exception('>'.join(list(map(str,key_path))))
    if len(data_type) > 0 and hasattr(sys.modules[__name__], data_type):
        return getattr(sys.modules[__name__], data_type)(
            data = data,
            schema = schema,
            key_path = key_path
        )
    else:
        raise ValueError(f"Data type '{data_type}' not found for {'>'.join(list(map(str,key_path)))}")


class Entry(object):
    def __init__(self,data,schema,key_path):
        self.data = data
        self.schema = schema
        self.key_path = key_path

        self.dtype = object

    def __str__(self) -> str:
        return "Entry()"
    
    def value(self) -> Any:
        return None if is_null(self.data) else self.data
    
    def check_type(self,data_type = None):
        # print("Check type")
        data_type = data_type if data_type is not None else self.schema.get("dtype",data_type)
        allow_nan = self.schema.get("allow-nan",False)
        if data_type is not None:
            try: 
                assert isinstance(self.data,self.dtype) or allow_nan
            except:
                raise InvalidTypeException(
                    message = f"Expected type {self.dtype}. Got {type(self.data)}",
                    key_path = self.key_path,
                    data = self.data
                )
        else:
            raise TypeNotFoundException(
                key_path = self.key_path
            )
        return True

    def check(self):
        self.check_type()
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#----------------------- Primitive Data Entry -----------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

class PrimitiveEntry(Entry):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)

    def __str__(self) -> str:
        return "PrimitiveEntry()"
    
    def check(self):
        pass
    
    def check_range(self):
        # print("Check range")
        # Check that correct range is used for each element
        valid_range = self.schema.get("valid-range","invalid")
        if valid_range != "invalid":
            # Get whether bounds are tight or not
            inclusive = self.schema.get("inclusive",True)
            allow_nan = self.schema.get("allow-nan",False)
            try: 
                assert in_range(self.data,valid_range,allow_nan = allow_nan,inclusive = inclusive)
            except:
                raise InvalidRangeException(
                    f"Data {self.data} is out of range {valid_range}",
                    key_path = self.key_path,
                    data = self.data
                )
        return True
    
    def check_scope(self):
        # print("Check scope")
        valid_scope = self.schema.get("is-any-of",[])
        if isinstance(valid_scope,list) and len(valid_scope) > 0:
            try: 
                assert self.data in valid_scope
            except:
                raise InvalidScopeException(
                    message = f"Data {self.data} is not in scope {valid_scope}",
                    key_path = self.key_path,
                    data = self.data
                )
        substrings = self.schema.get("contains",[])
        if isinstance(substrings,list) and len(substrings) > 0:
            try: 
                assert any([str(substr) in self.data for substr in substrings])
            except:
                raise InvalidScopeException(
                    f"None of the substrings {', '.join(substrings)} are contained in data {self.data}",
                    key_path = self.key_path,
                    data = self.data
                )
        return True

class Bool(PrimitiveEntry):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        self.dtype = bool
    
    def __str__(self) -> str:
        return "Boolean()"
    
    def check_boolean(self):
        # print("Check boolean")
        # Check that the data is either true or false
        try:
            assert str(self.data) in ['True','False']
        except:
            raise InvalidBooleanException(
                f"Boolean {str(self.data)} is not equal to 'True' or 'False'",
                key_path = self.key_path,
                data = self.data
            )

class Numeric(PrimitiveEntry):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        
    def check_numeric(self):
        # print("Check numeric")
        try:
            assert self.data.isnumeric()
        except:
            raise StringNotNumericException(
                f"String {self.data} is not numeric",
                key_path = self.key_path,
                data = self.data
            )
        return True

    def check_finiteness(self):
        # print("Check finiteness")
        # Get whether data is allowed to be infinite or not
        infinite = self.schema.get("is-infinite",True)
        if not infinite:
            try:
                assert isfinite(self.data)
            except:
                raise InfiniteNumericException(
                    f"Infinite data {self.data} provided",
                    key_path = self.key_path,
                    data = self.data
                )

    def check(self):
        # Check that correct type is used
        self.check_type()
        # Check that data is finite (if that requirement is specified)
        self.check_finiteness()
        # Check that correct range is used 
        self.check_range()
        # Check that input is in the scope of allowable inputs
        self.check_scope()

    def __str__(self) -> str:
        return "Numeric()"
    
class Int(Numeric):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        self.dtype = int
    
    def __str__(self) -> str:
        return "Integer()"
    
class Float(Numeric):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        self.dtype = float

    def __str__(self) -> str:
        return "Float()"
    
class Str(PrimitiveEntry):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        self.dtype = str
    
    def __str__(self) -> str:
        return "String()"

    def check(self):
        # Check that correct type is used
        self.check_type()
        # Check that input is in the scope of allowable inputs
        self.check_scope()

class StrInt(Numeric):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        self.dtype = str
    
    def __str__(self) -> str:
        return "StringInteger()"

    def check(self):
        # Check that correct type is used
        self.check_type()
        # Check that string is numeric
        self.data = int(self.data)
        # Check that correct range is used 
        self.check_range()
        # Check that input is in the scope of allowable inputs
        self.check_scope()

class Path(Str):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        self.dtype = str
    
    def __str__(self) -> str:
        return "Path(String)"

    def check_path_exists(self):
        # print("Check path exists")
        if self.schema.get("path-exists",False) or \
            self.schema.get("file-exists",False) or \
            self.schema.get("directory-exists",False) or \
            self.schema.get("exists",False):
            try:
                assert os.path.exists(os.path.join(ROOT,self.data))
            except:
                raise PathNotExistException(
                    f"Path to file {self.data} does not exist",
                    key_path = self.key_path,
                    data = self.data
                )
        return True

    def check_directory_exists(self):
        # print("Check directory exists")
        self.check_path_exists() 
        try:
            os.path.isdir(self.data)
        except:
            raise DirectoryNotFoundException(
                f"Path to file {self.data} is not a directory",
                key_path = self.key_path,
                data = self.data
            )
        return True

    def check_file_exists(self):
        # print("Check file exists")
        self.check_path_exists()
        try:
            os.path.isfile(self.data)
        except:
            raise FileNotFoundException(
                f"Path to file {self.data} is not a file",
                key_path = self.key_path,
                data = self.data
            )
        return True

    def check_extension(self):
        # print("Check extension")
        extension = self.schema.get("extension",None)
        if extension is not None:
            extension = self.schema.get("file-extension",None)
        if extension is not None:
            try:
                assert self.data.endswith(("."+extension))
            except:
                raise InvalidExtensionException(
                    f"File {self.data} does not have extension {extension}.",
                    key_path = self.key_path,
                    data = self.data
                )
        return True

    def value(self):
        return self.schema['default-value'] if len(self.data) <=0 else self.data


class File(Path):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        self.dtype = str

    def __str__(self) -> str:
        return "File(Path)"

    def check(self):
        # Check that correct type is used
        self.check_type()
        # Check that path exists
        self.check_file_exists()
        # Check that file extension exists
        self.check_extension()


class Directory(Path):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        self.dtype = str

    def __str__(self) -> str:
        return "Directory(Path)"

    def check(self):
        # Check that correct type is used
        self.check_type()
        # Check that path exists
        self.check_directory_exists()

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------- Non-Primitive Data Entry ---------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

class NonPrimitiveEntry(Entry):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        self.names = ['dict','list','list2D','customList','customDict']

    def __str__(self) -> str:
        return "NonPrimitiveEntry()"

    def __iter__(self):
        for datum in self.data:
            yield datum

    def check_length(self):
        # print("Check length")
        # Check that list has specified length
        length = self.schema.get("length",None)
        if length is not None:
            if isinstance(length,Iterable):
                try: 
                    assert in_range(len(self.data),length,allow_nan = False,inclusive = True)
                except:
                    raise InvalidLengthException(
                        f"Length {len(self.data)} is not in range {length}.",
                        key_path = self.key_path,
                        data = self.data
                    )
            else:
                try: 
                    assert len(self.data) == length
                except:
                    raise InvalidLengthException(
                        f"Length {len(self.data)} is not equal to {length}.",
                        key_path = self.key_path,
                        data = self.data
                    )
        return True

    def check_uniqueness(self,vals:Iterable):
        # print("Check uniqueness")
        # Check that unique values are provided
        unique_vals = self.schema.get("unique-vals",False)
        if unique_vals:
            # Express data in python primitive types
            data = [v.value() if isinstance(v.value(),PrimitiveEntry) else str(v.value()) for v in vals]
            try: 
                assert len(data) == len(set(data))
            except:
                raise DataUniquenessException(
                    f"Data {data} is not unique",
                    key_path = self.key_path,
                    data = data
                )
        return True

    def check_elements(self,vals:Iterable):
        # print("Check elements")
        # Check that all values are valid
        if isinstance(vals,NonPrimitiveEntry) or isinstance(vals,Iterable):
            for elem in vals:
                if isinstance(elem,NonPrimitiveEntry) or isinstance(elem,Iterable):
                    self.check_elements(elem)
                else:
                    elem.check()
        
        return True
    
    def check(self):
        pass


class List(NonPrimitiveEntry):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        self.dtype = list
        
        # First check that input is indeed a list
        self.check_type(data_type = self.dtype)

        # Then prep schema for list values
        # Remove all key prefixes that start with 'list'
        # Remove 'dtype' key
        self.schema = {} 
        for k,v in schema.items():
            if k in ["dtype","sweepable","target_name","optional"]:
                continue
            # Remove all non-primitive entry names from preffix
            stored = False
            for name in self.names:
                if k.startswith((name+'-')):
                    self.schema[k.replace((name+'-'),'',1)] = v
                    stored = True
                    continue
            # If not done so already, store schema setting
            if not stored:
                self.schema[k] = v
        # Get all value entries
        self.data = [instantiate_data_type(val,self.schema,key_path) for val in data]

    def value(self) -> list:
        return [self.schema['default-value'] if is_null(datum.value()) else datum.value() for datum in self.data]
    
    def check(self):
        # Check that correct length is provided
        self.check_length()
        # Check that all values are unique
        self.check_uniqueness(vals = self.data)
        # Check that each element of the list is valid 
        # by calling their respective check functions
        self.check_elements(vals = self.data)


class List2D(NonPrimitiveEntry):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        self.dtype = list

        # First check that input is indeed a list
        self.check_type(data_type = self.dtype)

        # Then prep schema for list values
        # Remove all key prefixes that start with 'list'
        # Remove 'dtype' key
        self.schema = {}
        for k,v in schema.items():
            if k in ['dtype','sweepable','target_name','optional']:
                continue
            # Remove all non-primitive entry names from preffix
            stored = False
            for name in self.names:
                if k.startswith((name+'-')):
                    self.schema[k.replace((name+'-'),'',1)] = v
                    stored = True
                    continue
            # If not done so already, store schema setting
            if not stored:
                self.schema[k] = v
        
        # Get all value entries
        self.data = [ [instantiate_data_type(val,self.schema,key_path) for val in row] for row in data]
    
    def __iter__(self) -> list:
        for row in self.data:
            for col in row:
                yield col

    def value(self) -> list:
        return [[self.schema['default-value'] if is_null(datum.value()) else datum.value() for datum in row] for row in self.data]
            
    def check_dims(self):
        # print("Check dims")
        # Check that list has specified length
        nrows,ncols = self.schema.get("nrows",None),self.schema.get("ncols",None)
        if ncols is not None and nrows is not None:
            if isinstance(ncols,Iterable) or isinstance(nrows,Iterable):
                if isinstance(nrows,Iterable):
                    try: 
                        assert in_range(shape(self.data)[0],nrows,allow_nan = False,inclusive = True)
                    except:
                        raise InvalidLengthException(
                            f"#rows {shape(self.data)[0]} is not in range {nrows}.",
                            key_path = self.key_path,
                            data = self.data
                        )
                if isinstance(ncols,Iterable):
                    try: 
                        assert in_range(shape(self.data)[1],ncols,allow_nan = False,inclusive = True)
                    except:
                        raise InvalidLengthException(
                            f"#cols {shape(self.data)[1]} is not in range {ncols}.",
                            key_path = self.key_path,
                            data = self.data
                        )
            else:
                try: 
                    assert shape(self.data) == (nrows,ncols)
                except:
                    raise InvalidLengthException(
                        f"Shape {shape(self.data)} is not equal to {(nrows,ncols)}.",
                        key_path = self.key_path,
                        data = self.data
                    )
        return True

    def check(self):
        # Check that correct length is provided
        self.check_dims()
        # Check that each element of the list is valid 
        # by calling their respective check functions
        self.check_elements(vals = self.data)


class CustomList(NonPrimitiveEntry):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        self.dtype = list
        # Parse data
        try:
            self.data, self.parsing_success = self.parse(self.data,self.schema)
        except:
            print(self.data)
            print(self.schema)
            print(traceback.format_exc())
            sys.exit()
        
        # First check that input is indeed a list
        self.check_type(data_type = self.dtype)

        # Then prep schema for list values
        # Remove all key prefixes that start with 'list'
        # Remove 'dtype' key
        self.schema = {} 
        for k,v in schema.items():
            if k in ['dtype','sweepable','target_name','optional']:
                continue
            # Remove all non-primitive entry names from preffix
            stored = False
            for name in self.names:
                if k.startswith((name+'-')):
                    self.schema[k.replace((name+'-'),'',1)] = v
                    stored = True
                    continue
            # If not done so already, store schema setting
            if not stored:
                self.schema[k] = v
        
        # Get all value entries
        self.data = [instantiate_data_type(val,self.schema,key_path) for val in self.data]
    
    def value(self) -> list:
        return [self.schema['default-value'] if is_null(datum.value()) else datum.value() for datum in self.data]


    def parse(self,data,schema):
        if isinstance(data,str):
            return self._parse(data,schema)
        elif isinstance(data,Iterable):
            results = []
            successes = []
            for datum in data:
                res = self._parse(datum,schema)
                results.append(res[0])
                successes.append(res[1])
            return list(flatten(results)), all(successes)
        else:
            return data,True
        
    def _parse(self,data,schema):
        if isinstance(data,str):
            if ":" in data:
                # Split by ':'
                d = data.split(":")
                # Count number of occurences of ':'
                column_count = data.count(":")
                # Get whether bounds are inclusive
                inclusive = schema.get('inclusive',True)
                if column_count > 0:
                    d = [string_to_numeric(elem) for elem in d]
                if column_count == 1:
                    if inclusive:
                        return arange(d[0],d[1]+1).tolist(),True
                    else:
                        return arange(d[0],d[1]).tolist(),True
                elif column_count == 2:
                    if inclusive:
                        return arange(d[0],d[1]+1,d[2]).tolist(),True
                    else:
                        return arange(d[0],d[1],d[2]).tolist(),True
                else:
                    return [],False
            elif "repeat" in data:
                # Clean string
                repetition_str = data.split('(')[-1].split(')')[0]
                # Elicit repetition arguments
                number,reps = repetition_str.split(',')
                # Convert to numeric
                number = string_to_numeric(number)
                reps = string_to_numeric(reps)
                return repeat(number,reps).tolist(),True
        else:
            return data,True

    def check_parsing(self):
        # print("Check parsing")
        try:
            assert self.parsing_success
        except:
            raise CustomListParsingException(
                f"Data {self.data} contain > 2 or < 0 ':' strings.",
                key_path = self.key_path,
                data = self.data
            )
        return True

    def check(self):
        # Check that parsing was completed successfully
        self.check_parsing()
        # Check that correct length is provided
        self.check_length()
        # Check that each element of the list is valid 
        # by calling their respective check functions
        self.check_elements(vals = self.data)

class Dict(NonPrimitiveEntry):
    def __init__(self,data,schema,key_path):
        super().__init__(data,schema,key_path)
        self.dtype = dict

        # First check that input is indeed a dict
        self.check_type(data_type = self.dtype)

        # Then prep schema for list values
        # Remove all key prefixes that start with 'dict'
        # Remove 'dtype' key
        self.schema = {} 
        for k,v in schema.items():
            if k in ['dtype','sweepable','target_name','optional']:
                continue
            # Remove all non-primitive entry names from preffix
            stored = False
            for name in self.names:
                # Find first instance of non-primitive entry
                # and remove non-primitive preffix. 
                if k.startswith((name+'-')):
                    self.schema[k.replace((name+'-'),'',1)] = v
                    stored = True
                    continue
            
            # If not done so already, store schema setting
            if not stored:
                self.schema[k] = v
        
        # Get key schema
        key_schema = {k.replace('key-','',1):v for k,v in self.schema.items() if k.startswith('key-')}
        # Get value schema
        value_schema = {k.replace('value-','',1):v for k,v in self.schema.items() if k.startswith('value-')}
        # Get value schema if provided
        if "schema" in value_schema:
            value_schema = value_schema["schema"]
        else:
            # Convert generic value schema to key-specific value schema
            value_schema = {k:value_schema for k in data.keys()}
        
        # Get all value entries
        self.data = {
            instantiate_data_type(key,key_schema,key_path) : (
                instantiate_data_type(val,value_schema[key],key_path+[key])
                if key in value_schema \
                else instantiate_data_type(val,value_schema,key_path+[key])
            )
            for key,val in data.items()
        }

    def value(self) -> dict:
        return {key.value():(self.schema['default-value'] if is_null(datum) else datum.value()) for key,datum in self.data.items()}
    

    def check(self):
        # Check that correct length is provided
        self.check_length()
        # Check that all keys are unique
        self.check_uniqueness(vals = self.data.keys())
        # Check that all values are unique
        self.check_uniqueness(vals = self.data.values())
        # Check that each key is valid
        self.check_elements(vals = self.data.keys())
        # Check that each value is valid
        self.check_elements(vals = self.data.values())