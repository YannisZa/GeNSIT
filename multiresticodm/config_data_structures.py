import os
import sys

from collections.abc import Iterable

from multiresticodm.utils import in_range

def instantiate_data_type(
    data,
    schema:dict
):
    data_type = schema.get("type","").capitalize()
    if len(data_type) > 0 and hasattr(sys.modules[__name__], data_type):
        return getattr(sys.modules[__name__], data_type)(
            data=data,
            schema=schema
        )
    else:
        raise ValueError(f"Data type'{data_type}' not found")


class Entry():
    def __init__(self,data,schema):
        self.data = data
        self.schema = schema

    def __repr__(self) -> str:
        return "Entry()"
    
    def check_type(self,data_type=None,key_path=[]):
        data_type = self.schema.get("type",data_type)
        if data_type is not None:
            try: 
                assert isinstance(self.data,self.type)
            except:
                raise InvalidTypeException(f"Expected type {self.type}. Got {type(self.data)}",key_path=key_path)
        else:
            raise TypeNotFoundException(key_path=key_path)
        return True
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#----------------------- Primitive Data Entry -----------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

class PrimitiveEntry(Entry):
    def __init__(self,data,schema):
        super().__init__(data,schema)

    def __repr__(self) -> str:
        return "PrimitiveEntry()"
    
    def check(self,key_path=[]):
        pass
    
    def check_range(self,key_path=[]):
        # Check that correct range is used for each element
        valid_range = self.schema.get("valid-range","invalid")
        if valid_range != "invalid":
            # Get whether bounds are tight or not
            inclusive = self.schema.get("inclusive",True)
            try: 
                assert in_range(self.data,valid_range,inclusive)
            except:
                raise InvalidRangeException(f"Data {self.data} is out of range {valid_range}",key_path=key_path)
        return True
    
    def check_scope(self,key_path=[]):
        valid_scope = self.schema.get("is-any-of",[])
        if isinstance(valid_scope,list) and len(valid_scope) > 0:
            try: 
                assert self.data in valid_scope
            except:
                raise InvalidScopeException(f"Data {self.data} is not in scope {valid_scope}",key_path=key_path)
        return True

class Bool(PrimitiveEntry):
    def __init__(self,data,schema):
        super().__init__(data,schema)
        self.type = bool
    
    def __repr__(self) -> str:
        return "Boolean()"
    
    def check_boolean(self,key_path=[]):
        # Check that the data is either true or false
        try:
            assert str(self.data) in ['True','False']
        except:
            raise InvalidBooleanException(f"Boolean {str(self.data)} is not equal to 'True' or 'False'",key_path=key_path)

class Numeric(PrimitiveEntry):
    def __init__(self,data,schema):
        super().__init__(data,schema)

    def check_numeric(self,key_path=[]):
        try:
            assert self.data.isnumeric()
        except:
            raise StringNotNumericException(f"String {self.data} is not numeric",key_path=key_path)
        return True

    def check(self,key_path=[]):
        # Check that correct type is used
        self.check_type(key_path=key_path)
        # Check that correct range is used 
        self.check_range(key_path=key_path)
        # Check that input is in the scope of allowable inputs
        self.check_scope(key_path=key_path)

    def __repr__(self) -> str:
        return "Numeric()"
    
class Int(Numeric):
    def __init__(self,data,schema):
        super().__init__(data,schema)
        self.type = int
    
    def __repr__(self) -> str:
        return "Integer()"
    
class Float(Numeric):
    def __init__(self,data,schema):
        super().__init__(data,schema)
        self.type = float
    
    def __repr__(self) -> str:
        return "Float()"
    
class Str(PrimitiveEntry):
    def __init__(self,data,schema):
        super().__init__(data,schema)
        self.type = str
    
    def __repr__(self) -> str:
        return "String()"

    def check(self,key_path=[]):
        # Check that correct type is used
        self.check_type(key_path=key_path)
        # Check that input is in the scope of allowable inputs
        self.check_scope(key_path=key_path)

class StrInt(Numeric):
    def __init__(self,data,schema):
        super().__init__(data,schema)
        self.type = str
    
    def __repr__(self) -> str:
        return "StringInteger()"

    def check(self,key_path=[]):
        # Check that correct type is used
        self.check_type(key_path=key_path)
        # Check that string is numeric
        self.data = int(self.data)
        # Check that correct range is used 
        self.check_range(key_path=key_path)
        # Check that input is in the scope of allowable inputs
        self.check_scope(key_path=key_path)

class Path(Str):
    def __init__(self,data,schema):
        super().__init__(data,schema)
        self.type = str

    def __repr__(self) -> str:
        return "Path(String)"

    def check_path_exists(self,key_path=[]):
        if self.schema.get("path-exists",False) or \
            self.schema.get("file-exists",False) or \
            self.schema.get("directory-exists",False) or \
            self.schema.get("exists",False):
            try:
                assert os.path.exists(self.data)
            except:
                raise PathNotExistException(f"Path to file {self.data} does not exist",key_path=key_path)
        return True

    def check_directory_exists(self,key_path=[]):
        self.check_path_exists(key_path=key_path) 
        try:
            os.path.isdir(self.data)
        except:
            raise DirectoryNotFoundException(f"Path to file {self.data} is not a directory",key_path=key_path)
        return True

    def check_file_exists(self,key_path=[]):
        self.check_path_exists(key_path=key_path)
        try:
            os.path.isfile(self.data)
        except:
            raise FileNotFoundException(f"Path to file {self.data} is not a file",key_path=key_path)
        return True

    def check_extension(self,key_path=[]):
        extension = self.schema.get("extension",None)
        if extension is not None:
            extension = self.schema.get("file-extension",None)
        if extension is not None:
            try:
                assert self.data.endswith(("."+extension))
            except:
                raise InvalidExtensionException(f"File {self.data} does not have extension {extension}.",key_path=key_path)
        return True



class File(Path):
    def __init__(self,data,schema):
        super().__init__(data,schema)
        self.type = str

    def __repr__(self) -> str:
        return "File(Path)"

    def check(self,key_path=[]):
        # Check that correct type is used
        self.check_type(key_path=key_path)
        # Check that path exists
        self.check_file_exists(key_path=key_path)
        # Check that file extension exists
        self.check_extension(key_path=key_path)

class Directory(Path):
    def __init__(self,data,schema):
        super().__init__(data,schema)
        self.type = str

    def __repr__(self) -> str:
        return "Directory(Path)"

    def check(self,key_path=[]):
        # Check that correct type is used
        self.check_type(key_path=key_path)
        # Check that path exists
        self.check_directory_exists(key_path=key_path)

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------- Non-Primitive Data Entry ---------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

class NonPrimitiveEntry(Entry):
    def __init__(self,data,schema):
        super().__init__(data,schema)
        self.names = ['dict','list']

    def __repr__(self) -> str:
        return "NonPrimitiveEntry()"

    def check_length(self,key_path=[]):
        # Check that list has specified length
        length = self.schema.get("length",None)
        if length is not None:
            if isinstance(length,Iterable):
                try: 
                    assert in_range(len(self.data),length,True)
                except:
                    raise InvalidLengthException(f"Length {len(self.data)} is not in range {length}.",key_path=key_path)
            else:
                try: 
                    assert len(self.data) == length
                except:
                    raise InvalidLengthException(f"Length {len(self.data)} is not equal to {length}.",key_path=key_path)
        return True
    
    def check(self,key_path=[]):
        pass


class List(NonPrimitiveEntry):
    def __init__(self,data,schema,key_path=[]):
        super().__init__(data,schema)
        self.type = list
        # print('before')
        # print(self.data,type(self.data))
        # print(self.schema)
        # First check that input is indeed a list
        self.check_type(key_path=key_path)
        # print('after')

        # Then prep schema for list values
        # Remove all key prefixes that start with 'list'
        # Remove 'type' key
        self.schema = {} 
        for k,v in schema.items():
            if k == 'type':
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
        self.data = [instantiate_data_type(val,self.schema) for val in data]
            
    def check_elements(self,key_path=[]):
        # Check that all elements of the list are valid
        for elem in self.data:
            elem.check(key_path=key_path)
        return True
            
    def check(self,key_path=[]):
        # Check that correct length is provided
        self.check_length(key_path=key_path)
        # Check that each element of the list is valid 
        # by calling their respective check functions
        self.check_elements(key_path=key_path)

class Dict(NonPrimitiveEntry):
    def __init__(self,data,schema,key_path=[]):
        super().__init__(data,schema)
        self.type = dict

        # First check that input is indeed a dict
        self.check_type(key_path=key_path)

        # Then prep schema for list values
        # Remove all key prefixes that start with 'list'
        # Remove 'type' key
        self.schema = {} 
        for k,v in schema.items():
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

        # Get key schema
        key_schema = {k.replace('key-','',1):v for k,v in self.schema.items() if k.startswith('key-')}
        # Get value schema
        value_schema = {k.replace('value-','',1):v for k,v in self.schema.items() if k.startswith('value-')}
        # Get all value entries
        self.data = {instantiate_data_type(key,key_schema):instantiate_data_type(val,value_schema) for key,val in data.items()}
            
    def check_elements(self,key_path=[]):
        # Check that all keys of the dictionary are valid
        for key in self.data.keys():
            key.check(key_path=key_path)
        # Check that all values of the dictionary are valid
        for val in self.data.values():
            val.check(key_path=key_path)
        return True

    def check(self,key_path=[]):
        # Check that correct length is provided
        self.check_length(key_path=key_path)
        # Check that each element of the list is valid 
        # by calling their respective check functions
        self.check_elements(key_path=key_path)

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#------------------------- Entry Exceptions -------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

class EntryException(Exception):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,**kwargs)
        self.message = message
        self.key_path = key_path

    def __str__(self):
        return f"""
            Error in {'>'.join(self.key_path)}
            {self.message}
        """

class InvalidLengthException(EntryException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

class InvalidRangeException(EntryException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

class InvalidTypeException(EntryException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

class InvalidElementTypeException(EntryException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

class InvalidScopeException(EntryException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

class StringNotNumericException(EntryException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

class PathNotExistException(EntryException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

class InvalidExtensionException(EntryException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

class InvalidBooleanException(EntryException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

class FileNotFoundException(PathNotExistException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

class DirectoryNotFoundException(PathNotExistException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

class InvalidElementException(EntryException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

class InvalidKeyException(InvalidElementException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

class InvalidValueException(InvalidElementException):
    def __init__(self,message,key_path=[],**kwargs):
        super().__init__(message,key_path=key_path,**kwargs)

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#------------------------ Schema Exceptions -------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

class SchemaException(Exception):
    def __init__(self,key_path=[],**kwargs):
        super().__init__(**kwargs)
        self.key_path = key_path

    def __str__(self):
        return f"""
            Missing key {">".join(self.key_path)} in schema
        """

class RangeNotFoundException(SchemaException):
    def __init__(self,key_path=[],**kwargs):
        super().__init__(key_path=key_path,**kwargs)

class TypeNotFoundException(SchemaException):
    def __init__(self,key_path=[],**kwargs):
        super().__init__(key_path=key_path,**kwargs)

class PathNotFoundException(SchemaException):
    def __init__(self,key_path=[],**kwargs):
        super().__init__(key_path=key_path,**kwargs)

class ExtensionNotFoundException(SchemaException):
    def __init__(self,key_path=[],**kwargs):
        super().__init__(key_path=key_path,**kwargs)

class EmptyOrInvalidScopeException(SchemaException):
    def __init__(self,key_path=[],**kwargs):
        super().__init__(key_path=key_path,**kwargs)
