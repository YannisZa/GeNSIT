class DataException(Exception):
    def __init__(self,message:str):
        super().__init__(message)

class CastingException(DataException):
    def __init__(self,data_name:str='',from_type:str='notype',to_type:str='notype',**kwargs):
        super().__init__('')
        self.data_name = data_name
        self.from_type = from_type
        self.to_type = to_type
        

    def __str__(self):
        return f"""
            Casting {self.data_name} from {self.from_type} to {self.to_type} failed!
        """

class MissingData(DataException):
    def __init__(self,missing_data_name:str='',data_names:str='none',location:str='nowhere',**kwargs):
        super().__init__('')
        self.missing_data_name = missing_data_name
        self.data_names = data_names
        self.location = location

    def __str__(self):
        return f"""
            Missing data {self.missing_data_name} in {self.location} ({self.data_names})!
        """

class InvalidDataLength(DataException):
    def __init__(self,data_name_lens:dict={},**kwargs):
        super().__init__('')
        self.data_name_lens = data_name_lens

    def __str__(self):
        return f"""
            Data {self.data_name_lens} does not have equal lengths!
        """

class DuplicateData(DataException):
    def __init__(self,message:str,len_data:int = 0,len_unique_data:int = 0,**kwargs):
        super().__init__(message)
        self.message = message
        self.len_data = len_data
        self.len_unique_data = len_unique_data
    
    def __str__(self):
        return f"""
            {self.message}
            # values: {self.len_data}
            # unique values: {self.len_unique_data}
        """


class EmptyData(DataException):
    def __init__(self,message:str,data_names:str='no_data',**kwargs):
        super().__init__(message)
        self.message = message
        self.data_names = data_names

    def __str__(self):
        return f"""
            Empty data {self.data_names} after {self.message}!
        """
    
class InvalidDataRange(DataException):
    def __init__(self,data,rang:list,**kwargs):
        super().__init__('')
        self.rang = rang
        self.data = data

    def __str__(self):
        return f"""
            Data {self.data} not ({self.rang})!
        """

class H5DataWritingFailed(DataException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)

class H5DataReadingFailed(DataException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)

class DataCollectionException(DataException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)

class DataConflict(DataException):
    def __init__(self,**kwargs):
        super().__init__('')
        self.property = kwargs.get('property','keys')
        self.problem = kwargs.get('problem','')
        self.data = kwargs.get('data','data')

    def __str__(self):
        return f"""
            Data has conflicting {self.property}.
            Conflict has to do with {self.data}.
            {self.problem}
        """

class IrregularDataCollectionSize(DataCollectionException):
    def __init__(self,message:str,sizes:dict={},**kwargs):
        super().__init__(message)
        self.sizes = sizes

    def __str__(self):
        return f"""
            Irregular DataCollection
            {self.sizes}
        """
# ---

class FunctionFailed(Exception):
    def __init__(self,**kwargs):
        super().__init__('')
        self.function_name = kwargs.get('name','')
        self.function_keys = kwargs.get('keys',[])
        self.message = kwargs.get('message','')

    def __str__(self):
        return f"""
            Function {self.function_name} with arguments {self.function_keys} has failed!
            {self.message}
        """

class MultiprocessorFailed(FunctionFailed):
    def __init__(self,keys:list=[],**kwargs):
        super().__init__(name = 'Multiprocessor', keys = keys, **kwargs)


# --

class InvalidDataNames(DataException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)

class MissingFiles(DataException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)

class CorruptedFileRead(DataException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)

class InvalidMetadataType(DataException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)

class MissingMetadata(DataException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)

class CoordinateSliceMismatch(DataException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)


# ---

class ConfigException(Exception):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)

class InvalidConfigType(ConfigException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)


# ---

class LoggerType(Exception):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)


class InvalidLoggerType(LoggerType):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)


# ---

class InstantiationException(Exception):
    def __init__(self,message:str,**kwargs):
        super().__init__(message)



#--------------------------------------------------------------------
#--------------------------------------------------------------------
#------------------------- Entry Exceptions -------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

class EntryException(Exception):
    def __init__(self,message:str,**kwargs):
        super().__init__()
        self.message = message
        self.key_path = kwargs.get('key_path',[])
        self.data = kwargs.get('data','[data-not-found]')

    def __str__(self):
        return f"""
            Error in {'>'.join(list(map(str,self.key_path)))}
            {self.message}
            Data: {self.data}
        """

class InvalidLengthException(EntryException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class InvalidRangeException(EntryException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class DataUniquenessException(EntryException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class InfiniteNumericException(EntryException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class InvalidTypeException(EntryException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class InvalidElementTypeException(EntryException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class InvalidScopeException(EntryException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class StringNotNumericException(EntryException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class PathNotExistException(EntryException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class InvalidExtensionException(EntryException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class InvalidBooleanException(EntryException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class CustomListParsingException(EntryException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class FileNotFoundException(PathNotExistException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class DirectoryNotFoundException(PathNotExistException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class InvalidElementException(EntryException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class InvalidKeyException(InvalidElementException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

class InvalidValueException(InvalidElementException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#------------------------ Schema Exceptions -------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

class SchemaException(Exception):
    def __init__(self,**kwargs):
        super().__init__()
        self.key_path = kwargs.get('key_path',[])
        self.data = kwargs.get('data','[data-not-found]')

    def __str__(self):
        return f"""
            Missing key {">".join(self.key_path)} in schema
        """

class RangeNotFoundException(SchemaException):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
class TypeNotFoundException(SchemaException):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

class PathNotFoundException(SchemaException):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

class ExtensionNotFoundException(SchemaException):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

class EmptyOrInvalidScopeException(SchemaException):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
