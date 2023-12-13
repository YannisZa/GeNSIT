class DataException(Exception):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class CastingException(DataException):
    def __init__(self,data_name:str,from_type:str,to_type:str,**kwargs):
        super().__init__('',**kwargs)
        self.data_name = data_name
        self.from_type = from_type
        self.to_type = to_type
        

    def __str__(self):
        return f"""
            Casting {self.data_name} from {self.from_type} to {self.to_type} failed!
        """

class MissingData(DataException):
    def __init__(self,missing_data_name:str,data_names:str,location:str,**kwargs):
        super().__init__('',**kwargs)
        self.missing_data_name = missing_data_name
        self.data_names = data_names
        self.location = location

    def __str__(self):
        return f"""
            Missing data {self.missing_data_name} in {self.location} ({self.data_names})!
        """

class InvalidDataLength(DataException):
    def __init__(self,data_name_lens:dict,**kwargs):
        super().__init__('',**kwargs)
        self.data_name_lens = data_name_lens

    def __str__(self):
        return f"""
            Data {self.data_name_lens} does not have equal lengths!
        """


class DuplicateData(DataException):
    def __init__(self,message:str,len_data:int,len_unique_data:int,**kwargs):
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
    def __init__(self,message:str,data_names:str,**kwargs):
        super().__init__('',**kwargs)
        self.message = message
        self.data_names = data_names

    def __str__(self):
        return f"""
            Empty data {self.data_names} after {self.message}!
        """

class H5DataWritingFailed(DataException):
    def __init__(self,message:str,**kwargs):
        super().__init__(message,**kwargs)
        self.message = message

class DataCollectionException(DataException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class IrregularDataCollectionSize(DataCollectionException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

# ---

class MultiprocessorFailed(Exception):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)
        self.message = message
        self.processor_name = kwargs.get('name','')

    def __str__(self):
        return f"""
            Multiprocessor {self.processor_name} has MultiprocessorFailed!
        """

# --

class InvalidDataNames(DataException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class MissingFiles(DataException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class CorruptedFileRead(DataException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InvalidMetadataType(DataException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class MissingMetadata(DataException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class CoordinateSliceMismatch(DataException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

# ---

class ConfigException(Exception):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InvalidConfigType(ConfigException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)


# ---

class LoggerType(Exception):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)


class InvalidLoggerType(LoggerType):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)


#--------------------------------------------------------------------
#--------------------------------------------------------------------
#------------------------- Entry Exceptions -------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

class EntryException(Exception):
    def __init__(self,message,**kwargs):
        super().__init__(message)
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
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InvalidRangeException(EntryException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class DataUniquenessException(EntryException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InfiniteNumericException(EntryException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InvalidTypeException(EntryException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InvalidElementTypeException(EntryException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InvalidScopeException(EntryException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class StringNotNumericException(EntryException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class PathNotExistException(EntryException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InvalidExtensionException(EntryException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InvalidBooleanException(EntryException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class CustomListParsingException(EntryException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class FileNotFoundException(PathNotExistException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class DirectoryNotFoundException(PathNotExistException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InvalidElementException(EntryException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InvalidKeyException(InvalidElementException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InvalidValueException(InvalidElementException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

#--------------------------------------------------------------------
#--------------------------------------------------------------------
#------------------------ Schema Exceptions -------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------

class SchemaException(Exception):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
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
