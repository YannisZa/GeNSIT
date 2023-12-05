class MissingData(Exception):
    def __init__(self,missing_data:str,data_names:str,location:str,**kwargs):
        super().__init__('',**kwargs)
        self.missing_data = missing_data
        self.data_names = data_names
        self.location = location

    def __str__(self):
        return f"""
            Missing data {self.missing_data} in {self.location} ({self.data_names})!
        """

class EmptyData(Exception):
    def __init__(self,message:str,data_names:str,**kwargs):
        super().__init__('',**kwargs)
        self.message = message
        self.data_names = data_names

    def __str__(self):
        return f"""
            Empty data {self.data_names} after {self.message}!
        """


class DataCollectionException(Exception):
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

class OutputsException(Exception):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InvalidDataNames(OutputsException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class MissingFiles(OutputsException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class CorruptedFileRead(OutputsException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class InvalidMetadataType(OutputsException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class MissingMetadata(OutputsException):
    def __init__(self,message,**kwargs):
        super().__init__(message,**kwargs)

class CoordinateSliceMismatch(OutputsException):
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