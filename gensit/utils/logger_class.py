"""Configures the CustomLogger 
obtained from https://gitlab.com/utopia-project/dantro/-/blob/main/dantro/logging.py
"""
import os
import inspect
import string
import random
import logging
import coloredlogs
from pathlib import Path

# Define the additional log levels
TRACE = 5
REMARK = 12
NOTE = 17
ITERATION = 18
PROGRESS = 19
CAUTION = 23
HILIGHT = 25
SUCCESS = 35
EMPTY = 60

LOG_LEVELS = ['trace','remark','note','iteration','progress','caution','hilight','success','empty']

logging.addLevelName(TRACE, "TRACE")
logging.addLevelName(REMARK, "REMARK")
logging.addLevelName(NOTE, "NOTE")
logging.addLevelName(ITERATION, "ITERATION")
logging.addLevelName(PROGRESS, "PROGRESS")
logging.addLevelName(CAUTION, "CAUTION")
logging.addLevelName(HILIGHT, "HILIGHT")
logging.addLevelName(SUCCESS, "SUCCESS")
logging.addLevelName(EMPTY, "EMPTY")



def random_string(N:int = 10):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k = N))


class CustomFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', delay = True, **kwargs):
        dirpath = Path(filename).absolute().parent
        if not os.path.exists(dirpath) or not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        
        super().__init__(filename = filename,mode = mode)
        self.filename = filename
        self.mode = mode
        self.delay = delay
        self.logger_name = kwargs.get('name','')
        self.buffer = []


    def emit(self, record):
        self.buffer.append(self.format(record))

    def flush(self):
        if not self.delay:
            self._write_to_file()

    def _write_to_file(self):
        # print("buffer size:",len(self.buffer),self.logger_name)
        with open(self.filename, self.mode) as f:
            for message in self.buffer:
                f.write(message + '\n')
        self.buffer = []

    def close(self):
        self._write_to_file()
        super().close()

class CustomLogger(logging.Logger):
    """The custom dantro logging class with additional log levels"""

    def __init__(self, name, level = logging.NOTSET):
        super().__init__(name, level)

    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)

    def remark(self, msg, *args, **kwargs):
        if self.isEnabledFor(REMARK):
            self._log(REMARK, msg, args, **kwargs)

    def note(self, msg, *args, **kwargs):
        if self.isEnabledFor(NOTE):
            self._log(NOTE, msg, args, **kwargs)

    def iteration(self, msg, *args, **kwargs):
        if self.isEnabledFor(ITERATION):
            self._log(ITERATION, msg, args, **kwargs)

    def progress(self, msg, *args, **kwargs):
        if self.isEnabledFor(PROGRESS):
            self._log(PROGRESS, msg, args, **kwargs)

    def caution(self, msg, *args, **kwargs):
        if self.isEnabledFor(CAUTION):
            self._log(CAUTION, msg, args, **kwargs)

    def hilight(self, msg, *args, **kwargs):
        if self.isEnabledFor(HILIGHT):
            self._log(HILIGHT, msg, args, **kwargs)

    def success(self, msg, *args, **kwargs):
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, msg, args, **kwargs)
    
    def empty(self, msg, *args, **kwargs):
        if self.isEnabledFor(EMPTY):
            self._log(EMPTY, msg, args, **kwargs)

class DualLogger():

    def __init__(self, name, level = logging.NOTSET):

        logging.setLoggerClass(CustomLogger)

        # Instantiate loggers
        self.console = logging.getLogger(name + "_console")
        self.file = logging.getLogger(name + "_file")

        # Console handler
        coloredlogs.install(
            fmt='%(asctime)s.%(msecs)03d %(modulename)-s %(levelname)-3s %(message)s',
            datefmt='%M:%S',
            logger = self.console,
            level = level
        )
        self.setLevels(console_level = level)

        # File handler
        file_handler = CustomFileHandler(f'logs/temp_{name}_{random_string(10)}.log',name = name)
        verbose_format = logging.Formatter(
            fmt='%(asctime)s.%(msecs)03d %(modulename)-s %(levelname)-s %(message)s',
            datefmt='%M:%S'
        )
        file_handler.setFormatter(verbose_format)
        
        # Remove default stderr handler
        for handler in self.file.handlers:
            if not isinstance(handler,CustomFileHandler):
                self.file.removeHandler(handler)
        
        if len(self.file.handlers) <= 0:
            self.file.addHandler(file_handler)
        
        for handler in self.file.handlers:
            handler.setLevel(logging.DEBUG)
        

    def getLevels(self,numeric:bool = False):
        if numeric:
            console_level = logging._nameToLevel[self.console.level] if isinstance(self.console.level,str) else self.console.level
            file_level = logging._nameToLevel[self.file.level] if isinstance(self.file.level,str) else self.file.level
        else:
            console_level = logging.getLevelName(self.console.level) if isinstance(self.console.level,int) else self.console.level
            file_level = logging.getLevelName(self.file.level) if isinstance(self.file.level,int) else self.file.level


        return {"console":console_level,
                "file":file_level}

    def setLevels(self, console_level = None, file_level = None) -> None:
        if console_level is not None:
            console_level = logging._nameToLevel[console_level.upper()] if isinstance(console_level,str) else console_level
            self.console.setLevel(console_level)
            for handler in self.console.handlers:
                handler.setLevel(console_level)
        if file_level is not None:
            file_level = logging._nameToLevel[file_level.upper()] if isinstance(file_level,str) else file_level
            self.file.setLevel(file_level)
            for handler in self.file.handlers:
                handler.setLevel(file_level)
    
    def getLoggers(self, name):
        self.file = logging.getLogger(name + "_file")
        self.console = logging.getLogger(name + "_console")
        return self
            
    def debug(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(logging.DEBUG):
            self.console.debug(msg,extra = dict(modulename = module))
        if self.file.isEnabledFor(logging.DEBUG):
            self.file.debug(msg,extra = dict(modulename = module))

    def info(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(logging.INFO):
            self.console.info(msg,extra = dict(modulename = module))
        if self.file.isEnabledFor(logging.INFO):
            self.file.info(msg,extra = dict(modulename = module))

    def warning(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(logging.INFO):
            self.console.warning(msg,extra = dict(modulename = module))
        if self.file.isEnabledFor(logging.WARNING):
            self.file.warning(msg,extra = dict(modulename = module))

    def error(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(logging.ERROR):
            self.console.error(msg,extra = dict(modulename = module))
        if self.file.isEnabledFor(logging.ERROR):
            self.file.error(msg,extra = dict(modulename = module))

    def critical(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(logging.CRITICAL):
            self.console.critical(msg,extra = dict(modulename = module))
        if self.file.isEnabledFor(logging.CRITICAL):
            self.file.critical(msg,extra = dict(modulename = module))

    def trace(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(TRACE):
            self.console.trace(msg, extra = dict(modulename = module))
        if self.file.isEnabledFor(TRACE):
            self.file.trace(msg, extra = dict(modulename = module))
    
    def remark(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(REMARK):
            self.console.remark(msg, extra = dict(modulename = module))
        if self.file.isEnabledFor(REMARK):
            self.file.remark(msg, extra = dict(modulename = module))

    def note(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(NOTE):
            self.console.note(msg, extra = dict(modulename = module))
        if self.file.isEnabledFor(NOTE):
            self.file.note(msg, extra = dict(modulename = module))
    
    def iteration(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(ITERATION):
            self.console.iteration(msg, extra = dict(modulename = module))
        if self.file.isEnabledFor(ITERATION):
            self.file.iteration(msg, extra = dict(modulename = module))
    
    def progress(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(PROGRESS):
            self.console.progress(msg, extra = dict(modulename = module))
        if self.file.isEnabledFor(PROGRESS):
            self.file.progress(msg, extra = dict(modulename = module))
    
    def caution(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(CAUTION):
            self.console.caution(msg, extra = dict(modulename = module))
        if self.file.isEnabledFor(CAUTION):
            self.file.caution(msg, extra = dict(modulename = module))
    
    def hilight(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(HILIGHT):
            self.console.hilight(msg, extra = dict(modulename = module))
        if self.file.isEnabledFor(HILIGHT):
            self.file.hilight(msg, extra = dict(modulename = module))
    
    def success(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(SUCCESS):
            self.console.success(msg, extra = dict(modulename = module))
        if self.file.isEnabledFor(SUCCESS):
            self.file.success(msg, extra = dict(modulename = module))
    
    def empty(self, msg):
        frame = inspect.stack()[1]
        module = frame.filename.split('.py')[0].split('/')[-1]
        if self.console.isEnabledFor(EMPTY):
            self.console.empty(msg, extra = dict(modulename = module))
        if self.file.isEnabledFor(EMPTY):
            self.file.empty(msg, extra = dict(modulename = module))
