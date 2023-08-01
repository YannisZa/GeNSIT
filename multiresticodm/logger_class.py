"""Configures the CustomLogger 
obtained from https://gitlab.com/utopia-project/dantro/-/blob/main/dantro/logging.py
"""

import logging

# Define the additional log levels
TRACE = 5
REMARK = 12
NOTE = 18
PROGRESS = 19
CAUTION = 23
HILIGHT = 25
SUCCESS = 35
EMPTY = 60


    
# Custom Formatter with additional options
class CustomFormatter(logging.Formatter):
    def __init__(self, *args, handler_name='', **kwargs):
        super().__init__(*args, **kwargs)
        self.handler_name = handler_name

    def format(self, record):
        record.handler_name = self.handler_name  # Add custom option to the record
        return super().format(record)
    

class CustomLogger(logging.getLoggerClass()):
    """The custom dantro logging class with additional log levels"""

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

        logging.addLevelName(TRACE, "TRACE")
        logging.addLevelName(REMARK, "REMARK")
        logging.addLevelName(NOTE, "NOTE")
        logging.addLevelName(PROGRESS, "PROGRESS")
        logging.addLevelName(CAUTION, "CAUTION")
        logging.addLevelName(HILIGHT, "HILIGHT")
        logging.addLevelName(SUCCESS, "SUCCESS")
        logging.addLevelName(EMPTY, "EMPTY")


    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)

    def remark(self, msg, *args, **kwargs):
        if self.isEnabledFor(REMARK):
            self._log(REMARK, msg, args, **kwargs)

    def note(self, msg, *args, **kwargs):
        if self.isEnabledFor(NOTE):
            self._log(NOTE, msg, args, **kwargs)

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



class CustomFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', delay=True, **kwargs):
        super().__init__(filename=filename,mode=mode)
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
