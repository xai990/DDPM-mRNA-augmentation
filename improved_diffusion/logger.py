import logging 
import os 


def log(*args, level=logging.INFO):
    get_current().log(*args, level=level)


def info(*args):
    log(*args, level=logging.INFO)

def warning(*args):
    log(*args, level=logging.WARN)

def error(*args):
    log(*args, level=logging.ERROR)

def debug(*args):
    log(*args, level=logging.DEBUG)



class OneTimeFormatter(logging.Formatter):
    def __init__(self, first_fmt, subsequent_fmt):
        super().__init__(first_fmt)
        self.first_fmt = first_fmt
        self.subsequent_fmt = subsequent_fmt
        self.first_call = True

    def format(self, record):
        if self.first_call:
            self.first_call = False
            return super().format(record)
        else:
            # Change to subsequent format after the first call
            self._style._fmt = self.subsequent_fmt
            return super().format(record)



class Logger:
    CURRENT = None
    def __init__(self, name, log_file, level=logging.INFO, formatter = None):
        self.level = level
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        # Create a formatter and set it for the handler
        #self.formatter = logging.Formatter(formatter, datefmt='%Y-%m-%d)
        self.formatter = OneTimeFormatter(formatter, '%(levelname)s - %(message)s')
        if formatter == None:
            #formatter = logging.Formatter('%(levelname)s - %(message)s')
            self.formatter = logging.Formatter('---- %(message)s')
        
        file_handler.setFormatter(self.formatter)

        # Add the handler to the logger
        self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)
    
    def log(self, *args, level=logging.INFO):
        if level==logging.INFO:
            self.info(*args)
        elif level==logging.DEBUG:
            self.debug(*args)
        elif level==logging.WARN:
            self.warning(*args)
        elif level==logging.ERROR:
            self.error(*args)
        else:
            raise NotImplementedError
        

def get_current():
    if Logger.CURRENT is None:
        _configure_default_logger()
    return Logger.CURRENT


def configure(dir=None, format_strs=None, comm=None, log_suffix="", level=None):
    Logger.CURRENT = Logger('DF','log/log.txt', formatter = format_strs)


def _configure_default_logger():
    configure()
    