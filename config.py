from logzero import logging, setup_logger

LOG_FILE = "trash_panda_cop.log"
LOG_LEVEL = logging.INFO


def getLogger():
    ''' Return a standard configured logger '''
    return setup_logger(logfile=LOG_FILE, 
                        level=LOG_LEVEL,
                        maxBytes=1024 * 1024 * 10, 
                        backupCount=10)
