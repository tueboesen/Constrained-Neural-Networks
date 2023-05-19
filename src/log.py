import logging
from datetime import datetime


def setup_custom_logger(name, logfile_loc, debug):
    '''
    Starts a logger that prints to a file and to the screen.
    '''
    sh = logging.StreamHandler()

    logger = logging.getLogger(name)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.addHandler(sh)
    fh = logging.FileHandler(logfile_loc)
    logger.addHandler(fh)
    logger.info('Starting logfile at {}.'.format(datetime.now()))
    return logger


def log_all_parameters(LOG, args):
    """
    Logs all parameters in the dictionary args
    """
    LOG.info('---------Listing all parameters-------')
    for key, value in args.items():
        LOG.info("{:30s} : {}".format(key, value))


def close_logger(log):
    '''
    Removes all handlers from logger and effectively closes the logger down.
    '''
    log.info('Closing logfile at {}.'.format(datetime.now()))
    handlers = log.handlers[:]
    for handler in handlers:
        handler.close()
        log.removeHandler(handler)
