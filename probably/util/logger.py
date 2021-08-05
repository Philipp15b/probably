import logging


def log_setup(name: str, level, file: str = 'test.log') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fhandler = logging.FileHandler(filename=file, mode='a')
    fhandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fhandler)
    return logger
