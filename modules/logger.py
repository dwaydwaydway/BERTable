import logging

console_fmt = "[%(asctime)s] [%(name)5s] %(message)s"
file_fmt = "%(message)s"
date_fmt = '%Y-%m-%d %H:%M:%S'

console_formatter = logging.Formatter(
    fmt=console_fmt, datefmt=date_fmt)
file_formatter = logging.Formatter(
    fmt=file_fmt, datefmt=date_fmt)


def create_logger(level=None, name="", filename=None):
    if filename is None:
        handler = logging.StreamHandler()
        handler.setFormatter(console_formatter)
        logger = logging.getLogger(f'{name}')
        logger.addHandler(handler)
        if level is not None:
            logger.setLevel(getattr(logging, level))
        else:
            logger.setLevel(logging.INFO)
    else:
        logger = logging.getLogger(f'{filename}')
        handler = logging.FileHandler(filename=filename)
        handler.setFormatter(file_formatter)
        logger.addHandler(handler)
        if level is not None:
            logger.setLevel(getattr(logging, level))
        else:
            logger.setLevel(logging.INFO)
    return logger
