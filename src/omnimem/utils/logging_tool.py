import os
import datetime
import logging
import sys
import torch.distributed as dist

logger_initialized = {}

TIMESTAMPS = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')


def get_logger(name="__main__", file_path=None, master_only=True, mode=None, level=logging.INFO):
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = int(os.getenv('RANK', 0))

    if rank == 0:
        logger_name = name
    else:
        logger_name = f"{name}:{rank}"
    logger = logging.getLogger(logger_name)

    format_str = f'[{logger_name}]:[%(asctime)s] [%(levelname)s] - %(message)s'
    formatter = logging.Formatter(format_str, "%Y-%m-%d %H:%M:%S")

    if rank == 0:
        log_level = level
        flog_level = logging.DEBUG
    else:
        if master_only:
            log_level = logging.WARNING
        else:
            log_level = level
        flog_level = logging.DEBUG

    has_file_handler = any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
    if logger_name in logger_initialized:
        if file_path is not None and not has_file_handler:
            os.makedirs(file_path, exist_ok=True)
            file_handler = logging.FileHandler(f"{file_path}/{TIMESTAMPS}.log", mode='a' if mode is None else mode)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(flog_level)
            logger.addHandler(file_handler)
            return logger
        else:
            return logger

    for handler in logger.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.ERROR)

    if file_path is not None:
        os.makedirs(file_path, exist_ok=True)
        file_handler = logging.FileHandler(f"{file_path}/{TIMESTAMPS}.log", mode='a' if mode is None else mode)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(flog_level)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)

    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)
    logger.setLevel(log_level)

    logger_initialized[logger_name] = True

    return logger
