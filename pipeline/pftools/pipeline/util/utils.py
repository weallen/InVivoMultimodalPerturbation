from typing import List,Callable
import dask.distributed as dd
import numpy as np
import logging
import os
import sys

def unique_unsorted(x:List) -> np.ndarray:
    """
    Return unique elements of an array, preserving order. 
    """
    _, idx = np.unique(np.array(x), return_index=True)
    return list(x[np.sort(idx)])

def scatter_gather(client: dd.Client, data: List, func: Callable, *args, **kwargs) -> List:
    """
    Apply a function to each tile in parallel.
    """
    futures = [client.submit(func, t, *args, **kwargs) for t in data]
    return client.gather(futures)

def create_logger(output_path:str, name:str, log_name:str="log.txt") -> logging.Logger:
    """
    Create a logger for the decoding.
    """
    logfile_path = os.path.join(output_path, log_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # or any other level you want
        
        # Create a file handler
        file_handler = logging.FileHandler(logfile_path, mode='w')
        file_handler.setLevel(logging.DEBUG)  # or any other level you want

        # Create a stream handler for stdout
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)  # or any other level you want

        # Create a formatter and set the formatter for the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
