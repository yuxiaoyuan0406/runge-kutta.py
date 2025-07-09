'''
Some useful utilities.
'''
# import yaml
import json
from datetime import datetime
import os
from functools import wraps
import numpy as np
import logging

now = datetime.now()
formatted_date_time = now.strftime('%Y%m%d-%H%M%S')


def save_dict(file_name: str, data: dict):
    '''
    Save the given data to a file.
    '''
    directory = os.path.dirname(file_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.close()


def vectorize(func):
    '''
    A vectorizing function to wrap any single value
    function(float to float for instance) to behave
    like a numpy function such as `np.sin`, which
    returns an array when given an array and returns
    a scalar value when given a scalar.
    '''

    @wraps(func)
    def wrapper(x, *args, **kwargs):
        if np.isscalar(x):
            return func(x, *args, **kwargs)
        else:
            vectorized_func = np.vectorize(func)
            return vectorized_func(x, *args, **kwargs)

    return wrapper

def default_logger(name: str = __name__, level: int = logging.INFO):
    '''
    Set up a default logger.
    '''
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    logger.addHandler(ch)

    return logger

def now_as_seed()->int:
    return int(datetime.now().timestamp() * 1e6)

if __name__ == '__main__':
    @vectorize
    def unit_pulse(x, offset: float = 0)-> float:
        if x == offset:
            return 1.
        return 0.

    t = np.linspace(0, 1, int(1e6), endpoint=False)
    print(unit_pulse(t).dtype)
    print(unit_pulse(t, 1e-6))
