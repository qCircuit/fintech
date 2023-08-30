import logging
import numpy as np
from scipy.stats import linregress

import configur

def get_coefs(x1:float, x2:float, y1:float, y2:float) -> tuple:
    # get slope intercept coefs
    slope, intercept, r_value, p_value, std_err = linregress([x1, x2], [y1, y2])
    
    return slope, intercept

def get_line(x1:float, x2:float, slope:float, intercept:float) -> np.array:
    # iterated linear function
    line = list()
    for x in range(x1, x2+1):
        line.append((x, slope * x + intercept))
        
    return np.array(line)

def setup_logger(log_file):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger("logs.log")