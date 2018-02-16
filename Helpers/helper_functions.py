import numpy as np

def find_closest(array, value):
    idx = (np.abs(array - value)).argmin()

    return array[idx],idx