import numpy as np

def find_closest(array, value):
    idx = (np.abs(array - value)).argmin()

    return array[idx],idx

def filter_good_neurons(accepted):
    neurons = [cell_number for cell_number, good in enumerate(accepted) if good]

    return neurons