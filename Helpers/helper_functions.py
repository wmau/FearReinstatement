import numpy as np

def find_closest(array, value):
    idx = (np.abs(array - value)).argmin()

    return array[idx],idx

def filter_good_neurons(accepted):
    neurons = [cell_number for cell_number, good in enumerate(accepted) if good]

    return neurons

def ismember(A,B):
    """
    Returns an array containing logical True where data in A is found in B. Also returns
    indices in B for every value of A.
    """

    A_in_B = np.in1d(A,B)
    B_idx = np.nonzero(A_in_B)[0]

    return A_in_B, B_idx