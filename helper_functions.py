import numpy as np
from itertools import groupby


def find_closest(array, value):
    value = float(value)
    idx = (np.abs(array - value)).argmin()

    return array[idx], idx


def ismember(A, B):
    """
    Returns an array containing logical True where data in A is found in B. Also returns
    indices in B for every value of A.
    """

    A_in_B = np.in1d(A, B)

    B_idx = np.zeros(len(B), dtype=int)
    A = list(A)
    for i,element in enumerate(B):
        try:
            B_idx[i] = int(A.index(element))
        except:
            B_idx[i] = 0

    return A_in_B, B_idx


def find_dict_index(list, key, value):
    idx = []
    for i, dic in enumerate(list):
        if dic[key] == value:
            idx.append(i)

    return idx

def shift_rows(data):
    n_rows, n_cols = data.shape
    max_shift = n_cols - 1

    col_start = np.random.randint(0, max_shift, n_rows)

    idx = np.mod(col_start[:,None] + np.arange(n_cols), n_cols)

    out = data[np.arange(n_rows)[:,None], idx]

    return out

def get_longest_run(list):
    length = max(sum(1 for _ in items) for val, items
        in groupby(list) if val)

    return length