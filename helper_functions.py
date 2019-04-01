import numpy as np
from itertools import groupby
from scipy.stats import zscore

def find_closest(array, value):
    value = float(value)
    idx = (np.abs(array - value)).argmin()

    return array[idx], idx


def ismember(A, B):
    """
    Returns an array containing logical True where data in B is found
    in A. Also returns indices in A for every value of B.
    """

    B_in_A = np.in1d(B, A)

    A = list(A)
    if isinstance(B, str):
        B = [B]
    try:
        B_idx = np.zeros(len(B),dtype=int)
        for i,element in enumerate(B):
            try:
                B_idx[i] = int(A.index(element))
            except:
                B_idx[i] = 0
    except:
        B_idx = int(A.index(B))

    return B_in_A, B_idx


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

def get_event_rate(events):
    assert np.max(events) <= 1, 'Binarize input first.'

    # Duration in seconds.
    if events.ndim == 1:
        duration = len(events) / 20
        event_rate = np.sum(events)
    elif events.ndim == 2:
        duration = events.shape[1] / 20
        event_rate = np.sum(events, axis=1)
    else:
        raise ValueError('Unknown operation.')

    event_rate = event_rate / duration

    return event_rate

def nan(size):
    """
    Makes nan array.

    Parameter
    ---
    size: tuple.

    Return
    a: nan array.
    """

    a = np.empty(size)
    a.fill(np.nan)

    return a

def bool_array(size, trues):
    """
    Makes a boolean array that's true where indicated.

    Parameters
    ---
    size: int OR tuple, size of the array.
    trues: list OR int OR tuple of lists, true values.

    Return
    ---
    arr: boolean array.

    """
    arr = np.zeros(size, dtype=bool)
    arr[trues] = True

    return arr


def sem(arr, axis=0):
    """
    Computes standard error of the mean of the array.

    Parameter
    ---
    arr: array, array to be sem'd.

    Return
    ---
    standard_error: array, standard error.

    """
    n = np.sum(~np.isnan(arr), axis=axis)
    standard_error = np.nanstd(arr, axis=axis)/np.sqrt(n)

    return standard_error


def detect_onsets(bool_arr):
    """
    Detect onsets of True values in a boolean array.

    :param bool_arr:
    :return:
    """
    int_arr = bool_arr.astype(int)
    onsets = np.diff(int_arr, axis=1)
    onsets[onsets < 0] = 0

    return onsets


def partial_z(arr, inds):
    """
    Computes z-score on a subset of the data, specified by inds.
    Currently only works on 2D array along columns.

    :param arr:
    :param axis:
    :return:
    """

    partial_arr = arr[:,inds]
    z = zscore(partial_arr, axis=1)
    arr[:,inds] = z
    arr[:,~inds] = np.nan

    return arr


def pad_and_stack(arrs, pad_lengths):
    x = [np.pad(arr, (0, pad_length), mode='constant',
                constant_values=np.nan)
         for arr, pad_length in zip(arrs, pad_lengths)]

    return np.hstack(x)