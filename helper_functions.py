import numpy as np
from itertools import groupby
from scipy.stats import zscore
import matplotlib.pyplot as plt

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


def pad_and_stack(arrs, pad_lengths, ax=0):
    """
    Stacks a list of arrays horizontally (column-wise) after
    padding or truncating, depending on the values in pad_lengths.
    If a pad length is positive, it will add nans. If negative,
    it will truncate up to that data point for each array.

    :param arrs:
    :param pad_lengths:
    :return:
    """
    for arr, pad_length in zip(arrs, pad_lengths):
        if pad_length < 0:
            if arr.ndim > 1:
                truncated = arr[:,:pad_length]
            else:
                truncated = arr[:pad_length]
            try:
                x.append(truncated)
            except:
                x = [truncated]
        else:
            if arr.ndim > 1:
                padded = np.pad(arr, [(0, 0), (0, pad_length)],
                                mode='constant', constant_values=np.nan)
            else:
                padded = np.pad(arr, (0, pad_length),
                                mode='constant',
                                constant_values=np.nan)
            try:
                x.append(padded)
            except:
                x = [padded]

    return np.hstack(x)

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, label=None):
    """
    Line show_plot with error bars except the error bars are filled in
    rather than the monstrosity from matplotlib.
    :parameters
    ---
    x: array-like
        x-axis values.
    y: array-like, same length as x
        y-axis values.
    yerr: array-like, same length as x and y
        Error around the y values.
    """
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.get_next_color()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)