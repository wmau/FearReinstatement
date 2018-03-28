import numpy as np


def filter_good_neurons(accepted):
    neurons = [cell_number for cell_number, good in enumerate(accepted) if good]

    return neurons


def trim_session(data, indices):
    if data.ndim is 1:
        data = data[indices]
    elif data.ndim is 2:
        data = data[:, indices]

    return data


def make_bins(data, samples_per_bin):
    length = len(data)

    bins = np.arange(samples_per_bin, length, samples_per_bin)
    bins = np.append(bins, length - 1)

    return bins


def bin_time_series(data, bins):
    binned_data = np.split(data, bins)

    return binned_data
