import numpy as np
import calcium_traces as ca_traces
import ff_video_fixer as ff
from scipy.stats import zscore

def filter_good_neurons(accepted):
    neurons = [cell_number for cell_number, good in
               enumerate(accepted) if good]

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

    return bins.astype(int)


def bin_time_series(data, bins):
    if data.ndim == 1:
        binned_data = np.split(data, bins)
    elif data.ndim == 2:
        binned_data = np.split(data, bins, axis=1)

    return binned_data

def trim_and_bin(session_index, dtype='trace', neurons=None,
                 samples_per_bin=200):
    session = ff.load_session(session_index)

    if dtype == 'trace':
        data, t = ca_traces.load_traces(session_index)
        data = zscore(data, axis=1)

    if neurons is not None:
        data = data[neurons]

    # Trim away home cage epochs.
    t = trim_session(t, session.mouse_in_cage)
    data = trim_session(data, session.mouse_in_cage)

    # Bin time series.
    bins = make_bins(t, samples_per_bin)
    t = bin_time_series(t, bins)
    data = bin_time_series(data, bins)

    return data, t, bins