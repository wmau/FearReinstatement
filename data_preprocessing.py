import numpy as np
from microscoPy_load import calcium_events as ca_events, calcium_traces as ca_traces, cell_reg, ff_video_fixer as ff
from scipy.stats import zscore
from session_directory import load_session_list, get_session
from helper_functions import find_closest
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

session_list = load_session_list()

def filter_good_neurons(accepted):
    """
    Pares down the cell list to only include accepted neurons.

    :param accepted:
    :return:
    """
    neurons = [cell_number for cell_number, good in
               enumerate(accepted) if good]

    return neurons


def trim_session(data, indices):
    if data.ndim is 1:
        data = data[indices]
    elif data.ndim is 2:
        data = data[:, indices]

    return data


def load_and_trim(session_index, dtype='traces', neurons=None,
                  session=None, start=None, end=None, do_zscore=True):
    """
    Load session data and trim it.

    Parameters
    ---
    session_index: scalar, session number.
    dtype: str, possible inputs:
        'traces'
        'events'
        'freezing'
        'velocity'
    neurons: list-like, neuron indices.
    session: FFObj, for performance if calling this function multiple times
    start: scalar, start index for session timestamp.
    end: scalar, end index.

    Returns
    ---
    data: time series, depending on dtype.
    t: time vector.

    """
    # If session is not pre-loaded, load it.
    if session is None:
        session = ff.load_session(session_index)

    # Find data type.
    if dtype == 'traces':
        data, t = ca_traces.load_traces(session_index)
        data = np.ma.masked_invalid(data)

        if do_zscore:
            data = zscore(data, axis=1)

    elif dtype == 'events':
        data, t = ca_events.load_events(session_index)
        data = np.ma.masked_invalid(data)
        data[data > 0] = 1
    elif dtype == 'freezing':
        data = session.imaging_freezing
        t = session.imaging_t
    elif dtype == 'velocity':
        data = session.imaging_v
        t = session.imaging_t
    else:
        raise ValueError('Invalid data type.')

    # Subset of neurons.
    if neurons is not None:
        data = data[neurons]

    # If start and end indices are not specified, take the mouse in cage
    # indices.
    indices = np.zeros_like(session.mouse_in_cage, dtype=bool)
    if start is None and end is None:
        indices = session.mouse_in_cage
    # Else if start is not specified, but end is, take the mouse in cage
    # start index and specified end index.
    elif start is None and end is not None:
        start = np.where(session.mouse_in_cage)[0][0]
        indices[start:end] = True
    # Else if end is not specified, but start is, take the specified start
    # index and mouse in cage end index.
    elif start is not None and end is None:
        end = np.where(session.mouse_in_cage)[0][-1]
        indices[start:end] = True
    # Else both start and end are specified, use those.
    else:
        indices[start:end] = True

    data = trim_session(data, indices)
    t = trim_session(t, indices)

    return data, t

def make_bins(data, samples_per_bin, axis=1):
    length = data.shape[axis]

    bins = np.arange(samples_per_bin, length, samples_per_bin)

    return bins.astype(int)


def bin_time_series(data, bins):
    if data.ndim == 1:
        binned_data = np.split(data, bins)
    elif data.ndim == 2:
        binned_data = np.split(data, bins, axis=1)

    return binned_data

def trim_and_bin(session_index, data=None, t=None, session=None, dtype='traces',
                 neurons=None, samples_per_bin=200, mask=None):
    """
    Trims and bins imaging from a session.

    Parameters
    ---
    session: int, session number OR FFObj from load_session.
    dtype: str, 'traces' or 'events'.
    neurons: list, neuron indices.
    samples_per_bin: int, # of samples per bin.
    mask: boolean array.

    Returns
    ---
    data: (N,B) array, trimmed and binned matrix.
    t: (B,) array, trimmed and binned time vector.
    bins: (B,) array, indices for bin boundaries.
    """
    if session is None:
        session = ff.load_session(session_index)

    if data is None:
        if dtype == 'traces':
            data, t = ca_traces.load_traces(session_index)
            masked = np.ma.masked_invalid(data)
            data = zscore(masked, axis=1)
        elif dtype == 'events':
            data, t = ca_events.load_events(session_index)
            data[data > 0] = 1
        else:
            raise ValueError('Wrong data type.')

    if neurons is not None:
        data = data[neurons]

    # Make mask for trimming. If from_this_time is None, just use the
    # entire session starting from when the mouse enters the chamber.
    # Otherwise, start from the specified time until when the mouse
    # leaves.
    if mask is None:
        mask = session.mouse_in_cage

    # Trim away home cage epochs or starting from some other specified
    # time point.
    t = trim_session(t, mask)
    data = trim_session(data, mask)

    # Bin time series.
    bins = make_bins(t, samples_per_bin)
    t = bin_time_series(t, bins)
    data = bin_time_series(data, bins)

    return data, t, bins

def concatenate_sessions(sessions, include_homecage=False, dtype='traces',
                         global_cell_idx=None):
    # Load cell map.
    mouse = session_list[sessions[0]]['Animal']
    match_map = cell_reg.load_cellreg_results(mouse)
    if global_cell_idx is not None:
        match_map = match_map[global_cell_idx]
    match_map = cell_reg.trim_match_map(match_map, sessions)


    neural_data = []
    all_t = []
    all_days = []
    all_freezing = []
    for idx, session in enumerate(sessions):
        # Only get these neurons.
        neurons = match_map[:, idx]

        if not include_homecage:
            data, t = load_and_trim(session, dtype=dtype, neurons=neurons)
            freezing, _ = load_and_trim(session, dtype='freezing')
            t -= t.min()

        else:
            if dtype == 'traces':
                data, t = ca_traces.load_traces(session)
                masked = np.ma.masked_invalid(data)
                data = zscore(masked, axis=1)
            elif dtype == 'events':
                data, t = ca_events.load_events(session)
                data = np.ma.masked_invalid(data)
            else:
                raise ValueError('Invalid data type.')

            ff_session = ff.load_session(session)
            freezing = ff_session.imaging_freezing

            data = data[neurons]

        all_freezing.extend(freezing)

        # Day index.
        day_idx = np.ones(t.size)*idx
        all_days.extend(day_idx)

        # Time stamps.
        all_t.extend(t)

        # Neural data.
        neural_data.append(data)

    neural_data = np.column_stack(neural_data)
    all_days = np.asarray(all_days)
    all_t = np.asarray(all_t)
    all_freezing = np.asarray(all_freezing)

    return neural_data, all_days, all_t, all_freezing

def get_avg_event_rate(mouse, stage, bin_size=1, data=None,
                       t=None, session=None,
                       neurons=None, mask=None):
    """
    Computes the average event rate (events/s) for each cell after
    binning.

    Parameters
    ---
    mouse: string, name of mouse.
    stage: string, session stage.
    bin_size: scalar, size of bin in seconds.
    session: FFObj, from load_session.
    mask: boolean array, mask for which timestamps to consider.
    from which to start computing.

    Return
    ---
    avg_event_rate: (N,) ndarray containing values for average event
    rate of each cell.

    """
    # Get session number.
    session_index = get_session(mouse, stage)[0]

    # Trim and bin.
    samples_per_bin = bin_size*20
    data, t, bins = trim_and_bin(session_index, session=session,
                                 data=data, t=t,
                                 dtype='events',
                                 samples_per_bin=samples_per_bin,
                                 mask=mask, neurons=neurons)

    # Compute average event rate.
    event_rates = [np.nanmean(neuron, axis=1) for neuron in data]
    avg_event_rates = np.nanmean(np.asarray(event_rates).T, axis=1)

    return avg_event_rates


def interp(A):
    ok = ~np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[ok]
    x = np.isnan(A).ravel().nonzero()[0]

    A[np.isnan(A)] = np.interp(x, xp, fp)

    return A

def convolve(A, win_size):
    mask = np.isnan(A)
    K = np.ones(win_size, dtype=int)
    A = np.convolve(np.where(mask, 0, A), K, mode='same') \
        / np.convolve(~mask, K, mode='same')

    return A

if __name__ == '__main__':
    for session_index in range(50):
        #session_index = get_session('Kepler','FC')[0]
        load_and_trim(session_index)