import numpy as np
import calcium_traces as ca_traces
import calcium_events as ca_events
import ff_video_fixer as ff
from scipy.stats import zscore
import cell_reg
from session_directory import load_session_list

session_list = load_session_list()

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


def load_and_trim(session_index, dtype='traces', neurons=None):
    session = ff.load_session(session_index)

    if dtype == 'traces':
        data, t = ca_traces.load_traces(session_index)
        data = zscore(data, axis=1)
    elif dtype == 'events':
        data, t = ca_events.load_events(session_index)
    elif dtype == 'freezing':
        data = session.imaging_freezing
        t = session.imaging_t
    else:
        raise ValueError('Wrong data type.')

    if neurons is not None:
        data = data[neurons]

    data = trim_session(data, session.mouse_in_cage)
    t = trim_session(t, session.mouse_in_cage)

    return data, t

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
                data = zscore(data, axis=1)
            elif dtype == 'events':
                data, t = ca_events.load_events(session)
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

if __name__ == '__main__':
    concatenate_sessions([0,1,2,4])