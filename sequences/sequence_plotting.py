from sequences.seqNMF_data import seqNMF
from session_directory import load_session_list
import numpy as np
import matplotlib.pyplot as plt
import calcium_traces as ca_traces
from scipy.stats import zscore
import data_preprocessing as d_pp
import ff_video_fixer as ff
from matplotlib import gridspec
import calcium_events as ca_events
from cell_reg import load_cellreg_results, find_match_map_index, \
    find_cell_in_map

session_list = load_session_list()

def plot_ordered_cells(session_index, cells, order=None, dtype='event'):
    if order is None:
        order = range(len(cells))

    session = ff.load_session(session_index)
    n_cells = len(cells)

    if type(cells) is list:
        cells = np.asarray(cells)

    if dtype == 'trace':
        data, t = ca_traces.load_traces(session_index)
        data = zscore(data, axis=1)
    elif dtype == 'event':
        data, t  = ca_events.load_events(session_index)
        data[data > 0] = 1
    else:
        raise ValueError('Wrong dtype value.')

    # Trim all the time series.
    traces = d_pp.trim_session(data, session.mouse_in_cage)
    t = d_pp.trim_session(t, session.mouse_in_cage)
    t -= min(t)
    v = d_pp.trim_session(session.imaging_v, session.mouse_in_cage)
    freezing = d_pp.trim_session(session.imaging_freezing,
                                 session.mouse_in_cage)

    if dtype == 'event':
        events = []
        for cell in traces:
            events.append(np.where(cell)[0]/20)

    # Plot.
    gs = gridspec.GridSpec(4, 1)
    fig = plt.figure()

    # Velocity plot.
    velocity_plot = fig.add_subplot(gs[0,:])
    plt.plot(t, v, 'k')
    velocity_plot.set_ylabel('Velocity',color='k')
    velocity_plot.fill_between(t, 0, v.max(), freezing,
                               facecolor='r')
    plt.setp(velocity_plot.get_xticklabels(), visible=False)

    # Raster plot.
    event_plot = fig.add_subplot(gs[1:,0], sharex=velocity_plot)

    if dtype == 'trace':
        plt.imshow(traces[cells[order]],
                   extent=[t[0], t[-1], n_cells, 0])
    elif dtype == 'event':
        plt.eventplot([events[x] for
                       x in cells[order]])

        event_plot.fill_between(t, -0.5, n_cells+0.5, freezing,
                                facecolor='r', alpha=0.4)
        event_plot.invert_yaxis()

    plt.axis('tight')
    plt.xlabel('Time (s)')
    plt.ylabel('Cell #')

def plot_ordered_cells_across_days(session_1, session_2, cells,
                                   order=None, dtype='event'):
    if order is None:
        order = range(len(cells))

    # Get mouse name and ensure that you're matching correct animals.
    mouse = session_list[session_1]["Animal"]
    assert mouse == session_list[session_2]["Animal"], \
        "Animal names do not match."

    # Load cell matching map.
    cell_map = load_cellreg_results(mouse)
    map_idx = find_match_map_index([session_1, session_2])

    global_cell_idx = find_cell_in_map(cell_map, map_idx[0], cells)
    local_s2_cells = cell_map[global_cell_idx, map_idx[1]]
    sorted_s2_cells = local_s2_cells[order]

    sorted_s2_cells = sorted_s2_cells[sorted_s2_cells > -1]
    n_cells = len(sorted_s2_cells)

    plot_ordered_cells(session_2, sorted_s2_cells, range(n_cells),
                       dtype=dtype)


def plot_sequences(session_index, dtype='event',
                   mat_file='seqNMF_results.mat'):
    # Load data.
    seqNMF_results = seqNMF(session_index, mat_file=mat_file)

    # Get significantly contributing cells.
    significant_cells = seqNMF_results.get_sequential_cells()

    # Determine the order of the cells.
    sequence_orders = seqNMF_results.get_sequence_order()

    # Plot.
    for cells, order in zip(significant_cells, sequence_orders):
        plot_ordered_cells(session_index, cells, order, dtype=dtype)

def plot_sequences_across_days(session_1, session_2, dtype='event',
                               mat_file='seqNMF_results.mat'):
    # Get mouse name and ensure that you're matching correct animals.
    mouse = session_list[session_1]["Animal"]
    assert mouse == session_list[session_2]["Animal"], \
        "Animal names do not match."

    # Get sequence information from session 1 and traces from session 2.
    s1_sequence = seqNMF(session_1, mat_file=mat_file)

    if dtype == 'trace':
        data, t = ca_traces.load_traces(session_2)
        data = zscore(data, axis=1)
    elif dtype == 'event':
        data,t  = ca_events.load_events(session_2)
        data[data > 0] = 1
    else:
        raise ValueError('Wrong dtype value.')

    session = ff.load_session(session_2)

    # Trim and process trace information and freezing/velocity.
    traces = d_pp.trim_session(data, session.mouse_in_cage)

    if dtype == 'event':
        events = []
        for cell in traces:
            events.append(np.where(cell)[0]/20)

    # Get statistically significant cells in sequence and their rank.
    significant_cells = s1_sequence.get_sequential_cells()
    sequence_orders = s1_sequence.get_sequence_order()

    # Load cell matching map.
    cell_map = load_cellreg_results(mouse)
    map_idx = find_match_map_index([session_1, session_2])

    # For each sequence from seqNMF...
    for cells, order in zip(significant_cells, sequence_orders):
        # Get cell index and sort them.
        global_cell_idx = find_cell_in_map(cell_map, map_idx[0], cells)
        local_s2_cells = cell_map[global_cell_idx, map_idx[1]]
        sorted_s2_cells = local_s2_cells[order]

        # Delete cells that didn't match.
        sorted_s2_cells = sorted_s2_cells[sorted_s2_cells > -1]
        n_cells = len(sorted_s2_cells)

        plot_ordered_cells(session_2, sorted_s2_cells, range(n_cells),
                           dtype=dtype)


if __name__ == '__main__':
    from single_cell_analyses.freezing_selectivity import FreezingCellFilter
    cells, p = FreezingCellFilter(5, 'trace').get_freezing_cells()
    plot_ordered_cells(5, cells)