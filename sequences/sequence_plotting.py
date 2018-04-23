from sequences.seqNMF_data import seqNMF
from session_directory import load_session_list
import numpy as np
import matplotlib.pyplot as plt
import calcium_traces as ca_traces
from scipy.stats import zscore
import data_preprocessing as d_pp
import ff_video_fixer as ff
from matplotlib import gridspec

session_list = load_session_list()

def plot_sequences(session_index, zero_time=True):
    # Load data.
    seqNMF_results = seqNMF(session_index, ['H','W','XC','thres'])
    session = ff.load_session(session_index)
    traces, t = ca_traces.load_traces(session_index)

    # Trim all the time series.
    traces = d_pp.trim_session(traces, session.mouse_in_cage)
    t = d_pp.trim_session(t, session.mouse_in_cage)
    v = d_pp.trim_session(session.imaging_v, session.mouse_in_cage)
    freezing = d_pp.trim_session(session.imaging_freezing,
                                 session.mouse_in_cage)

    # Can either zero the time so start of the session is when mouse
    # enters chamber or don't zero the time in which case teh start of
    # the session is when recording begins.
    if zero_time:
        t -= min(t)

    # Normalize.
    traces = zscore(traces, axis=1)

    # Dump results.
    XC, H, W, thres = seqNMF_results.data['XC'], \
                      seqNMF_results.data['H'], \
                      seqNMF_results.data['W'], \
                      seqNMF_results.data['thres']

    # Get significantly contributing cells.
    significant_cells = np.where((XC > thres.T).any(axis=1))[0]
    n_significant_cells = len(significant_cells)

    # Determine the order of the cells.
    W = np.squeeze(W[significant_cells,1,:])
    peaks = np.argmax(W, axis=1).astype(int)
    order = np.argsort(peaks).astype(int)

    # Plot.
    gs = gridspec.GridSpec(4, 1)
    fig = plt.figure()

    # Velocity plot.
    velocity_plot = fig.add_subplot(gs[0,:])
    plt.plot(t, v, 'k')
    velocity_plot.set_ylabel('Velocity',color='k')

    # Freezing plot.
    freezing_plot = velocity_plot.twinx()
    plt.plot(t, freezing, 'r')
    freezing_plot.set_ylabel('Freezing',color='r')
    plt.setp(velocity_plot.get_xticklabels(), visible=False)

    # Raster plot.
    fig.add_subplot(gs[1:,0], sharex=velocity_plot)
    plt.imshow(traces[significant_cells[order]],
               extent=[t[0], t[-1], n_significant_cells, 0])
    plt.axis('tight')
    plt.xlabel('Time (s)')
    plt.ylabel('Cell #')

    pass

if __name__ == '__main__':
    plot_sequences(4)