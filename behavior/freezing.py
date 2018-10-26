from session_directory import load_session_list, get_session
from microscoPy_load.ff_video_fixer import load_session as load_ff
import data_preprocessing as d_pp
from microscoPy_load import calcium_events as ca_events
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import nan
import itertools

session_list = load_session_list()

def compute_percent_freezing(session_index, bin_length=60, plot_flag=False):
    session = load_ff(session_index)
    _, t = ca_events.load_events(session_index)

    t = d_pp.trim_session(t, session.mouse_in_cage)
    freezing = d_pp.trim_session(session.imaging_freezing,
                                 session.mouse_in_cage)

    samples_per_bin = bin_length * 20
    bins = d_pp.make_bins(t, samples_per_bin)

    # Elapsed time in minutes.
    t -= t.min()
    t /= 60
    t_binned = (bins-bins.min()) / 1200
    t_binned = np.append(t_binned, t.max())

    binned_freezing = d_pp.bin_time_series(freezing, bins)

    percent_freezing = np.zeros((len(binned_freezing)))
    for epoch_number, epoch in enumerate(binned_freezing):
        percent_freezing[epoch_number] = np.sum(epoch) / len(epoch)

    if plot_flag:
        plt.figure()
        plt.plot(t_binned, percent_freezing)

    return percent_freezing, t_binned


def plot_freezing_percentages(mouse, bin_length=60, plot=True):
    """
    Plot freezing percentages over time and sessions for a mouse. Will
    automatically truncate EXT sessions to 30 min and Recall sessions to
    8 min.

    Parameters
    ---
    mouse: str, mouse name.
    bin_length: scalar, size of bin (in seconds) to compute freezing %.
    plot: logical, plot or not.

    Returns
    ---
    freezing: array, freezing percentages over time.
    boundaries: array, indices demarcating different sessions.
    freezing_p: dict, untruncated freezing percentages.
    """

    # Session names.
    context_1 = ('E1_1', 'E2_1', 'RE_1')
    context_2 = ('E1_2', 'E2_2', 'RE_2')
    session_stages = ('FC','E1_1','E2_1','RE_1',
                           'E1_2','E2_2','RE_2')

    # Bin size in minutes.
    bin_size_min = bin_length/60

    # Get the sessions.
    session_idx, session_stages = get_session(mouse, session_stages)
    if 'E1_2' in session_stages:
        n_contexts = 2
    else:
        n_contexts = 1

    # Get freezing percentages in a dict.
    freezing_p = {}
    for session_number, session in zip(session_idx, session_stages):
        freezing_p[session], t = compute_percent_freezing(session_number,
                                                          bin_length=bin_length)

    # Compute the final size of the truncated array in indices.
    ext_size = 30/bin_size_min
    rc_size = 8/bin_size_min
    limits = [0, ext_size, ext_size, rc_size]
    boundaries = np.cumsum(limits)

    # Preallocate for truncated freezing percentages.
    freezing = nan((n_contexts, boundaries[-1]))

    # For each session, truncate. First row is context 1, second row is
    # context 2.
    for session_1, session_2, limit, start, end in \
            zip(context_1,
                context_2,
                limits[1:],
                boundaries[:-1],
                boundaries[1:]):
        freezing[0, start:end] = freezing_p[session_1][0:limit]

        if n_contexts is 2:
            freezing[1, start:end] = freezing_p[session_2][0:limit]

    # Plot.
    if plot:
        f, ax = plt.subplots(1,1)
        for context in freezing:
            plt.plot(context)

        for boundary in boundaries:
            ax.axvline(x=boundary)

        ax.set_xticks([0,
                       ext_size,
                       ext_size*2,
                       ext_size*2 + rc_size])
        ax.set_xticklabels([0, 30, 30, 8])
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Freezing')

    return freezing, boundaries, freezing_p



def plot_freezing(mouse, stages_tuple):
    session_index, _ = get_session(mouse, stages_tuple)
    session = load_ff(session_index)

    t = d_pp.trim_session(session.imaging_t, session.mouse_in_cage)
    t-= min(t)
    v = d_pp.trim_session(session.imaging_v, session.mouse_in_cage)
    freezing = d_pp.trim_session(session.imaging_freezing,
                                 session.mouse_in_cage)

    fig, velocity_plot = plt.subplots()
    plt.plot(t, v, 'k')
    velocity_plot.set_ylabel('Velocity', color='k')
    velocity_plot.fill_between(t, 0, v.max(), freezing,
                               facecolor='r')


if __name__ == '__main__':
    plot_freezing_percentages('Helene', bin_length=30)
    plt.show()
    pass