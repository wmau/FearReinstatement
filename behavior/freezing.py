from session_directory import load_session_list, get_session
from microscoPy_load.ff_video_fixer import load_session as load_ff
import data_preprocessing as d_pp
from microscoPy_load import calcium_events as ca_events
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import nan

session_list = load_session_list()

def compute_percent_freezing(session_index, bin_length=60, plot=False):
    """
    Computes freezing percentage for a session.

    Parameters
    ---
    session_index: scalar, session number.
    bin_length: scalar, size of bin (in seconds).
    plot: logical, plot or not.

    Returns
    ---
    percent_freezing: array, freezing percentages from binned session.
    t_binned: array, spaced timestamps (in seconds).
    """

    # Load session.
    session = load_ff(session_index)
    _, t = ca_events.load_events(session_index)

    # Trim session.
    t = d_pp.trim_session(t, session.mouse_in_cage)
    freezing = d_pp.trim_session(session.imaging_freezing,
                                 session.mouse_in_cage)

    # Get bins.
    samples_per_bin = bin_length * 20
    bins = d_pp.make_bins(t, samples_per_bin, axis=0)

    # Elapsed time in minutes.
    t -= t.min()
    t /= 60
    t_binned = (bins-bins.min()) / 1200
    t_binned = np.append(t_binned, t.max())

    # Freezing vector segmented into bins.
    binned_freezing = d_pp.bin_time_series(freezing, bins)

    # Compute freezing percentage for each bin.
    percent_freezing = np.zeros((len(binned_freezing)))
    for epoch_number, epoch in enumerate(binned_freezing):
        percent_freezing[epoch_number] = np.sum(epoch) / len(epoch)

    # Turn into percentage.
    percent_freezing = percent_freezing*100

    # Plot.
    if plot:
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
    freezing_untruncated: dict, untruncated freezing percentages.
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
    freezing_untruncated = {}
    for session_number, session in zip(session_idx, session_stages):
        freezing_untruncated[session], t = compute_percent_freezing(session_number,
                                                          bin_length=bin_length)

    # Compute the final size of the truncated array in indices.
    cfc_size = int(8/bin_size_min)
    ext_size = int(30/bin_size_min)
    limits = [0, cfc_size, ext_size, ext_size, cfc_size]
    boundaries = np.cumsum(limits).astype(int)

    # Preallocate for truncated freezing percentages.
    freezing = nan((n_contexts, boundaries[-1]))

    # First, get the fear conditioning session freezing percentages.
    freezing[0, boundaries[0]:boundaries[1]] = freezing_untruncated['FC'][:limits[1]]

    # For each session, truncate. First row is context 1, second row is
    # context 2.
    for session_1, session_2, limit, start, end in \
            zip(context_1,
                context_2,
                limits[2:],
                boundaries[1:-1],
                boundaries[2:]):
        try:
            try:
                freezing[0, start:end] = \
                    freezing_untruncated[session_1][:limit]
            except:
                end = start + len(freezing_untruncated[session_1])
                freezing[0, start:end] = \
                    freezing_untruncated[session_1]
        except:
            pass

        if n_contexts is 2:
            try:
                freezing[1, start:end] = \
                    freezing_untruncated[session_2][:limit]
            except:
                end = start + len(freezing_untruncated[session_2])
                freezing[1, start:end] = \
                    freezing_untruncated[session_2]

    tick_locations = [0,
                      cfc_size,
                      cfc_size + ext_size,
                      cfc_size + ext_size*2,
                      cfc_size + ext_size*2 + cfc_size]

    # Plot.
    if plot:
        f, ax = plt.subplots(1,1)
        for context in freezing:
            ax.plot(context)

        for boundary in boundaries:
            ax.axvline(x=boundary)

        ax.set_xticks(tick_locations)
        ax.set_xticklabels([0, 8, 30, 30, 8])
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Freezing')

    return freezing, freezing_untruncated, boundaries, tick_locations



def plot_freezing(mouse, stage):
    """
    Plots the velocity time series and highlight freezing epochs.

    Parameters
    ---
    mouse: str, mouse name.
    stage: str, experiment stage.

    Returns
    ---
    Plots the velocity and freezing.
    """

    # Get session.
    session_index, _ = get_session(mouse, stage)
    session = load_ff(session_index)

    # Trim session and zero timestamps.
    t = d_pp.trim_session(session.imaging_t, session.mouse_in_cage)
    t-= min(t)
    v = d_pp.trim_session(session.imaging_v, session.mouse_in_cage)
    freezing = d_pp.trim_session(session.imaging_freezing,
                                 session.mouse_in_cage)

    # Plot.
    fig, velocity_plot = plt.subplots()
    plt.plot(t, v, 'k')
    velocity_plot.set_ylabel('Velocity', color='k')
    velocity_plot.fill_between(t, 0, v.max(), freezing,
                               facecolor='r')


if __name__ == '__main__':
    plot_freezing('Skoll','RE_2')