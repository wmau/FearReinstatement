from session_directory import load_session_list, find_mouse_sessions
from ff_video_fixer import load_session as load_ff
import data_preprocessing as d_pp
import calcium_events as ca_events
import numpy as np
import matplotlib.pyplot as plt

session_list = load_session_list()

def compute_percent_freezing(session_index, bin_length=100, plot_flag=False):
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

def plot_freezing_percentages(mouse, bin_length=100):
    session_idx, _ = find_mouse_sessions(mouse)
    del session_idx[3]

    _, ax = plt.subplots(1,4,sharey=True)
    titles = ['Fear conditioning', 'Ext1', 'Ext2', 'Recall']
    for i, session in enumerate(session_idx):
        p, t = compute_percent_freezing(session,bin_length=bin_length)

        ax[i].plot(t, p)
        ax[i].set_title(titles[i])

    ax[0].set_ylabel('% freezing')

def plot_freezing(session_index):
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
    plot_freezing('Kerberos')