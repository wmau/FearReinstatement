from session_directory import load_session_list, get_session
from microscoPy_load.ff_video_fixer import load_session as load_ff
import data_preprocessing as d_pp
from microscoPy_load import calcium_events as ca_events
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import nan

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

def plot_freezing_percentages(mouse, bin_length=60):
    session_stages = ('FC','E1_1','E2_1','RE_1',
                      'E1_2','E2_2','RE_2')
    subplot_number = [0, 1, 2, 3, 1, 2, 3]

    session_idx, _ = get_session(mouse, session_stages)

    f, ax = plt.subplots(1,4,sharey=True, sharex=True, figsize=(6,2))
    titles = ['Fear conditioning', 'Ext1', 'Ext2', 'Recall']

    for i, session in zip(subplot_number, session_idx):
        p, t = compute_percent_freezing(session,bin_length=bin_length)

        ax[i].plot(t, p)
        ax[i].set_title(titles[i])

    ax[0].set_ylabel('% freezing')

    return f, ax

def plot_freezing_percentages2(mouse, bin_length=60):
    context_1 = ('E1_1', 'E2_1', 'RE_1')
    context_2 = ('E1_2', 'E2_2', 'RE_2')
    session_stages = ('FC','E1_1','E2_1','RE_1',
                           'E1_2','E2_2','RE_2')
    slice_size_min = bin_length/60

    session_idx, session_stages = get_session(mouse, session_stages)

    freezing_p = {}
    for session_number, session in zip(session_idx, session_stages):
        freezing_p[session], t = compute_percent_freezing(session_number,
                                                          bin_length=bin_length)


    context_1_freezing, context_2_freezing = [], []
    limits = [30/slice_size_min, 30/slice_size_min, 8/slice_size_min]
    boundaries = np.cumsum(limits)
    if 'E1_2' in session_stages:
        for session_1, session_2, limit in \
                zip(context_1, context_2, limits):
            context_1_freezing.append(freezing_p[session_1][0:limit])
            context_2_freezing.append(freezing_p[session_2][0:limit])


    pass



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
    plot_freezing_percentages2('Helene', bin_length=30)
    plt.show()
    pass