from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from session_directory import load_session_list
import calcium_traces as ca_traces
import numpy as np
from matplotlib import pyplot as plt
import ff_video_fixer as FF
from mpl_toolkits.mplot3d import Axes3D
import data_preprocessing as d_pp
import calcium_events as ca_events
from scipy.stats import zscore, mode
from session_directory import find_mouse_sessions

session_list = load_session_list()


def PCA_session(session_index, bin_length=1, dtype='traces'):
    session = FF.load_session(session_index)

    # Get accepted neurons.
    if dtype == 'traces':
        traces, t = ca_traces.load_traces(session_index)
        traces = zscore(traces, axis=1)

        scaler = StandardScaler()
        traces = scaler.fit_transform(traces)
    elif dtype == 'events':
        traces, t = ca_events.load_events(session_index)
    else:
        raise ValueError('Wrong dtype input.')

    n_neurons = len(traces)

    # Trim the traces to only include instances where mouse is in the chamber.
    t = d_pp.trim_session(t, session.mouse_in_cage)
    traces = d_pp.trim_session(traces, session.mouse_in_cage)
    freezing = d_pp.trim_session(session.imaging_freezing, session.mouse_in_cage)

    samples_per_bin = bin_length * 20
    bins = d_pp.make_bins(t, samples_per_bin)
    n_samples = len(bins) + 1

    X = np.zeros([n_samples, n_neurons])
    for n, trace in enumerate(traces):
        binned_activity = d_pp.bin_time_series(trace, bins)
        avg_activity = [np.mean(chunk) for chunk in binned_activity]

        X[:, n] = avg_activity

    binned_freezing = d_pp.bin_time_series(freezing, bins)
    binned_freezing = [i.any() for i in binned_freezing]

    # lda = LinearDiscriminantAnalysis(solver='eigen',n_components=2,shrinkage='auto')
    # lda.fit(X,binned_freezing)
    # Y = lda.transform(X)

    pca = PCA(n_components=3)
    pca.fit(X)
    Y = pca.transform(X)

    fig = plt.figure()
    ax = Axes3D(fig)
    s = ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=binned_freezing)
    fig.show()

def PCA_concatenated_sessions(mouse, bin_length=1, dtype='traces',
                              global_cell_idx=None, plot_flag=True):
    sessions, _ = find_mouse_sessions(mouse)

    sessions = sessions[[0, 1, 2, 4]]

    neural_data, days, t, freezing = \
        d_pp.concatenate_sessions(sessions,
                                  dtype=dtype,
                                  global_cell_idx=global_cell_idx)

    #neural_data = zscore(neural_data,axis=1)
    bins = d_pp.make_bins(t, bin_length*20)
    neural_data = d_pp.bin_time_series(neural_data, bins)

    X = np.mean(np.asarray(neural_data[0:-1]), axis=2)
    X = np.append(X, np.mean(neural_data[-1], axis=1)[None, :],
                  axis=0)

    # Bin freezing vector.
    binned_freezing = d_pp.bin_time_series(freezing, bins)
    freezing = np.asarray([i.any() for i in binned_freezing])

    binned_days = d_pp.bin_time_series(days, bins)
    day_id, _ = mode(np.asarray(binned_days[0:-1]), axis=1)
    day_id = np.append(day_id, mode(binned_days[-1])[0])


    # pca = PCA(n_components=3)
    # pca.fit(X)
    # Y = pca.transform(X)
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    #
    #
    pca = PCA(n_components=2)
    pca.fit(X)
    Y = pca.transform(X)

    if plot_flag:
        fig, ax = plt.subplots()
        ax.scatter(Y[freezing, 0], Y[freezing, 1],
                   c=day_id[freezing], s=40,
                   marker='P', cmap='summer',alpha=0.2)
        ax.scatter(Y[~freezing, 0], Y[~freezing, 1],
                   c=day_id[~freezing], s=40,
                   marker='.', cmap='summer',alpha=0.2)

        freezing_center = np.zeros((len(sessions),2))
        nonfreezing_center = np.zeros((len(sessions),2))
        unique_days = np.unique(day_id)
        for i, day in enumerate(unique_days):
            points = Y[(day_id == day) & (freezing)]
            freezing_center[i] = np.mean(points, axis=0)

            points = Y[(day_id == day) & (~freezing)]
            nonfreezing_center[i] = np.mean(points, axis=0)

        ax.scatter(freezing_center[:,0], freezing_center[:,1],
                   marker='P', c=unique_days, cmap='summer', s=200,
                   edgecolors='k', linewidth=2)
        ax.scatter(nonfreezing_center[:,0], nonfreezing_center[:,0],
                   marker='.', c=unique_days, cmap='summer', s=200,
                   edgecolors='k', linewidth=2)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        fig.show()

    return Y, freezing



    pass


if __name__ == '__main__':
    PCA_concatenated_sessions('Kerberos')
