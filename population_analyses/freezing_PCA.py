from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from session_directory import load_session_list
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import data_preprocessing as d_pp
from microscoPy_load import calcium_events as ca_events, calcium_traces as ca_traces, ff_video_fixer as FF
from scipy.stats import zscore, mode
from session_directory import get_session

session_list = load_session_list()


def PCA_session(session_index, bin_length=5, dtype='traces'):
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

def PCA_concatenated_sessions(mouse, bin_length=10, dtype='traces',
                              global_cell_idx=None, plot_flag=True,
                              ax=None):
    session_types = ('FC',
                     'E1_1',
                     'E2_1',
                     'RE_1',
                     )
    sessions, _ = get_session(mouse, session_types)

    neural_data, days, t, freezing = \
        d_pp.concatenate_sessions(sessions,
                                  dtype=dtype,
                                  global_cell_idx=global_cell_idx)

    #neural_data = zscore(neural_data,axis=1)
    bins = d_pp.make_bins(t, bin_length*20)
    neural_data = d_pp.bin_time_series(neural_data, bins)

    X = np.nanmean(np.asarray(neural_data[0:-1]), axis=2)
    X = np.append(X, np.nanmean(neural_data[-1], axis=1)[None, :],
                  axis=0)
    X.mask = np.ma.nomask
    good = np.where(~np.isnan(X))[0]
    X = X[good,:]

    # Bin freezing vector.
    binned_freezing = d_pp.bin_time_series(freezing, bins)
    freezing = np.asarray([i.any() for i in binned_freezing])
    freezing = freezing[good]

    binned_days = d_pp.bin_time_series(days, bins)
    day_id, _ = mode(np.asarray(binned_days[0:-1]), axis=1)
    day_id = np.append(day_id, mode(binned_days[-1])[0])
    day_id = day_id[good]


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
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(Y[freezing, 0], Y[freezing, 1],
                   c=day_id[freezing], s=10,
                   marker='P', cmap='summer',alpha=0.2)
        ax.scatter(Y[~freezing, 0], Y[~freezing, 1],
                   c=day_id[~freezing], s=10,
                   marker='.', cmap='summer',alpha=0.2)

        freezing_center = np.zeros((len(sessions),2))
        nonfreezing_center = np.zeros_like(freezing_center)
        center = np.zeros_like(freezing_center)
        unique_days = np.unique(day_id)
        for i, day in enumerate(unique_days):
            points = Y[(day_id == day) & (freezing)]
            freezing_center[i] = np.mean(points, axis=0)

            points = Y[(day_id == day) & (~freezing)]
            nonfreezing_center[i] = np.mean(points, axis=0)

            points = Y[(day_id == day)]
            center[i] = np.mean(points, axis=0)


        # ax.scatter(freezing_center[:,0], freezing_center[:,1],
        #            marker='P', c=unique_days, cmap='summer', s=200,
        #            edgecolors='k', linewidth=2)
        # ax.scatter(nonfreezing_center[:,0], nonfreezing_center[:,1],
        #            marker='.', c=unique_days, cmap='summer', s=200,
        #            edgecolors='k', linewidth=2)
        ax.scatter(center[:,0], center[:,1], marker='o', c=unique_days,
                   cmap='summer', s=30, edgecolors='k', linewidth=2)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')


    return Y, freezing


if __name__ == '__main__':
    mice = ('Janus',
            'Kepler',
            'Mundilfari',
            'Aegir',
            )

    f, ax = plt.subplots(2,2,figsize=(6,6))
    ax = ax.ravel()
    for i, mouse in enumerate(mice):
        PCA_concatenated_sessions(mouse, ax=ax[i])

    f.show()
    pass