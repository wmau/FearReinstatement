import calcium_traces as ca_traces
from session_directory import load_session_list
import data_preprocessing as d_pp
import ff_video_fixer as ff
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.cluster import k_means

def time_slice_correlation(session_index, bin_length, neurons):

    session = ff.load_session(session_index)

    traces, t = ca_traces.load_traces(session_index)
    if neurons is None:
        neurons = range(len(traces))

    t = d_pp.trim_session(t, session.mouse_in_cage)
    traces = d_pp.trim_session(traces, session.mouse_in_cage)
    n_neurons = len(neurons)

    samples_per_bin = bin_length * 20
    bins = d_pp.make_bins(t, samples_per_bin)
    n_samples = len(bins) + 1

    corr_matrices = np.zeros((n_samples, n_neurons, n_neurons))
    binned_traces = d_pp.bin_time_series(traces, bins)
    for i, sample in enumerate(binned_traces):
        corr_matrices[i] = np.corrcoef(sample)

    return corr_matrices

def cluster_corr_matrices(corr_matrices, time_slice=0, n_clusters=5):
    _, cluster_idx, _ = k_means(corr_matrices[time_slice], n_clusters)
    order = np.argsort(cluster_idx)

    return cluster_idx, order

def plot_corr_matrices(session_idx, bin_length=60, neurons=None,
                       cluster=True, time_slice=0, n_clusters=5):
    corr_matrices = time_slice_correlation(session_idx,
                                           bin_length=bin_length,
                                           neurons=neurons)

    if cluster:
        _, order = cluster_corr_matrices(corr_matrices, time_slice=time_slice,
                                        n_clusters=n_clusters)

        for i in range(11):
            plt.subplot(4,3,i+1)
            plt.imshow(corr_matrices[i,order,:][order])

    else:
        for i in range(11):
            plt.subplot(4,3,i+1)
            plt.imshow(corr_matrices[i])

    pass
if __name__ == '__main__':
    plot_corr_matrices(0, n_clusters=2)