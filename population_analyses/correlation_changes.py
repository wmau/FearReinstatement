import calcium_traces as ca_traces
from session_directory import load_session_list
import data_preprocessing as d_pp
import ff_video_fixer as ff
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.cluster import k_means
from scipy.stats import zscore
import seaborn as sns
import cell_reg
import helper_functions as helper
from scipy.spatial import distance

session_list = load_session_list()

def do_sliced_correlation(session_index, bin_length, neurons):
    traces, t, bins = d_pp.trim_and_bin(session_index, neurons=neurons,
                                        samples_per_bin=bin_length * 20)

    n_neurons = len(neurons)
    n_samples = len(bins) + 1

    corr_matrices = np.zeros((n_samples, n_neurons, n_neurons))
    for i, sample in enumerate(traces):
        corr_matrices[i] = np.corrcoef(sample)

    return corr_matrices, t

def cluster_corr_matrices(corr_matrices, cluster_here, n_clusters, t):
    for idx, time_slice in enumerate(t):
        if cluster_here in time_slice:
            break

    _, cluster_idx, _ = k_means(corr_matrices[idx], n_clusters)
    order = np.argsort(cluster_idx)

    return cluster_idx, order

def plot_corr_matrices_over_time(session_idx, bin_length=10,
                                 neurons=None, cluster_here=0,
                                 n_clusters=5):

    # Do correlations on time slices.
    corr_matrices,t  = do_sliced_correlation(session_idx,
                                             bin_length=bin_length,
                                             neurons=neurons)

    if n_clusters > 0:
        _, order = cluster_corr_matrices(corr_matrices, cluster_here,
                                         n_clusters, t)
    else:
        order = range(corr_matrices.shape[1])

    plot_corr_matrices(corr_matrices, order)

def plot_corr_matrices(corr_matrices, order, clusters=None):
    n_time_slices = len(corr_matrices)
    n_cols = int(np.floor(n_time_slices**(.5)))
    n_rows = int(np.ceil(n_time_slices / n_cols))
    gs = gridspec.GridSpec(n_rows, n_cols, wspace = 0.025, hspace=0.05)
    fig = plt.figure(figsize=(n_cols, n_rows))

    # Plot correlation matrices.
    for i, matrix in enumerate(corr_matrices):
        ax = fig.add_subplot(gs[i])
        ax.set_yticks([])
        ax.set_xticks([])
        plt.imshow(matrix[:,order][order],cmap='bwr',vmin=-1,vmax=1)

        # Plot clusters.
        if clusters is not None:
            clusters.sort()
            cluster_boundaries = np.where(np.diff(clusters))[0]
            for boundary in cluster_boundaries:
                plt.axvline(x=boundary, c='k', lw=1)
                plt.axhline(y=boundary, c='k', lw=1)


def compute_matrices_across_days(session_1, session_2, bin_length=10,
                                 neurons=None, n_clusters=5,
                                 cluster_here=0):
    # Load cell map.
    mouse = session_list[session_1]["Animal"]
    assert mouse == session_list[session_2]["Animal"], "Mice don't match."
    match_map = cell_reg.load_cellreg_results(mouse)
    map_idx = cell_reg.find_match_map_index((session_1, session_2))

    # If neurons are specified, narrow down the list.
    if neurons is not None:
        global_cell_idx = cell_reg.find_cell_in_map(match_map,
                                                    map_idx[0],
                                                    neurons)
        match_map = match_map[global_cell_idx]

    # Only take cells that persisted across the sessions.
    match_map = cell_reg.trim_match_map(match_map, map_idx)
    neurons_ref = match_map[:, 0]
    neurons_test = match_map[:, 1]

    # Do correlations on time slices.
    corr_matrices_1, t = do_sliced_correlation(session_1, bin_length,
                                               neurons_ref)
    corr_matrices_2, _ = do_sliced_correlation(session_2, bin_length,
                                               neurons_test)

    # Cluster correlation matrix at specified time slice.
    if n_clusters > 0:
        clusters, order = cluster_corr_matrices(corr_matrices_1,
                                                cluster_here,
                                                n_clusters, t)
    else:
        order = range(len(neurons_ref))
        clusters = None

    return corr_matrices_1, corr_matrices_2, order, clusters

def plot_matrices_across_days(session_1, session_2, bin_length=10,
                              neurons=None, n_clusters=5, cluster_here=0):

    corr_matrices_1, corr_matrices_2, order, clusters = \
        compute_matrices_across_days(session_1, session_2,
                                     bin_length=bin_length,
                                     neurons=neurons,
                                     n_clusters=n_clusters,
                                     cluster_here=cluster_here)

    plot_corr_matrices(corr_matrices_1, order, clusters=clusters)
    plot_corr_matrices(corr_matrices_2, order, clusters=clusters)


def cosine_distance_between_matrices(session_1, session_2, bin_length=10,
                                     neurons=None, ref_time=0):
    corr_matrices_1, corr_matrices_2, order, clusters = \
        compute_matrices_across_days(session_1, session_2,
                                     bin_length=bin_length,
                                     neurons=neurons, n_clusters=0)

    _, t, _ = d_pp.trim_and_bin(session_1, samples_per_bin=bin_length*20)
    for idx, time_slice in enumerate(t):
        if ref_time in time_slice:
            break

    reference_matrix = corr_matrices_1[idx].flatten()

    distances = np.zeros(len(corr_matrices_2))
    for i, matrix in enumerate(corr_matrices_2):
        distances[i] = distance.cosine(reference_matrix, matrix.flatten())

    plt.figure()
    plt.plot(distances)

    return distances

def pairwise_correlate_traces(session_index, neurons=None):
    """
    Perform pairwise correlations between all (specified) cells in
    a session.

    Parameters
    ---
    session_index: scalar
    neurons: (N,) array-like

    Returns
    ---
    corr_matrix: (N,N) matrix containing correlation coefficients
    """
    # Load data and trim.
    traces, t = d_pp.load_and_trim(session_index, neurons=neurons)
    corr_matrix = np.corrcoef(traces)

    return corr_matrix


def cosine_distance_between_days(session_1, session_2, neurons=None):
    # Load cell map.
    mouse = session_list[session_1]["Animal"]
    assert mouse == session_list[session_2]["Animal"], "Mice don't match."
    match_map = cell_reg.load_cellreg_results(mouse)
    map_idx = cell_reg.find_match_map_index((session_1, session_2))

    # If neurons are specified, narrow down the list.
    if neurons is not None:
        global_cell_idx = cell_reg.find_cell_in_map(match_map,
                                                    map_idx[0],
                                                    neurons)
        match_map = match_map[global_cell_idx]

    # Only take cells that persisted across the sessions.
    match_map = cell_reg.trim_match_map(match_map, map_idx)
    neurons_ref = match_map[:, 0]
    neurons_test = match_map[:, 1]

    corr_matrix_1 = pairwise_correlate_traces(session_1, neurons_ref)
    corr_matrix_2 = pairwise_correlate_traces(session_2, neurons_test)

    d = distance.cosine(corr_matrix_1.flatten(), corr_matrix_2.flatten())

    return d

if __name__ == '__main__':
    # cell_map = cell_reg.load_cellreg_results('Fenrir')
    # cell_map = cell_reg.trim_match_map(cell_map, [10, 11, 12, 14])
    # neurons = cell_map[:, 0]
    # cosine_distance_between_matrices(10,11,time_slice=-2,neurons=neurons)
    # plot_matrices_across_days(10,11,cluster_here=698, n_clusters=10)

    s1 = [0, 5, 10, 15]
    s2 = [[1, 2, 4], [6, 7, 9], [11, 12, 14], [16, 17, 19]]
    all_sessions = [[0, 1, 2, 4], [5, 6, 7, 9], [10, 11, 12, 14],
                    [15, 16, 17, 19]]

    d = np.zeros((4,3))
    for i, fc in enumerate(s1):
        match_map = cell_reg.load_cellreg_results(session_list[fc]['Animal'])
        trimmed = cell_reg.trim_match_map(match_map,all_sessions[i])
        neurons = trimmed[:,0]
        for j, s in enumerate(s2[i]):
            d[i,j] = cosine_distance_between_days(fc, s, neurons=neurons)

    pass