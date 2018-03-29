from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from session_directory import load_session_list
import calcium_traces as ca_traces
import numpy as np
from matplotlib import pyplot as plt
import ff_video_fixer as FF
from mpl_toolkits.mplot3d import Axes3D
import data_preprocessing as d_pp
import calcium_events as ca_events
from scipy.stats import zscore

session_list = load_session_list()

def PCA_session(session_index, bin_length=2):
    session = FF.load_session(session_index)

    # Get accepted neurons.
    traces, accepted, t = ca_traces.load_traces(session_index)
    traces = zscore(traces, axis=0)
    #traces = ca_events.make_event_matrix(session_index)       # If you want events (not traces).
    scaler = StandardScaler()
    traces = scaler.fit_transform(traces)
    neurons = d_pp.filter_good_neurons(accepted)
    n_neurons = len(neurons)

    # Trim the traces to only include instances where mouse is in the chamber.
    t = d_pp.trim_session(t,session.mouse_in_cage)
    traces = d_pp.trim_session(traces,session.mouse_in_cage)
    freezing = d_pp.trim_session(session.imaging_freezing,session.mouse_in_cage)

    samples_per_bin = bin_length * 20
    bins = d_pp.make_bins(t,samples_per_bin)
    n_samples = len(bins)+1

    X = np.zeros([n_samples,n_neurons])
    for n,this_neuron in enumerate(neurons):
        binned_activity = d_pp.bin_time_series(traces[this_neuron],bins)
        avg_activity = [np.mean(chunk) for chunk in binned_activity]

        X[:,n] = avg_activity

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
    s = ax.scatter(Y[:,0], Y[:,1], Y[:,2], c=binned_freezing)
    fig.show()

    pass

if __name__ == '__main__':
    PCA_session(14)