from sklearn.decomposition import PCA
from session_directory import load_session_list
import calcium_traces as ca_traces
import numpy as np
from matplotlib import pyplot as plt
import ff_video_fixer as FF
from mpl_toolkits.mplot3d import Axes3D

session_list = load_session_list()

def bin_session(session_index, bin_length=5):
    session = FF.load_session(session_index)

    traces, accepted, t = ca_traces.load_traces(session_index)
    neurons = [cell_number for cell_number,good in enumerate(accepted) if good]
    n_neurons = len(neurons)

    n_samples_per_bin = bin_length * 20
    bins = np.arange(n_samples_per_bin, len(t), n_samples_per_bin)
    bins = np.append(bins,len(t)-1)
    n_samples = len(bins)+1

    X = np.zeros([n_samples,n_neurons])
    for n,this_neuron in enumerate(neurons):
        binned_activity = np.split(traces[this_neuron],bins)
        avg_activity = [np.mean(chunk) for chunk in binned_activity]

        X[:,n] = avg_activity

    binned_freezing = np.split(session.imaging_freezing, bins)
    binned_freezing = [i.any() for i in binned_freezing]

    pca = PCA(n_components=3)
    pca.fit(X)
    Y = pca.transform(X)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(Y[:,0],Y[:,1],Y[:,2],c=binned_freezing)

    pass

if __name__ == '__main__':
    bin_session(1)