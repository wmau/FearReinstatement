from session_directory import load_session_list
import data_preprocessing as d_pp
import calcium_traces as ca_traces
import calcium_events as ca_events
import numpy as np
import ff_video_fixer as ff
import scipy.stats as stats

session_list = load_session_list()

def detect_engram(session_index, neurons='all', bin_length=1):
    traces, accepted, t = ca_traces.load_traces(session_index)
    events = ca_events.make_event_matrix(session_index)
    session = ff.load_session(session_index)

    samples_per_bin = bin_length * 20

    bins = d_pp.make_bins(session.imaging_t,samples_per_bin)
    n_samples = len(bins) + 1

    if neurons == 'all':
        neurons = d_pp.filter_good_neurons(accepted)
    n_neurons = len(neurons)

    binned_events = np.zeros([n_neurons, n_samples])
    for n, this_neuron in enumerate(neurons):
        binned = d_pp.bin_time_series(events[this_neuron], bins)
        binned_events[n, :] = [np.sum(chunk > 0) for chunk in binned]

    binned_in_cage = d_pp.bin_time_series(session.mouse_in_cage, bins)
    binned_in_cage = [i.all() for i in binned_in_cage]
    binned_out_cage = [not i for i in binned_in_cage]

    significant_neurons = []
    for n,this_neuron in enumerate(neurons):
        in_chamber = binned_events[n, binned_in_cage]
        out_chamber = binned_events[n, binned_out_cage]

        _, pval = stats.ranksums(in_chamber, out_chamber)

        if pval < 0.05:
            if np.mean(in_chamber) > np.mean(out_chamber):
                significant_neurons.append(n)

    return significant_neurons

if __name__ == "__main__":
    detect_engram(1)