import data_preprocessing as d_pp
import calcium_traces as ca_traces
import calcium_events as ca_events
from scipy.spatial.distance import pdist
from session_directory import load_session_list, find_mouse_sessions
import numpy as np

session_list = load_session_list()

def mdist_all(mouse, bin_length=1):
    sessions, _ = find_mouse_sessions(mouse)
    samples_per_bin = 20*bin_length

    neural_data, days, t, freezing = d_pp.concatenate_sessions(sessions)

    bins = d_pp.make_bins(t,samples_per_bin=samples_per_bin)
    neural_data = d_pp.bin_time_series(neural_data, bins)

    X = np.mean(np.asarray(neural_data[0:-1]), axis=2)
    X = np.append(X, np.mean(neural_data[-1], axis=1)[None,:], axis=0)
    Y = pdist(X, metric='mahalanobis')
    pass

if __name__ == '__main__':
    mdist_all('Kerberos')