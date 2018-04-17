import scipy.signal as signal
from pickle import load, dump
import calcium_traces as ca_traces
import numpy as np
from helper_functions import ismember
import itertools

def xcorr_all_neurons(traces, xcorr_window=[-15,15]):
    xcorr_window.sort()
    xcorr_window = np.asarray(xcorr_window)

    n_neurons = len(traces)
    n_samples = len(traces[0])
    x_axis = np.arange(n_samples * 2 - 1)
    lags = (x_axis - (n_samples - 1)) * 0.05

    # Define your window and get its indices.
    window = np.arange(xcorr_window[0], xcorr_window[1])
    _, window_idx = ismember(lags, window)

    best_lags = np.zeros((n_neurons, n_neurons))
    xcorrs = np.zeros((n_neurons,
                       n_neurons,
                       len(window)))

    matrix_indices = np.unravel_index(np.arange(xcorrs.shape[0]**2),
                                      xcorrs.shape[0:2])

    # Get all pairwise combinations.
    combinations = list(itertools.product(traces, traces))
    for i, (trace_1, trace_2) in enumerate(combinations):
        # Perform cross-correlation then normalize.
        xcorr = signal.correlate(trace_1, trace_2)

        # Get the appropriate index.
        row = matrix_indices[0][i]
        col = matrix_indices[1][i]

        # Dump the maximum cross-correlation coefficient
        # and the corresponding time lag in seconds.
        xcorrs[row, col] = xcorr[window_idx]
        best_lags[row, col] = window[np.argmax(xcorr[window_idx])]

    return xcorrs, best_lags, window