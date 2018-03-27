from session_directory import load_session_list
import calcium_traces as ca_traces
import numpy as np
from helper_functions import find_closest
import data_preprocessing as d_pp
from plot_helper import ScrollPlot, neuron_number_title
import plot_functions as plot_funcs
from scipy.stats import zscore

session_list = load_session_list()


class ShockObj:
    def __init__(self, session_index):
        self.frame_rate = 20
        self.footshock_times = [698, 778, 858, 938]
        traces, self.accepted, self.t = \
            ca_traces.load_traces(session_index)
        self.traces = zscore(traces, axis=0)
        self.align_neurons()

    def align_trace(self, window, trace):
        n_back = window[0]
        n_front = window[1]
        aligned_trace = np.zeros([len(self.footshock_times),
                                  abs(n_back) + n_front])
        for shock_number, timestamp in enumerate(self.footshock_times):
            _, shock_idx = find_closest(self.t, timestamp)
            idx = np.arange(shock_idx + n_back, shock_idx + n_front)

            aligned_trace[shock_number, :] = trace[idx]

        return aligned_trace

    def align_neurons(self, window=[-1,20]):
        window.sort()
        self.window = np.asarray(window)

        assert len(self.window) is 2, "Window length must be 2."
        assert any(self.window < 0), "One input must be negative."
        assert any(self.window > 0), "One input must be positive."

        window = self.frame_rate * self.window
        self.aligned_traces = []
        for this_trace in self.traces:
            trace = self.align_trace(window, this_trace)

            self.aligned_traces.append(trace)

    def plot_traces(self, neurons='all'):
        if neurons == 'all':
            neurons = d_pp.filter_good_neurons(self.accepted)

        selected_traces = [self.aligned_traces[n] for n in neurons]

        titles = neuron_number_title(neurons)
        ref_time = np.arange(self.window[0], self.window[1],
                                 1 / self.frame_rate)
        f = ScrollPlot(plot_funcs.plot_multiple_traces,
                       t=ref_time,
                       traces=selected_traces,
                       xlabel='Time from shock (s)',
                       ylabel='Calcium activity (s.d.)',
                       titles=titles)


        return f


if __name__ == '__main__':
    S = ShockObj(0)
    S.plot_traces()
