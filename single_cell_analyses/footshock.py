from session_directory import load_session_list
import calcium_traces as ca_traces
import calcium_events as ca_events
import numpy as np
from helper_functions import find_closest, shift_rows, get_longest_run
import data_preprocessing as d_pp
from plot_helper import ScrollPlot, neuron_number_title
import plot_functions as plot_funcs
from scipy.stats import zscore, sem
import matplotlib.pyplot as plt

session_list = load_session_list()


class ShockObj:
    def __init__(self, session_index):
        self.frame_rate = 20
        self.footshock_times = [698, 778, 858, 938]
        traces, self.t = \
            ca_traces.load_traces(session_index)
        self.events = ca_events.make_event_matrix(session_index)
        self.traces = zscore(traces, axis=0)
        self.n_neurons = len(traces)
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

    def align_neurons(self, window=[-1,15]):
        window.sort()
        self.window = np.asarray(window)

        assert len(self.window) is 2, "Window length must be 2."
        assert any(self.window < 0), "One input must be positive."
        assert any(self.window > 0), "One input must be negative."

        window = self.frame_rate * self.window
        self.aligned_traces = []
        self.aligned_events = []
        for this_trace, this_event_trace \
                in zip(self.traces, self.events):
            trace = self.align_trace(window, this_trace)
            event = self.align_trace(window, this_event_trace)

            self.aligned_traces.append(trace)
            self.aligned_events.append(event)

        # for this_event_trace in self.events:
        #     event_trace = self.align_trace(window, this_event_trace)
        #
        #     self.aligned_events.append(event_trace)


    def plot_traces(self, neurons='all'):
        if neurons == 'all':
            neurons = np.arange(len(self.traces))
            selected_traces = self.aligned_traces
        else:
            selected_traces = [self.aligned_traces[n]
                               for n in neurons]

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

    def create_shuffle_tuning_curve(self, B=1000):
        # Preallocate.
        n_neurons = len(self.traces)
        trace_size = self.aligned_traces[0].shape[1]
        self.surrogate = np.zeros((n_neurons, B, trace_size))

        # Create shuffled tuning curves.
        for neuron, trace in enumerate(self.aligned_traces):
            shuffled = np.zeros((B, trace_size))
            for iteration in range(B):
                shuffled[iteration] = np.mean(shift_rows(trace),axis=0)

            self.surrogate[neuron] = shuffled

    def find_modulated_cells(self):
        self.shock_modulated_cells = []
        self.tuning_curves = []
        for trace, shuffled, neuron in \
                zip(self.aligned_traces, self.surrogate,
                    np.arange(self.n_neurons)):
            tuning_curve = np.mean(trace, axis=0)
            self.tuning_curves.append(tuning_curve)

            p_value = np.sum(tuning_curve < shuffled,axis=0)/shuffled.shape[0]

            if any(p_value < 0.01):
                if get_longest_run(p_value < 0.01) > 5:
                    self.shock_modulated_cells.append(neuron)


if __name__ == '__main__':
    S = ShockObj(5)
    S.create_shuffle_tuning_curve()
    S.find_modulated_cells()
    S.plot_traces(neurons=S.shock_modulated_cells)

    pass