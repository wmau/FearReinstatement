from session_directory import load_session_list
import calcium_traces as ca_traces
import calcium_events as ca_events
import numpy as np
from helper_functions import find_closest, shift_rows, get_longest_run, \
    ismember
import data_preprocessing as d_pp
from plot_helper import ScrollPlot, neuron_number_title
import plot_functions as plot_funcs
from scipy.stats import zscore, sem, pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from os import path
from pickle import load, dump
import ff_video_fixer as ff
from cell_reg import load_cellreg_results, find_cell_in_map, \
    find_match_map_index
import matplotlib.pyplot as plt
import scipy.signal as signal
import itertools
from single_cell_analyses.computations import xcorr_all_neurons

session_list = load_session_list()


def load_aligned_data(session_index):
    directory = session_list[session_index]["Location"]
    file_name = path.join(directory, 'FootshockResponses.pkl')

    with open(file_name, 'rb') as file:
        shock_cells, aligned_traces, aligned_events = load(file)

    return shock_cells, aligned_traces, aligned_events

def plot_xcorr_over_days(S1, S2, neuron_pair):
    map = S1.map

    map_idx = find_match_map_index([S1.session_index, S2.session_index])
    global_cell_1 = find_cell_in_map(map, map_idx[0], neuron_pair[0])
    global_cell_2 = find_cell_in_map(map, map_idx[0], neuron_pair[1])

    plt.figure()

    plt.plot(S1.window, S1.xcorrs[neuron_pair[0], neuron_pair[1]], c='b')

    cell_1 = map[global_cell_1, map_idx[1]]
    cell_2 = map[global_cell_2, map_idx[1]]
    plt.plot(S2.window, S2.xcorrs[cell_1, cell_2], c='k')



class ShockResponse:
    def __init__(self, session_index, window=[-1, 15]):
        self.session_index = session_index
        self.frame_rate = 20
        window.sort()
        self.window = np.asarray(window)

        assert len(self.window) is 2, "Window length must be 2."
        assert any(self.window < 0), "One input must be positive."
        assert any(self.window > 0), "One input must be negative."

        self.ref_time = np.arange(self.window[0], self.window[1],
                                  1 / self.frame_rate)

        self.footshock_times = [698, 778, 858, 938]
        traces, self.t = \
            ca_traces.load_traces(session_index)
        self.events = ca_events.make_event_matrix(session_index)
        self.traces = zscore(traces, axis=0)
        self.n_neurons = len(traces)
        self.align_neurons()
        self.create_shuffle_tuning_curve()
        self.find_modulated_cells()

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

    def align_neurons(self):
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

        f = ScrollPlot(plot_funcs.plot_multiple_traces,
                       t=self.ref_time,
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
                shuffled[iteration] = np.mean(shift_rows(trace), axis=0)

            self.surrogate[neuron] = shuffled

    def find_modulated_cells(self):
        """
        Find cells that are modulated by the shock with a shuffling
        method.
        :return:
        """
        B = self.surrogate[0].shape[0]
        self.shock_modulated_cells = []
        self.tuning_curves = []
        for traces, events, shuffled, neuron in \
                zip(self.aligned_traces, self.aligned_events,
                    self.surrogate, np.arange(self.n_neurons)):

            # Take the mean of the response across shocks.
            tuning_curve = np.mean(traces, axis=0)
            self.tuning_curves.append(tuning_curve)

            # Get the p-value, the percentage of shuffled tuning curves
            # with a higher amplitude response than the real data.
            p_value = np.sum(tuning_curve < shuffled, axis=0) / B

            # If there are any bins where p < 0.01, there must be at
            # least 5 consecutive bins that satisfy this. Also must be
            # active for at least 3 of the 4 shocks.
            if any(p_value < 0.01):
                if get_longest_run(p_value < 0.01) > 3 and \
                        np.sum(events.any(axis=1)) > 1:
                    self.shock_modulated_cells.append(neuron)

    def plot_sequence(self):
        # Convert to array. Then normalize.
        tuning_curves = np.asarray(self.tuning_curves)[self.shock_modulated_cells]
        tuning_curves = normalize(tuning_curves, norm='max')

        # Get the peaks.
        peaks = np.argmax(tuning_curves, axis=1)

        # Sort the peaks and get their order.
        order = np.argsort(peaks)

        n_modulated_cells = len(self.shock_modulated_cells)

        # Plot.
        plt.figure(figsize=(4, 5))
        plt.imshow(tuning_curves[order],
                   extent=[self.ref_time[0], self.ref_time[-1],
                           n_modulated_cells, 0])
        plt.axis('tight')
        plt.yticks([1, n_modulated_cells])
        plt.xlabel('Time from shock (s)')
        plt.ylabel('Cell #')

    def save_data(self):
        directory = session_list[self.session_index]["Location"]
        file_name = path.join(directory, 'FootshockResponses.pkl')

        with open(file_name, 'wb') as file:
            dump([self.shock_modulated_cells,
                  self.aligned_traces,
                  self.aligned_events], file)


class ShockXCorr:
    def __init__(self, session_index, xcorr_window=[-15,15]):
        self.session_index = session_index

        # Load shock-aligned cells, traces, and events.
        try:
            shock_cells, self.aligned_traces, self.aligned_events = \
                load_aligned_data(session_index)
            self.shock_cells = np.asarray(shock_cells)
        except:
            print("Shock-aligned traces not detected.")


        # Load full traces.
        traces, _ = ca_traces.load_traces(session_index)
        self.n_neurons = len(traces)

        # Get mouse in chamber indices.
        session = ff.load_session(session_index)
        self.traces = d_pp.trim_session(traces, session.mouse_in_cage)

        mouse = session_list[session_index]["Animal"]
        self.map = load_cellreg_results(mouse)

        xcorr_window.sort()
        self.xcorr_window = np.asarray(xcorr_window)

        assert len(self.xcorr_window) is 2, "Window length must be 2."
        assert any(self.xcorr_window < 0), "One input must be positive."
        assert any(self.xcorr_window > 0), "One input must be negative."

        directory = session_list[session_index]["Location"]
        xcorr_file = path.join(directory, 'CrossCorrelations.pkl')

        try:
            with open(xcorr_file, 'rb') as file:
                self.xcorrs, self.best_lags, self.window = load(file)
        except:
            self.xcorrs, self.best_lags, self.window = \
                xcorr_all_neurons(self.traces, self.xcorr_window)

            self.save_xcorrs()


    def save_xcorrs(self):
        directory = session_list[self.session_index]["Location"]
        file_path = path.join(directory, 'CrossCorrelations.pkl')

        with open(file_path, 'wb') as file:
            dump([self.xcorrs, self.best_lags, self.window], file)


if __name__ == '__main__':
    FC = ShockXCorr(0)
    EXT1 = ShockXCorr(1)

    pass
