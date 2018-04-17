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


session_list = load_session_list()


def load_aligned_data(session_index):
    directory = session_list[session_index]["Location"]
    file_name = path.join(directory, 'FootshockResponses.pkl')

    with open(file_name, 'rb') as file:
        shock_cells, aligned_traces, aligned_events = load(file)

    return shock_cells, aligned_traces, aligned_events


class ShockObj:
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


class ShockNetwork:
    def __init__(self, session_index):
        # Load shock-aligned cells, traces, and events.
        shock_cells, self.aligned_traces, self.aligned_events = \
            load_aligned_data(session_index)
        self.shock_cells = np.asarray(shock_cells)

        # Load full traces.
        traces, _ = ca_traces.load_traces(session_index)

        # Get mouse in chamber indices.
        session = ff.load_session(session_index)
        self.traces = d_pp.trim_session(traces, session.mouse_in_cage)

        mouse = session_list[session_index]["Animal"]
        self.map = load_cellreg_results(mouse)

        self.build_corr_matrix()

    def build_corr_matrix(self, mode='xcorr'):
        # Get the traces and number of cells.
        shock_traces = self.traces[self.shock_cells]
        n_shock_cells = len(self.shock_cells)

        if mode == 'corr':
            # If simple linear correlation, corcoeff populates the
            # matrix in one go.
            coefficients = np.corrcoef(shock_traces)
            best_lag = np.zeros(n_shock_cells)

        elif mode == 'xcorr':
            # If cross-correlation, allocate some matrices first.
            best_lag = np.zeros((n_shock_cells, n_shock_cells))
            coefficients = np.full((n_shock_cells,
                                    n_shock_cells),np.nan)
            matrix_indices = np.unravel_index(
                                        np.arange(coefficients.size),
                                        coefficients.shape)

            # Define the lag vector.
            n_samples = len(shock_traces[0])
            x_axis = np.arange(n_samples * 2 - 1)
            lags = (x_axis - (n_samples - 1))*0.05

            # Define your window and get its indices.
            window = np.arange(-15,15)
            _, window_idx = ismember(lags, window)

            # Get all pairwise combinations.
            combinations = list(itertools.product(shock_traces, shock_traces))
            for i, (trace_1, trace_2) in enumerate(combinations):
                # Perform cross-correlation then normalize.
                xcorr = signal.correlate(trace_1, trace_2)
                #xcorr /= max(xcorr)

                # Get the appropriate index.
                row = matrix_indices[0][i]
                col = matrix_indices[1][i]

                # Dump the maximum cross-correlation coefficient
                # and the corresponding time lag in seconds.
                coefficients[row, col] = max(xcorr[window_idx])
                best_lag[row, col] = window[np.argmax(xcorr[window_idx])]

        # Populate identity with NaNs.
        np.fill_diagonal(coefficients, np.nan)

        return coefficients, best_lag

    def get_max_corr(self, coeffs):
        max_coeff = np.nanmax(coeffs, axis=1)
        most_corr_cell = self.shock_cells[np.nanargmax(coeffs, axis=1)]

        return max_coeff, most_corr_cell

    def corr_again(self, session_index, mode='corr'):
        # Of all the sessions from this mouse, get the session number
        # from the input.
        map_index = find_match_map_index(session_index)

        # Get the cell numbers for all the shock-responsive cells from
        # fear conditioning day.
        shock_cell_idx = find_cell_in_map(self.map, 0, self.shock_cells)
        shock_cell_number = self.map[shock_cell_idx, map_index]

        # Get the correlation matrix from fear conditioning day.
        conditioning_coeffs, best_lag = self.build_corr_matrix(mode=mode)

        # Get the correlation coefficient of the highest correlated cell
        # and the cell number.
        max_coeffs, most_corr_cell_fc = \
            self.get_max_corr(conditioning_coeffs)

        # Get those cells' numbers on this session.
        most_corr_cell_idx = \
            find_cell_in_map(self.map, 0, most_corr_cell_fc)
        most_corr_cell_number = self.map[most_corr_cell_idx, map_index]

        # Omit the cells that weren't registered.
        shock_cell_number = shock_cell_number[shock_cell_number > -1]
        most_corr_cell_number = \
            most_corr_cell_number[most_corr_cell_number > -1]

        traces, _ = ca_traces.load_traces(session_index)
        session = ff.load_session(session_index)
        traces = d_pp.trim_session(traces, session.mouse_in_cage)

        new_coeffs = []
        if mode == 'corr':
            for cell_1, cell_2 in zip(shock_cell_number, most_corr_cell_number):
                r, _ = pearsonr(traces[cell_1], traces[cell_2])
                new_coeffs.append(r)
        elif mode == 'xcorr':
            # Define the lag vector.
            n_samples = len(traces[0])
            x_axis = np.arange(n_samples * 2 - 1)
            lags = -(x_axis - (n_samples - 1)) * 0.05

            for cell_1, cell_2 in zip(shock_cell_number,
                                      most_corr_cell_number):
                xcorr = signal.correlate(traces[cell_1], traces[cell_2])
                #xcorr /= max(xcorr)

                _, i = ismember(self.map[:, map_index], cell_1)
                _, i = ismember(self.shock_cells, self.map[i, 0])

                _, j = ismember(self.map[:, map_index], cell_2)
                _, j = ismember(self.shock_cells, self.map[j, 0])

                _, idx = ismember(lags, best_lag[i, j])
                new_coeffs.append(xcorr[idx])

        return new_coeffs, max_coeffs



if __name__ == '__main__':
    S = ShockNetwork(5)
    ext1, max_coeffs_ext1 = S.corr_again(6, 'xcorr')
    ext2, max_coeffs_ext2 = S.corr_again(7, 'xcorr')
    recall, max_coeffs_recall = S.corr_again(9, 'xcorr')

    pass
