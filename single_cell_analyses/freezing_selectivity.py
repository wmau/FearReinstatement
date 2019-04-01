from session_directory import load_session_list, get_session
from microscoPy_load.calcium_events import load_events
import numpy as np
import helper_functions as helper
from scipy.stats import ttest_1samp, pearsonr, kendalltau, spearmanr
from microscoPy_load.calcium_traces import load_traces
from scipy.stats import zscore
from plotting.plot_helper import ScrollPlot, neuron_number_title
from plotting import plot_functions as plot_funcs
from microscoPy_load import calcium_events as ca_events, calcium_traces as ca_traces, ff_video_fixer as ff
import matplotlib.pyplot as plt
import data_preprocessing as d_pp
from statsmodels.stats.multitest import fdrcorrection

session_list = load_session_list()

class FreezingCellFilter:
    def __init__(self, mouse, session_stage, dtype='trace'):
        session_index = get_session(mouse, session_stage)[0]
        self.session_index = session_index
        self.dtype = dtype

        if dtype == 'event':
            self.events, _ = load_events(session_index)
            self.events[self.events > 0] = 1
        elif dtype == 'trace':
            self.events, _ = load_traces(session_index)

        self.session = ff.load_session(session_index)
        self.freezing_epochs = \
            self.session.get_freezing_epochs_imaging_framerate()
        self.non_freezing_idx = self.get_non_freezing_epochs()
        self.n_neurons = self.events.shape[0]

    def get_non_freezing_epochs(self):
        return np.where(np.logical_not(self.session.imaging_freezing)
                        & self.session.mouse_in_cage)[0]

    def get_non_freezing_event_rate(self):
        """
        Returns the average event rate during non-freezing epochs.
        :return:
        """
        if self.dtype == 'event':
            non_freezing_event_rate = \
                helper.get_event_rate(self.events[:, self.non_freezing_idx])
        elif self.dtype == 'trace':
            non_freezing_event_rate = \
                np.mean(self.events[:, self.non_freezing_idx], axis=1)
        else:
            raise ValueError('dtype not recognized.')

        return non_freezing_event_rate

    def get_freezing_event_rate(self):
        n_freezing_epochs = self.freezing_epochs.shape[0]

        freezing_event_rate = np.zeros((self.n_neurons, n_freezing_epochs))
        for epoch_number, epoch in enumerate(self.freezing_epochs):
            for cell, events in enumerate(self.events):
                if self.dtype == 'event':
                    freezing_event_rate[cell, epoch_number] = \
                        helper.get_event_rate(events[epoch[0]:epoch[1]])
                elif self.dtype == 'trace':
                    freezing_event_rate[cell, epoch_number] = \
                        np.mean(events[epoch[0]:epoch[1]])


        return freezing_event_rate

    def get_freezing_cells(self):
        """
        Gets cells responsive to freezing by comparing activity during
        freezing to activity during non-freezing.

        :return:
        """
        non_freezing_event_rate = self.get_non_freezing_event_rate()
        freezing_event_rate = self.get_freezing_event_rate()

        p = np.zeros((self.n_neurons))
        for cell, (freeze, non_freeze) in \
                enumerate(zip(freezing_event_rate,
                          non_freezing_event_rate)):

            if np.max(freeze) > 0:
                p[cell] = ttest_1samp(freeze, non_freeze)[1]

        # Gets cells that have higher fluorescence during freezing.
        more_active_during_freezing = np.mean(freezing_event_rate, axis=1) > \
                                      non_freezing_event_rate

        # Gets significant cells.
        significant_cells = np.where((p < (0.05 / self.n_neurons)) &
                                     more_active_during_freezing)[0]

        return significant_cells, p


class FreezingCellFilter2:
    """
    Freezing cell filter.

    """
    def __init__(self, mouse, session_stage, dtype='traces', window=(-2,2),
                 freeze_duration_thresh = 1.25):
        # Metadata.
        self.mouse = mouse
        self.session_stage = session_stage
        self.dtype = dtype
        self.window = window
        self.session_index = get_session(mouse, session_stage)[0]
        self.session = ff.load_session(self.session_index)

        # Imaging data.
        if self.dtype == 'traces':
            self.act_matrix = load_traces(self.session_index)[0]
            self.act_matrix = np.ma.masked_invalid(self.act_matrix)
            in_chamber = self.session.mouse_in_cage
            self.act_matrix = helper.partial_z(self.act_matrix,
                                               in_chamber)

        # Freezing epochs.
        self.freeze_epochs = self.session.get_freezing_epochs_imaging_framerate()
        good_epochs = np.squeeze(np.diff(self.freeze_epochs) >
                                 freeze_duration_thresh * 20)
        self.freeze_epochs = self.freeze_epochs[good_epochs]

        # Useful variables.
        self.n_neurons = self.act_matrix.shape[0]
        self.n_freezes = self.freeze_epochs.shape[0]
        self.freeze_duration = abs(np.ceil(np.diff(window)*20)).astype(int)[0]

        # Compile the traces centered around the start of a freezing bout.
        self.compile_freezing_traces()

        pass

    def compile_freezing_traces(self):
        """
        Gathers all the traces centered around the start of a freezing bout.

        :return:
        """
        self.freezing_traces = np.zeros((self.n_neurons,
                                         self.n_freezes,
                                         self.freeze_duration))

        for n, trace in enumerate(self.act_matrix):
            for i, epoch in enumerate(self.freeze_epochs):
                start = epoch[0] - (abs(self.window[0]) * 20)
                stop =  epoch[0] + (abs(self.window[1]) * 20)

                self.freezing_traces[n, i, :] = trace[start:stop]


    def plot_traces(self, neurons=None):
        """
        Plots traces centered around the start of a freezing bout.

        :param neurons:
        :return:
        """
        if neurons is None:
            neurons = range(self.n_neurons)

        # Make time vector and titles.
        t = np.arange(self.window[0], self.window[1], 1/20)
        titles = neuron_number_title(neurons)

        f = ScrollPlot(plot_funcs.plot_multiple_traces,
                       traces=self.freezing_traces[neurons],
                       t=t, xlabel='Time from start of freezing (s)',
                       ylabel='Fluorescence amplitude (z)',
                       titles=titles)


    #def get_freezing_cells(self):



def plot_prefreezing_traces(mouse, session_stage, neurons=None,
                            dtype='traces', window=(-2,2),
                            freeze_duration_threshold=1.25,
                            plot_bool=True):
    """
    Plots the average activity for each cell centered around the start of
    freezing bouts. ONLY TESTED WITH DTYPE="TRACES".

    Parameters
    ---
    mouse: str, mouse name.
    session_stage: str, session stage.
    neurons: list-like: specified neuron indices.
    dtype: str, data type ('events' or 'traces')
    window: tuple, (-seconds before freezing, seconds after)
    freeze_duration_threshold: scalar, freezing duration must be longer
        than this value (seconds)
    plot_bool: boolean, whether or not to plot

    Returns
    ---
    freeze_traces: dict with the following fields
        mouse, seession_stage, neurons, window: same as inputs
        data: untrimmed (includes home cage) traces or events
        epoch: Fx2 array with first column being the index of the start
            of the freezing epoch, where F = # of freezing epochs
        windowed_data: NxFxT matrix depicting the traces centered around
            the start of a freezing epoch, where N = # neurons and
            T = window length
        f: ScrollPlot object
    """

    # Load data and get freezing timestamps.
    session_index = get_session(mouse, session_stage)[0]
    session = ff.load_session(session_index)
    freeze_epochs = session.get_freezing_epochs_imaging_framerate()

    # Eliminate freeze epochs that don't pass the duration threshold.
    good_epochs = np.squeeze(np.diff(freeze_epochs) >
                             freeze_duration_threshold * 20)
    freeze_epochs = freeze_epochs[good_epochs, :]

    # Process data depending on data type.
    if dtype == 'traces':
        data, _ = ca_traces.load_traces(session_index)
        data = np.ma.masked_invalid(data)
        data = zscore(data, axis=1)
    elif dtype == 'events':
        data, _ = ca_events.load_events(session_index)
        data[data > 0] = 1
    else:
        raise ValueError("Invalid data type.")

    # Index the list if neurons were specified.
    if neurons is not None:
        data = data[neurons]
        titles = neuron_number_title(neurons)
    else:
        titles = neuron_number_title(range(len(data)))

    # Get sizes for all dimensions of our array.
    n_neurons = len(data)
    n_freezes = freeze_epochs.shape[0]
    freeze_duration = abs(np.ceil(np.diff(window)*20)).astype(int)[0]

    # Plot each cell and each freeze epoch.
    prefreezing_traces = np.zeros((n_neurons, n_freezes, freeze_duration))
    t = np.arange(window[0], window[1], 1/20)
    for n, trace in enumerate(data):
        for i, epoch in enumerate(freeze_epochs):
            start = epoch[0] - (abs(window[0]) * 20)
            stop = epoch[0] + (abs(window[1]) * 20)
            prefreezing_traces[n, i, :] = trace[start:stop]

    f = None        # Necessary if plot_flag is False.
    if dtype == 'events':
        events = [[(np.where(bout)[0] - freeze_duration)/20
                   for bout in cell]
                   for cell in prefreezing_traces]

        if plot_bool:
            f = ScrollPlot(plot_funcs.plot_raster,
                           events=events,
                           window=window,
                           xlabel='Time from start of freezing (s)',
                           ylabel='Freezing bout #',
                           titles=titles)

    elif dtype == 'traces':
        if plot_bool:
            f = ScrollPlot(plot_funcs.plot_multiple_traces,
                           traces=prefreezing_traces,
                           t=t,
                           xlabel='Time from start of freezing (s)',
                           ylabel='Fluorescence amplitude (z)',
                           titles=titles)

    else:
        raise ValueError("Invalid data type.")

    # Summarize.
    freeze_traces = {
        "mouse": mouse,
        "session_stage": session_stage,
        "data": data,
        "neurons": neurons,
        "window": window,
        "epochs": freeze_epochs,
        "windowed_data": prefreezing_traces,
        "figure": f,
    }

    return freeze_traces



def speed_modulation(mouse, stage, neurons=None, dtype='events'):
    session_index = get_session(mouse, stage)[0]
    session = ff.load_session(session_index)
    first_shock = 698*20

    data, t = d_pp.load_and_trim(session_index,
                                 dtype=dtype,
                                 neurons=neurons,
                                 session=session,
                                 end=first_shock)

    v, _ = d_pp.load_and_trim(session_index,
                              dtype='velocity',
                              neurons=None,
                              session=session,
                              end=first_shock)

    p = []
    for neuron in data:
        p.append(spearmanr(neuron, v)[1])

    p = np.asarray(p)
    modulated, p = fdrcorrection(p, 0.05)

    #modulated = p < 0.01/len(p)

    print(str(sum(modulated)) + ' neurons removed from ' + mouse)

    return modulated




if __name__ == '__main__':
    FreezingCellFilter2('Mundilfari','E1_1')