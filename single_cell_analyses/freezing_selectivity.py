from session_directory import load_session_list
import ff_video_fixer as ff
from calcium_events import load_events
import numpy as np
import helper_functions as helper
from scipy.stats import ttest_1samp
from calcium_traces import load_traces
from scipy.stats import zscore
from plot_helper import ScrollPlot, neuron_number_title
import calcium_traces as ca_traces
import plot_functions as plot_funcs
import calcium_events as ca_events

session_list = load_session_list()

class FreezingCellFilter:
    def __init__(self, session_index, dtype='event'):
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
        non_freezing_event_rate = self.get_non_freezing_event_rate()
        freezing_event_rate = self.get_freezing_event_rate()

        p = np.zeros((self.n_neurons))
        for cell, (freeze, non_freeze) in \
                enumerate(zip(freezing_event_rate,
                          non_freezing_event_rate)):

            if np.max(freeze) > 0:
                p[cell] = ttest_1samp(freeze, non_freeze)[1]

        more_active_during_freezing = np.mean(freezing_event_rate, axis=1) > \
                                      non_freezing_event_rate
        significant_cells = np.where((p < (0.05 / self.n_neurons)) &
                                     more_active_during_freezing)[0]

        return significant_cells, p

def plot_prefreezing_traces(session_index, neurons=None, dtype='events',
                            window=1):
    # Load data and get freezing timestamps.
    session = ff.load_session(session_index)
    freeze_epochs = session.get_freezing_epochs_imaging_framerate()

    if dtype == 'traces':
        data, _ = ca_traces.load_traces(session_index)
        data = zscore(data, axis=1)
    elif dtype == 'events':
        data, _ = ca_events.load_events(session_index)
        data[data > 0] = 1
    else:
        raise ValueError("Invalid data type.")

    if neurons is not None:
        data = data[neurons]
        titles = neuron_number_title(neurons)
    else:
        titles = neuron_number_title(range(len(data)))

    n_neurons = len(data)
    n_freezes = freeze_epochs.shape[0]
    freeze_duration = np.ceil(window*20).astype(int)

    prefreezing_traces = np.zeros((n_neurons, n_freezes, freeze_duration))

    for n, trace in enumerate(data):
        for i, epoch in enumerate(freeze_epochs):
            start = epoch[0]-freeze_duration
            stop = epoch[0]
            prefreezing_traces[n, i, :] = trace[start:stop]

    if dtype == 'events':
        events = [[(np.where(bout)[0] - freeze_duration)/20
                   for bout in cell]
                   for cell in prefreezing_traces]

        f = ScrollPlot(plot_funcs.plot_raster, events=events,
                       window=window,
                       xlabel='Time from start of freezing (s)',
                       ylabel='Freezing bout #', titles=titles)

    elif dtype == 'traces':
        f = ScrollPlot(plot_funcs.plot_heatmap,
                       heatmap = prefreezing_traces,
                       xlabel = 'Time from start of freezing (s)',
                       ylabel = 'Freezing bout #', titles=titles)

    else:
        raise ValueError("Invalid data type.")



if __name__ == '__main__':
    #FreezingCellFilter(0, 'trace').get_freezing_cells()
    plot_prefreezing_traces(1, window=10)
    pass