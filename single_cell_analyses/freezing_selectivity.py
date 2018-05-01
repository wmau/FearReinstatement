from session_directory import load_session_list
from ff_video_fixer import load_session as load_ff
from calcium_events import load_events
import numpy as np
import helper_functions as helper
from scipy.stats import ttest_1samp
from calcium_traces import load_traces

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

        self.session = load_ff(session_index)
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


if __name__ == '__main__':
    FreezingCellFilter(0, 'trace').get_freezing_cells()