# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:09:29 2018

@author: William Mau
"""

from cell_data_compiler import CellData
from session_directory import load_session_list
from plot_helper import ScrollPlot, neuron_number_title
import plot_functions as plot_funcs
import matplotlib.pyplot as plt
import calcium_traces as ca_traces
from helper_functions import find_closest
import numpy as np

session_list = load_session_list()


def load_events(session_index):
    """
    Load calcium events and save to disk if not already saved.

    :param
        session_index: Number corresponding to a session.
    :return
        events:
    """
    data = CellData(session_index)

    return data.event_times, data.event_values

def plot_events(session_index, neurons):
    """
        Plot events as a scatter plot.
        :param
            session_index: Number corresponding to a session.
            neurons: List of neurons.
        :return
            f: ScrollPlot class.
        """
    # Load events.
    event_times, event_values = load_events(session_index)

    # Plot and scroll through calcium events.
    titles = neuron_number_title(neurons)
    f = ScrollPlot(plot_funcs.plot_events,
                   event_times=event_times[neurons], event_values=event_values[neurons],
                   xlabel='Time (s)', ylabel='Event magnitude', titles=titles)
    return f

def overlay_events(session_index, neurons):
    """
    Plot events on top of traces.
    :param
        session_index: Number corresponding to a session.
        neurons: List of neurons.
    :return
        f: ScrollPlot class.
    """
    # Load events and traces.
    event_times, event_values = load_events(session_index)
    traces, _, t = ca_traces.load_traces(session_index)

    # Plot and scroll.
    titles = neuron_number_title(neurons)
    f = ScrollPlot(plot_funcs.overlay_events,
                   t=t, traces=traces[neurons],
                   event_times=event_times[neurons], event_values=event_values[neurons],
                   xlabel='Time (s)', ylabel='% DF/F', titles=titles)

    return f

def make_event_matrix(session_index):
    event_times, event_values = load_events(session_index)

    traces, accepted, t = ca_traces.load_traces(session_index)

    events = np.zeros(traces.shape)

    for cell,timestamps in enumerate(event_times):
        for i,this_time in enumerate(timestamps):
            _,idx = find_closest(t,this_time)
            events[cell, idx] = event_values[cell][i]

    return events


if __name__ == '__main__':
    make_event_matrix(0)