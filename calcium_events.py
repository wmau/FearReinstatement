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
from os import path
from pickle import load, dump
import data_preprocessing as d_pp
import scipy.io as sio

session_list = load_session_list()


def load_event_times(session_index):
    """
    Load calcium events and save to disk if not already saved.

    :param
        session_index: Number corresponding to a session.
    :return
        events:
    """
    data = CellData(session_index)

    return data.event_times, data.event_values


def save_events(session_index):
    from ff_video_fixer import load_session

    entire_session_events, _ = load_events(session_index)
    session = load_session(session_index)
    events = d_pp.trim_session(entire_session_events,
                               session.mouse_in_cage)

    directory = session_list[session_index]["Location"]
    file = path.join(directory, 'Events.mat')

    sio.savemat(file,{'events': events,
                      'events_all': entire_session_events})


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
    event_times, event_values = load_event_times(session_index)

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
    event_times, event_values = load_event_times(session_index)
    traces, t = ca_traces.load_traces(session_index)

    # Plot and scroll.
    titles = neuron_number_title(neurons)
    f = ScrollPlot(plot_funcs.overlay_events,
                   t=t, traces=traces[neurons],
                   event_times=event_times[neurons], event_values=event_values[neurons],
                   xlabel='Time (s)', ylabel='% DF/F', titles=titles)

    return f


def load_events(session_index):
    directory = session_list[session_index]["Location"]
    file_path = path.join(directory, "EventMatrix.pkl")
    try:
        with open(file_path, 'rb') as file:
            events = load(file)

        _, t = ca_traces.load_traces(session_index)

    except:
        event_times, event_values = load_event_times(session_index)

        traces, t = ca_traces.load_traces(session_index)

        events = np.zeros(traces.shape)

        for cell, timestamps in enumerate(event_times):
            for i, this_time in enumerate(timestamps):
                _, idx = find_closest(t, this_time)
                events[cell, idx] = event_values[cell][i]

        with open(file_path, 'wb') as file:
            dump(events, file, protocol=4)

    return events, t

if __name__ == '__main__':
    overlay_events(0,[1,2,3,4,5])
