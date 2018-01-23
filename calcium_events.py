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

session_list = load_session_list()

def load_events(session_index):
    """
    Load calcium events and save to disk if not already saved.

    :param
        session_index: Number corresponding to a session.
    :return
        events:
    """
    session_directory = session_list[session_index]["Location"]

    data = CellData(session_index)

    return data.event_times, data.event_values

def plot_events(session_index,neurons):
    event_times,event_values = load_events(session_index)

    # Scroll through calcium events.
    titles = neuron_number_title(neurons)
    f = ScrollPlot(plot_funcs.plot_events,
                   event_times = event_times[neurons], event_values = event_values[neurons],
                   xlabel = 'Time (s)', ylabel = 'Event magnitude', titles = titles)
    return f

def overlay_events(session_index,neurons):
    event_times,event_values = load_events(session_index)
    traces, _, t = ca_traces.load_traces(session_index)

    titles = neuron_number_title(neurons)
    f = ScrollPlot(plot_funcs.overlay_events,
                   t = t, traces = traces[neurons],
                   event_times = event_times[neurons], event_values = event_values[neurons],
                   xlabel = 'Time (s)', ylabel = '% DF/F', titles = titles)
