# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:11:14 2018

@author: William Mau
"""

from cell_data_compiler import CellData
from plot_helper import ScrollPlot, neuron_number_title
from session_directory import load_session_list
from os import path
from pickle import load
import plot_functions as plot_funcs
import ff_video_fixer as FF
import cell_stats
import numpy as np

session_list = load_session_list()

def load_traces(session_index):
    """
    Load traces from data of single sessions saved via Inscopix Data Processing
    software. This may take a few seconds to run if running for the first time 
    on a session.

    :param
        session_index: number corresponding to a session.
    :return
        traces: Array of cell traces.
        accepted: List of accepted cells.
        t: Vector of timestamps.
    """

    # Gather data.
    data = CellData(session_index)

    return data.traces.astype(np.float), data.accepted, data.t.astype(np.float)

def plot_traces(session_index, neurons):
    """
    Plot the traces of multiple neurons. You can scroll between neurons!

    :param
        session_index: Number corresponding to a session.
        neurons: List of neurons.
    :return
    """

    # Load the traces.
    [traces, accepted, t] = load_traces(session_index)

    # Scroll through.
    titles = neuron_number_title(neurons)
    f = ScrollPlot(plot_funcs.plot_traces,
                   t=t, traces=traces[neurons],
                   xlabel='Time (s)', ylabel='%DF/F', titles=titles)

    # Gets the ScrollPlot object.
    return f

def plot_freezing_traces(session_index):
    # Load the position data.
    session = FF.load_session(session_index)

    # Get freezing epochs.
    freeze_epochs = session.get_freezing_epochs_imaging_framerate()

    # Load trace data.
    [traces, accepted, t] = load_traces(session_index)

    f = ScrollPlot(plot_funcs.plot_freezing_traces,
                   t=t, traces=traces, epochs=freeze_epochs, n_rows=freeze_epochs.shape[0],
                   share_y=True,  xlabel='Time (s)', ylabel='%DF/F')

def freezing_trace_heatmap(session_index, neurons='all'):
    if neurons == 'all':
        n_neurons = cell_stats.get_number_of_ICs(session_index)
        neurons = np.arange(n_neurons)
    else:
        n_neurons = len(neurons)

    # Load the position data.
    session = FF.load_session(session_index)

    # Get freezing epochs.
    freeze_epochs = session.get_freezing_epochs_imaging_framerate()

    # Load trace data.
    [traces, accepted, t] = load_traces(session_index)

    # Get dimensions for heatmap.
    n_freezes = freeze_epochs.shape[0]
    freeze_durations = np.diff(freeze_epochs)
    longest_freeze = freeze_durations.max()

    # Preallocate heatmap.
    freezing_traces = np.full([n_neurons, n_freezes, longest_freeze], np.nan)

    for n,this_neuron in enumerate(neurons):
        for i,epoch in enumerate(freeze_epochs):
            freezing_traces[n,i,np.arange(freeze_durations[i])] = traces[this_neuron,epoch[0]:epoch[1]]

    titles = neuron_number_title(neurons)
    f = ScrollPlot(plot_funcs.heatmap,
                   heatmap = freezing_traces,
                   xlabel='Time from start of freezing (s)', ylabel='Freezing bout #', titles=titles)


if __name__ == '__main__':
    freezing_trace_heatmap(1,[1,2,5])