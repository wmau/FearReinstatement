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
import ff_video_fixer as ff
import numpy as np
from cell_reg import load_cellreg_results, find_match_map_index
from session_directory import find_mouse_sessions
from helper_functions import ismember
from scipy.stats import zscore
import scipy.io as sio
import data_preprocessing as d_pp

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

    return data.traces.astype(np.float), data.t.astype(np.float)

def save_traces(session_index):
    from ff_video_fixer import load_session

    entire_session_traces, _ = load_traces(session_index)
    session = load_session(session_index)
    traces = d_pp.trim_session(entire_session_traces,
                               session.mouse_in_cage)

    directory = session_list[session_index]["Location"]
    file = path.join(directory, 'Traces.mat')

    sio.savemat(file,{'traces': traces,
                      'traces_all': entire_session_traces})


def plot_traces(session_index, neurons):
    """
    Plot the traces of multiple neurons. You can scroll between neurons!

    :param
        session_index: Number corresponding to a session.
        neurons: List of neurons.
    :return
    """

    # Load the traces.
    traces, t = load_traces(session_index)

    # Scroll through.
    titles = neuron_number_title(neurons)
    f = ScrollPlot(plot_funcs.plot_traces,
                   t=t, traces=traces[neurons],
                   xlabel='Time (s)', ylabel='%DF/F', titles=titles)

    # Gets the ScrollPlot object.
    return f

def plot_freezing_traces(session_index):
    # Load the position data.
    session = ff.load_session(session_index)

    # Get freezing epochs.
    freeze_epochs = session.get_freezing_epochs_imaging_framerate()

    # Load trace data.
    traces, t = load_traces(session_index)

    f = ScrollPlot(plot_funcs.plot_freezing_traces,
                   t=t, traces=traces, epochs=freeze_epochs, n_rows=freeze_epochs.shape[0],
                   share_y=True,  xlabel='Time (s)', ylabel='%DF/F')

def freezing_trace_heatmap(session_index, neurons='all'):
    # Load the position data.
    session = ff.load_session(session_index)

    # Get freezing epochs.
    freeze_epochs = session.get_freezing_epochs_imaging_framerate()

    # Load trace data.
    traces, t = load_traces(session_index)

    if neurons == 'all':
        n_neurons = len(traces)
        neurons = arange(n_neurons)
    else:
        n_neurons = len(neurons)

    # Get dimensions for heatmap.
    n_freezes = freeze_epochs.shape[0]
    freeze_durations = np.diff(freeze_epochs)
    longest_freeze = freeze_durations.max()

    # Preallocate heatmap.
    freezing_traces = np.full([n_neurons, n_freezes, longest_freeze],
                              np.nan)

    for n in neurons:
        for i,epoch in enumerate(freeze_epochs):
            freezing_traces[n,i,np.arange(freeze_durations[i])] = \
                traces[n,epoch[0]:epoch[1]]

    titles = neuron_number_title(neurons)
    f = ScrollPlot(plot_funcs.heatmap,
                   heatmap = freezing_traces,
                   xlabel='Time from start of freezing (s)',
                   ylabel='Freezing bout #', titles=titles)


def plot_traces_over_days(session_index, neurons):
    # Get the mouse.
    mouse = session_list[session_index]["Animal"]

    # Load the cell map.
    cell_map = load_cellreg_results(mouse)
    n_sessions = cell_map.shape[1]

    # Find all the sessions from this mouse.
    sessions_from_this_mouse, _ = find_mouse_sessions(mouse)

    # Make sure this matches the number of sessions in the cell map.
    assert len(sessions_from_this_mouse) == n_sessions, \
        "Number of sessions do not agree."

    traces = []
    t = []
    # Compile all the traces and time vectors
    for session in sessions_from_this_mouse:
        day_traces, day_t = load_traces(session)
        traces.append(zscore(day_traces, axis=1))
        t.append(day_t)

    # Get the column of cell map that corresponds to the specified session.
    map_index = find_match_map_index(session_index)

    # Get the cell numbers.
    _, cell_index = ismember(cell_map[:, map_index], neurons)

    # Compile traces, matching by cell.
    traces_to_plot = []
    for cell in cell_index:
        this_cell_traces = []

        for day in range(n_sessions):
            cell_number = cell_map[cell, day]

            # Make sure there was actually a match, otherwise insert
            # a placeholder.
            if cell_number > -1:
                this_cell_traces.append(traces[day][cell_number])
            else:
                placeholder = np.zeros(t[day].shape)
                this_cell_traces.append(placeholder)

        traces_to_plot.append(this_cell_traces)

    # Plot.
    f = ScrollPlot(plot_funcs.plot_traces_over_days,
                   n_rows=1, n_cols=n_sessions, share_y=True, share_x=True,
                   t=t, traces=traces_to_plot, figsize=(12,3))

    return f


if __name__ == '__main__':
    #plot_prefreezing_traces(1)
