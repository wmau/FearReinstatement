# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:11:14 2018

@author: William Mau
"""

from cell_data_compiler import CellData
from plot_helper import ScrollPlot
from session_directory import load_session_list

session_list = load_session_list()

def check_session(session_index):
    """
    Displays all the details of that session as recorded in the CSV file.

    :param
        session_index: number corresponding to a session.
    :return
        Printed session information.
    """

    print("Mouse: " + session_list[session_index]["Animal"])
    print("Date: " + session_list[session_index]["Date"])
    print("Session # that day: " + session_list[session_index]["Session"])
    print("Location: " + session_list[session_index]["Location"])
    print("Notes: " + session_list[session_index]["Notes"])

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

    data = CellData(session_index)

    return data.traces, data.accepted, data.t


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
    f = ScrollPlot(t, traces[neurons], 'Time (s)', '%DF/F')

    # Gets the ScrollPlot object.
    return f