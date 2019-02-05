# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:23:20 2018

@author: William Mau
"""

from os import path, chdir
from pickle import load
from csv import DictReader
from pickle import dump
from helper_functions import find_dict_index
import numpy as np

master_directory = 'U:\Fear conditioning project_Mosaic2\SessionDirectories'


def make_session_list(csv_directory=master_directory):
    """

    Make a list of recording sessions by reading from an editable CSV found in
    csv_directory.

    """

    # Go to the directory containing the CSV file.
    chdir(csv_directory)

    # Define "structure array". Not going to bother to learn how to preallocate
    # yet; this should only ever be a few entries long per project.
    session_directories = []
    with open('SessionDirectories.csv', 'r') as file:
        reader = DictReader(file)

        # Consolidate entries.
        for entry in reader:
            session_directories.append({"Animal": entry['Animal'],
                                        "Date": entry['Date'],
                                        "Location": entry['Location'],
                                        "Session": entry['Session'],
                                        "Region": entry['Region'],
                                        "Notes": entry['Notes']})

    # Save.
    with open('SessionDirectories.pkl', 'wb') as output:
        dump(session_directories, output)

    return session_directories


def load_session_list():
    file = path.join(master_directory, 'SessionDirectories.pkl')
    session_list = load(open(file, 'rb'))

    return session_list


def check_session(session_index):
    """
    Displays all the details of that session as recorded in the CSV file.

    :param
        session_index: number corresponding to a session.
    :return
        Printed session information.
    """
    session_list = load_session_list()

    print("Mouse: " + session_list[session_index]["Animal"])
    print("Date: " + session_list[session_index]["Date"])
    print("Session # that day: " + session_list[session_index]["Session"])
    print("Location: " + session_list[session_index]["Location"])
    print("Region: " + session_list[session_index]["Region"])
    print("Notes: " + session_list[session_index]["Notes"])


def find_mouse_directory(mouse):
    session_list = load_session_list()

    # Seems really inefficient but functional for now. Searches the directory containing that
    # mouse's data folders.
    mouse_not_found = True
    while mouse_not_found:
        for session in session_list:
            if session["Animal"] == mouse:
                mouse_directory = path.split(session["Location"])[0]
                mouse_not_found = False
                break

    return mouse_directory


def find_mouse_sessions(mouse):
    session_list = load_session_list()

    sessions = list(filter(lambda sessions: sessions["Animal"] == mouse,
                    session_list))

    idx = np.asarray(find_dict_index(session_list, "Animal", mouse))

    return idx, sessions


def get_session_stage(stages_tuple):
    """
    Gets session index in session_list by searching for strings in
    the Session key. Can input multiple session stages.

    Parameters
    ---
    stages_tuple: tuple of string keywords, valid inputs:
        FC: fear conditioning
        E1_1: extinction day 1, fear conditioning context.
        E1_2: extinction day 1, harmless context.
        E2_1: extinction day 2, fear conditioning context.
        E2_2: extinction day 2, harmless context.
        RI: reinstatement.
        RE_1: recall, fear conditioning context.
        RE_2: recall, harmless context.

    Return
    ---
    indices: indices of session_list matching str.
    types: session types for each entry in indices.

    """
    session_list = load_session_list()

    # Make string a tuple.
    if isinstance(stages_tuple, str):
        stages_tuple = (stages_tuple,)

    indices = []
    stages = []
    for session_stage in stages_tuple:
        # Get all the indices matching the type.
        indices_for_this_stage = find_dict_index(session_list,
                                                 "Session",
                                                 session_stage)
        indices.extend(indices_for_this_stage)

        # Compile the list of session_types.
        stages.extend([session_stage
                       for session in indices_for_this_stage])

    return indices, stages


def get_session(mouse, stages_tuple):
    """
    Gets a specific session for a mouse.

    Parameters
    ---
    mouse: str, mouse name.
    stages_tuple: tuple of strings, session stages. See
    get_session_type() for valid inputs.

    Return
    ---
    sessions: indices of session_list.
    stages: strings of stages.

    """
    mouse_sessions, _ = find_mouse_sessions(mouse)
    sessions_at_that_stage, stages = get_session_stage(stages_tuple)

    # Get intersection.
    sessions = [i for i in sessions_at_that_stage
                if i in mouse_sessions]
    stages = [stage for i, stage in zip(sessions_at_that_stage, stages)
              if i in mouse_sessions]

    if len(sessions) is 1:
        sessions = sessions[0]
        stages = stages[0]

    return sessions, stages


def get_region(mouse):
    """
    Gets the recording region for that mouse.

    Parameters
    ---
    mouse: str, mouse name.

    Return
    ---
    region: str, region name.
    """
    mouse_sessions = find_mouse_sessions(mouse)[0]
    session_list = load_session_list()

    region = session_list[mouse_sessions[0]]["Region"]

    return region


if __name__ == '__main__':
    session_list = load_session_list()
