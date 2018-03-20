# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:23:20 2018

@author: William Mau
"""

from os import path
from pickle import load

master_directory = 'U:\Fear conditioning project_Mosaic2\SessionDirectories'

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
    print("Notes: " + session_list[session_index]["Notes"])