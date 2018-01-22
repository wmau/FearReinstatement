# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:09:29 2018

@author: William Mau
"""

from cell_data_compiler import CellData
from session_directory import load_session_list


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

    return data.events