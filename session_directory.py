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