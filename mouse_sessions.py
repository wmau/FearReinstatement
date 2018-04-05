# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:42:45 2018

@author: William Mau
"""

# Import libraries
from os import chdir
from csv import DictReader
from pickle import dump


def make_session_list(csv_directory):
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
                                        "Notes": entry['Notes']})

    # Save.
    with open('SessionDirectories.pkl', 'wb') as output:
        dump(session_directories, output)

    return session_directories
