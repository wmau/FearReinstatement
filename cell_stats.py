from os import path
from glob import glob
from session_directory import load_session_list

session_list = load_session_list()

def get_number_of_ICs(session_index):
    directory = path.join(session_list[session_index]["Location"],'ROIs')
    ROIs = glob(path.join(directory,'ROIs_C*.*'))

    n_ICs = len(ROIs)
    return n_ICs
