from os import path, rename
from session_directory import load_session_list, find_mouse_directory, \
    find_mouse_sessions
import glob
import h5py
import numpy as np
import pickle
from helper_functions import find_dict_index, ismember
from cell_data_compiler import CellData
from plot_helper import ScrollPlot, neuron_number_title
import plot_functions as plot_funcs

session_list = load_session_list()


def rename_rejected_ROIs(session_index):
    """
    Rename the bad ROIs so that they don't get registered.

    :param session_index:
    :return:
    """
    # Get directory of session.
    directory = session_list[session_index]["Location"]
    directory = path.join(directory, "ROIs")

    data = CellData(session_index)

    tiffs = glob.glob(path.join(directory, 'ROIs_????.*'))
    # Handles BLA sessions.
    if not tiffs:
        tiffs = glob.glob(path.join(directory, 'ROIs_???.*'))

    # Rename the file so formatFootprints2.m doesn't regiser it.
    for cell, good in enumerate(data.accepted):
        if not good:
            new_name = tiffs[cell]  # Python strings are immutable.
            new_name = new_name + '_'  # So use this silly method instead.

            rename(tiffs[cell], new_name)


def exclude_bad_cells_in_this_mouse(mouse):
    """
    Rename all bad ROIs for a mouse.

    :param mouse:
    :return:
    """
    for session_index, entry in enumerate(session_list):
        if entry["Animal"] == mouse:
            rename_rejected_ROIs(session_index)


def load_cellreg_results(mouse, mode='map'):
    """
    After having already running CellRegObj, load the saved pkl file.
    :param mouse:
    :return:
    """
    # Find the directory and navigate to the pkl file.
    mouse_directory = find_mouse_directory(mouse)
    cellreg_directory = path.join(mouse_directory, 'CellRegResults')
    cellreg_file = path.join(cellreg_directory, 'CellRegResults.pkl')
    cellreg_footprints_file = path.join(cellreg_directory, 'CellRegFootprints.pkl')
    cellreg_centroids_file = path.join(cellreg_directory, 'CellRegCentroids.pkl')

    # Open pkl file.
    if mode == 'map':
        with open(cellreg_file, 'rb') as file:
            match_map = pickle.load(file)
        return match_map

    elif mode == 'footprints':
        with open(cellreg_footprints_file, 'rb') as file:
            footprints = pickle.load(file)
        return footprints

    elif mode == 'centroids':
        with open(cellreg_centroids_file, 'rb') as file:
            centroids = pickle.load(file)
        return centroids

    else:
        raise ValueError('Wrong mode.')


def find_match_map_index(session_indices):
    """
    Find the index reference for the match_map matrix.
    :param session_indices:
    :return:
    """
    # If it's just one session...
    if isinstance(session_indices, int):
        # Get all session data.
        mouse = session_list[session_indices]["Animal"]
        date = session_list[session_indices]["Date"]
        session = session_list[session_indices]["Session"]

        # Find sessions from this mouse.
        _, sessions = find_mouse_sessions(mouse)

        # Find all sessions with the specified date and session number.
        idx_date = find_dict_index(sessions, "Date", date)
        idx_session = find_dict_index(sessions, "Session", session)

        # Get session that matches the specified session_index.
        matched_session = list(set(idx_date) & set(idx_session))

        # Make sure there's only one.
        assert len(matched_session) is 1, "Multiple sessions with these fields!"

        idx = matched_session[0]
    else:
        idx = []
        for session_index in session_indices:
            # Get all session data.
            mouse = session_list[session_index]["Animal"]
            date = session_list[session_index]["Date"]
            session = session_list[session_index]["Session"]

            # Find sessions from this mouse.
            _, sessions = find_mouse_sessions(mouse)

            # Find all sessions with the specified date and session number.
            idx_date = find_dict_index(sessions, "Date", date)
            idx_session = find_dict_index(sessions, "Session", session)

            # Get session that matches the specified session_index.
            matched_session = list(set(idx_date) & set(idx_session))

            # Make sure there's only one.
            assert len(matched_session) is 1, "Multiple sessions with these fields!"

            idx.append(matched_session[0])

    return idx


def trim_match_map(match_map, session_indices, active_all_days=True):
    idx = find_match_map_index(session_indices)

    trimmed_map = match_map[:, idx]

    if active_all_days:
        detected_all_days = (trimmed_map > -1).all(axis=1)
        trimmed_map = trimmed_map[detected_all_days, :]

    return trimmed_map


def plot_footprints_over_days(session_index, neurons):
    """
    Plots specified cells across all sessions.
    :param session_index:
    :param neurons:
    :return:
    """
    # Get the mouse name.
    mouse = session_list[session_index]["Animal"]

    # Load the footprints and map.
    footprints = load_cellreg_results(mouse, mode='footprints')
    cell_map = load_cellreg_results(mouse)
    n_sessions = cell_map.shape[1]

    # Get the session in map that corresponds to the session_index.
    # Then get all row numbers corresponding to the neurons.
    map_index = find_match_map_index(session_index)
    _, cell_index = ismember(cell_map[:, map_index], neurons)

    # Create list of arrays containing footprints.
    # List of size N where N is number of neurons. Each list item is
    # a set of footprints, one for each day, of the same neuron.
    footprints_to_plot = []
    x_dim = footprints[0].shape[1]
    y_dim = footprints[0].shape[2]
    # For each row number...
    for cell in cell_index:
        # Preallocate.
        this_cell_footprints = np.zeros((n_sessions, x_dim, y_dim))
        # For each day, get the cell number (index for footprints).
        for day in range(n_sessions):
            cell_number = cell_map[cell, day]

            # Only get the footprint if the cell was matched that day.
            if cell_number > -1:
                this_cell_footprints[day] = footprints[day][cell_number]

        # Append to list.
        footprints_to_plot.append(this_cell_footprints)

    # Plot.
    f = ScrollPlot(plot_funcs.plot_footprints_over_days,
                   n_rows=1, n_cols=n_sessions,
                   share_x=True, share_y=True,
                   footprints=footprints_to_plot,
                   figsize=(12, 3))

    return f

def find_cell_in_map(map, map_index, neurons):
    _, global_cell_index = ismember(map[:, map_index], neurons)

    return global_cell_index.astype(int)


class CellRegObj:
    def __init__(self, mouse):
        self.mouse = mouse
        self.data, self.file, self.mouse_directory = \
            self.read_cellreg_output()
        self.compile_cellreg_data()

    def read_cellreg_output(self):
        """
        Reads the .mat file.
        :return:
        """
        # Get directory.
        mouse_directory = find_mouse_directory(self.mouse)

        # Get the .mat file name.
        self.cellreg_results_directory = path.join(mouse_directory, 'CellRegResults')
        cellreg_file = glob.glob(path.join(self.cellreg_results_directory, 'cellRegistered*.mat'))
        assert len(cellreg_file) is 1, "Multiple cell registration files!"
        cellreg_file = cellreg_file[0]

        # Load it.
        file = h5py.File(cellreg_file)
        data = file['cell_registered_struct']

        return data, file, mouse_directory

    def process_registration_map(self):
        # Get the cell_to_index_map. Reading the file transposes the
        # matrix. Transpose it back.
        cell_to_index_map = self.data['cell_to_index_map'].value.T

        # Matlab indexes starting from 1. Correct this.
        match_map = cell_to_index_map - 1

        return match_map.astype(int)

    def process_spatial_footprints(self):
        # Get the spatial footprints after translations.
        footprints_reference = self.data['spatial_footprints_corrected'].value[0]

        footprints = []
        for idx in footprints_reference:
            # Float 32 takes less memory.
            session_footprints = np.float32(np.transpose(self.file[idx].value, (2, 0, 1)))
            footprints.append(session_footprints)

        return footprints

    def process_centroids(self):
        # Also get centroid positions after translations.
        centroids_reference = self.data['centroid_locations_corrected'].value[0]

        centroids = []
        for idx in centroids_reference:
            session_centroids = self.file[idx].value.T
            centroids.append(session_centroids)

        return centroids

    def compile_cellreg_data(self):
        # Gets registration information. So far, this consists of the
        # cell to index map, centroids, and spatial footprints.
        match_map = self.process_registration_map()
        centroids = self.process_centroids()
        footprints = self.process_spatial_footprints()

        filename = path.join(self.cellreg_results_directory,
                             'CellRegResults.pkl')
        filename_footprints = path.join(self.cellreg_results_directory,
                                        'CellRegFootprints.pkl')
        filename_centroids = path.join(self.cellreg_results_directory,
                                       'CellRegCentroids.pkl')

        with open(filename, 'wb') as output:
            pickle.dump(match_map, output, protocol=4)
        with open(filename_footprints, 'wb') as output:
            pickle.dump(footprints, output, protocol=4)
        with open(filename_centroids, 'wb') as output:
            pickle.dump(centroids, output, protocol=4)


if __name__ == '__main__':
    plot_footprints_over_days(0, [1, 2, 5])
