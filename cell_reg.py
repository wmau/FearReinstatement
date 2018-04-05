from os import path, rename
from session_directory import load_session_list, find_mouse_directory, \
    find_mouse_sessions
import calcium_traces as ca_traces
import glob
import h5py
import numpy as np
import pickle
from helper_functions import find_dict_index

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

    # Get list of accepted neurons.
    _, accepted, _ = ca_traces.load_traces(session_index)

    tiffs = glob.glob(path.join(directory, 'ROIs_????.*'))
    # Handles BLA sessions.
    if not tiffs:
        tiffs = glob.glob(path.join(directory, 'ROIs_???.*'))

    # Rename the file so formatFootprints2.m doesn't regiser it.
    for cell, good in enumerate(accepted):
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


def load_cellreg_results(mouse):
    """
    After having already running CellRegObj, load the saved pkl file.
    :param mouse:
    :return:
    """
    # Find the directory and navigate to the pkl file.
    mouse_directory = find_mouse_directory(mouse)
    cellreg_directory = path.join(mouse_directory, 'CellRegResults')
    cellreg_file = path.join(cellreg_directory, 'CellRegResults.pkl')

    # Open pkl file.
    with open(cellreg_file, 'rb') as file:
        match_map, centroids, footprints = pickle.load(file)

    return match_map, centroids, footprints


def find_match_map_index(session_indices):
    """
    Find the index reference for the match_map matrix.
    :param session_indices:
    :return:
    """
    idx = []
    for session_index in session_indices:
        # Get all session data.
        mouse = session_list[session_index]["Animal"]
        date = session_list[session_index]["Date"]
        session = session_list[session_index]["Session"]

        # Find sessions from this mosue.
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

        filename = path.join(self.cellreg_results_directory, 'CellRegResults.pkl')

        with open(filename, 'wb') as output:
            pickle.dump([match_map, centroids, footprints],
                        output, protocol=4)


if __name__ == '__main__':
    find_match_map_index(11)
