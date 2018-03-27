from os import path, rename
from session_directory import load_session_list, find_mouse_directory
import calcium_traces as ca_traces
import glob
import h5py
import numpy as np
import pickle

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
            new_name = tiffs[cell]       # Python strings are immutable.
            new_name = new_name + '_'    # So use this silly method instead.

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
        mouse_directory = find_mouse_directory(mouse)
        cellreg_directory = path.join(mouse_directory,'CellRegResults')
        cellreg_file = path.join(cellreg_directory,'CellRegResults.pkl')
        with open(cellreg_file, 'rb') as file:
            match_map, centroids, footprints = pickle.load(file)

        return match_map, centroids, footprints

class CellRegObj:
    def __init__(self,mouse):
        self.mouse = mouse
        self.data,self.file,self.mouse_directory = self.read_cellreg_output()
        self.compile_cellreg_data()

    def read_cellreg_output(self):
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
        match_map = self.data['cell_to_index_map'].value.T

        return match_map

    def process_spatial_footprints(self):
        footprints_reference = self.data['spatial_footprints_corrected'].value[0]

        footprints = []
        for idx in footprints_reference:
            session_footprints = np.float32(np.transpose(self.file[idx].value,(2,0,1)))
            footprints.append(session_footprints)

        return footprints

    def process_centroids(self):
        centroids_reference = self.data['centroid_locations_corrected'].value[0]

        centroids = []
        for idx in centroids_reference:
            session_centroids = self.file[idx].value.T
            centroids.append(session_centroids)

        return centroids

    def compile_cellreg_data(self):
        match_map = self.process_registration_map()
        centroids = self.process_centroids()
        footprints = self.process_spatial_footprints()

        filename = path.join(self.cellreg_results_directory,'CellRegResults.pkl')

        with open(filename,'wb') as output:
            pickle.dump([match_map, centroids, footprints],output,protocol=4)


if __name__ == '__main__':
    load_cellreg_results('Kerberos')