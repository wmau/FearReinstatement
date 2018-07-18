from population_analyses.freezing_classifier import preprocess as preprocess
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
import numpy as np
from session_directory import find_mouse_sessions
from microscoPy_load import cell_reg


def RFE_single_session(session_index, bin_size=2, predictor='traces'):
    """
    Do recursive feature elimination on a single session, the criterion
    being decoding accuracy.
    Parameters
    ---
    session_index: session to analyze.
    bin_size: size of bin in seconds.
    predictor: 'traces' or 'events'

    Returns
    ---
    rfecv: rfecv object.

    """
    # Preprocess the data into predictor and response variables.
    X, Y = preprocess(session_index, bin_length=bin_size, predictor=predictor)

    # Define classifier.
    classifier = SVC(kernel='linear')

    # Specify parameters for RFECV and fit.
    rfecv = RFECV(estimator=classifier, step=1, cv=3,
                  scoring='accuracy')
    rfecv.fit(X,Y)

    return rfecv


def get_best_cells(rfe_obj):
    best_cells = np.where(rfe_obj.support_)[0]

    return best_cells


def rfe_overlap(mouse, bin_size=2, predictor='traces'):
    session_indices, _ = find_mouse_sessions(mouse)
    session_indices = session_indices[[0, 1, 2, 4]]

    rfe_objs, best_cells, = [], []
    for i,session in enumerate(session_indices):
        rfe_objs.append(RFE_single_session(session,
                                           bin_size=bin_size,
                                           predictor=predictor))

        best_cells.append(get_best_cells(rfe_objs[i]))

    map = cell_reg.load_cellreg_results(mouse)
    map_idx = cell_reg.find_match_map_index(session_indices)

    global_cell_idx = cell_reg.find_cell_in_map(map, map_idx[0],
                                                best_cells[0])
    overlap = []
    for session, idx, cells in zip(session_indices[1:],
                                   map_idx[1:], best_cells[1:]):
        gci = cell_reg.find_cell_in_map(map, idx, cells)

        in_both = np.asarray(list(set(global_cell_idx) & set(gci)))

        overlap.append(in_both)

    pass

if __name__ == '__main__':
    rfe_overlap('Kerberos')