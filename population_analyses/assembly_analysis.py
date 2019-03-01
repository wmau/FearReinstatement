import population_analyses.assembly as assembly
from session_directory import get_session
from data_preprocessing import load_and_trim
from microscoPy_load.calcium_events import load_events
from microscoPy_load.calcium_traces import load_traces
import microscoPy_load.ff_video_fixer as ff
import matplotlib.pyplot as plt
import data_preprocessing as d_pp
import numpy as np
from scipy.stats import zscore
from helper_functions import ismember
from microscoPy_load.cell_reg import load_cellreg_results, trim_match_map

def single_session_assembly(mouse, stage, neurons=None, dtype='traces',
                            session=None, trim=True, start=None,
                            end=None, method='ica', nullhyp='circ',
                            n_shuffles=1000, percentile=99,
                            compute_activity=True):
    """
    Detects assemblies from a single session using specified neurons.

    Parameters
    ---
    mouse: str
        Mouse name.

    stage: str or tuple of strs
        Stage codes.

    neurons: list-like, default all neurons
        Neurons to include.

    dtype: 'traces' (default) or 'events'
        Flag for which data type to use.

    session: FFObj, default (load it from disk)
        Supply FFObj if convenient, decreases run time.

    trim: boolean, default: True
        Flag to only include indices when mouse was in chamber.

    method: 'ica' (default) or 'pca'
        Method for getting cell assemblies.

    nullhyp:
        'mp' (default):  Marcenko-Pastur distribution
        'bin': shuffle time bins independently.
        'circ': shift time bins in time.

    n_shuffles: float, default: 1000
        Number of shuffles to perform (not used if nullhyp='mp')

    percentile: float, default: 99.5
        Defines which percentile to use (not used if nullhyp='mp')

    compute_activity: boolean, default: True
        Flag to compute ensemble activity.


    Returns
    ---
    patterns: (assemblies, neurons) ndarray
        Weights for neurons per assembly.

    significance:
        PCA() object with new attributes, see documentation for runPatterns()

    ensemble_timecourse: (assemblies, time) ndarray
        Activation levels for each assembly.

    """
    # Get session index and load data.
    session_index = get_session(mouse, stage)[0]

    # Only look at timestamps when the mouse is in the fear conditioning box
    if trim:
        activity_matrix = load_and_trim(session_index, dtype=dtype, session=session,
                                        neurons=neurons, do_zscore=False, start=start,
                                        end=end)[0]

    # Or the whole recording session
    else:
        if dtype is 'events':
            activity_matrix = load_events(session_index)[0]
            activity_matrix = activity_matrix[neurons]
        elif dtype is 'traces':
            activity_matrix = load_traces(session_index)[0]
            activity_matrix = activity_matrix[neurons]
        else:
            raise ValueError('Data type not recognized')

    # Find patterns.
    patterns, significance, zscored_activity = \
        assembly.runPatterns(activity_matrix, method=method, nullhyp=nullhyp,
                             nshu=n_shuffles, percentile=percentile)

    # Get activity of patterns.
    if compute_activity:
        ensemble_timecourse = assembly.computeAssemblyActivity(patterns, zscored_activity)
    else:
        ensemble_timecourse = None

    return patterns, significance, ensemble_timecourse


def assembly_freezing(mouse, stage, neurons=None, dtype='traces',
                      trim=True, method='ica', nullhyp='circ',
                      n_shuffles=1000, percentile=99):
    """
    Plot ensemble activation strength over freezing bouts.

    """
    # Get session data.
    session_index = get_session(mouse, stage)[0]
    session = ff.load_session(session_index)

    # Get time and freezing vectors then trim.
    t = session.imaging_t.copy()
    freezing = session.imaging_freezing.copy()
    if trim:
        t = d_pp.trim_session(t, session.mouse_in_cage)
        t -= t.min()

        freezing = d_pp.trim_session(freezing, session.mouse_in_cage)

    # Get ensemble patterns and timecourse.
    patterns, significance, ensemble_timecourse = \
        single_session_assembly(mouse, stage, neurons=neurons, dtype=dtype,
                                trim=trim, method=method, nullhyp=nullhyp,
                                n_shuffles=n_shuffles, percentile=percentile,
                                session=session)

    # Get activation threshold for each ensemble.
    ensemble_active = threshold_ensemble_activity(ensemble_timecourse)

    # Plot ensemble timecourse with freezing.
    for ensemble, active in zip(ensemble_timecourse,
                                ensemble_active):
        fig, ax = plt.subplots(1)
        ax.plot(t, ensemble, linewidth=0.2, color='k')
        ax.fill_between(t, -0.5, ensemble.max()+0.5, freezing,
                        facecolor='r', alpha=0.4)
        ax.fill_between(t, -0.5, ensemble.max()+0.5, active,
                        facecolor='g', alpha=0.4)


def threshold_ensemble_activity(ensemble_timecourse, n=2):
    """
    Thresholds ensemble activity n standard deviations above the mean.

    Parameters
    ---
    ensemble_timecourse: (ensembles, time) array from single_session_assembly
        Time series of ensemble activation strength.

    n: scalar, default: 2
        Number of standard deviations above the mean to be considered active.

    """
    activation_thresh = np.mean(ensemble_timecourse, axis=1) + \
                        n*np.std(ensemble_timecourse, axis=1)
    thr = np.matlib.repmat(activation_thresh,
                           ensemble_timecourse.shape[1], 1).T
    ensemble_active = ensemble_timecourse > thr

    return ensemble_active


def cross_day_ensemble_activity(mouse, template_session, test_sessions,
                                neurons=None, dtype='traces',
                                session=None, trim=True, start=None,
                                end=None, method='ica', nullhyp='circ',
                                n_shuffles=1000, percentile=99):


    # Make list of template and test sessions then get session numbers.
    all_sessions = [template_session]
    all_sessions.extend(test_sessions)
    session_indices = get_session(mouse, all_sessions)[0]

    # Load the map of cell matchings.
    map = load_cellreg_results(mouse)

    # Extract the neuron numbers from template_session from those that
    # were active on all days.
    trimmed_map = trim_match_map(map, session_indices)

    if neurons is None:
        template_neurons = trimmed_map[:,0]
    else: # Handles cases where you specify neurons, some may be eliminated.
        # Convert this to an array if it's a list.
        if type(neurons) is not np.ndarray:
            neurons = np.asarray(neurons)

        # Trim the matching map some more to only include specified neurons
        # from template session.
        is_in, idx = ismember(trimmed_map[:,0], neurons)
        trimmed_map = trimmed_map[idx[is_in]]
        template_neurons = trimmed_map[:,0]

    # Load all the session data and neural data (and zscore).
    session_dict = {}
    z_activity_matrices = {}
    t_vectors = {}
    for i, this_session, neuron_order in zip(session_indices,
                                             all_sessions,
                                             trimmed_map.T):
        session_dict[this_session] = ff.load_session(i)

        # If only including fear chamber timestamps. Be sure to reorder
        # the neurons here to match the template.
        if trim:
            traces, t = load_and_trim(i, do_zscore=True)
            t -= t.min()
            z_activity_matrices[this_session] = traces[neuron_order]
            t_vectors[this_session] = t

        else:
            traces, t = load_traces(i)
            traces = zscore(traces, axis=1)
            z_activity_matrices[this_session] = traces[neuron_order]
            t_vectors[this_session] = t

    # Get assembly structure.
    patterns, significance, ensemble_timecourse = \
        single_session_assembly(mouse, template_session,
                                neurons=template_neurons, dtype=dtype,
                                session=session_dict[template_session],
                                trim=trim, start=start, end=end,
                                method=method, nullhyp=nullhyp,
                                n_shuffles=n_shuffles,
                                percentile=percentile,
                                compute_activity=False)

    # Get ensemble activity on other days.
    ensemble_strengths = {}
    ensemble_activations = {}
    for this_session, activity_matrix in z_activity_matrices.items():
        ensemble_strengths[this_session] = \
            assembly.computeAssemblyActivity(patterns, activity_matrix)

        ensemble_activations[this_session] = \
            threshold_ensemble_activity(ensemble_strengths[this_session])

    # Plot activations of ensembles for each session.
    fig, ax = plt.subplots(len(all_sessions), significance.nassemblies,
                           squeeze=False, sharey=True)
    for i, this_session in enumerate(all_sessions):
        ensembles = ensemble_strengths[this_session]
        t = t_vectors[this_session]
        for j, ensemble in enumerate(ensembles):
            ax[i, j].plot(t, ensemble)
            ax[i, j].fill_between(t, -0.5, ensemble.max() + 0.5,
                                  ensemble_activations[this_session][j],
                                  alpha=0.4, facecolor='g')
            ax[i, j].set_title(this_session)

    return ensemble_strengths, ensemble_activations,



if __name__ == '__main__':
    cross_day_ensemble_activity('Mundilfari','E1_1',['E2_1','RE_1'])