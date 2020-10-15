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
from scipy.stats import wilcoxon
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
    for ensemble_strengths, ensemble_activations \
            in zip(ensemble_timecourse, ensemble_active):
        plot_ensemble_activations_with_freezing(t, ensemble_strengths,
                                                ensemble_activations, freezing)


def plot_ensemble_activations_with_freezing(t, ensemble_strengths,
                                            ensemble_activations,
                                            freezing,
                                            ax=None):
    """
    Plots ensemble activation strength timecourse overlaid with
    freezing bouts.

    Parameters
    ---
    t: list-like
        Time vector.

    ensemble_strengths: (time,) array
        Matrix of ensemble activation strengths.

    ensemble_activations: (time,) array
        Boolean matrix of time bins above threshold.

    freezing: list-like
        Boolean array of freezing status.

    ax: Axes object, default: None
        Axis to plot on. If not specified, make a new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1)

    ax.plot(t, ensemble_strengths, linewidth=0.2, color='k')
    ax.fill_between(t, -0.5, ensemble_strengths.max()+0.5, freezing,
                    facecolor='r', alpha=0.4)
    ax.fill_between(t, -0.5, ensemble_strengths.max()+0.5,
                    ensemble_activations,
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
    activation_thresh = np.nanmean(ensemble_timecourse, axis=1) + \
                        n*np.nanstd(ensemble_timecourse, axis=1)
    thr = np.matlib.repmat(activation_thresh,
                           ensemble_timecourse.shape[1], 1).T
    ensemble_active = ensemble_timecourse > thr

    return ensemble_active


def cross_day_ensemble_activity(mouse, template_session, test_sessions,
                                neurons=None, dtype='traces',
                                session=None, trim=True, start=None,
                                end=None, method='ica', nullhyp='circ',
                                n_shuffles=500, percentile=99.5, plot=True):
    """
    Gets the ensemble activity across days, keeping the neuron weights
    from a template session and applying it to test sessions.

    Parameters
    ---
    mouse: str
        Mouse name.

    template_session: str
        Session type (e.g., 'E1_1').

    test_sessions: list
        List of session types for that mouse.

    Returns
    ---


    """

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
    freezing = {}
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
            freezing[this_session] = load_and_trim(i, dtype='freezing')[0]

        else:
            traces, t = load_traces(i)
            traces = zscore(traces, axis=1)
            z_activity_matrices[this_session] = traces[neuron_order]
            t_vectors[this_session] = t
            freezing[this_session] = \
                session_dict[this_session].imaging_freezing

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
    ensemble_strengths = {}     # Ensemble activation strength.
    ensemble_activations = {}   # Supra-threshold epochs.
    for this_session, activity_matrix in z_activity_matrices.items():
        ensemble_strengths[this_session] = \
            assembly.computeAssemblyActivity(patterns, activity_matrix)

        ensemble_activations[this_session] = \
            threshold_ensemble_activity(ensemble_strengths[this_session])

    # Plot activations of ensembles for each session.
    if plot:
        for n_assembly in range(significance.nassemblies):
            fig, ax = plt.subplots(len(all_sessions), sharey=True)

            for i, this_session in enumerate(all_sessions):
                # Get ensemble activation strength and time vectors.
                ensembles = ensemble_strengths[this_session][n_assembly]
                activity = ensemble_activations[this_session][n_assembly]
                t = t_vectors[this_session]
                frozen = freezing[this_session]

                plot_ensemble_activations_with_freezing(t, ensembles,
                                                        activity, frozen,
                                                        ax=ax[i])

    # for i, this_session in enumerate(all_sessions):
    #     # Get ensemble activation strength and time vectors.
    #     ensembles = ensemble_strengths[this_session]
    #     t = t_vectors[this_session]
    #
    #     for j, ensemble in enumerate(ensembles):
    #         # Get supra-threshold epochs.
    #         active = ensemble_activations[this_session][j]
    #
    #         ax[i, j].plot(t, ensemble)      # Plot activation strength.
    #         ax[i, j].scatter(t[active],     # Mark supra-threshold epochs.
    #                          np.matlib.repmat(np.nanmax(ensemble) + 0.5, 1,
    #                                           np.nansum(active)),
    #                          s=1, c='r')
    #         ax[i, j].set_title(this_session)

    # Normalize number of activations by session duration.
    norm_activations = {this_session: np.nansum(activations, axis=1)/activations.shape[1]
                        for this_session, activations
                        in ensemble_activations.items()}

    return ensemble_strengths, ensemble_activations, patterns,\
           significance, norm_activations, freezing, session_dict


def prefreezing_assembly_activations(mouse, template_session, test_sessions,
                                     neurons=None, dtype='traces',
                                     session=None, trim=True, start=None,
                                     end=None, method='ica', nullhyp='circ',
                                     n_shuffles=500, percentile=99.5
                                     ):

    (ensemble_strengths,
     ensemble_activations,
     patterns,
     significance,
     norm_activations,
     freezing,
     session_dict) = \
        cross_day_ensemble_activity(mouse, template_session, test_sessions,
                                    neurons=neurons, dtype=dtype, trim=trim,
                                    start=start, end=end, method=method,
                                    nullhyp=nullhyp, n_shuffles=n_shuffles,
                                    percentile=percentile)

    plot_prefreezing_ensemble_activations(ensemble_strengths['E1_1'],
                                          session_dict['E1_1'])

    return


def plot_prefreezing_ensemble_activations(ensemble_strength, session,
                                          window=(-2,2),
                                          freeze_duration_threshold=1.25,
                                          plot_bool=True):
    """
    Plots the average activity for an ensemble centered around the start of
    freezing bouts.

    Parameters
    ---
    window: tuple, (-seconds before freezing, seconds after)
    freeze_duration_threshold: scalar, freezing duration must be longer
        than this value (seconds)
    plot_bool: boolean, whether or not to plot

    Returns
    ---

    """

    # Load data and get freezing timestamps.

    freeze_epochs = session.get_freezing_epochs_imaging_framerate()

    # Chop off indices where mouse is not in chamber.
    mouse_in_chamber = np.where(session.mouse_in_cage)[0]
    freeze_epochs = freeze_epochs[np.any(freeze_epochs > mouse_in_chamber[0],
                                         axis=1),:]
    freeze_epochs = freeze_epochs[np.any(freeze_epochs < mouse_in_chamber[-1],
                                         axis=1),:]
    freeze_epochs -= mouse_in_chamber[0]

    # Eliminate freeze epochs that don't pass the duration threshold.
    good_epochs = np.squeeze(np.diff(freeze_epochs) >
                             freeze_duration_threshold * 20)
    freeze_epochs = freeze_epochs[good_epochs, :]

    # Get sizes for all dimensions of our array.
    n_assemblies = ensemble_strength.shape[0]
    n_freezes = freeze_epochs.shape[0]
    freeze_duration = abs(np.ceil(np.diff(window)*20)).astype(int)[0]

    # Plot each cell and each freeze epoch.
    prefreezing_traces = np.zeros((n_assemblies,
                                   n_freezes,
                                   freeze_duration))
    t = np.arange(window[0], window[1], 1/20)
    for n, trace in enumerate(ensemble_strength):
        for i, epoch in enumerate(freeze_epochs):
            start = epoch[0] - (abs(window[0]) * 20)
            stop = epoch[0] + (abs(window[1]) * 20)
            prefreezing_traces[n, i, :] = trace[start:stop]

    for trace in prefreezing_traces:
        plt.figure()
        plt.plot(trace.T)

    return


## IDEAS:
    # Count rate of time-normalized ensemble activations
    # (from either FC or Ext1 templates) and correlate to freezing
    # during Recall. DONE - Corr_Activations_to_Freezing, positive
    # correlation for BLA mice, not significant for CA1.

    # Do the above for neutral context. DONE - not significant for neutral
    # context.

    # Count rate of time-normalized ensemble activations (from Recall)
    # over Ext1 and Ext2. DONE - RecallEnsembleTimecourse, ambiguous results,
    # probably need to separate ensembles.

    # Compare Ext1 ensemble template activity in shock recall versus
    # neutral recall. Done, sort of: Corr_Activations_to_Freezing, no
    # no difference in BLA.

    # Try setting thresholds from the entire mouse, not just within session. x

    # Look at weights for cells that respond to freezing. Use the recall
    # session. Using the same extinction session to find freezing cells
    # and assembly weights might be circular.

    # Look at pre-freezing traces, but instead of traces, ensemble activations


if __name__ == '__main__':
    (strengths,
     activations,
     patterns,
     significance,
     norm,
     freezing,
     session_dict) = \
        prefreezing_assembly_activations('Helene','FC',
                                         ['E1_1','E2_1','RE_1'])

    # mice = (
    #     'Kerberos',
    #     'Nix',
    #     'Pandora',
    #     'Calypso',
    #     'Helene',
    #     'Hyperion',
    #     'Janus',
    #     'Kepler',
    #     'Mundilfari',
    #     'Aegir',
    #     'Skoll',
    #     'Telesto',
    # )
    #
    # strengths, activations, patterns, significance, norm, session_dict = [], [], [], [], [], []
    # for mouse in mice:
    #     try:
    #         a, b, c, d, e, f = \
    #             cross_day_ensemble_activity(mouse,'FC', ['E1_1', 'E2_1','RE_1'],
    #                                         n_shuffles=500)
    #     except:
    #         a, b, c, d, e, f = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    #
    #     strengths.append(a)
    #     activations.append(b)
    #     patterns.append(c)
    #     significance.append(d)
    #     norm.append(e)
    #     session_dict.append(f)
    #
    #     print(mouse + ' done')
    #
    # BLA = mice[6:]
    # FC, E1_1, E2_1, RE_1 = [], [], [], []
    # for mouse, n in zip(mice[:6], norm[:6]):
    #     try:
    #         for fc, e1, e2, re in zip(n['FC'],
    #                                   n['E1_1'],
    #                                   n['E2_1'],
    #                                   n['RE_1']):
    #             FC.append(fc)
    #             E1_1.append(e1)
    #             E2_1.append(e2)
    #             RE_1.append(re)
    #
    #     except:
    #         FC.append(np.nan)
    #         E1_1.append(np.nan)
    #         E2_1.append(np.nan)
    #         RE_1.append(np.nan)
    #
    # X = np.stack((np.asarray(FC),
    #               np.asarray(E1_1),
    #               np.asarray(E2_1),
    #               np.asarray(RE_1)),
    #              axis=1)
    # plt.plot(X.T)
    #
    # stat, p = wilcoxon(X[:,1], X[:,2])
