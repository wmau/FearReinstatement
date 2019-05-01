from session_directory import get_session
import data_preprocessing as d_pp
from microscoPy_load.ff_video_fixer import load_session
from helper_functions import find_closest, ismember
import numpy as np
import matplotlib.pyplot as plt
import microscoPy_load.cell_reg as cell_reg
from scipy.stats import pearsonr, spearmanr
import microscoPy_load.calcium_events as ca_events
from scipy.stats.mstats import zscore

def time_lapse_corr(mouse, session, ref_session='FC', bin_size=1,
                    slice_size=60, ref_mask_start=None, plot_flag=True,
                    ref_indices=None, ref_neurons=None, corr=pearsonr):
    """
    Takes the reference session and computes the average event rate for
    each cell during that session. Then correlate those rates to rates
    during a session of interest, taken from progressive slices.

    Parameters
    ---
    mouse: string, mouse name.
    session: string, session name.
    ref_session: string, session name for the reference, usually the
    fear conditioning session.
    bin_size: scalar, size of bin, in seconds.
    slice_size: scalar, size of slices of sessions, in seconds.
    ref_mask_start: scalar, timestamp from which to calculate reference
    firing rate vector, from start of session.
    plot_flag: boolean, whether to plot correlation vector.
    """

    session_index = get_session(mouse, (ref_session, session))[0]

    # If ref_mask_start is a scalar, clip the time series starting from
    # the specified timestamp.
    ff_ref = load_session(session_index[0])
    data, t = ca_events.load_events(session_index[0])
    data[data > 0] = 1

    if ref_mask_start is not None:
        ref_mask = np.zeros(ff_ref.mouse_in_cage.shape, dtype=bool)
        start_idx = find_closest(ff_ref.imaging_t, ref_mask_start)[1]
        end_idx = np.where(ff_ref.mouse_in_cage)[0][-1]
        ref_mask[start_idx:end_idx] = True
    else:
        ref_mask = None

    if ref_indices is not None:
        assert ref_mask is None, "ref_mask_start must be None to use this feature"

        ref_mask = np.zeros(ff_ref.mouse_in_cage.shape, dtype=bool)
        if ref_indices == 'homecage1':
            end_idx = np.where(ff_ref.mouse_in_cage)[0][0]
            ref_mask[:end_idx] = True
        elif ref_indices == 'homecage2':
            start_idx = np.where(ff_ref.mouse_in_cage)[0][-1]
            ref_mask[start_idx:] = True

    map = cell_reg.load_cellreg_results(mouse)
    trimmed_map = cell_reg.trim_match_map(map, session_index)

    if ref_neurons is None:
        ref_neurons = trimmed_map[:,0]
        neurons = trimmed_map[:,1]
    else:
        in_there, idx = ismember(trimmed_map[:, 0], ref_neurons)
        ref_neuron_rows = idx[in_there]
        neurons = trimmed_map[ref_neuron_rows, 1]
        ref_neurons = trimmed_map[ref_neuron_rows, 0]
        assert len(neurons) == len(np.unique(neurons)), 'Error.'

    # Get average event rates from the reference session.
    ref_event_rates = d_pp.get_avg_event_rate(mouse, ref_session,
                                              data=data, t=t,
                                              session=ff_ref,
                                              bin_size=bin_size,
                                              mask=ref_mask,
                                              neurons=ref_neurons)
    # if z:
    #     ref_event_rates = zscore(ref_event_rates)

    # Load other session.
    ff_session = load_session(session_index[1])
    data, t = ca_events.load_events(session_index[1])
    data[data > 0] = 1

    # Get indices for when the mouse is in the chamber, then slice them.
    in_cage = np.where(ff_session.mouse_in_cage)[0]
    bins = d_pp.make_bins(in_cage, slice_size*20)
    binned_in_cage = d_pp.bin_time_series(in_cage, bins)

    # Make slice masks.
    masks = np.zeros((len(binned_in_cage),
                      len(ff_session.mouse_in_cage)), dtype=bool)
    for i, indices in enumerate(binned_in_cage):
        masks[i,indices] = True

    event_rates = np.zeros((masks.shape[0], len(neurons)))
    for i, mask in enumerate(masks):
        event_rates[i,:] = d_pp.get_avg_event_rate(mouse, session,
                                                   data=data, t=t,
                                                   session=ff_session,
                                                   bin_size=bin_size,
                                                   mask=mask,
                                                   neurons=neurons)

        # if z:
        #     event_rates[i,:] = zscore(event_rates[i,:])

    correlations = np.zeros((len(event_rates)))
    for i, vector in enumerate(event_rates):
        correlations[i] = corr(vector, ref_event_rates)[0]

    if len(binned_in_cage[-1]) < len(binned_in_cage[0])/2:
        correlations[-1] = np.nan

    if plot_flag:
        plt.plot(correlations)
        plt.show()

    return correlations, ref_event_rates, event_rates

def session_corr(mouse, session, ref_session='FC', corr=pearsonr):
    session_index = get_session(mouse, (ref_session, session))[0]

    map = cell_reg.load_cellreg_results(mouse)
    trimmed_map = cell_reg.trim_match_map(map, session_index)
    ref_neurons = trimmed_map[:,0]
    neurons = trimmed_map[:,1]

    ref_event_rates = d_pp.get_avg_event_rate(mouse, ref_session,
                                              neurons=ref_neurons)

    event_rates = d_pp.get_avg_event_rate(mouse, session,
                                          neurons=neurons)

    correlation = corr(ref_event_rates, event_rates)[0]

    return correlation


def sort_PVs(mouse, session, ref_session='FC', bin_size=1,
             slice_size=60, ref_mask_start=None, plot_flag=True,
             corr=pearsonr):

    _, ref_event_rates, event_rates = time_lapse_corr(mouse, session,
                                                      ref_session=ref_session,
                                                      bin_size=bin_size,
                                                      slice_size=slice_size,
                                                      ref_mask_start=ref_mask_start,
                                                      plot_flag=False, corr=corr)

    # Sort by neuron activity in reference, then reorder.
    neurons = np.arange(len(ref_event_rates))
    order = np.argsort(ref_event_rates)
    event_rates = event_rates[:,order]
    n_slices = event_rates.shape[0]

    f, axs = plt.subplots(n_slices, figsize=(3,30), sharey=True)
    axs[0].bar(neurons, ref_event_rates[order])
    axs[0].tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
    )


    for i, vector in enumerate(event_rates[:-1]):
        axs[i+1].bar(neurons, vector)


        if i+1 != n_slices-1:
            axs[i+1].tick_params(
                axis='x',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False
            )
        else:
            axs[i+1].set_xlabel('Cell #')
            axs[i+1].set_xticks([0, np.max(neurons)])


    f.show()

    #f, ax = plt.subplots(1,1)
    #X = event_rates[:-1]
    #X = X.T / np.amax(X, axis=1)
    #ax.imshow(X)
    #ax.set_xlabel('Time')
    #ax.set_ylabel('Cell #')

    f.show()
    pass

if __name__ == '__main__':
    sort_PVs('Mundilfari','RE_1',slice_size=30)