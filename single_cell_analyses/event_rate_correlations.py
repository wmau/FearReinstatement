from session_directory import get_session
import data_preprocessing as d_pp
from microscoPy_load.ff_video_fixer import load_session
from helper_functions import find_closest
import numpy as np
import matplotlib.pyplot as plt
import microscoPy_load.cell_reg as cell_reg
from scipy.stats import pearsonr, spearmanr
import microscoPy_load.calcium_events as ca_events

def time_lapse_corr(mouse, session, ref_session='FC', bin_size=1,
                    slice_size=60, ref_mask_start=None, plot_flag=True,
                    corr=spearmanr):
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

    map = cell_reg.load_cellreg_results(mouse)
    trimmed_map = cell_reg.trim_match_map(map, session_index)
    ref_neurons = trimmed_map[:,0]
    neurons = trimmed_map[:,1]

    # Get average event rates from the reference session.
    ref_event_rates = d_pp.get_avg_event_rate(mouse, ref_session,
                                              data=data, t=t,
                                              session=ff_ref,
                                              bin_size=bin_size,
                                              mask=ref_mask,
                                              neurons=ref_neurons)

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


    correlations = np.zeros((len(event_rates)))
    for i, vector in enumerate(event_rates):
        correlations[i] = corr(vector, ref_event_rates)[0]

    if len(binned_in_cage[-1]) < len(binned_in_cage[0])/2:
        correlations[-1] = np.nan

    if plot_flag:
        plt.plot(correlations)
        plt.show()

    return correlations

if __name__ == '__main__':
    time_lapse_corr('Calypso','E2_1',slice_size=30)