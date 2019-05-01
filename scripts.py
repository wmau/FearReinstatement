import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 12})
import numpy as np
import seaborn as sns
from pandas import MultiIndex
from scipy.stats import pearsonr, mannwhitneyu, spearmanr, kendalltau, \
    ttest_ind, wilcoxon, ks_2samp, linregress
from sklearn.linear_model import LinearRegression
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import fdrcorrection
from helper_functions import ismember, nan, bool_array, sem, detect_onsets, \
    pad_and_stack
from plotting.plot_functions import scatter_box
from population_analyses.freezing_classifier import classify_cross_session
from session_directory import get_session, \
    get_region, load_session_list
from single_cell_analyses.event_rate_correlations \
    import time_lapse_corr, session_corr
from single_cell_analyses.freezing_selectivity import speed_modulation
from microscoPy_load import cell_reg
from behavior.freezing import compute_percent_freezing, plot_freezing_percentages
from population_analyses.assembly_analysis import cross_day_ensemble_activity
from data_preprocessing import convolve
import data_preprocessing as d_pp
from itertools import zip_longest
from pickle import dump, load
import os

session_list = load_session_list()

# Specify mice and days to be included in analysis.
mice = (
        'Kerberos',
        'Nix',
        'Pandora',
        'Calypso',
        'Helene',
        'Hyperion',
        'Pan',
        'Janus',
        'Kepler',
        'Mundilfari',
        'Aegir',
        'Skoll',
        'Telesto',
)
CA1_mice = ('Kerberos',
            'Nix',
            'Pandora',
            'Calypso',
            'Helene',
            'Hyperion',
            'Pan',
           )

BLA_mice = ('Janus',
            'Kepler',
            'Mundilfari',
            'Aegir',
            'Skoll',
            'Telesto',
            )

days = (
        'E1_1',
        'E2_1',
        'RE_1',
        'E1_2',
        'E2_2',
        'RE_2',
        )

regions = [get_region(mouse) for mouse in mice]

# Set up.
n_mice = len(mice)
n_days = len(days)

# Preallocate.
session_1 = [get_session(mouse, 'FC')[0]
             for mouse in mice]
session_2 = [get_session(mouse, days)[0]
             for mouse in mice]
session_2_stages = [get_session(mouse, days)[1]
                    for mouse in mice]

master_dir = 'U:\\Fear conditioning project_Mosaic2\\SessionDirectories'

def Fix_Calypso_E2b():
    import os
    import csv

    session = get_session('Calypso', 'E2_1')[0]

    filename = os.path.join(session_list[session]["Location"],
                            "Events.csv")
    new_name = os.path.join(session_list[session]["Location"],
                            "New_Events.csv")

    r = csv.reader(open(filename, 'r'))

    lines = list(r)

    for i, row in enumerate(lines):
        try:
            t = float(row[0])
            if t > 888.9:
                new_value = round(t - 3838.707, 2)
                lines[i][0] = str(new_value)
        except:
            pass

    writer = csv.writer(open(new_name, 'w', newline=''))
    writer.writerows(lines)


def FixKeplerShockTraces():
    import microscoPy_load.calcium_traces as ca_traces
    from microscoPy_load.cell_data_compiler import CellData
    import os
    from pickle import dump

    session_index = get_session('Kepler','FC')[0]
    path = session_list[session_index]['Location']

    # Load traces.
    data, t = ca_traces.load_traces(session_index)

    # Fix bad indices based on raw fluorescence.
    bad_idx = np.logical_or(data > 2500, data < -500)
    data[bad_idx] = np.nan

    # Get DF/DT.
    dfdt = np.hstack([np.zeros((data.shape[0], 1)),
                      np.diff(data, axis=1)])

    # Fix bad values based on df/dt.
    bad_idx = np.logical_or(dfdt > 200, dfdt < -150)
    data[bad_idx] = np.nan

    # Convolve based on rolling average.
    for i, cell in enumerate(data):
        data[i] = convolve(cell, 20)

    C = CellData(session_index)
    C.traces = data

    with open(os.path.join(path, 'CellData_new.pkl'), 'wb') as file:
        dump(C, file)


def CrossSessionEventRateCorr_wholesession(corr=pearsonr):
    correlations = np.zeros((n_mice, n_days))
    for i, mouse in enumerate(mice):
        for j, session in enumerate(session_2_stages[i]):
            try:
                correlations[i,j] = session_corr(mouse, session,
                                                 corr=corr)
            except:
                pass

    return correlations


def FindGeneralizers():
    discriminator = nan((n_mice, 2))
    for n, mouse in enumerate(mice):
        session_idx, session_type = \
            get_session(mouse, ('E1_1','E1_2','RE_1','RE_2'))

        if 'E1_2' in session_type:
            _, i = ismember(session_type, 'E1_1')
            shock_freezing = compute_percent_freezing(session_idx[i[0]],
                                                      bin_length=1)[0]
            _, i = ismember(session_type, 'E1_2')
            neutral_freezing = compute_percent_freezing(session_idx[i[0]],
                                                        bin_length=1)[0]

            p = mannwhitneyu(shock_freezing, neutral_freezing).pvalue

            if p < 0.05:
                discriminator[n,0] = True
            else:
                discriminator[n,0] = False

        if 'RE_2' in session_type:
            _, i = ismember(session_type, 'RE_1')
            shock_freezing = compute_percent_freezing(session_idx[i[0]],
                                                      bin_length=1)[0]
            _, i = ismember(session_type, 'RE_2')
            neutral_freezing = compute_percent_freezing(session_idx[i[0]],
                                                      bin_length=1)[0]

            p = mannwhitneyu(shock_freezing, neutral_freezing).pvalue

            if p < 0.05:
                discriminator[n,1] = True
            else:
                discriminator[n,1] = False
    pass


def CrossSessionEventRateCorr2(bin_size=1, slice_size=30,
                               ref_mask_start=None,
                               corr=pearsonr, ref_indices=None,
                               omit_speed_modulated=False):
    min_per_slice = slice_size/60

    # Get all the correlation time series.
    correlations = {}
    freezing = {}
    for mouse in mice:
        # Omit speed modulated neurons, if specified.
        if omit_speed_modulated:
            modulated = speed_modulation(mouse, 'FC')
            ref_neurons = np.where(~modulated)[0]
        else:
            ref_neurons = None

        # Preallocate a dict per mouse.
        correlations[mouse] = {}
        freezing[mouse] = {}
        for session in days:
            try:
                correlations[mouse][session] = \
                    time_lapse_corr(mouse, session,
                                    ref_session='FC', bin_size=bin_size,
                                    slice_size=slice_size,
                                    ref_neurons=ref_neurons,
                                    ref_mask_start=ref_mask_start,
                                    ref_indices=ref_indices,
                                    plot_flag=False,
                                    corr=corr)[0]


            except:
                correlations[mouse][session] = nan((0))


            try:
                freezing[mouse][session] = \
                    compute_percent_freezing(get_session(mouse, session)[0],
                                             bin_length=slice_size, plot=False)[0]
            except:
                freezing[mouse][session] = nan((0))


    # The mice are in the chambers for different lengths of time, even
    # for the same session type (+/- 1 min). Find the largest session
    # and use that size to build the matrix. Pad everything else with
    # nans.
    # longest_sessions = {}
    # for session2s in days:
    #     longest_sessions[session2s] = np.max([len(correlations[mouse][session2s])
    #                                           for mouse in mice])
    longest_sessions = {'E1_1': int(np.round(30 / min_per_slice)),
                        'E2_1': int(np.round(30 / min_per_slice)),
                        'E1_2': int(np.round(30 / min_per_slice)),
                        'E2_2': int(np.round(30 / min_per_slice)),
                        'RE_1': int(np.round(8 / min_per_slice)),
                        'RE_2': int(np.round(8 / min_per_slice))}

    # Get session boundaries in accumulated array.
    session_boundaries = {}
    session_boundaries['shock'] = np.cumsum([longest_sessions[session]
                                             for session in ('E1_1','E2_1','RE_1')])
    session_boundaries['neutral'] = np.cumsum([longest_sessions[session]
                                               for session in ('E1_2','E2_2','RE_2')])

    # Arrange correlations nicely into a (mouse, time) array.
    big_dict = {}
    big_dict['shock'] = nan((n_mice, session_boundaries['shock'][-1]))
    big_dict['neutral'] = nan((n_mice, session_boundaries['neutral'][-1]))

    big_freezing = {}
    big_freezing['shock'] = nan((n_mice, session_boundaries['shock'][-1]))
    big_freezing['neutral'] = nan((n_mice, session_boundaries['neutral'][-1]))

    contexts = {'shock': ['E1_1', 'E2_1', 'RE_1'],
                'neutral': ['E1_2', 'E2_2', 'RE_2']}
    # Pad and append.
    for context in contexts.keys():
        for i, mouse in enumerate(mice):
            length_difference = [longest_sessions[session] - len(correlations[mouse][session])
                                 for session in contexts[context]]

            big_dict[context][i] = pad_and_stack([correlations[mouse][session]
                                                  for session in contexts[context]],
                                                 length_difference)

            big_freezing[context][i] = pad_and_stack([freezing[mouse][session]
                                                      for session in contexts[context]],
                                                     length_difference)

    big_dict['boundaries'] = session_boundaries
    big_dict['min_per_slice'] = min_per_slice
    with open(os.path.join(master_dir, 'PV_Corrs.pkl'), 'wb') as file:
        dump(big_dict, file)

    print('Correlate R values to freezing')
    for i, mouse in enumerate(mice):
        f = big_freezing['shock'][i]
        r = big_dict['shock'][i]

        f = f[np.isfinite(r)]
        r = r[np.isfinite(r)]

        print(mouse)
        print(pearsonr(r, f))

    return big_dict


def RegressCorrTimeSeries():
    try:
        with open(os.path.join(master_dir, 'PV_Corrs.pkl'), 'rb') as file:
            big_dict = load(file)
    except:
        big_dict = CrossSessionEventRateCorr2()
    session_boundaries = big_dict['boundaries']
    min_per_slice = big_dict['min_per_slice']

    # Regress extinction time bins, then regress recall time bins.
    # Predict with extinction model, compare to y-intercept of recall model.
    regressions = {}
    values = {'shock':      nan((n_mice, 2)),
              'neutral':    nan((n_mice, 2))}
    ext_boundary = {}
    ext_boundary['shock'] = session_boundaries['shock'][1]
    ext_boundary['neutral'] = session_boundaries['neutral'][1]
    recall_start = {}
    recall_start['shock'] = session_boundaries['shock'][1]
    recall_start['neutral'] = session_boundaries['neutral'][1]
    for i, mouse in enumerate(mice):
        regressions[mouse] = {}

        # For both contexts..
        for context in ['shock','neutral']:
            regressions[mouse][context] = {}

            try:
                y = big_dict[context][i, :ext_boundary[context]]
                X = np.arange(ext_boundary[context])
                X = X[np.isfinite(y)]
                y = y[np.isfinite(y)]

                # Do regression on extinction.
                regressions[mouse][context]['ext'] = LinearRegression().fit(X.reshape(-1,1),
                                                                            y.reshape(-1,1))
                regressions[mouse][context]['ext'].pval = linregress(X, y)[3]

                y = big_dict[context][i, recall_start[context]:]
                X = np.arange(session_boundaries[context][-1] - session_boundaries[context][-2])
                X = X[np.isfinite(y)].reshape(-1,1)
                y = y[np.isfinite(y)].reshape(-1,1)

                # Do regression on recall.
                regressions[mouse][context]['recall'] = LinearRegression().fit(X,y)
            except:
                regressions[mouse][context]['ext'] = np.nan
                regressions[mouse][context]['recall'] = np.nan


            try:
                values[context][i,0] = regressions[mouse][context]['ext'].predict(np.array(
                    [recall_start[context]+1]).reshape(-1,1))
                values[context][i,1] = regressions[mouse][context]['recall'].intercept_
            except:
                values[context][i] = (np.nan, np.nan)


    #Get the CA1 and BLA mice by hard-code for now.
    CA1 = values['shock'][:7]
    CA1 = np.delete(CA1, np.where(~np.isfinite(CA1[:, 0])), axis=0)
    BLA = values['shock'][7:]
    BLA = np.delete(BLA, np.where(~np.isfinite(BLA[:, 0])), axis=0)

    # Do stats.
    pCA1 = wilcoxon(CA1[:,0], CA1[:,1])
    pBLA = wilcoxon(BLA[:,0], BLA[:,1])

    #Get the CA1 and BLA mice by hard-code for now.
    CA1_neutral = values['neutral'][:7]
    CA1_neutral = np.delete(CA1_neutral, np.where(~np.isfinite(CA1_neutral[:, 0])), axis=0)
    BLA_neutral = values['neutral'][7:]
    BLA_neutral = np.delete(BLA_neutral, np.where(~np.isfinite(BLA_neutral[:, 0])), axis=0)

    # Do stats.
    pCA1_neutral = wilcoxon(CA1_neutral[:,0], CA1_neutral[:,1])
    pBLA_neutral = wilcoxon(BLA_neutral[:,0], BLA_neutral[:,1])

    # Plot individual animals.
    for i, mouse in enumerate(mice):
        f, ax = plt.subplots(1,2)

        for j, context in enumerate(['shock','neutral']):
            ax[j].plot(big_dict[context][i], '.')
            ax[j].set_title(mouse + ' ' + context)

            try:
                # Get x and y coordinates.
                X_ext = np.arange(ext_boundary[context]).reshape(-1, 1)
                y_ext = regressions[mouse][context]['ext'].predict(X_ext)

                X_recall = np.arange(ext_boundary[context],
                                     session_boundaries[context][2]).reshape(-1, 1) - ext_boundary[context]
                y_recall = regressions[mouse][context]['recall'].predict(X_recall)

                # Plot the points.
                ax[j].plot(X_ext, y_ext, 'b')
                ax[j].plot(X_recall + ext_boundary[context], y_recall, 'g')
                ax[j].set_xticks([0,
                               30 / min_per_slice,
                               30 / min_per_slice + 30 / min_per_slice,
                               30 / min_per_slice + 30 / min_per_slice + 8 / min_per_slice])
                ax[j].set_xticklabels([0, 30, 30, 8])
                ax[j].set_xlabel('Time [min]')

                if context == 'shock':
                    ax[j].set_ylabel('PV similarity to CFC [r]')

                # Plot session boundaries.
                for boundary in session_boundaries[context]:
                    ax[j].axvline(boundary)
            except:
                pass

    # Plot.
    f, ax = plt.subplots(1, 2, figsize=(4, 5))
    ax[0].boxplot(BLA, positions=[0, 1])
    for pair in BLA:
        ax[0].plot(pair, 'grey')

    ax[0].set_ylabel('Population similarity (r)')
    ax[0].set_xticklabels(('Predicted', 'Real'))
    ax[0].set_title('BLA')

    ax[1].boxplot(CA1, positions=[0, 1])
    for pair in CA1:
        ax[1].plot(pair, 'grey')

    ax[1].set_xticklabels(('Predicted', 'Real'))
    ax[1].set_title('CA1')

    # Plot.
    f, ax = plt.subplots(1, 2, figsize=(4, 5))
    ax[0].boxplot(BLA_neutral, positions=[0, 1])
    for pair in BLA_neutral:
        ax[0].plot(pair, 'grey')

    ax[0].set_ylabel('Population similarity (r)')
    ax[0].set_xticklabels(('Predicted', 'Real'))
    ax[0].set_title('BLA')

    ax[1].boxplot(CA1_neutral, positions=[0, 1])
    for pair in CA1_neutral:
        ax[1].plot(pair, 'grey')

    ax[1].set_xticklabels(('Predicted', 'Real'))
    ax[1].set_title('CA1')

    for mouse in mice:
        print(mouse)
        try:
            print('coef = ' + str(regressions[mouse]['neutral']['ext'].coef_) +
                  ', p = ' + str(regressions[mouse]['neutral']['ext'].pval))
        except:
            pass


    return pCA1, pBLA, pCA1_neutral, pBLA_neutral




def CrossSessionEventRateCorr(bin_size=1, slice_size=30,
                              ref_mask_start=None, corr=pearsonr,
                              truncate=True,
                              omit_speed_modulated=False):

    slice_size_min = slice_size/60
    # Get correlations and session identities.
    correlations = []
    mouse_id = []
    session_id = []
    for i, mouse in enumerate(mice):
        if omit_speed_modulated:
            modulated = speed_modulation(mouse, 'FC')
            ref_neurons = np.where(~modulated)[0]
        else:
            ref_neurons = None


        # if active_all_days:
        #     map = cell_reg.load_cellreg_results(mouse)
        #     all_sessions = ['FC']
        #     all_sessions.extend(session_2_stages[i])
        #     trimmed_map = \
        #         cell_reg.trim_match_map(map,
        #                                 get_session(mouse,
        #                                             all_sessions)[0])
        #     ref_neurons = trimmed_map[:,0]
        # else:
        #     ref_neurons = None

        for session in session_2_stages[i]:
            correlations.append(time_lapse_corr(mouse, session,
                                                ref_session='FC',
                                                bin_size=bin_size,
                                                slice_size=slice_size,
                                                ref_neurons=ref_neurons,
                                                ref_mask_start=ref_mask_start,
                                                plot_flag=False,
                                                corr=corr)[0])
            mouse_id.append(mouse)
            session_id.append(session)

    # Arrange correlations nicely into a (mouse, time) array.
    max_sizes = []
    for day in days:
        # The mice are in the chambers for different lengths of time, even
        # for the same session type (+/- 1 min). Find the largest session
        # and use that size to build the matrix. Pad everything else with
        # nans.
        idx = np.where(ismember(day, session_id)[0])[0]
        max_sizes.append(np.max([len(correlations[i]) for i in idx]))

    session_boundaries = np.cumsum(max_sizes)

    # Preallocate a column array. We'll use this to append our session
    # matrices to.
    all_correlations = nan((n_mice, 1))
    for i, day in enumerate(days):
        # Make a nan matrix whose length is the longest session of that
        # type.
        stacked_correlations = nan((n_mice, max_sizes[i]))

        for j, mouse in enumerate(mice):
            # For each mouse, get the session and fill the matrix row.
            # If that session doesn't exist, skip it.
            try:
                # Get the index for that session, which is the intersect
                # between the mouse and session type.
                idx = list(set(np.where(ismember(day, session_id)[0])[0]) & \
                           set([k for k, x in enumerate(mouse_id) if x == mouse]))[0]
                c = correlations[idx]

                stacked_correlations[j, :len(c)] = c
            except:
                pass

        # Append the matrix along columns.
        all_correlations = np.append(all_correlations,
                                     stacked_correlations, axis=1)

    # Delete the first column, which was used to append.
    all_correlations = np.delete(all_correlations, 0, axis=1)
    E1_1, E2_1, RE_1, E1_2, E2_2, RE_2 = \
        make_condition_logicals(all_correlations,
                                slice_size, session_boundaries,
                                truncate=truncate)

    # Get indices for regions.
    CA1_mice = ismember('CA1', regions)[0]
    BLA_mice = ismember('BLA', regions)[0]

    # Partition the matrix into BLA and CA1.
    CA1 = all_correlations[CA1_mice]
    BLA = all_correlations[BLA_mice]

    # Get indices for context 1 and context 2.
    context_1 = E1_1 | E2_1 | RE_1
    context_2 = E1_2 | E2_2 | RE_2

    if truncate:
        context1_boundaries = np.array([30/slice_size_min,
                                        (30/slice_size_min)*2,
                                        (30/slice_size_min)*2 + 8/slice_size_min])
        context2_boundaries = context1_boundaries
        t = np.concatenate((np.arange(0, 30, slice_size_min),
                            np.arange(0, 30, slice_size_min),
                            np.arange(0, 8, slice_size_min)))
    else:
        context1_boundaries = session_boundaries[:3]
        context2_boundaries = np.cumsum(max_sizes[3:])

    CA1_E1_1 = CA1[:, E1_1].flatten()
    CA1_E2_1 = CA1[:, E2_1].flatten()
    CA1_RE_1 = CA1[:, RE_1].flatten()
    BLA_E1_1 = BLA[:, E1_1].flatten()
    BLA_E2_1 = BLA[:, E2_1].flatten()
    BLA_RE_1 = BLA[:, RE_1].flatten()

    CA1_E1_2 = CA1[:, E1_2].flatten()
    CA1_E2_2 = CA1[:, E2_2].flatten()
    CA1_RE_2 = CA1[:, RE_2].flatten()
    BLA_E1_2 = BLA[:, E1_2].flatten()
    BLA_E2_2 = BLA[:, E2_2].flatten()
    BLA_RE_2 = BLA[:, RE_2].flatten()

    # Plot time series of correlation coefficients for CA1.
    f, ax = plt.subplots(2, 2, figsize=(9, 7))
    plot_Rs(ax[0, 0], CA1[:, context_1], context1_boundaries, slice_size_min)
    ax[0, 0].set_title('CA1')
    ax[0, 0].set_ylabel('Correlation Coefficient')

    # Plot time series of correlation coefficients for BLA.
    plot_Rs(ax[0, 1], BLA[:, context_1], context1_boundaries, slice_size_min)
    ax[0, 1].set_title('BLA')

    # Boxplot by session type for CA1.
    data = [CA1_E1_1,
            CA1_E2_1,
            CA1_RE_1]
    data = [x[~np.isnan(x)] for x in data]
    scatter_box(data, xlabels=['Ext1', 'Ext2', 'Recall'],
                ylabel='Correlation Coefficient', ax=ax[1, 0])

    # Boxplot by session type for BLA.
    data = [BLA_E1_1,
            BLA_E2_1,
            BLA_RE_1]
    data = [x[~np.isnan(x)] for x in data]
    scatter_box(data, xlabels=['Ext1', 'Ext2', 'Recall'], ax=ax[1, 1])

    ###
    f, ax = plt.subplots(2, 2, figsize=(9, 7))
    plot_Rs(ax[0, 0], CA1[:, context_2], context2_boundaries, slice_size_min, color='navy')
    ax[0, 0].set_ylabel('Correlation Coefficient')
    ax[0, 0].set_title('CA1')

    # Plot time series of correlation coefficients for BLA.
    plot_Rs(ax[0, 1], BLA[:, context_2], context2_boundaries, slice_size_min, color='navy')
    ax[0, 1].set_title('BLA')

    # Boxplot by session type for CA1.
    data = [CA1_E1_2,
            CA1_E2_2,
            CA1_RE_2]
    data = [x[~np.isnan(x)] for x in data]
    scatter_box(data, xlabels=['Ext1', 'Ext2', 'Recall'],
                ylabel='Correlation Coefficient', ax=ax[1, 0], box_color='navy')

    # Boxplot by session type for BLA.
    data = [BLA_E1_2,
            BLA_E2_2,
            BLA_RE_2]
    data = [x[~np.isnan(x)] for x in data]
    scatter_box(data, xlabels=['Ext1', 'Ext2', 'Recall'], ax=ax[1, 1], box_color='navy')

    # Do lots of stats.
    CA1_E1_1 = CA1_E1_1[~np.isnan(CA1_E1_1)]
    CA1_E2_1 = CA1_E2_1[~np.isnan(CA1_E2_1)]
    CA1_RE_1 = CA1_RE_1[~np.isnan(CA1_RE_1)]

    CA1_E1_2 = CA1_E1_2[~np.isnan(CA1_E1_2)]
    CA1_E2_2 = CA1_E2_2[~np.isnan(CA1_E2_2)]
    CA1_RE_2 = CA1_RE_2[~np.isnan(CA1_RE_2)]

    BLA_E1_1 = BLA_E1_1[~np.isnan(BLA_E1_1)]
    BLA_E2_1 = BLA_E2_1[~np.isnan(BLA_E2_1)]
    BLA_RE_1 = BLA_RE_1[~np.isnan(BLA_RE_1)]

    BLA_E1_2 = BLA_E1_2[~np.isnan(BLA_E1_2)]
    BLA_E2_2 = BLA_E2_2[~np.isnan(BLA_E2_2)]
    BLA_RE_2 = BLA_RE_2[~np.isnan(BLA_RE_2)]

    E1_E2_CA1 = mannwhitneyu(CA1_E1_1, CA1_E2_1).pvalue
    E1_RE_CA1 = mannwhitneyu(CA1_E1_1, CA1_RE_1).pvalue
    E2_RE_CA1 = mannwhitneyu(CA1_E2_1, CA1_RE_1).pvalue

    E1_E2_CA1_2 = mannwhitneyu(CA1_E1_2, CA1_E2_2).pvalue
    E1_RE_CA1_2 = mannwhitneyu(CA1_E1_2, CA1_RE_2).pvalue
    E2_RE_CA1_2 = mannwhitneyu(CA1_E2_2, CA1_RE_2).pvalue

    E1_E2_BLA = mannwhitneyu(BLA_E1_1, BLA_E2_1).pvalue
    E1_RE_BLA = mannwhitneyu(BLA_E1_1, BLA_RE_1).pvalue
    E2_RE_BLA = mannwhitneyu(BLA_E2_1, BLA_RE_1).pvalue

    E1_E2_BLA_2 = mannwhitneyu(BLA_E1_2, BLA_E2_2).pvalue
    E1_RE_BLA_2 = mannwhitneyu(BLA_E1_2, BLA_RE_2).pvalue
    E2_RE_BLA_2 = mannwhitneyu(BLA_E2_2, BLA_RE_2).pvalue

    reject_CA1 = fdrcorrection((E1_E2_CA1, E1_RE_CA1, E2_RE_CA1,
                                E1_E2_CA1_2, E1_RE_CA1_2, E2_RE_CA1_2),0.05)[0]
    reject_BLA = fdrcorrection((E1_E2_BLA, E1_RE_BLA, E2_RE_BLA,
                                E1_E2_BLA_2, E1_RE_BLA_2, E2_RE_BLA_2),0.05)[0]


    plt.show()

    return reject_CA1, reject_BLA


def Plot_Freezing(bin_length=30):
    """
    Plot freezing over the experiment for all mice.

    Parameter
    ---
    bin_length: scalar, size of bin (in seconds)

    Return
    ---
    Plot of freezing.
    """

    # Get freezing percentages for each mouse.
    freezing, freezing_untruncated = {}, {}
    for mouse in mice:
        freezing[mouse], freezing_untruncated[mouse], boundaries,\
            tick_locations = \
            plot_freezing_percentages(mouse,
                                      bin_length=bin_length,
                                      plot=False)

    # Rearrange data into a mouse x time matrix.
    context1_freezing = nan((n_mice, boundaries[-1]))
    context2_freezing = nan((n_mice, boundaries[-1]))
    for i, mouse in enumerate(mice):
        context1_freezing[i] = freezing[mouse][0]

        try:
            context2_freezing[i] = freezing[mouse][1]
        except:
            pass

    # Compute the mean and sem freezing.
    c1_freezing_mean = np.nanmean(context1_freezing, axis=0)
    c1_freezing_sem = sem(context1_freezing)
    c2_freezing_mean = np.nanmean(context2_freezing, axis=0)
    c2_freezing_sem = sem(context2_freezing)
    x = np.r_[0:c1_freezing_mean.shape[0]]

    # Plot.
    f, ax = plt.subplots(1, 1)
    ax.errorbar(x, c2_freezing_mean, yerr=c2_freezing_sem, fmt='o',
                ecolor='gray', markersize=3, capsize=0, mfc='gray',
                mew=0, alpha=0.5)
    ax.errorbar(x, c1_freezing_mean, yerr=c1_freezing_sem, fmt='o',
                ecolor='navy', markersize=3, capsize=0,
                mfc='navy', mew=0, alpha=0.5)

    for boundary in boundaries:
        ax.axvline(x=boundary)

    ax.set_xticks(tick_locations)
    ax.set_xticklabels([0, 8, 30, 30, 8])
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Freezing (%)')

    # Separate into different sessions.
    E1_1 = context1_freezing[:, boundaries[1]:boundaries[2]]
    E2_1 = context1_freezing[:, boundaries[2]:boundaries[3]]
    RE_1 = context1_freezing[:, boundaries[3]:boundaries[4]]

    E1_2 = context2_freezing[:, boundaries[1]:boundaries[2]]
    E2_2 = context2_freezing[:, boundaries[2]:boundaries[3]]
    RE_2 = context2_freezing[:, boundaries[3]:boundaries[4]]

    E1_1_mean = np.nanmean(np.array_split(E1_1,5,axis=1)[0], axis=1)
    E2_1_mean = np.nanmean(np.array_split(E2_1,5,axis=1)[-1], axis=1)
    RE_1_mean = np.nanmean(RE_1[:,2:], axis=1)

    E1_2_mean = np.nanmean(np.array_split(E1_2,5,axis=1)[0], axis=1)
    E2_2_mean = np.nanmean(np.array_split(E2_2,5,axis=1)[-1], axis=1)
    RE_2_mean = np.nanmean(RE_2[:,2:], axis=1)

    E1_test = wilcoxon(E1_1_mean[~np.isnan(E1_2_mean)],
                       E1_2_mean[~np.isnan(E1_2_mean)]).pvalue
    E2_test = wilcoxon(E2_1_mean[~np.isnan(E2_2_mean)],
                       E2_2_mean[~np.isnan(E2_2_mean)]).pvalue
    RE_test = wilcoxon(RE_1_mean[~np.isnan(RE_2_mean)],
                       RE_2_mean[~np.isnan(RE_2_mean)]).pvalue
    RE_test2 = wilcoxon(RE_1_mean[~np.isnan(RE_1_mean)],
                        E2_1_mean[~np.isnan(RE_1_mean)]).pvalue

    E1vsE2 = wilcoxon(E1_1_mean[~np.isnan(E2_1_mean)],
                      E2_1_mean[~np.isnan(E2_1_mean)]).pvalue

    context1_means = [np.nanmean(E1_1_mean),
                      np.nanmean(E2_1_mean),
                      np.nanmean(RE_1_mean)]
    context2_means = [np.nanmean(E1_2_mean),
                      np.nanmean(E2_2_mean),
                      np.nanmean(RE_2_mean)]

    context1_sem = [sem(E1_1_mean), sem(E2_1_mean), sem(RE_1_mean)]
    context2_sem = [sem(E1_2_mean), sem(E2_2_mean), sem(RE_2_mean)]

    ind = np.arange(3)
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(ind, context1_means, width, edgecolor='k' * 3, color='navy',
           yerr=[(0, 0, 0), context1_sem], capsize=10, linewidth=1)
    ax.bar(ind + width, context2_means, width, edgecolor='k' * 3,
           color='gray', yerr=[(0, 0, 0), context2_sem], capsize=10, linewidth=1)

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('Ext1', 'Ext2', 'Recall'))
    ax.set_ylabel('Freezing (%)')

    plt.setp(ax.spines.values(), linewidth=1)


    return f, context1_freezing, context2_freezing


def CrossSessionClassify(bin_length=1, I=100,
                         classifier=None,
                         predictor='traces', shuffle='scramble'):
    scores, pvals = np.empty((n_mice, n_days)), \
                    np.empty((n_mice, n_days))
    permuted = np.empty((n_mice, n_days, I))
    scores.fill(np.nan)
    pvals.fill(np.nan)
    permuted.fill(np.nan)

    # Find scores and p-values.
    for mouse, train_session in enumerate(session_1):
        # match_map = load_cellreg_results(session_list[fc]['Animal'])
        # trimmed = trim_match_map(match_map,all_sessions[i])
        # neurons = trimmed[:,0]
        for session, test_session in enumerate(session_2[mouse]):
            score, permuted_scores, p_value = \
                classify_cross_session(train_session, test_session,
                                       bin_length=bin_length,
                                       predictor=predictor, I=I,
                                       classifier=classifier,
                                       shuffle=shuffle)

            scores[mouse, session] = score
            pvals[mouse, session] = p_value
            permuted[mouse, session, :] = permuted_scores

    # Define labels for dataframe.
    day_label = np.tile(np.repeat(days, I), n_mice)
    day_label = np.concatenate([day_label, np.tile(days, n_mice)])

    mouse_label = np.repeat(mice, I * n_days)
    mouse_label = np.concatenate([mouse_label,
                                  np.repeat(mice, n_days)])

    condition_label = np.repeat("Shuffled", n_mice * n_days * I)
    condition_label = np.concatenate([condition_label,
                                      np.repeat("Real", n_mice * n_days)])

    region_label = np.repeat(regions, I * n_days)
    region_label = np.concatenate([region_label,
                                   np.repeat(regions, n_days)])

    data = np.concatenate([permuted.flatten(),
                           scores.flatten()])

    # Build dataframe.
    df = MultiIndex.from_arrays([data,
                                 mouse_label,
                                 day_label,
                                 condition_label,
                                 region_label],
                                names=['Accuracy',
                                       'Mouse',
                                       'Day',
                                       'Condition',
                                       'Region',
                                       ]).to_frame(False)

    # Plot distributions of shuffled decoder results with real result.
    f = sns.catplot(x="Day",
                    y="Accuracy",
                    col="Mouse",
                    data=df.loc[df['Condition'] == 'Shuffled'],
                    kind="violin",
                    bw=.2,
                    col_wrap=True)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.setGeometry(2400, -420, 420, 1610)

    for i, mouse in enumerate(mice):
        sns.stripplot(x="Day",
                      y="Accuracy",
                      data=df.loc[(df['Condition'] == 'Real') &
                                  (df['Mouse'] == mouse)],
                      jitter=False,
                      size=10,
                      marker='X',
                      linewidth=2,
                      ax=f.axes[i])

    # Only look at the first context.
    one_context = df.loc[df['Day'].isin(('E1_1', 'E2_1', 'RE_1'))]

    # Get the means and sem of the scores and permuted scores.
    # Pool across mice.
    # collapse_by_mouse = one_context.groupby(['Mouse','Day','Condition'],
    #                                           sort=False).mean().reset_index()

    # means = collapse_by_mouse.groupby(['Day', 'Condition'],
    #                                   sort=False).mean()
    # sems = collapse_by_mouse.groupby(['Day', 'Condition'],
    #                                  sort=False).sem()['Accuracy']

    # Take the mean across groups.
    means = one_context.groupby(['Day', 'Condition', 'Region'],
                                sort=False).mean().reset_index()
    sems = one_context.groupby(['Day', 'Condition', 'Region'],
                               sort=False).sem()['Accuracy'].reset_index()

    # CA1.
    CA1_real = (means['Condition'] == 'Real') & (means['Region'] == 'CA1')
    CA1_shuffle = (means['Condition'] == 'Shuffled') & (means['Region'] == 'CA1')

    y_real = means.loc[CA1_real]['Accuracy']
    yerr_real = sems.loc[CA1_real]['Accuracy']

    y_shuffle = means.loc[CA1_shuffle]['Accuracy']
    yerr_shuffle = sems.loc[CA1_shuffle]['Accuracy']

    plt.figure()
    plt.errorbar(days[0:3], y_real, yerr=yerr_real)
    plt.errorbar(days[0:3], y_shuffle, yerr=yerr_shuffle)

    formula = 'Accuracy ~ C(Day) + C(Condition) + C(Day):C(Condition)'
    # model = ols(formula, collapse_by_mouse).fit() # Pool across mice.
    model = ols(formula,
                one_context.loc[one_context['Region'] == 'CA1']).fit()
    anova = anova_lm(model, type=2)

    # Compare conditions individually.
    E1_CA1 = one_context.loc[(one_context['Condition'] == 'Real') &
                             (one_context['Day'] == 'E1_1') &
                             (one_context['Region'] == 'CA1')]['Accuracy']
    E2_CA1 = one_context.loc[(one_context['Condition'] == 'Real') &
                             (one_context['Day'] == 'E2_1') &
                             (one_context['Region'] == 'CA1')]['Accuracy']
    RE_CA1 = one_context.loc[(one_context['Condition'] == 'Real') &
                             (one_context['Day'] == 'RE_1') &
                             (one_context['Region'] == 'CA1')]['Accuracy']
    plt.show()
    return scores, pvals, permuted


def plot_Rs(ax, Rs, boundaries, slice_size_min, color='lightgray', alpha=1):
    #ax.plot(Rs.T, linewidth=0.5, alpha=0.3)
    #ax.plot(np.nanmean(Rs, axis=0), linewidth=2, color='k')
    yerr = np.nanstd(Rs, axis=0)/np.sqrt(Rs.shape[0])
    m = np.nanmean(Rs, axis=0)
    x = np.r_[0:Rs.shape[1]]
    ax.errorbar(x, m, yerr=yerr, fmt='o', ecolor=color,
                markersize=3, capsize=0, mfc='k', mew=0, alpha=alpha)
    for boundary in boundaries:
        ax.axvline(x=boundary)
    ax.set_xticks([0,
                   30/slice_size_min,
                   30/slice_size_min+30/slice_size_min,
                   30/slice_size_min + 30/slice_size_min + 8/slice_size_min])
    ax.set_xticklabels([0, 30, 30, 8])
    ax.set_xlabel('Time (min)')

def make_condition_logicals(all_correlations, slice_size, session_boundaries,
                            truncate=False):
    # Get indices for each session type.
    ratio = int(60 / slice_size)
    array_size = all_correlations.shape[1]

    if truncate:
        E1_1 = bool_array(array_size,
                           range(ratio * 30))
        E2_1 = bool_array(array_size,
                          range(session_boundaries[0],
                                ratio*30 + session_boundaries[0]))
        RE_1 = bool_array(array_size,
                          range(session_boundaries[1],
                                ratio*8 + session_boundaries[1]))
        E1_2 = bool_array(array_size,
                          range(session_boundaries[2],
                                ratio * 30 + session_boundaries[2]))
        E2_2 = bool_array(array_size,
                          range(session_boundaries[3],
                                ratio * 30 + session_boundaries[3]))
        RE_2 = bool_array(array_size,
                          range(session_boundaries[4],
                                ratio * 8 +session_boundaries[4]))

    else:
        E1_1 = bool_array(array_size,
                          range(session_boundaries[0]))
        E2_1 = bool_array(array_size,
                          range(session_boundaries[0],
                                session_boundaries[1]))
        RE_1 = bool_array(array_size,
                          range(session_boundaries[1],
                                session_boundaries[2]))
        E1_2 = bool_array(array_size,
                          range(session_boundaries[2],
                                session_boundaries[3]))
        E2_2 = bool_array(array_size,
                          range(session_boundaries[3],
                                session_boundaries[4]))
        RE_2 = bool_array(array_size,
                          range(session_boundaries[4],
                                session_boundaries[5]))

    return E1_1, E2_1, RE_1, E1_2, E2_2, RE_2


def Corr_Activations_to_Freezing(region='BLA', template_session='FC', save_fig=False):
    """
    Correlate the number of ensemble activations (averaged across ensembles)
    during Recall with the amount of freezing in that session.

    Parameters
    ---
    region: tuple of strs, default: BLA mice
        Tuple of mouse names.

    template_session: str, default: 'FC'
        Session to build ensemble patterns from.

    """
    if region == 'BLA':
        mice = BLA_mice
    elif region == 'CA1':
        mice = CA1_mice
    else:
        raise TypeError('Not recognized')

    all_activations = {}
    all_freezing = {}
    recall_sessions = ['RE_1', 'RE_2']
    contexts = ['shock','neutral']
    for context, recall in zip(contexts, recall_sessions):
        all_activations[context] = []
        all_freezing[context] = []

        for mouse in mice:
            try:
                (activation_strengths,
                 activations,
                 patterns,
                 significance,
                 norm_activations,
                 freezing,
                 session_dict,
                 ) = cross_day_ensemble_activity(mouse, template_session, [recall])

                print(mouse + ': ' + str(patterns.shape[0]) + ' assemblies detected')

                all_activations[context].append(np.mean(norm_activations[recall]))
                all_freezing[context].append(np.sum(freezing[recall]/len(freezing[recall])))
            except:
                all_activations[context].append(np.nan)
                all_freezing[context].append(np.nan)

    fig, ax = plt.subplots()
    for context, color in zip(['shock', 'neutral'], ['b', 'xkcd:grey']):
        ax.scatter(all_activations[context], all_freezing[context], color=color)

        X = np.asarray(all_activations[context])
        y = np.asarray(all_freezing[context])
        y = y[np.isfinite(X)]
        X = X[np.isfinite(X)]
        model = LinearRegression().fit(X.reshape(-1, 1), y.reshape(-1, 1))
        predicted = model.predict(X.reshape(-1, 1))

        order = np.argsort(X)
        ax.plot(X[order], predicted[order], color=color)

    ax.set_ylabel('Freezing')
    ax.set_xlabel('Norm. ensemble activation')

    p = {}
    for context in contexts:
        x = np.asarray(all_activations[context])
        y = np.asarray(all_freezing[context])

        y = y[np.isfinite(x)]
        x = x[np.isfinite(x)]

        p[context] = pearsonr(x,y)

    ax.set_title('R = ' + str(np.round(p['shock'][0], 3)) +
                 ', p = ' + str(np.round(p['shock'][1], 3)) + '\n'
                 'R = ' + str(np.round(p['neutral'][0], 3)) +
                 ', p = ' + str(np.round(p['neutral'][1], 3)))

    fig.show()

    if save_fig:
        fname = os.path.join('C:\\Users\\William Mau\\Documents\\Projects\\S. Ramirez Fear Conditioning\\Figures',
                             'Ensemble correlation to freezing_' + region + '_' + template_session + '.pdf')
        fig.savefig(fname)

    return fig


def RecallEnsembleTimecourse(region=BLA_mice, test_sessions = ('E1_1', 'E2_1'),
                             samples_per_bin=2400):
    """
    Track the number of ensemble activations from recall over extinction.

    Parameters
    ---
    region: str, default: 'BLA_mice'
        'BLA_mice' or 'CA1_mice'

    test_sessions: list-like, default: ('E1_1', 'E2_1')
        Sessions over which to track.

    """
    # Initialize some lists and dicts.
    all_sessions = [*test_sessions, 'RE_1']
    n_sessions = len(all_sessions)
    all_activations = []
    all_freezing = []
    all_summed = {}
    min_per_bin = samples_per_bin/60/20

    # Get template ensembles and activations from each mouse.
    for mouse in region:
        try:
            (activation_strengths,
             activations,
             patterns,
             significance,
             norm_activations,
             freezing,
             session_dict,
             ) = cross_day_ensemble_activity(mouse, 'RE_1',
                                             test_sessions,
                                             n_shuffles=500)

            all_activations.append(activations)
            all_freezing.append(freezing)

        # NaN if missing a session.
        except:
            all_activations.append(np.nan)
            all_freezing.append(np.nan)

    # Plot.
    scatter_fig, scatter_ax = plt.subplots(1, n_sessions, sharex=True)
    line_fig, line_ax = plt.subplots(1, n_sessions, sharex=True, sharey=True)
    for i, session in enumerate(all_sessions):
        all_summed[session] = []

        for activations in all_activations:
            try:
                onsets = detect_onsets(activations[session])

                # Bin and sum time bins where ensemble was activated.
                bins = d_pp.make_bins(onsets, samples_per_bin)
                binned_activations = d_pp.bin_time_series(onsets, bins)
                summed = [np.sum(b) for b in binned_activations]
                # Append this to the mega-dict.
                all_summed[session].append(summed)
                # Scatter plot.
                scatter_ax[i].scatter([np.int32(0), *bins], summed, s=5)
            except:
                pass

        # Combine the nested lists.
        all_summed[session] = np.vstack(zip_longest(*all_summed[session], fillvalue=np.nan)).T

        # Make plot values.
        m = np.nanmean(all_summed[session], axis=0)
        t = np.arange(0, len(m) * min_per_bin, min_per_bin)
        std_err = sem(all_summed[session], axis=0)

        # Plot the line.
        line_ax[i].errorbar(t, m, ecolor='gray', yerr=std_err,
                            capsize=0, markersize=3, mfc='gray',
                            mew=0, fmt='o')
        line_ax[i].set_title(session)

    pass



if __name__ == '__main__':
    #Plot_Freezing()
    # for session in ['FC', 'E1_1', 'E2_1']:
    #     Corr_Activations_to_Freezing(template_session=session, save_fig=True)

    RegressCorrTimeSeries()
