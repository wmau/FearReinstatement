import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels
from pandas import MultiIndex
from scipy.stats import pearsonr, mannwhitneyu
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import fdrcorrection
from helper_functions import ismember, nan, bool_array
from plotting.plot_functions import scatter_box
from population_analyses.freezing_classifier import classify_cross_session
from session_directory import get_session, \
    get_region, load_session_list
from single_cell_analyses.event_rate_correlations import time_lapse_corr

session_list = load_session_list()

# Specify mice and days to be included in analysis.
mice = ('Kerberos',
        'Nix',
        'Calypso',
        'Hyperion',
        'Pandora',
        'Janus',
        'Kepler',
        'Mundilfari',
        'Aegir',
        )

days = ('E1_1',
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


def CrossSessionEventRateCorr(bin_size=1, slice_size=30,
                              ref_mask_start=None, corr=pearsonr):
    # Get correlations and session identities.
    correlations = []
    mouse_id = []
    session_id = []
    for i, mouse in enumerate(mice):
        for session in session_2_stages[i]:
            correlations.append(time_lapse_corr(mouse, session,
                                                bin_size=bin_size,
                                                slice_size=slice_size,
                                                ref_mask_start=ref_mask_start,
                                                plot_flag=False,
                                                corr=corr))
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

    # Get indices for each session type.
    session_boundaries = np.cumsum(max_sizes)
    E1_1 = bool_array(all_correlations.shape[1],
                      range(session_boundaries[0]))
    E2_1 = bool_array(all_correlations.shape[1],
                      range(session_boundaries[0],
                            session_boundaries[1]))
    RE_1 = bool_array(all_correlations.shape[1],
                      range(session_boundaries[1],
                            session_boundaries[2]))
    E1_2 = bool_array(all_correlations.shape[1],
                      range(session_boundaries[2],
                            session_boundaries[3]))
    E2_2 = bool_array(all_correlations.shape[1],
                      range(session_boundaries[3],
                            session_boundaries[4]))
    RE_2 = bool_array(all_correlations.shape[1],
                      range(session_boundaries[4],
                            session_boundaries[5]))

    # Get indices for regions.
    CA1_mice = ismember('CA1', regions)[0]
    BLA_mice = ismember('BLA', regions)[0]

    # Partition the matrix into BLA and CA1.
    CA1 = all_correlations[CA1_mice]
    BLA = all_correlations[BLA_mice]

    # Get indices for context 1 and context 2.
    context_1 = E1_1 | E2_1 | RE_1
    context_2 = E1_2 | E2_2 | RE_2

    context1_boundaries = session_boundaries[:3]
    context2_boundaries = np.cumsum(max_sizes[3:])

    # Plot time series of correlation coefficients for CA1.
    f, ax = plt.subplots(2, 2)
    ax[0, 0].plot(CA1[:, context_1].T)
    ax[0, 0].plot(np.nanmean(CA1[:, context_1], axis=0),
                  linewidth=2, color='k')
    for boundary in context1_boundaries:
        ax[0, 0].axvline(x=boundary)
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Correlation Coefficient')
    ax[0, 0].set_title('CA1')

    # Plot time series of correlation coefficients for BLA.
    ax[0, 1].plot(BLA[:, context_1].T)
    ax[0, 1].plot(np.nanmean(BLA[:, context_1], axis=0),
                  linewidth=2, color='k')
    for boundary in context1_boundaries:
        ax[0, 1].axvline(x=boundary)

    ax[0, 1].set_xlabel('Time')
    ax[0, 1].set_title('BLA')

    # Boxplot by session type for CA1.
    data = [CA1[:, E1_1].flatten(),
            CA1[:, E2_1].flatten(),
            CA1[:, RE_1].flatten()]
    data = [x[~np.isnan(x)] for x in data]
    scatter_box(data, xlabels=['Ext1', 'Ext2', 'Recall'],
                ylabel='Correlation Coefficient', ax=ax[1, 0])

    # Boxplot by session type for BLA.
    data = [BLA[:, E1_1].flatten(),
            BLA[:, E2_1].flatten(),
            BLA[:, RE_1].flatten()]
    data = [x[~np.isnan(x)] for x in data]
    scatter_box(data, xlabels=['Ext1', 'Ext2', 'Recall'], ax=ax[1, 1])

    ###
    f, ax = plt.subplots(2, 2)
    ax[0, 0].plot(CA1[:, context_2].T)
    ax[0, 0].plot(np.nanmean(CA1[:, context_2], axis=0),
                  linewidth=2, color='k')
    for boundary in context2_boundaries:
        ax[0, 0].axvline(x=boundary)

    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Correlation Coefficient')
    ax[0, 0].set_title('CA1')

    # Plot time series of correlation coefficients for BLA.
    ax[0, 1].plot(BLA[:, context_2].T)
    ax[0, 1].plot(np.nanmean(BLA[:, context_2], axis=0),
                  linewidth=2, color='k')
    for boundary in context2_boundaries:
        ax[0, 1].axvline(x=boundary)

    ax[0, 1].set_xlabel('Time')
    ax[0, 1].set_title('BLA')

    # Boxplot by session type for CA1.
    data = [CA1[:, E1_2].flatten(),
            CA1[:, E2_2].flatten(),
            CA1[:, RE_2].flatten()]
    data = [x[~np.isnan(x)] for x in data]
    scatter_box(data, xlabels=['Ext1', 'Ext2', 'Recall'],
                ylabel='Correlation Coefficient', ax=ax[1, 0])

    # Boxplot by session type for BLA.
    data = [BLA[:, E1_2].flatten(),
            BLA[:, E2_2].flatten(),
            BLA[:, RE_2].flatten()]
    data = [x[~np.isnan(x)] for x in data]
    scatter_box(data, xlabels=['Ext1', 'Ext2', 'Recall'], ax=ax[1, 1])

    CA1_E1_1 = CA1[:, E1_1].flatten()
    CA1_E1_1 = CA1_E1_1[~np.isnan(CA1_E1_1)]

    CA1_E2_1 = CA1[:, E2_1].flatten()
    CA1_E2_1 = CA1_E2_1[~np.isnan(CA1_E2_1)]

    CA1_RE_1 = CA1[:, RE_1].flatten()
    CA1_RE_1 = CA1_RE_1[~np.isnan(CA1_RE_1)]

    BLA_E1_1 = BLA[:, E1_1].flatten()
    BLA_E1_1 = BLA_E1_1[~np.isnan(BLA_E1_1)]

    BLA_E2_1 = BLA[:, E2_1].flatten()
    BLA_E2_1 = BLA_E2_1[~np.isnan(BLA_E2_1)]

    BLA_RE_1 = BLA[:, RE_1].flatten()
    BLA_RE_1 = BLA_RE_1[~np.isnan(BLA_RE_1)]

    E1_E2_CA1 = mannwhitneyu(CA1_E1_1, CA1_E2_1).pvalue
    E1_RE_CA1 = mannwhitneyu(CA1_E1_1, CA1_RE_1).pvalue
    E2_RE_CA1 = mannwhitneyu(CA1_E2_1, CA1_RE_1).pvalue
    E1_E2_BLA = mannwhitneyu(BLA_E1_1, BLA_E2_1).pvalue
    E1_RE_BLA = mannwhitneyu(BLA_E1_1, BLA_RE_1).pvalue
    E2_RE_BLA = mannwhitneyu(BLA_E2_1, BLA_RE_1).pvalue

    plt.show()
    pass


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


if __name__ == '__main__':
    CrossSessionClassify()
