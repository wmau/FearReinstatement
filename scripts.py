from population_analyses.freezing_classifier import classify_cross_session
import numpy as np
from single_cell_analyses.footshock import ShockSequence
from microscoPy_load.calcium_events import load_events
import random
from session_directory import get_session_stage, get_session, get_region
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from pandas import MultiIndex
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind, ttest_rel

def CompareShockCellDecoderContribution():

    bin_length = 2
    session_1 = [0, 5, 10, 15]
    session_2 = [[1, 2, 4], [6, 7, 9], [11, 12, 14], [16, 17, 19]]
    all_sessions = [[0, 1, 2, 4], [5, 6, 7, 9], [10, 11, 12, 14],
                    [15, 16, 17, 19]]

    scores_with = np.zeros((len(session_1),3))
    scores_without = np.zeros((len(session_1),3))
    scores_without_random = np.zeros((len(session_1),3))

    for i, fc in enumerate(session_1):
        S = ShockSequence(fc)
        shock_neurons = S.shock_modulated_cells

        events, _ = load_events(fc)
        neurons = list(range(len(events)))
        shock_neurons_omitted = list(set(neurons).difference(shock_neurons))

        neurons_to_omit = random.sample(shock_neurons_omitted,
                                        len(shock_neurons))
        random_neurons_omitted = list(set(neurons).difference(neurons_to_omit))

        for j, ext in enumerate(session_2[i]):
            scores_with[i,j], _, _ = \
                classify_cross_session(fc, ext, bin_length=bin_length,
                                       neurons=shock_neurons)

            scores_without[i,j], _, _ = \
                classify_cross_session(fc, ext, bin_length=bin_length,
                                       neurons=shock_neurons_omitted)

            scores_without_random[i,j], _, _ = \
                classify_cross_session(fc, ext, bin_length=bin_length,
                                       neurons=random_neurons_omitted)


def CrossSessionClassify(bin_length = 1, I = 100,
                         classifier=None,
                         predictor='traces', shuffle='scramble'):

    # Specify mice and days to be included in analysis.
    mice = ('Kerberos',
            'Nix',
            'Hyperion',
            'Calypso',
            'Pandora',
            'Janus',
            'Kepler',
            )

    days = ('E1_1',
            'E2_1',
            'RE_1',
            'E1_2',
            'E2_2',
            'RE_2',
             )

    # Get region for each mouse.
    regions = [get_region(mouse) for mouse in mice]

    # Set up.
    n_mice = len(mice)
    n_days = len(days)

    scores, pvals = np.empty((n_mice,n_days)), \
                    np.empty((n_mice,n_days))
    permuted = np.empty((n_mice,n_days,I))
    scores.fill(np.nan)
    pvals.fill(np.nan)
    permuted.fill(np.nan)

    # Preallocate.
    session_1 = [get_session(mouse,'FC')[0]
                 for mouse in mice]
    session_2 = [get_session(mouse,days)[0]
                 for mouse in mice]

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

    mouse_label = np.repeat(mice,I*n_days)
    mouse_label = np.concatenate([mouse_label,
                                  np.repeat(mice, n_days)])

    condition_label = np.repeat("Shuffled", n_mice*n_days*I)
    condition_label = np.concatenate([condition_label,
                                    np.repeat("Real", n_mice*n_days)])

    region_label = np.repeat(regions,I*n_days)
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
                    data=df.loc[df['Condition']=='Shuffled'],
                    kind="violin",
                    bw=.2,
                    col_wrap=True)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.setGeometry(2400, -420, 420, 1610)

    for i, mouse in enumerate(mice):
        sns.stripplot(x="Day",
                      y="Accuracy",
                      data=df.loc[(df['Condition']=='Real') &
                                  (df['Mouse']==mouse)],
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
    CA1_real = (means['Condition']=='Real') & (means['Region']=='CA1')
    CA1_shuffle = (means['Condition']=='Shuffled') & (means['Region']=='CA1')

    y_real = means.loc[CA1_real]['Accuracy']
    yerr_real = sems.loc[CA1_real]['Accuracy']

    y_shuffle = means.loc[CA1_shuffle]['Accuracy']
    yerr_shuffle = sems.loc[CA1_shuffle]['Accuracy']

    plt.figure()
    plt.errorbar(days[0:3], y_real, yerr=yerr_real)
    plt.errorbar(days[0:3], y_shuffle, yerr=yerr_shuffle)

    formula = 'Accuracy ~ C(Day) + C(Condition) + C(Day):C(Condition)'
    #model = ols(formula, collapse_by_mouse).fit() # Pool across mice.
    model = ols(formula,
                one_context.loc[one_context['Region']=='CA1']).fit()
    anova = statsmodels.stats.anova.anova_lm(model, type=2)

    # Compare conditions individually.
    y = one_context.loc[(one_context['Condition'] == 'Shuffled') &
                        (one_context['Day'] == 'E1_1') &
                        (one_context['Region'] == 'CA1')]['Accuracy']
    x = one_context.loc[(one_context['Condition'] == 'Real') &
                        (one_context['Day'] == 'E1_1') &
                        (one_context['Region'] == 'CA1')]['Accuracy']

    plt.show()
    return scores, pvals, permuted

if __name__ == '__main__':
    CrossSessionClassify()