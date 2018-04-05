import calcium_traces as ca_traces
import data_preprocessing as d_pp
import ff_video_fixer as ff
import numpy as np
from scipy.stats import zscore
from session_directory import load_session_list
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, \
    permutation_test_score, GroupKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cell_reg import load_cellreg_results, find_match_map_index
from random import randint
import calcium_events as ca_events

session_list = load_session_list()


def preprocess_NB(session_index, bin_length=2, predictor='traces'):
    session = ff.load_session(session_index)

    # Get accepted neurons.
    predictor_var, accepted, t = ca_traces.load_traces(session_index)
    if predictor == 'traces':
        predictor_var = zscore(predictor_var, axis=0)
    elif predictor == 'events':
        predictor_var = ca_events.make_event_matrix(session_index)
    else:
        raise ValueError('Predictor incorrectly defined.')

    neurons = d_pp.filter_good_neurons(accepted)
    n_neurons = len(neurons)

    # Trim the traces to only include instances where mouse is in the chamber.
    t = d_pp.trim_session(t, session.mouse_in_cage)
    predictor_var = d_pp.trim_session(predictor_var, session.mouse_in_cage)
    freezing = d_pp.trim_session(session.imaging_freezing,
                                 session.mouse_in_cage)

    # Define bin limits.
    samples_per_bin = bin_length * 20
    bins = d_pp.make_bins(t, samples_per_bin)
    n_samples = len(bins) + 1

    # Bin the imaging data.
    X = np.zeros([n_samples, n_neurons])
    for n, this_neuron in enumerate(neurons):
        binned_activity = d_pp.bin_time_series(predictor_var[this_neuron], bins)
        # X[:, n] = [np.mean(chunk) for chunk in binned_activity]
        X[:, n] = [np.sum(chunk > 0) for chunk in binned_activity]

    # Bin freezing vector.
    binned_freezing = d_pp.bin_time_series(freezing, bins)
    Y = [i.all() for i in binned_freezing]

    return X, Y


def NB_session(X, Y):
    # Build train and test sets.
    X_train, X_test, y_train, y_test = \
        train_test_split(X, Y, test_size=0.2)
    classifier = make_pipeline(StandardScaler(), GaussianNB())
    classifier.fit(X_train, y_train)
    predict_test = classifier.predict(X_test)

    # Classify.
    accuracy = metrics.accuracy_score(y_test, predict_test)

    return accuracy


def NB_session_permutation(X, Y):
    # Build classifier and cross-validation object.
    classifier = make_pipeline(StandardScaler(), GaussianNB())
    cv = StratifiedKFold(2)

    # Classify and permutation tests.
    score, permutation_scores, p_value = \
        permutation_test_score(classifier, X, Y, scoring='accuracy',
                               cv=cv, n_permutations=500, n_jobs=1)

    return score, permutation_scores, p_value

    # lda = LinearDiscriminantAnalysis(solver='eigen',n_components=2,shrinkage='auto')
    # lda.fit(X,binned_freezing)
    # Y = lda.transform(X)


def preprocess_NB_cross_session(train_session, test_session,
                                bin_length=2, predictor='traces'):
    # Make sure the data comes from the same mouse.
    mouse = session_list[train_session]["Animal"]
    assert mouse == session_list[test_session]["Animal"], \
        "Mouse names don't match!"

    # Trim and bin data from both sessions.
    X_train, y_train = preprocess_NB(train_session, bin_length=bin_length,
                                     predictor=predictor)
    X_test, y_test = preprocess_NB(test_session, bin_length=bin_length,
                                   predictor=predictor)

    # Get registration map.
    match_map, _, _ = load_cellreg_results(mouse)

    idx = find_match_map_index([train_session, test_session])

    trimmed_map = match_map[:, [idx[0], idx[1]]]
    detected_both_days = (trimmed_map > -1).all(axis=1)
    trimmed_map = trimmed_map[detected_both_days, :]

    X_train = X_train[:, trimmed_map[:, 0]]
    X_test = X_test[:, trimmed_map[:, 1]]

    return X_train, X_test, y_train, y_test


def cross_session_NB(train_session, test_session, bin_length=2,
                     shuffle=False):
    X_train, X_test, y_train, y_test = \
        preprocess_NB_cross_session(train_session, test_session,
                                    bin_length=bin_length)

    if shuffle:
        y_train = np.random.permutation(y_train)

    classifier = make_pipeline(StandardScaler(), MultinomialNB())
    classifier.fit(X_train, y_train)
    predict_test = classifier.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, predict_test)

    return accuracy


def cross_session_NB2(train_session, test_session, bin_length=2,
                      predictor='traces'):
    X_train, X_test, y_train, y_test = \
        preprocess_NB_cross_session(train_session, test_session,
                                    bin_length=bin_length,
                                    predictor=predictor)

    X = np.concatenate((X_train, X_test))
    y = y_train + y_test

    train_label = np.zeros(len(y_train), dtype=int)
    test_label = np.ones(len(y_test), dtype=int)
    groups = np.concatenate((train_label, test_label))

    cv = GroupKFold(n_splits=2)

    if predictor == 'traces':
        classifier = make_pipeline(StandardScaler(), GaussianNB())
    elif predictor == 'events':
        classifier = make_pipeline(MultinomialNB())
    else:
        raise ValueError('Predictor incorrectly defined.')

    score, permutation_scores, p_value = \
        permutation_test_score(classifier, X, y, scoring='accuracy',
                               groups=groups, cv=cv, n_permutations=500,
                               n_jobs=1)

    return score, permutation_scores, p_value


if __name__ == '__main__':
    # X, Y = preprocess_NB(0)
    # score, permutation_scores, p_value = NB_session_permutation(X, Y)
    # accuracy = NB_session(X, Y)
    s1 = 0
    s2 = [1,2,4]
    # shuffled = []
    # for i in np.arange(100):
    #     accuracy = cross_session_NB(s1,s2,shuffle=True)
    #     shuffled.append(accuracy)
    #
    # accuracy = cross_session_NB(s1,s2)

    score_kerberos_events = []
    pval_kerberos_events = []
    score_kerberos_traces = []
    pval_kerberos_traces = []
    for s in s2:
        score, permutation_scores, p_value = \
            cross_session_NB2(s1, s, bin_length=2, predictor='events')

        score_kerberos_events.append(score)
        pval_kerberos_events.append(p_value)

        score, permutation_scores, p_value = \
            cross_session_NB2(s1, s, bin_length=2, predictor='traces')

        score_kerberos_traces.append(score)
        pval_kerberos_traces.append(p_value)

    s1 = 5
    s2 = [6, 7, 9]

    score_nix_events = []
    pval_nix_events = []
    score_nix_traces = []
    pval_nix_traces = []
    for s in s2:
        score, permutation_scores, p_value = \
            cross_session_NB2(s1, s, bin_length=2, predictor='events')

        score_nix_events.append(score)
        pval_nix_events.append(p_value)

        score, permutation_scores, p_value = \
            cross_session_NB2(s1, s, bin_length=2, predictor='traces')

        score_nix_traces.append(score)
        pval_nix_traces.append(p_value)

    s1 = 10
    s2 = [11, 12, 14]

    score_atlas_events = []
    pval_atlas_events = []
    score_atlas_traces = []
    pval_atlas_traces = []
    for s in s2:
        score, permutation_scores, p_value = \
            cross_session_NB2(s1, s, bin_length=2, predictor='events')

        score_atlas_events.append(score)
        pval_atlas_events.append(p_value)

        score, permutation_scores, p_value = \
            cross_session_NB2(s1, s, bin_length=2, predictor='traces')

        score_atlas_traces.append(score)
        pval_atlas_traces.append(p_value)

    pass