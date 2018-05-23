import calcium_traces as ca_traces
import data_preprocessing as d_pp
import ff_video_fixer as ff
import numpy as np
from scipy.stats import zscore
from session_directory import load_session_list
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, \
    permutation_test_score, LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cell_reg import load_cellreg_results, find_match_map_index, \
    find_cell_in_map, trim_match_map
from random import randint
import calcium_events as ca_events

session_list = load_session_list()


def preprocess_NB(session_index, bin_length=2, predictor='traces'):
    session = ff.load_session(session_index)

    # Get accepted neurons.
    if predictor == 'traces':
        predictor_var, t = ca_traces.load_traces(session_index)
        predictor_var = zscore(predictor_var, axis=1)
    elif predictor == 'events':
        predictor_var, t = ca_events.load_events(session_index)
        predictor_var[predictor_var > 0] = 1
    else:
        raise ValueError('Predictor incorrectly defined.')

    # Trim the traces to only include instances where mouse is in the chamber.
    t = d_pp.trim_session(t, session.mouse_in_cage)
    predictor_var = d_pp.trim_session(predictor_var,
                                      session.mouse_in_cage)
    freezing = d_pp.trim_session(session.imaging_freezing,
                                 session.mouse_in_cage)

    # Define bin limits.
    samples_per_bin = bin_length * 20
    bins = d_pp.make_bins(t, samples_per_bin)
    binned_activity = d_pp.bin_time_series(predictor_var, bins)

    if predictor == 'traces':
        X = np.mean(np.asarray(binned_activity[0:-1]),axis=2)
        X = np.append(X, np.mean(binned_activity[-1],axis=1)[None, :],
                      axis=0)

    elif predictor == 'events':
        X = np.sum(np.asarray(binned_activity[0:-1]),axis=2)
        X = np.append(X, np.sum(binned_activity[-1],axis=1)[None, :],
                      axis=0)

    else:
        raise ValueError('Invalid data type.')

    # Bin freezing vector.
    binned_freezing = d_pp.bin_time_series(freezing, bins)
    Y = [i.any() for i in binned_freezing]

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
                                bin_length=2, predictor='traces',
                                neurons=None):
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
    match_map = load_cellreg_results(mouse)

    idx = find_match_map_index([train_session, test_session])

    if neurons is not None:
        global_cell_idx = find_cell_in_map(match_map, idx[0], neurons)
        match_map = match_map[global_cell_idx,:]

    trimmed_map = match_map[:, [idx[0], idx[1]]]
    detected_both_days = (trimmed_map > -1).all(axis=1)
    trimmed_map = trimmed_map[detected_both_days, :]

    X_train = X_train[:, trimmed_map[:, 0]]
    X_test = X_test[:, trimmed_map[:, 1]]

    return X_train, X_test, y_train, y_test



def fit_cross_session_NB(X_train, X_test, y_train, y_test,
                         classifier=
                         make_pipeline(StandardScaler(), GaussianNB())):

    classifier.fit(X_train, y_train)
    predict_test = classifier.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, predict_test)

    return accuracy


def cross_session_NB(train_session, test_session, bin_length=2,
                     predictor='traces', neurons=None, I=1000):

    X_train, X_test, y_train, y_test = \
        preprocess_NB_cross_session(train_session, test_session,
                                    bin_length=bin_length,
                                    predictor=predictor,
                                    neurons=neurons)

    if predictor == 'traces':
        classifier = make_pipeline(StandardScaler(), GaussianNB())
    elif predictor == 'events':
        classifier = MultinomialNB()

    score = fit_cross_session_NB(X_train, X_test, y_train, y_test,
                                 classifier=classifier)

    permutation_scores = np.zeros((I))
    for i in range(I):
        y_shuffle = np.random.permutation(y_train)

        permutation_scores[i] = fit_cross_session_NB(X_train, X_test,
                                                     y_shuffle, y_test,
                                                     classifier=classifier)

    p_value = np.sum(score < permutation_scores)/I

    return score, permutation_scores, p_value

def cross_session_NB2(train_session, test_session, bin_length=2,
                      predictor='traces', neurons=None):
    X_train, X_test, y_train, y_test = \
        preprocess_NB_cross_session(train_session, test_session,
                                    bin_length=bin_length,
                                    predictor=predictor,
                                    neurons=neurons)

    X = np.concatenate((X_train, X_test))
    y = y_train + y_test

    train_label = np.zeros(len(y_train), dtype=int)
    test_label = np.ones(len(y_test), dtype=int)
    groups = np.concatenate((train_label, test_label))

    cv = LeaveOneGroupOut()

    if predictor == 'traces':
        classifier = make_pipeline(StandardScaler(), GaussianNB())
    elif predictor == 'events':
        classifier = make_pipeline(MultinomialNB())
    else:
        raise ValueError('Predictor incorrectly defined.')

    score, permutation_scores, p_value = \
        permutation_test_score(classifier, X, y, scoring='accuracy',
                               groups=groups, cv=cv, n_permutations=1000,
                               n_jobs=1)

    return score, permutation_scores, p_value


if __name__ == '__main__':
    #from single_cell_analyses.footshock import ShockSequence
    bin_length = 1
    # X, Y = preprocess_NB(0)
    # score, permutation_scores, p_value = NB_session_permutation(X, Y)
    # accuracy = NB_session(X, Y)
    session_1 = [0, 5, 10, 15]
    session_2 = [[1, 2, 4], [6, 7, 9], [11, 12, 14], [16, 17, 19]]
    all_sessions = [[0, 1, 2, 4], [5, 6, 7, 9], [10, 11, 12, 14],
                    [15, 16, 17, 19]]
    # shuffled = []
    # for i in np.arange(100):
    #     accuracy = cross_session_NB(s1,s2,shuffle=True)
    #     shuffled.append(accuracy)
    #
    # accuracy = cross_session_NB(s1,s2)

    scores_events = np.zeros((len(session_1),3))
    pvals_events = np.zeros((len(session_1),3))
    scores_traces = np.zeros((len(session_1),3))
    pvals_traces = np.zeros((len(session_1),3))
    #S = ShockSequence(s1)

    for i, fc in enumerate(session_1):
        match_map = load_cellreg_results(session_list[fc]['Animal'])
        trimmed = trim_match_map(match_map,all_sessions[i])
        neurons = trimmed[:,0]
        for j, ext in enumerate(session_2[i]):
            score, _, p_value = \
            cross_session_NB(fc, ext, bin_length=bin_length, predictor='events',
                              neurons=neurons)

            scores_events[i, j] = score
            pvals_events[i, j] = p_value

            score, _, p_value = \
            cross_session_NB(fc, ext, bin_length=bin_length, predictor='traces',
                             neurons=neurons)

            scores_traces[i, j] = score
            pvals_traces[i, j] = p_value

    pass