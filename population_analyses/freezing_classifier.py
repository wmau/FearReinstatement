import data_preprocessing as d_pp
import numpy as np
from scipy.stats import zscore
from session_directory import load_session_list
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, \
    permutation_test_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from microscoPy_load.cell_reg import load_cellreg_results, find_match_map_index, \
    find_cell_in_map
from microscoPy_load import calcium_events as ca_events, calcium_traces as ca_traces, ff_video_fixer as ff

session_list = load_session_list()


def preprocess(session_index, bin_length=2, predictor='traces'):
    session = ff.load_session(session_index)

    # Get accepted neurons.
    if predictor == 'traces':
        predictor_var, t = ca_traces.load_traces(session_index)
        masked = np.ma.masked_invalid(predictor_var)
        predictor_var = zscore(masked, axis=1)
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
        X = np.nanmean(np.asarray(binned_activity[0:-1]),axis=2)
        X = np.append(X, np.nanmean(binned_activity[-1],axis=1)[None, :],
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

    # Handle missing values.
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp = imp.fit(X)
    X = imp.transform(X)

    return X, Y


def classify(X, Y):
    """
    Simple within-session naive Bayes decoder. CV done with default
    0.2 leave-out.
    """
    # Build train and test sets.
    X_train, X_test, y_train, y_test = \
        train_test_split(X, Y, test_size=0.2)
    classifier = make_pipeline(StandardScaler(), GaussianNB())
    classifier.fit(X_train, y_train)
    predict_test = classifier.predict(X_test)

    # Classify.
    accuracy = metrics.accuracy_score(y_test, predict_test)

    return accuracy


def classify_with_permutation(X, Y):
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


def preprocess_cross_session(train_session, test_session,
                             bin_length=2, predictor='traces',
                             neurons=None):
    # Make sure the data comes from the same mouse.
    mouse = session_list[train_session]["Animal"]
    assert mouse == session_list[test_session]["Animal"], \
        "Mouse names don't match!"

    # Trim and bin data from both sessions.
    X_train, y_train = preprocess(train_session, bin_length=bin_length,
                                  predictor=predictor)
    X_test, y_test = preprocess(test_session, bin_length=bin_length,
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



def fit_cross_session(X_train, X_test, y_train, y_test,
                      classifier=
                      make_pipeline(StandardScaler(), GaussianNB())):

    classifier.fit(X_train, y_train)
    predict_test = classifier.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, predict_test)

    return accuracy


def classify_cross_session(train_session, test_session, bin_length=2,
                           predictor='traces', neurons=None, I=1000,
                           classifier=None):

    X_train, X_test, y_train, y_test = \
        preprocess_cross_session(train_session, test_session,
                                 bin_length=bin_length,
                                 predictor=predictor,
                                 neurons=neurons)

    if classifier is None:
        if predictor == 'traces':
            classifier = make_pipeline(StandardScaler(), GaussianNB())
        elif predictor == 'events':
            classifier = MultinomialNB()
        else:
            raise ValueError('Wrong predictor data type.')


    score = fit_cross_session(X_train, X_test, y_train, y_test,
                              classifier=classifier)

    permutation_scores = np.zeros((I))
    for i in range(I):
        y_shuffle = np.random.permutation(y_train)

        permutation_scores[i] = fit_cross_session(X_train, X_test,
                                                  y_shuffle, y_test,
                                                  classifier=classifier)

    p_value = np.sum(score < permutation_scores)/I

    return score, permutation_scores, p_value

def cross_session_classify_timelapse(train_session, test_session,
                                     resolution=0.05,
                                     score_avg_bin_length=30,
                                     predictor_type='traces', neurons=None):
    X_train, X_test, y_train, y_test = \
        preprocess_cross_session(train_session, test_session,
                                 bin_length=resolution,
                                 predictor=predictor_type,
                                 neurons=neurons)

    if predictor_type == 'traces':
        classifier = make_pipeline(StandardScaler(), GaussianNB())
    elif predictor_type == 'events':
        classifier = MultinomialNB()
    else:
        raise ValueError('Wrong predictor data type.')

    bin_size = int(score_avg_bin_length / resolution)
    score_avg_bins = d_pp.make_bins(X_test[:,0], bin_size)
    n_bins = len(score_avg_bins) + 1
    binned_predictor = d_pp.bin_time_series(X_test.T, score_avg_bins)
    binned_response = d_pp.bin_time_series(np.asarray(y_test),
                                           score_avg_bins)

    scores = np.zeros((n_bins))
    for i, (predictor, response) in enumerate(zip(binned_predictor,
                                                binned_response)):

        scores[i] = fit_cross_session(X_train, predictor.T, y_train,
                                      response, classifier=classifier)

    return scores

if __name__ == '__main__':
    classify_cross_session(23, 30, predictor='traces', bin_length=1)
