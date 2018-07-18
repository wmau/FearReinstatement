from population_analyses.freezing_classifier import classify_cross_session
import numpy as np
from single_cell_analyses.footshock import ShockSequence
from microscoPy_load.calcium_events import load_events
import random
from session_directory import get_session_stage, get_session
from sklearn.svm import SVC

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


def CrossSessionNaiveBayes(bin_length = 1, I = 1000):
    mice = ['Kerberos',
            'Nix',
            'Titan',
            'Hyperion',
            'Calypso',
            'Pandora',
            'Janus',
            ]

    session_1 = get_session_stage('FC')[0]
    session_2 = [get_session(mouse,'E1_1','E2_1','RE_1')[0]
                 for mouse in mice]

    scores = np.zeros((len(session_1),3))
    pvals = np.zeros((len(session_1),3))
    permuted = np.zeros((len(session_1),3,I))
    for i, fc in enumerate(session_1):
        # match_map = load_cellreg_results(session_list[fc]['Animal'])
        # trimmed = trim_match_map(match_map,all_sessions[i])
        # neurons = trimmed[:,0]
        for j, ext in enumerate(session_2[i]):
            score, permuted_scores, p_value = \
            classify_cross_session(fc, ext, bin_length=bin_length,
                                   predictor='traces', I=I,
                                   classifier=SVC(kernel='linear'))

            scores[i, j] = score
            pvals[i, j] = p_value
            permuted[i, j, :] = permuted_scores

    return scores, pvals, permuted

if __name__ == '__main__':
    CrossSessionNaiveBayes()