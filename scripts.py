from population_analyses.freezing_NB import cross_session_NB
import numpy as np
from single_cell_analyses.footshock import ShockSequence
from calcium_events import load_events
import random

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
                cross_session_NB(fc, ext, bin_length=bin_length,
                                 neurons=shock_neurons)

            scores_without[i,j], _, _ = \
                cross_session_NB(fc, ext, bin_length=bin_length,
                                 neurons=shock_neurons_omitted)

            scores_without_random[i,j], _, _ = \
                cross_session_NB(fc, ext, bin_length=bin_length,
                                 neurons=random_neurons_omitted)

     pass

if __name__ == '__main__':
    CompareShockCellDecoderContribution()