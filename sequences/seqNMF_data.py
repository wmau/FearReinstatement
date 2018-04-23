from os import path
from session_directory import load_session_list
import scipy.io as sio
import numpy as np

session_list = load_session_list()

class seqNMF:
    def __init__(self, session_index, var_names):
        self.session_index = session_index
        self.mouse = session_list[session_index]["Animal"]
        self.directory = session_list[session_index]["Location"]
        self.filename = path.join(self.directory,'seqNMF_results.mat')

        self.data = self.load_data(var_names)

    def load_data(self, var_names):
        data = sio.loadmat(self.filename, variable_names=var_names)

        return data

    def get_number_of_sequences(self):
        try:
            self.n_sequences = np.sum(self.data['H'].any(axis=1))
        except:
            data = sio.loadmat(self.filename, variable_names='H')
            self.n_sequences = np.sum(data['H'].any(axis=1))


if __name__ == '__main__':
    S = seqNMF(0, ['XC','H','W'])
    pass