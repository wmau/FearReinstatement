from os import path
from session_directory import load_session_list
import scipy.io as sio
import numpy as np

session_list = load_session_list()

class seqNMF:
    def __init__(self, session_index, var_names=('H','W','XC','thres'),
                 mat_file='seqNMF_results.mat'):
        self.session_index = session_index
        self.mouse = session_list[session_index]["Animal"]
        self.directory = session_list[session_index]["Location"]
        self.filename = path.join(self.directory, mat_file)

        self.data = self.load_data(var_names)
        self.count_sequences()
        self.sequence_cells = self.get_sequential_cells()
        self.order = self.get_sequence_order()

    def load_data(self, var_names):
        data = sio.loadmat(self.filename, variable_names=var_names)

        return data

    def count_sequences(self):
        try:
            self.n_sequences = np.sum(self.data['H'].any(axis=1))
        except:
            data = sio.loadmat(self.filename, variable_names='H')
            self.n_sequences = np.sum(data['H'].any(axis=1))

    def get_sequential_cells(self):
        try:
            XC = self.data['XC']
            thres = self.data['thres']
        except:
            raise ValueError('Variables XC and/or thres do not exist')

        # Get statistically significant cells for each sequence.
        significant_cells = []
        n_time_bins = XC.shape[1]
        for sequence, thresholds in zip(range(self.n_sequences), thres):
            # Cross correlation of each cell.
            if self.n_sequences == 1:
                xcorr = XC
            else:
                xcorr = XC[:,:,sequence]
            thresholds = np.tile(thresholds,[n_time_bins,1]).T

            # Time bins where cross correlation exceeds null.
            significant_bins = np.where((xcorr > thresholds).any(axis=1))[0]
            significant_cells.append(significant_bins)

        return significant_cells

    def get_sequence_order(self):
        try:
            W = self.data['W']
        except:
            raise ValueError('Variable W does not exist')

        # Get statistically significant cells.
        significant_cells = self.get_sequential_cells()
        order = []

        # Get their rank based on their power in that factor.
        for factor_power, cells in \
                zip(np.reshape(W, (W.shape[1],W.shape[0],-1)),
                    significant_cells):
            peaks = np.argmax(factor_power[cells],
                              axis=1).astype(int)
            order.append(np.argsort(peaks).astype(int))

        return order


if __name__ == '__main__':
    S = seqNMF(5)
    S.get_sequential_cells()