from os import path
from pandas import read_csv
from csv import DictReader
from pickle import dump, load
from numpy import delete,array
from session_directory import load_session_list
import numpy as np
from glob import glob

session_list = load_session_list()

def get_number_of_ICs(session_index):
    directory = path.join(session_list[session_index]["Location"],'ROIs')
    ROIs = glob(path.join(directory,'ROIs_C*.*'))

    n_ICs = len(ROIs)
    return n_ICs


class CellData:
    def __init__(self, session_number):
        # File directories and paths.
        self.directory = session_list[session_number]["Location"]
        self.cell_data_file = path.join(self.directory, 'CellData.pkl')
        self.trace_file = path.join(self.directory, 'Traces.csv')
        self.event_file = path.join(self.directory, 'Events.csv')

        # Try opening up a pickled file. This is faster than compiling all the data again.
        try:
            with open(self.cell_data_file, 'rb') as file:
                data = load(file)
                self.traces = data.traces
                self.n_ICs = data.n_ICs
                self.accepted = data.accepted
                self.accepted_neurons = data.accepted_neurons
                self.event_times = data.event_times
                self.event_values = data.event_values
                self.t = data.t

        # If that didn't work, recompile.
        except:
            # Initialize.
            self.traces = []
            self.t = []
            self.n_ICs = get_number_of_ICs(session_number)
            self.accepted = [False] * self.n_ICs

            # Compile.
            self.compile_all()

    def compile_accepted_list(self):
        """
        Compile the accepted cell list.
        """

        # Open the trace CSV.
        with open(self.trace_file, "r") as csv_file:
            accepted_csv = read_csv(csv_file, nrows=1).T

        # Delete the first row, which is like a header.
        accepted_csv = accepted_csv.iloc[1:]

        # Turn this thing into a logical.
        for cell in range(self.n_ICs):
            if accepted_csv.iloc[cell, 0] == ' accepted':
                self.accepted[cell] = True

        self.accepted_neurons = np.asarray(
                                [cell_number for cell_number, good in
                                 enumerate(self.accepted) if good]
                                )

    def compile_traces(self):
        """
        Compile calcium traces.
        """
        with open(self.trace_file, 'r') as csv_file:
            traces = read_csv(csv_file, skiprows=1).T.as_matrix()  # Need to transpose here.

        # First row is the time vector so extract that then add 1 to
        # the neuron indices.
        self.t = traces[0,:]
        self.traces = traces[self.accepted_neurons + 1]

    def compile_events(self):
        """
        Compile calcium events.
        """
        # Open calcium events file.
        with open(self.event_file) as csv_file:
            events_csv = DictReader(csv_file)

            # Preallocate.
            self.event_times = [[] for x in range(self.n_ICs)]
            self.event_values = [[] for x in range(self.n_ICs)]

            # Gather calcium events.
            for row in events_csv:

                # Get the cell number then append the event timestamps to the nested list.
                ind = int(row[" Cell Name"][2:])
                self.event_times[ind].append(row["Time (s)"])
                self.event_values[ind].append(row[" Value"])

        # Turn a list of lists into an array of lists. This will make indexing easier.
        event_times = array(self.event_times)
        self.event_times = event_times[self.accepted_neurons]
        event_values = array(self.event_values)
        self.event_values = event_values[self.accepted_neurons]

    def compile_all(self):
        """
        Compile everything.
        """
        self.compile_accepted_list()
        self.compile_traces()
        self.compile_events()

        # Pickle the class instance.
        with open(self.cell_data_file, 'wb') as output:
            dump(self, output)


if __name__ == '__main__':
    CellData(0)