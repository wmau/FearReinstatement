import cell_stats
from os import path
from pandas import read_csv
from csv import DictReader
from pickle import dump, load
from numpy import delete,array
from session_directory import load_session_list

session_list = load_session_list()


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
                self.event_times = data.event_times
                self.event_values = data.event_values
                self.t = data.t

        # If that didn't work, recompile.
        except:
            # Initialize.
            self.traces = []
            self.t = []
            self.n_ICs = cell_stats.get_number_of_ICs(session_number)
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

    def compile_traces(self):
        """
        Compile calcium traces.
        """
        with open(self.trace_file, 'r') as csv_file:
            self.traces = read_csv(csv_file, skiprows=2).T.as_matrix()  # Need to transpose here.

    def compile_time(self):
        """
        Extract the time vector.
        """
        self.t = self.traces[0, :]
        self.traces = delete(self.traces, 0, axis=0)  # Delete time vector from traces.

    def compile_events(self):
        """
        Compile calcium events.
        """
        # Open calcium events file.
        with open(self.event_file) as csv_file:
            events_csv = DictReader(csv_file)

            self.event_times = [[] for x in range(self.n_ICs)]
            self.event_values = [[] for x in range(self.n_ICs)]
            # Gather calcium events.
            for row in events_csv:
                ind = int(row[" Cell Name"][2:])
                self.event_times[ind].append(row["Time (s)"])
                self.event_values[ind].append(row[" Value"])

        # Turn a list of lists into an array of lists. This will make indexing easier.
        self.event_times = array(self.event_times)
        self.event_values = array(self.event_values)

    def compile_all(self):
        """
        Compile everything.
        """
        self.compile_accepted_list()
        self.compile_traces()
        self.compile_time()
        self.compile_events()

        # Pickle the class instance.
        with open(self.cell_data_file, 'wb') as output:
            dump(self, output)


if __name__ == '__main__':
    CellData(11)