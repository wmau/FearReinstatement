"""
List of plotting functions to pass through ScrollPlot
"""

def plot_traces(obj):
    """
    For plotting calcium traces.
    :param
        obj: ScrollPlot class object.
    """
    obj.ax.plot(obj.t,obj.traces[obj.current_position,:])
    obj.last_position = len(obj.traces) - 1

def plot_events(obj):
    """
    For plotting calcium events.
    :param
        obj: ScrollPlot class object.
    :return:
    """
    obj.ax.plot(obj.event_times[obj.current_position], obj.event_values[obj.current_position],'.')
    obj.last_position = len(obj.event_values) - 1

def overlay_events(obj):
    """
    For plotting calcium events over traces.
    :param
        obj: ScrollPlot class object.
    :return:
    """
    plot_traces(obj)
    plot_events(obj)