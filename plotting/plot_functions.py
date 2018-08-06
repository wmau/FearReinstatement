"""
List of plotting functions to pass through ScrollPlot
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_traces(obj):
    """
    For plotting calcium traces.
    :param
        obj: ScrollPlot class object.
    """
    obj.ax.plot(obj.t, obj.traces[obj.current_position])
    obj.last_position = len(obj.traces) - 1


def plot_events(obj):
    """
    For plotting calcium events.
    :param
        obj: ScrollPlot class object.
    :return:
    """
    obj.ax.scatter(obj.event_times[obj.current_position], obj.event_values[obj.current_position],marker='.')
    obj.last_position = len(obj.event_values) - 1


def overlay_events(obj):
    """
    For plotting calcium events over traces.
    :param
        obj: ScrollPlot class object.
    :return:
    """
    plot_traces(obj)
    plt.hold(True)
    plot_events(obj)


def display_frame(obj):
    """
    For plotting FreezeFrame video frames.
    :param obj:
    :return:
    """

    if obj.current_position == 0:
        obj.current_position = round(obj.n_frames / 2)

    obj.ax.imshow(obj.movie[obj.current_position], cmap='gray')
    obj.last_position = obj.n_frames - 1


def display_frame_and_position(obj):
    if obj.current_position == 0:
        obj.current_position = round(obj.n_frames / 2)

    obj.ax.imshow(obj.movie[obj.current_position], cmap='gray')
    obj.ax.plot(obj.position[obj.current_position, 0], obj.position[obj.current_position, 1], 'ro')
    obj.last_position = obj.n_frames - 1


def display_frame_and_freezing(obj):
    if obj.current_position == 0:
        obj.current_position = round(obj.n_frames / 2)

    obj.ax.imshow(obj.movie[obj.current_position], cmap='gray')
    if obj.freezing[obj.current_position]:
        obj.ax.plot(obj.position[obj.current_position, 0], obj.position[obj.current_position, 1], 'bo')
    else:
        obj.ax.plot(obj.position[obj.current_position, 0], obj.position[obj.current_position, 1], 'ro')
    obj.last_position = obj.n_frames - 1

def plot_freezing_traces(obj):
    for i,this_epoch in enumerate(obj.epochs):
        obj.ax[i].plot(obj.t[this_epoch[0]:this_epoch[1]],
                       obj.traces[obj.current_position, this_epoch[0]:this_epoch[1]])

    obj.last_position = len(obj.traces) - 1

def plot_heatmap(obj):
    obj.ax.imshow(obj.heatmap[obj.current_position])
    plt.axis('tight')

    obj.last_position = len(obj.heatmap) - 1

def plot_multiple_traces(obj):
    for i,trace in enumerate(obj.traces[obj.current_position]):
        c = cm.gray(i/len(obj.traces[obj.current_position]),1)
        obj.ax.plot(obj.t,trace,color=c)

    obj.ax.plot(obj.t,np.mean(obj.traces[obj.current_position],
                              axis=0),
                color='r', linewidth=2)

    obj.last_position = len(obj.traces) - 1

def plot_footprints_over_days(obj):
    for i,footprint in enumerate(obj.footprints[obj.current_position]):
        obj.ax[i].imshow(footprint, cmap='gray')

    for ax in obj.ax:
        ax.set_yticks([])
        ax.set_xticks([])

    plt.setp(obj.ax.flat, aspect='equal')
    obj.last_position = len(obj.footprints) - 1

def plot_traces_over_days(obj):
    for i,trace in enumerate(obj.traces[obj.current_position]):
        obj.ax[i].plot(obj.t[i],trace)

    obj.last_position = len(obj.traces) - 1

def plot_raster(obj):
    obj.ax.eventplot(obj.events[obj.current_position])
    obj.ax.set_xlim([-obj.window, 0])

    obj.last_position = len(obj.events) - 1