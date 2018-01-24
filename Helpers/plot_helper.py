# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:53:29 2018

@author: William Mau
"""
import matplotlib.pyplot as plt


class ScrollPlot:
    """ 
    Plot stuff then scroll through it!

    :param
        x: X axis data.
        y: Y axis data.
        xlabel = 'x': X axis label.
        ylabel = 'y': Y axis label.


    """

    # Initialize the class. Gather the data and labels.
    def __init__(self, plot_func, xlabel = 'x', ylabel = 'y', titles = (["Title"] * 10000), **kwargs):
        self.plot_func = plot_func
        [self.fig, self.ax] = plt.subplots()
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.titles = titles

        # Dump all arguments into ScrollPlot.
        for key,value in kwargs.items():
            setattr(self,key,value)

        # Necessary for scrolling.
        self.current_position = 0

        # Plot the first time series and label.
        self.plot_func(self)
        self.apply_labels()

        # Connect the figure to keyboard arrow keys.
        self.fig.canvas.mpl_connect('key_press_event',
                                    lambda event: self.update_plots(event))

    # Go up or down the list. Left = down, right = up.
    def scroll(self, event):
        if (event.key == 'right') and (self.current_position < self.last_position):
            self.current_position += 1
        elif (event.key == 'left') and (self.current_position > 0):
            self.current_position -= 1

    # Apply axis labels.
    def apply_labels(self):
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.titles[self.current_position])

    # Update the plot based on keyboard inputs.
    def update_plots(self, event):
        # Clear axis.
        self.ax.cla()

        # Scroll then update plot.
        self.scroll(event)

        # Run the plotting function.
        self.plot_func(self)

        # Draw.
        self.fig.canvas.draw()
        self.apply_labels()


def neuron_number_title(neurons):
    titles = ["Neuron: " + str(n) for n in neurons]

    return titles