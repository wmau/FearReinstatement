# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:53:29 2018

@author: William Mau
"""
import matplotlib.pyplot as plt
import numpy as np

class ScrollPlot():
    """ 
    Plot stuff then scroll through it!
    """
    
    #Initialize the class. Gather the data and labels. 
    def __init__(self,x,y,xlabel='x',ylabel='y'):
        self.fig, self.ax = plt.subplots()
        self.data = [x,y]
        self.labels = [xlabel,ylabel]
        
        #Necessary for scrolling. 
        self.last_position = len(y)-1
        self.current_position = 0
        
        #Plot the first time series.
        self.ax.plot(x,y[0,:])
        self.apply_labels()
        
        #Connect the figure to keyboard arrow keys. 
        self.fig.canvas.mpl_connect('key_press_event',lambda event:
                                    self.update_plots(event))

    #Go up or down the list. Left = down, right = up. 
    def scroll(self,event):
        if (event.key == 'right') and (self.current_position < self.last_position):
            self.current_position += 1
        elif (event.key == 'left') and (self.current_position > 0):
            self.current_position -= 1

    #Apply axis labels. 
    def apply_labels(self):
        plt.xlabel(self.labels[0])
        plt.ylabel(self.labels[1])
            
    #Update the plot based on keyboard inputs. 
    def update_plots(self,event):
        #Clear axis. 
        self.ax.cla()
        
        #Scroll then update plot.
        self.scroll(event)
        self.ax.plot(self.data[0],self.data[1][self.current_position,:])
        
        #Reset axes. 
        self.ax.relim()
        self.ax.autoscale_view(True,True,True)
        
        #Draw.
        self.fig.canvas.draw()
        self.apply_labels()