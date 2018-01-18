# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:11:14 2018

@author: William Mau
"""

from os import path
from pickle import load, dump
from pandas import read_csv
from numpy import delete
from plot_helper import scroll
import matplotlib.pyplot as plt

master_directory = 'U:\Fear conditioning project_Mosaic2\SessionDirectories'
file = path.join(master_directory,'SessionDirectories.pkl')
session_list = load(open(file,'rb'))

def load_traces(session_index):
    """ 
    
    Load traces from data of single sessions saved via Inscopix Data Processing
    software. This may take a few seconds to run if running for the first time 
    on a session.
    
    """
    
    #Get the directory.
    session_directory = session_list[session_index]["Location"]
    
    #Reading the CSV takes a few seconds. If we've already done this step, 
    #instead just load the saved data. 
    try:
        with open(path.join(session_directory,'CelLData.pkl'),'rb') as data:
            [traces,accepted,t] = load(data)
            
    except:
        #Get file name.
        trace_file = path.join(session_directory,'Traces.csv')
    
        #Get accepted/rejected list. 
        with open(trace_file,'r') as csv_file:
            accepted_csv = read_csv(csv_file,nrows=1).T  
            
        #Delete the first row, which is like a header. 
        accepted_csv = accepted_csv.iloc[1:]
        
        #Turn this thing into a logical. 
        neuron_count = len(accepted_csv)
        accepted = [False]*neuron_count
        for cell in range(0,neuron_count):
            if accepted_csv.iloc[cell,0] == ' accepted':
                accepted[cell] = True
        
        #For reasons I don't understand yet, read_csv modifies your position on 
        #the CSV file so we need to reload the file. Now, actually get the traces. 
        with open(trace_file,'r') as csv_file:
            traces = read_csv(csv_file,skiprows=2).T.as_matrix()
            
        #Extract the time vector. 
        t = traces[0,:] 
        
        #Delete time vector from traces. 
        traces = delete(traces,(0),axis=0)
               
        #Save data.
        with open(path.join(session_directory,'CellData.pkl'),'wb') as output:
            dump([traces,accepted,t],output)
        
    return traces,accepted,t

def plot_traces(session_index,neurons):
    """
    
    Plot the traces of multiple neurons. You can scroll between neurons!
    
    """
 
    #Load the traces.
    [traces,accepted,t] = load_traces(session_index)
    
    #Make the figure then connect it to keyboard inputs. 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.canvas.mpl_connect('key_press_event',lambda event: 
                            scroll(event,t,traces[neurons],ax,fig))
        
    #Plot the first time series. Then subsequent arrow key strokes will scroll
    #through different neurons.
    ax.plot(t,traces[neurons[0]])
    plt.xlabel('Time (s)')
    plt.ylabel('% DF/F')
    
 