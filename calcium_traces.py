# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:11:14 2018

@author: William Mau
"""

from os import path
from pickle import load
from pandas import read_csv

master_directory = 'U:\Fear conditioning project_Mosaic2\SessionDirectories'
file = path.join(master_directory,'SessionDirectories.pkl')
session_list = load(open(file,'rb'))

def load_traces(session_index):
    #Get file name.
    trace_file = path.join(session_list[session_index]["Location"],'Traces.csv')

    #Get accepted/rejected list. 
    with open(trace_file,'r') as csv_file:
        accepted = read_csv(csv_file,nrows=1).T
        
    #For reasons I don't understand yet, read_csv modifies your position on 
    #the CSV file so we need to reload the file. Now, actually get the traces. 
    with open(trace_file,'r') as csv_file:
        traces = read_csv(csv_file,skiprows=2).T
           
    return traces,accepted
