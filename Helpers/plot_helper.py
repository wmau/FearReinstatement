# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:53:29 2018

@author: William Mau
"""

global current_position
current_position = 0

def scroll(e,x,y,ax,fig):
    global current_position
    n_plots = len(y)
    
    if e.key == 'right':
        current_position += 1
    elif e.key == 'left':
        current_position -= 1
    else: 
        return
    current_position = current_position % n_plots
    
    ax.cla()
    ax.plot(x,y[current_position])
    fig.canvas.draw()
    
#t = np.linspace(start=0, stop=2*np.pi, num=100)
#y1 = np.cos(t)
#y2 = np.sin(t)
#fig = plt.figure()
#fig.canvas.mpl_connect('key_press_event',lambda event: 
#                        scroll(event,t,(y1,y2)))