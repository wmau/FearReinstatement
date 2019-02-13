# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:16:38 2019

@author: Eichenbaum lab
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import TextBox
import tifffile as TF
import os
import numpy as np
import time

def crop_and_write(fnames, destination):
    # Load the tif.
    tif = TF.TiffFile(fnames[0]).asarray()
    initial_text = '(Crop from left, crop to right, crop from top, crop to bottom)'

    # Define crop function
    def do_crop(text):
        # Get the values from the textbox.
        split_values = text.split(",")
        coordinates = [int(i) for i in split_values]
        x0, x1, y0, y1 = coordinates

        # for each file, crop then save to new location.
        for file in fnames:
            tif = TF.TiffFile(file).asarray()
            cropped = tif[:, x0:x1, y0:y1]

            tif_name = os.path.split(file)[-1]
            TF.imsave(os.path.join(destination, tif_name), cropped)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
    ax.imshow(np.std(tif,0))
    txt_box = TextBox(axbox, 'Crop coords: ', initial=initial_text)
    txt_box.on_submit(do_crop)



class Cropper:
    def __init__(self, ax):
        self.ax = ax
        self.rect = Rectangle((0, 0), 1, 1, alpha=0.1)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

        while plt.get_fignums():
            plt.waitforbuttonpress()

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()

        time.sleep(5)

        plt.close()

if __name__ == '__main__':
    fnames = ['G:\\Skoll\\11_12_2018_FC\\recording_20181112_091939.tif',
              'G:\\Skoll\\11_12_2018_FC\\recording_20181112_091939.tif']
    destination = 'G:\\Skoll\\Test'
    crop_and_write(fnames, destination)