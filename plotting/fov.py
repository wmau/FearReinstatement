import os
from session_directory import get_session, load_session_list
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle

session_list = load_session_list()

def plot_fov(mouse, session_stage, filename='MaxProj.tif'):
    """
    Plots the field of view, usually the maximum projection.

    Parameters:
        mouse: str, mouse name.
        session_stage: str, session name.
        filename: str, name of file (not the full path).

    Returns:
        f: figure object.
        ax: axes object.
        img: image object.

    """
    # Get image location.
    session_index = get_session(mouse, session_stage)[0]
    folder = session_list[session_index]['Location']
    file = os.path.join(folder, filename)

    # Read image.
    img = np.array(Image.open(file))

    # Plot.
    f, ax = plt.subplots(1,1)
    ax.imshow(img, cmap=plt.cm.gray)
    ax.axis('equal')
    ax.axis('off')

    return f, ax, img


def plot_rois(mouse, session_stage, ax=None, color='xkcd:azure',
              alpha=0.5, neurons=None):
    """
    Plots ROI outlines.

    Parameters:
        mouse; str, mouse name.
        session_stage: str, session name.
        ax: axis object, plots on this. If None, makes an new axis.
        color: color.
        alpha: scalar, transparency.

    Returns:
        ax: axis object.
    """
    # Get session and pickle file.
    session_index = get_session(mouse, session_stage)[0]
    roi_file = os.path.join(session_list[session_index]["Location"],
                            'ROI_Outlines.pkl')

    # Open saved ROI outlines or make them.
    try:
        with open(roi_file, 'rb') as f:
            contours = pickle.load(f)
    except:
        from microscoPy_load.cell_reg import build_cell_rois
        contours = build_cell_rois(mouse, session_stage)

    # Make axis if None.
    if ax is None:
        f, ax = plt.subplots(1,1)

    # If neurons are specified, get those.
    if neurons is not None:
        contours = contours[neurons]

    # Plot.
    for cell in contours:
        ax.plot(cell[:,1], cell[:,0], color=color, alpha=alpha)

    return ax


def place_scale_bar(ax, pix_to_microns=1.1, length_in_microns=100):
    """
    Puts a scale bar on an axis.

    Parameters:
        ax: axis object.
        pix_to_microns: scalar, pixels to micron conversion.
        length_in_microns: scalar, length of scale bar in microns.

    """
    n_pixels = length_in_microns/pix_to_microns

    x = [0, n_pixels]
    y = [0, 0]

    ax.plot(x, y, color='w', linewidth=5)


def overlay_proj_rois(mouse, session_stage):
    """
    Overlays the ROI outlines on the field of view.

    Parameters:
        mouse: str, mouse name.
        session_stage: str, session name.

    """
    f, ax, img = plot_fov(mouse, session_stage)
    ax = plot_rois(mouse, session_stage, ax=ax)
    place_scale_bar(ax)

    f.show()

if __name__ == '__main__':
    overlay_proj_rois('Pandora', 'FC')