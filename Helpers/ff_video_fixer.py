from glob import glob
from os import path
from plot_helper import ScrollPlot
import matplotlib.pyplot as plt
import pygame
import skvideo.io
from skimage import color
from pandas import read_csv
from PIL import Image
import numpy as np
import cv2
from session_directory import load_session_list
import plot_functions as plot_funcs
from scipy.ndimage.filters import gaussian_filter as gfilt
import calcium_traces as ca_traces

session_list = load_session_list()


class FFObj:
    def __init__(self, session_index, stitch=True):
        self.session_index = session_index
        self.csv_location, self.avi_location = self.get_ff_files()
        self.movie = np.squeeze(skvideo.io.vread(self.avi_location,as_grey=True))
        self.n_frames = len(self.movie)
        self.video_t = self.get_timestamps()

        if stitch:
            self.get_baseline_frame()

    def get_ff_files(self):
        """
        Find FreezeFrame files by searching the directory.
        """
        ff_dir = path.join(session_list[self.session_index]["Location"], "FreezeFrame")
        csv_location = glob(path.join(ff_dir, '*.csv'))
        avi_location = glob(path.join(ff_dir, '*.avi'))

        # Make sure there's only one of each file.
        assert len(csv_location) is 1
        assert len(avi_location) is 1

        csv_location = csv_location[0]
        avi_location = avi_location[0]

        return csv_location, avi_location

    def get_timestamps(self):
        """
        Open the CSV file imported from FreezeFrame.
        """
        with open(self.csv_location, 'r') as csv_file:
            data = read_csv(csv_file)

        t = np.array(data.iloc[:, 0])
        return t

    def scroll_through_frames(self):
        """
        Scroll through all the frames of the movie. Currently the default starting point
        is the middle.
        """
        titles = ["Frame " + str(n) for n in range(self.n_frames)]

        f = ScrollPlot(plot_funcs.display_frame,
                       movie=self.movie, n_frames=self.n_frames,
                       titles=titles)

        return f

    def select_region(self, frame_num):
        """
        Selects a region from a frame of your choice.
        :param
            frame_num: frame number.
        """
        # Convert frame into image object then pass through the FrameSelector class.
        frame = Image.fromarray(self.movie[frame_num],mode='P')
        frameObj = FrameSelector(frame)
        frameObj.setup()
        region = frameObj.loop_through()
        frameObj.terminate()

        # Obtain the image from the selected ROI.
        chunk = frame.crop(region)
        return chunk, region

    def stitch_baseline_frame(self, from_frame, to_frame):
        """
        Crop a region from from_frame and paste it to to_frame.
        :param
            from_frame: frame number you want to paste onto to_frame.
            to_frame: frame number you want from_Frame to paste to.
        """
        # Get the selected area and the rectangle associated with it.
        chunk, region = self.select_region(from_frame)
        paste_onto_me = Image.fromarray(self.movie[to_frame])

        # Paste and also convert the image to grayscale.
        paste_onto_me.paste(chunk, region)
        self.baseline_frame = np.array(paste_onto_me)

    def get_baseline_frame(self):
        """
        Build the baseline frame.
        """
        # Scroll through the frames.
        f = self.scroll_through_frames()
        fig_num = f.fig.number

        # While the figure is still open, keep going.
        while plt.fignum_exists(fig_num):
            plt.waitforbuttonpress(0)

        # When you hit ESC, prompt for frame numbers.
        else:
            from_frame = int(input("From what frame do you want to cut out a region?"))
            to_frame = int(input("On what frame do you want to paste that region?"))

        # Stitch the partial frames together.
        self.stitch_baseline_frame(from_frame, to_frame)

    def auto_detect_mouse(self, smooth_sigma=6, threshold=80):
        """
        Find the mouse automatically using preset settings and blob detection.
        :param
            smooth_sigma: factor to smooth delta frames by.
            threshold: intensity cut-off for blob detection.
        """
        MouseDetectorObj = MouseDetector(self.baseline_frame, self.movie, smooth_sigma)
        self.position = MouseDetectorObj.detect_mouse(threshold)

    def interpolate(self):
        """
        Match timestamps and position to imaging.
        """
        _,_,imaging_t = ca_traces.load_traces(self.session_index)
        x = np.interp(imaging_t, self.video_t, self.position[:, 0])
        y = np.interp(imaging_t, self.video_t, self.position[:, 1])

        return x,y,imaging_t

    def process_video(self):
        """
        Main function for detecting mouse and correcting video.
        """
        self.auto_detect_mouse()
        self.x,self.y,self.imaging_t = self.interpolate()


class FrameSelector:
    """
    Draws a rectangle over the frame and collects its coordinates.
    https://stackoverflow.com/questions/20349901/interactively-drawing-rectangles-on-the-image
    """

    def __init__(self, frame):
        self.frame = frame
        pygame.init()

    def display_frame(self, topleft, prior):
        """
        Displays the frame and bounded rectangle.
        :param topleft:
        :param prior:
        :return:
        """
        x, y = topleft

        # Get position of computer mouse.
        width = pygame.mouse.get_pos()[0] - topleft[0]
        height = pygame.mouse.get_pos()[1] - topleft[1]

        # Fix numbers if I clicked bottom right then top left instead of the other way around.
        if width < 0:
            x += width
            width = abs(width)
        if height < 0:
            y += height
            height = abs(height)

        # Not sure what the point of this chunk does tbh.
        current = x, y, width, height
        if not (width and height):
            return current
        if current == prior:
            return current

        # Display the frame and the drawn rectangle.
        self.screen.blit(self.frame, self.frame.get_rect())
        im = pygame.Surface((width, height))
        im.fill((128, 128, 128))
        pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
        im.set_alpha(128)
        self.screen.blit(im, (x, y))
        pygame.display.flip()

        # Return the rectangle coordinates.
        return (x, y, width, height)

    def setup(self):
        """
        Build initial display.
        :return:
        """

        palette = self.frame.getpalette()
        palette = np.reshape(palette,(len(palette)//3,3))

        # Convert PIL Image to pygame image.
        mode = self.frame.mode
        size = self.frame.size
        data = self.frame.tobytes()
        self.frame = pygame.image.fromstring(data, size, mode)
        self.frame.set_palette(palette)

        # Get the whole frame, mostly for its dimensions.
        whole = self.frame.get_rect()

        # Display it.
        self.screen = pygame.display.set_mode((whole.width, whole.height))
        self.screen.blit(self.frame, whole)
        pygame.display.flip()

    def loop_through(self):
        """
        Main function that assigns mouse clicks to variables.
        :return:
        """
        topleft = bottomright = prior = None
        n = 0

        # Always scan for mouse clicks.
        while n != 1:

            # Every time an "event" happens...
            for event in pygame.event.get():

                # If it was a mouse click...
                if event.type == pygame.MOUSEBUTTONUP:

                    # And there's no topleft value yet...
                    if not topleft:

                        # Assign the first click to topleft.
                        topleft = event.pos

                    # Otherwise, assign it to bottom right.
                    else:
                        bottomright = event.pos
                        n = 1
            if topleft:
                prior = self.display_frame(topleft, prior)
        return (topleft + bottomright)

    def terminate(self):
        pygame.display.quit()


class MouseDetector:
    def __init__(self, baseline_frame, movie, sigma):
        self.movie = movie
        self.n_frames = len(movie)
        self.sigma = sigma
        self.baseline = baseline_frame
        self.d_movie = self.make_difference_movie()

    def delta_baseline(self, frame):
        """
        Find the difference of a frame from the baseline, then smooth.
        :param
            frame: frame that you want to subtract from baseline.
        """
        # Subtract and smooth.
        delta = gfilt(self.baseline - frame, self.sigma)

        return delta

    def make_difference_movie(self):
        """
        Take the difference between every frame and the baseline.
        """
        # Preallocate then for each frame, subtract it from the baseline.
        d_movie = np.zeros(self.movie.shape)
        for i, frame in enumerate(self.movie):
            d_movie[i] = self.delta_baseline(frame)

        return d_movie

    def threshold_movie(self, d_movie, threshold):
        """
        Threshold the difference movie.
        :param
            d_movie: stacked frames of the difference between frames and baseline.
            threshold: scalar [0,1].
        """
        thresh_movie = [cv2.inRange(frame, threshold, 100) for frame in d_movie]

        return thresh_movie

    def build_blob_detector(self):
        """
        Set the parameters for the blob detector here.
        """
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.maxThreshold = 100
        params.maxArea = 100000

        detector = cv2.SimpleBlobDetector_create(params)

        return detector

    def detect_mouse(self, threshold):
        d_movie = self.make_difference_movie()                      # Make difference movie.
        #thresh_movie = self.threshold_movie(d_movie, threshold)     # Make thresholded movie.
        detector = self.build_blob_detector()                       # Build blob detector.

        # Detect blobs on each frame.
        position = np.zeros([self.n_frames, 2])
        for i, frame in enumerate(d_movie):
            blobs = detector.detect(frame)

            blob_sizes = [this_blob.size for this_blob in blobs]

            # Find the biggest blob. This is usually where the mouse is most likely.
            try:
                biggest = max(blob_sizes)
                biggest_blob = blob_sizes.index(biggest)

                position[i] = blobs[biggest_blob].pt
            except:
                position[i] = [0, 0]

        # Plot frame and position of mouse.
        titles = ["Frame " + str(n) for n in range(self.n_frames)]

        f = ScrollPlot(plot_funcs.display_frame_and_position,
                       movie=self.movie, n_frames=self.n_frames, position=position, titles=titles)

        return position


FF = FFObj(0)
# FF.scroll_through_frames()
# FF.inquire_user_for_baseline_inputs()
# FF.auto_detect_mouse()
FF.process_video()