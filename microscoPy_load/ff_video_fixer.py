from glob import glob
from os import path
from plotting.plot_helper import ScrollPlot
import matplotlib.pyplot as plt
import pygame
import skvideo.io
from skimage import color
from pandas import read_csv
import pickle
from PIL import Image
import numpy as np
import cv2
from session_directory import load_session_list
from plotting import plot_functions as plot_funcs
from scipy.ndimage.filters import gaussian_filter as gfilt
from microscoPy_load import calcium_traces as ca_traces
from helper_functions import find_closest

session_list = load_session_list()


def load_session(session_index):
    directory = session_list[session_index]["Location"]
    position_path = path.join(directory, 'FreezeFrame', 'Position.pkl')

    with open(position_path, 'rb') as file:
        FF = pickle.load(file)

    return FF


def load_movie(session_index):
    directory = session_list[session_index]["Location"]
    position_path = path.join(directory, 'FreezeFrame', 'Movie.pkl')

    with open(position_path, 'rb') as file:
        FF = pickle.load(file)

    return FF


class FFObj:
    def __init__(self, session_index, stitch=True, pix_to_cm=0.15):
        self.session_index = session_index
        self.csv_location, self.avi_location = self.get_ff_files()

        directory, _ = path.split(self.avi_location)
        self.location = path.join(directory, 'Position.pkl')

        self.pix_to_cm = pix_to_cm
        self.converted = False
        self.movie = skvideo.io.vread(self.avi_location)
        self.n_frames = len(self.movie)
        self.video_t = self.get_timestamps()
        self.get_baseline_frame(stitch)

    def disp_baseline(self):
        """
        Display the baseline frame.
        """
        plt.figure()
        plt.imshow(self.baseline_frame, cmap='gray')

    def get_ff_files(self):
        """
        Find FreezeFrame files by searching the directory.
        """
        ff_dir = path.join(session_list[self.session_index]["Location"],
                           "FreezeFrame")
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
        Scroll through all the frames of the movie. Currently the default
        starting point is the middle.
        """
        # titles = ["Frame " + str(n) for n in range(self.n_frames)]
        #
        # f = ScrollPlot(plot_funcs.display_frame,
        #                movie=self.movie, n_frames=self.n_frames,
        #                titles=titles)
        reader = cv2.VideoCapture(self.avi_location)

        def onChange(track_bar_value):
            reader.set(cv2.CAP_PROP_POS_FRAMES, track_bar_value)
            err, img = reader.read()
            cv2.imshow('Video', img)
            pass

        cv2.namedWindow('Video')
        cv2.createTrackbar('Frame 1', 'Video', 0, self.n_frames, onChange)
        cv2.createTrackbar('Frame 2', 'Video', 0, self.n_frames, onChange)

        onChange(0)
        keep_going = True
        while keep_going:
            k = cv2.waitKey()
            if k == 27:
                keep_going = False

        frame1 = cv2.getTrackbarPos('Frame 1', 'Video')
        frame2 = cv2.getTrackbarPos('Frame 2', 'Video')

        cv2.destroyAllWindows()

        return frame1, frame2

    def select_region(self, frame_num):
        """
        Selects a region from a frame of your choice.
        :param frame_num:
        :return:
        """
        # Convert frame into image object then pass through the
        # FrameSelector class.
        frame = Image.fromarray(self.movie[frame_num])
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
        self.baseline_frame = color.rgb2gray(np.array(paste_onto_me))

    def get_baseline_frame(self, stitch):
        """
        Build the baseline frame.
        """
        # Scroll through the frames.
        if stitch:
            print('Define the frames to stitch. Your crop from the top'
                  ' frame goes onto the bottom frame.')
            from_frame, to_frame = self.scroll_through_frames()

            self.stitch_baseline_frame(from_frame, to_frame)

        else:
            print('Define the baseline frame on the top trackbar.')
            frame, _ = self.scroll_through_frames()
            self.baseline_frame = color.rgb2gray(self.movie[frame])

    def auto_detect_mouse(self, smooth_sigma, threshold):
        """
        Find the mouse automatically using preset settings and blob
        detection.
        :param
            smooth_sigma: factor to smooth delta frames by.
            threshold: intensity cut-off for blob detection.
        """
        MouseDetectorObj = MouseDetector(self.baseline_frame, self.movie,
                                         smooth_sigma)
        self.position = MouseDetectorObj.detect_mouse(threshold)

    def interpolate(self):
        """
        Match timestamps and position to imaging.
        """
        _, self.imaging_t = ca_traces.load_traces(self.session_index)
        self.x = np.interp(self.imaging_t, self.video_t,
                           self.position[:, 0])
        self.y = np.interp(self.imaging_t, self.video_t,
                           self.position[:, 1])
        self.imaging_v = np.interp(self.imaging_t, self.video_t,
                                   self.velocity)

        # Convert to cm.
        try:
            if not self.converted:
                self.x = self.x * self.pix_to_cm
                self.y = self.y * self.pix_to_cm
                self.imaging_v = self.imaging_v * self.pix_to_cm
                self.velocity = self.velocity * self.pix_to_cm
                self.converted = True
        except:
            self.x *= 0.15
            self.y *= 0.15
            self.imaging_v *= 0.15
            self.velocity *= 0.15
            self.converted = True

        self.imaging_freezing = np.zeros(self.imaging_t.shape,
                                         dtype=bool)
        freezing_epochs = self.get_freezing_epochs()

        # Interpolate the freezing bouts.
        for this_epoch in freezing_epochs:
            _, start_idx = find_closest(self.imaging_t,
                                        self.video_t[this_epoch[0]])
            _, end_idx = find_closest(self.imaging_t,
                                      self.video_t[this_epoch[1] - 1])
            self.imaging_freezing[start_idx:end_idx] = True

    def detect_freezing(self, velocity_threshold=7,
                        min_freeze_duration=1.25,
                        plot_freezing=True):
        """
        Detect freezing epochs.
        :param
            velocity_threshold: anything below this threshold is considered
            freezing.
            min_freeze_duration: also, the epoch needs to be longer than
            this scalar.
            plot_freezing: logical, whether you want to see the results.
        """
        pos_diff = np.diff(self.position, axis=0)  # For calculating distance.
        time_diff = np.diff(self.video_t)  # Time difference.
        distance = np.hypot(pos_diff[:, 0], pos_diff[:, 1])  # Displacement.
        self.velocity = np.concatenate(([0], distance // time_diff))  # Velocity.
        self.freezing = self.velocity < velocity_threshold

        freezing_epochs = self.get_freezing_epochs()

        # If any freezing epochs were less than ~1.25 seconds long, get rid of
        # them.
        for this_epoch in freezing_epochs:
            start = this_epoch[0]
            stop = this_epoch[1]
            if self.video_t[stop] - self.video_t[start] < \
                    min_freeze_duration:
                self.freezing[this_epoch[0]:this_epoch[1]] = False

        self.min_freeze_duration = min_freeze_duration
        self.velocity_threshold = velocity_threshold

        if plot_freezing:
            self.plot_freezing()

    def get_freezing_epochs(self):
        padded_freezing = np.concatenate(([0], self.freezing, [0]))
        status_changes = np.abs(np.diff(padded_freezing))

        # Find where freezing begins and ends.
        freezing_epochs = np.where(status_changes == 1)[0].reshape(-1, 2)

        freezing_epochs[freezing_epochs >= self.video_t.shape[0]] = \
            self.video_t.shape[0] - 1

        # Only take the middle.
        freezing_epochs = freezing_epochs[1:-1]

        return freezing_epochs

    def get_freezing_epochs_imaging_framerate(self):
        # Get freezing epochs in video time.
        epochs = self.get_freezing_epochs()

        # Get the imaging indices for freezing.
        freeze_epochs = np.zeros(epochs.shape, dtype=int)
        for i, this_epoch in enumerate(epochs):
            _, start = find_closest(self.imaging_t, self.video_t[this_epoch[0]])
            _, end = find_closest(self.imaging_t, self.video_t[this_epoch[1]])
            freeze_epochs[i, :] = [start, end]

        return freeze_epochs

    def get_mouse_in_cage_epoch(self, frame1=None, frame2=None):
        # Get epoch (as a frame # of the tracking video) where mouse is in
        # the fear conditioning chamber.
        self.mouse_in_cage_imaging_t = np.zeros(2)

        if frame1 is None:
            print('Define frames where mouse enters the chamber (top) or '
                  'exits the chamber (bottom).')
            frame1, frame2 = self.scroll_through_frames()

        # Get timestamps.
        self.mouse_in_cage_imaging_t[0], start = \
            find_closest(self.imaging_t, self.video_t[frame1])
        self.mouse_in_cage_imaging_t[1], end = \
            find_closest(self.imaging_t, self.video_t[frame2])
        self.mouse_in_cage = np.zeros(self.imaging_t.shape, dtype=bool)
        self.mouse_in_cage[start:end] = True


    def plot_position(self, current_position=0):
        # Plot frame and position of mouse.
        titles = ["Frame " + str(n) for n in range(self.n_frames)]

        self.f = ScrollPlot(plot_funcs.display_frame_and_position,
                            current_position=current_position,
                            movie=self.movie, n_frames=self.n_frames,
                            position=self.position, titles=titles)

    def correct_position(self, current_position=0, detect_freezing=True):
        self.plot_position(current_position)
        self.f.fig.canvas.mpl_connect('button_press_event',
                                      self.on_press)

        while plt.get_fignums():
            plt.waitforbuttonpress()

        if detect_freezing:
            self.detect_freezing(plot_freezing=False)
            self.interpolate()

    def on_press(self, event):
        self.position[self.f.current_position] = \
            (event.xdata, event.ydata)
        self.f.fig.axes[0].plot(event.xdata, event.ydata, 'go')
        self.f.fig.canvas.draw()

    def plot_freezing(self):
        # Plot frame and position of mouse. Blue dots indicate freezing epochs.
        titles = ["Frame " + str(n) for n in range(self.n_frames)]

        f = ScrollPlot(plot_funcs.display_frame_and_freezing,
                       movie=self.movie, n_frames=self.n_frames,
                       position=self.position, freezing=self.freezing,
                       titles=titles)

    def plot_velocity(self):
        plt.plot(self.velocity)

    def process_video(self, smooth_sigma=6, mouse_threshold=0.15,
                      velocity_threshold=7, min_freeze_duration=1.25,
                      plot_freezing=True, manual_correct=True):
        """
        Main function for detecting mouse and correcting video.
        """
        print('Select frame where mouse enters (top) and leaves '
              '(bottom) the box.')
        mouse_in, mouse_out = self.scroll_through_frames()
        self.auto_detect_mouse(smooth_sigma, mouse_threshold)

        # Do manual correction around when mouse enters the cage.
        # This is normally where the tracker fails.
        if manual_correct:
            print('Do manual corrections now.')
            self.correct_position(mouse_in, detect_freezing=False)
            while plt.get_fignums():
                plt.waitforbuttonpress()

            print('Do manual corrections again.')
            self.correct_position(mouse_out, detect_freezing=False)
            while plt.get_fignums():
                plt.waitforbuttonpress()

        self.detect_freezing(velocity_threshold, min_freeze_duration,
                             plot_freezing)
        self.interpolate()
        self.get_mouse_in_cage_epoch(mouse_in, mouse_out)

    def save_data(self):
        directory, _ = path.split(self.avi_location)
        movie_file = path.join(directory, 'Movie.pkl')

        if hasattr(self, 'movie'):
            with open(movie_file, 'wb') as output:
                pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)
            del self.movie

        if hasattr(self, 'f'):
            del self.f

        with open(self.location, 'wb') as output:
            pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)


class FrameSelector:
    """
    Draws a rectangle over the frame and collects its coordinates.
    https://stackoverflow.com/questions/20349901/interactively-drawing-rectangles-on-the-image
    """

    def __init__(self, frame):
        self.frame = frame

    def display_frame(self, topleft, prior):
        """
        Displays the frame and bounded rectangle.
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
        # Convert PIL Image to pygame image.
        mode = self.frame.mode
        size = self.frame.size
        data = self.frame.tobytes()
        self.frame = pygame.image.fromstring(data, size, mode)

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

    def make_delta_baseline(self, frame):
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
        d_movie = np.zeros(self.movie.shape[0:3])
        for i, frame in enumerate(self.movie):
            frame = color.rgb2gray(frame)
            d_movie[i] = self.make_delta_baseline(frame)

        return d_movie

    def threshold_movie(self, d_movie, threshold):
        """
        Threshold the difference movie.
        :param
            d_movie: stacked frames of the difference between frames and baseline.
            threshold: scalar [0,1].
        """
        thresh_movie = [cv2.inRange(frame, threshold, 1)
                        for frame in d_movie]

        return thresh_movie

    def build_blob_detector(self):
        """
        Set the parameters for the blob detector here.
        """
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.maxThreshold = 255
        params.maxArea = 100000

        detector = cv2.SimpleBlobDetector_create(params)

        return detector

    def detect_mouse(self, threshold):
        d_movie = self.make_difference_movie()  # Make difference movie.
        thresh_movie = self.threshold_movie(d_movie, threshold)  # Make thresholded movie.
        detector = self.build_blob_detector()  # Build blob detector.

        # Detect blobs on each frame.
        position = np.zeros([self.n_frames, 2])
        for i, frame in enumerate(thresh_movie):
            blobs = detector.detect(frame)

            blob_sizes = [this_blob.size for this_blob in blobs]

            # Find the biggest blob. This is usually where the mouse is most likely.
            try:
                biggest = max(blob_sizes)
                biggest_blob = blob_sizes.index(biggest)

                position[i] = blobs[biggest_blob].pt
            except:
                position[i] = [0, 0]

        return position


if __name__ == '__main__':
    FF = load_movie(1)
    f = FF.plot_position()
    FF.correct_position(f)