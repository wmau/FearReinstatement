from glob import glob
from os import path
from plot_helper import ScrollPlot
import matplotlib.pyplot as plt
import pygame
import skvideo.io
from pandas import read_csv
from PIL import Image
from session_directory import load_session_list
import plot_functions as plot_funcs

session_list = load_session_list()

class FFObj:
    def __init__(self, session_index):
        self.session_index = session_index
        self.csv_location, self.avi_location = self.get_ff_files()
        self.movie = skvideo.io.vread(self.avi_location)
        self.n_frames = len(self.movie)
        self.t = self.get_timestamps()

    def get_ff_files(self):
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
        with open(self.csv_location, 'r') as csv_file:
            data = read_csv(csv_file)

        t = data.iloc[:, 0]
        return t

    def scroll_through_frames(self):
        titles = ["Frame " + str(n) for n in range(self.n_frames)]

        ScrollPlot(plot_funcs.display_frame,
                   movie = self.movie, n_frames = self.n_frames,
                   titles = titles)

        pass

    def select_region(self):
        frameObj = FrameSelector(Image.fromarray(self.movie[1, :, :]))

        frameObj.setup()
        rectangle = frameObj.loop_through()

        return rectangle


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
        while n!= 1:

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

#FF = FFObj(0)
#FF.scroll_through_frames()