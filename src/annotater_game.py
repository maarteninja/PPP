import pygame

import os

from pygame.locals import *

from StringIO import StringIO
from PIL import Image

import argparse
import textwrap

# constant for the verbosity level, assumes an integer value
VERBOSE = None


class Annotater(object):
    """ Requires a data folder as input. The folder structure is as
    followed:
        - INPUT_FOLDER
            - raw (contains the raw images of a book)
            - annotated
                - pic (contains annotated images)
                - text (contains all the pages that are just text)

    The following text Assumes the BEGIN_TAG is set to 500_ and the END_TAG is
    set to .png:

    The images require a format of 500_*.png. The leading 500_ are enforced
    because we only want to work with images that have been converted by
    convert.sh.

    The output formad is as followed:
        - text images: 500_*.png
        - pic images: pic_x_500_*.png, where x is an integere (0, 1, n) that
            indicates the amount of earlier annotated pictures from the same
            page
    """

    def __init__(self, folder, begin_tag, end_tag):
        """ creates the folders where necessary, loads the image names,
        initiates pygame and loads the first image """

        # make sure the folders are in order
        self.raw_folder = os.path.join(folder, 'raw')
        self.out_folder = os.path.join(folder, 'annotated')
        self.out_pic_folder = os.path.join(self.out_folder, 'pic')
        self.out_text_folder = os.path.join(self.out_folder, 'text')
        Annotater.create_folder_if_not_exist(self.out_folder)
        Annotater.create_folder_if_not_exist(self.out_pic_folder)
        Annotater.create_folder_if_not_exist(self.out_text_folder)

        # get a list of the images
        self.images = [x for x in os.listdir(self.raw_folder) if \
            x[:len(begin_tag)] == begin_tag and x[-len(end_tag):] == end_tag]
        if VERBOSE > 0:
            print 'found %d images in %s' % (len(self.images), self.raw_folder)

        # sorting not really necessary, but meh, lets sort anyway
        self.images.sort()

        # image index, stores the index of the NEXT image
        self.next_image_index = 0

        pygame.init()
        self.next_image()

    def loop(self):
        """ main loop, checks for input and calls the necessary functions to
        process that input"""

        self.mouse_pressed_coords = None

        self.rectangles = []

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return

                elif event.type == KEYDOWN:
                    # load next image
                    if event.key == K_f:
                        self.process_rectangles()
                        self.rectangles = []
                        self.next_image()

                    # load previous image
                    if event.key == K_d:
                        self.remove_previous_annotated_data()
                        self.rectangles = []
                        self.previous_image()

                    # remove previously added rectangle
                    elif event.key == K_b:
                        if len(self.rectangles) == 0:
                            continue
                        self.rectangles = self.rectangles[:-1]
                        self.reload_image()
                        for pos, size in self.rectangles:
                            self.draw_rectangle(pos, size)

                elif event.type == MOUSEBUTTONDOWN:
                    self.mouse_pressed_coords = pygame.mouse.get_pos()
                elif event.type == MOUSEBUTTONUP:

                    pos, size = Annotater.get_rectangle_pos_size(\
                        self.mouse_pressed_coords, pygame.mouse.get_pos())

                    self.draw_rectangle(pos, size)
                    self.rectangles.append((pos, size))

                    self.mouse_pressed_coords = None

    def process_rectangles(self):
        """ actually cuts the image into pieces, if necessary, and places em in
        the correct folders """

        if self.next_image_index < 0:
            print 'next image index can not be 0 in process_rectangles in process_rectangles (quit)'
            exit()

        current_image_name = self.images[self.next_image_index - 1]

        # if no image found on page, store whole image in text folder
        if len(self.rectangles) == 0:
            out = os.path.join(self.out_text_folder, current_image_name)
            if VERBOSE > 1:
                print '(text image) saving %s' % out
            pygame.image.save(self.screen, out)
            return

        # otherwise, cut the rectangle out of the orignal images and store those
        for i, (pos, size) in enumerate(self.rectangles):
            sub_surface = self.current_image.subsurface(pos, size)
            out = os.path.join(self.out_pic_folder, 'pic_%d_%s' % (i,
                current_image_name))
            if VERBOSE > 1:
                print '(pic image) saving %s' % out
            pygame.image.save(sub_surface, out)

    def remove_previous_annotated_data(self):
        """ removes the possible derivatives of image from self.out_pic_folder
        and self.out_text_folder if they are present there"""
        previous_image_index = self.next_image_index - 2
        if previous_image_index < 0:
            print 'previous image index can not be < 0 in remove_annotated_data_by_image (quit)'
            exit()

        previous_image_name = self.images[self.next_image_index - 2]

        Annotater.empty_folder_containing(self.out_pic_folder,
            previous_image_name, 'pic')
        Annotater.empty_folder_containing(self.out_text_folder,
            previous_image_name, 'text')

    @staticmethod
    def empty_folder_containing(folder, name, cat):
        """ removes all files in folder that contains a part of name
            cat is the category (ie pic or text)"""
        to_delete = [x for x in os.listdir(folder) if name in x]
        for f in to_delete:
            if VERBOSE > 1:
                print '(%s image) deleting %s' % (cat, f)
            os.remove(os.path.join(folder, f))


    def next_image(self):
        """ loads the next image or stops the program if it was the last one """
        if self.next_image_index >= len(self.images):
            print 'zero based image index (%d) exceeds number of images (%d) (quit)'\
                 % (self.next_image_index, len(self.images))
            exit()

        # load image
        self.current_image = pygame.image.load(os.path.join(self.raw_folder,
            self.images[self.next_image_index]))

        # update image index
        self.next_image_index += 1

        # render image
        self.screen = pygame.display.set_mode(self.current_image.get_size())
        self.screen.blit(self.current_image, (0,0))
        pygame.display.flip()

    def reload_image(self):
        """ reloads the current image (makes sure the rectangles are all gone)"""
        if self.next_image_index > 0:
            self.next_image_index -= 1
        self.next_image()

    def previous_image(self):
        """ loads the previous image """
        if self.next_image_index > 1:
            self.next_image_index -= 2
        else:
            print 'this is madness (quit)'
            exit()

        self.next_image()

    def draw_rectangle(self, pos, size):
        """ draws a rectangle on the current screen """
        s = pygame.Surface(size)
        s.set_alpha(100)
        s.fill((255, 0, 0))
        self.screen.blit(s, pos)
        pygame.display.flip()

    @staticmethod
    def get_rectangle_coords(a, b):
        """ orders coordinates of a rectangle: top_left, bottom_right """
        max_w = max(a[0], b[0])
        min_w = min(a[0], b[0])
        max_h = max(a[1], b[1])
        min_h = min(a[1], b[1])
        return (min_w, min_h), (max_w, max_h)

    @staticmethod
    def get_rectangle_pos_size(a, b):
        """ returns the coordinates as a rectangle: pos, size"""
        max_w = max(a[0], b[0])
        min_w = min(a[0], b[0])
        max_h = max(a[1], b[1])
        min_h = min(a[1], b[1])
        pos = min_w, min_h
        size = max_w - min_w, max_h - min_h
        return pos, size

    @staticmethod
    def create_folder_if_not_exist(folder):
        """ creates a folder if it exists. Stops the program if the folder
        specified existed, but not as a folder """
        if os.path.exists(folder):
            if not os.path.isdir(folder):
                print 'supplied path is no folder (quit)'
                exit()
            if VERBOSE > 0:
                print '%s already exists' % folder
            return False
        if VERBOSE > 0:
            print 'creating folder %s' % folder
        os.mkdir(folder)
        return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=Annotater.__doc__)
    parser.add_argument("input_folder", metavar='INPUT_FOLDER', type=str,
                    help="""The input folder.""")
    parser.add_argument("--verbose", '-v', action='count', help="""Set verbosity level (output
general status messages of the program). For example, -v for level 1 and -vv for
level 2""")
    parser.add_argument('--begin_tag', '-b', type=str,
        help='The string that all images should start with (for example 500_)', default='500_')
    parser.add_argument('--end_tag', '-e', type=str,
        help='The string that all images should start with (for example .png)', default='.png')

    args = vars(parser.parse_args())
    input_folder = args['input_folder']
    if not(os.path.exists(input_folder) and os.path.isdir(input_folder)):
        print 'INPUT_FOLDER should be a folder'
        exit()

    VERBOSE = args['verbose']

    annotater = Annotater(input_folder, args['begin_tag'], args['end_tag'])
    annotater.loop()
