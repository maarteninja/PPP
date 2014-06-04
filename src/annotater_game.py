import pygame

import os

from pygame.locals import *


class Annotater(object):
    """ Requires a data folder as input. The folder structure is as
    followed:
        - input_folder
            - raw (contains the raw images of a book)
            - annotated
                - pic (contains annotated images)
                - text (contains all the pages that are just text)

    The images require a format of 500_*.png. The leading 500_ are enforced
    because we only want to work with images that have been converted by
    convert.sh.

    The output formad is as followed:
        - text images: 500_*.png
        - pic images: pic_x_500_*.png, where x is an integere (0, 1, n) that
            indicates the amount of earlier annotated pictures from the same
            page
    """

    def __init__(self, folder):
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
        self.images = [x for x in os.listdir(self.raw_folder) if x[:4] == '500_' and \
            x[-4:] == '.jpg']

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
                        self.next_image()

                    # load previous image
                    if event.key == K_d:
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
            print 'next image index can not be 0 in process_rectangles'
            exit()

        current_image_name = self.images[self.next_image_index - 1]

        # if no image found on page, store whole image in text folder
        if len(self.rectangles) < 1:
            self.current_image.tostring(os.path.join(self.out_text_folder,
                current_image_name))
            return

       # otherwise, cut the rectangle out of the orignal images and store those



    def remove_annotated_data_by_image(self, image):
        """ removes the possible derivatives of image from self.out_pic_folder
        and self.out_text_folder if they are present there"""
        pass
            #previous_image_index = self.next_image_index - 2
            #if previous_image_index < 0:
            #    print 'previous image index can not be < 0'
            #    exit()

    def next_image(self):
        """ loads the next image or stops the program if it was the last one """
        if self.next_image_index >= len(self.images):
            print 'zero based image index (%d) exceeds number of images (%d)' % \
                (self.next_image_index, len(self.images))
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
                print 'supplied path is no folder'
                exit()
            print '%s already exists' % folder
            return False
        os.mkdir(folder)
        return True


if __name__ == '__main__':
    annotater = Annotater('../data/test/')
    annotater.loop()