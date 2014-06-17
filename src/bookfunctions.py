""" This file contains functions that are used by both the page classifier and
image localizer """

from skimage import color
from skimage.feature import hog
from scipy import misc
import os
import numpy as np
from pystruct.utils import SaveLogger

def remove_unannotated_books(input_folder, books):
	""" Removes the books from the array 'books' that do not have a
	subfolder called 'annotated' """
	return_books = []
	for book in books:
		path = input_folder + os.sep + book + os.sep + 'annotated'
		if os.path.exists(path) and os.path.isdir(path):
			return_books.append(book)
	return return_books


def calculate_hog(image, number_of_blocks):
	pixels_per_cell = calculate_pixels_per_cell(image, number_of_blocks)
	image = color.rgb2gray(image)
	# The number of orientations should be kept fixed as long as we don't
	# save them to the metafiles
	return hog(image, orientations=8, pixels_per_cell=pixels_per_cell,
		cells_per_block=(1, 1))


def calculate_pixels_per_cell(image, number_of_blocks):
	s = np.shape(image)
	pixels_vertical = int(s[0]/number_of_blocks[0])
	pixels_horizontal = int(s[1]/number_of_blocks[1])
	return (pixels_horizontal, pixels_vertical)

def calculate_hog_locations(image, number_of_blocks):
	""" Calculates the center of all hogs, given the image (used for the
	size) Returns a list of tuples of size np.prod(number_of_blocks)
	 """
	pixels_per_cell = calculate_pixels_per_cell(image, number_of_blocks)
	y_half = pixels_per_cell[0]/2
	# All Y coordinates are starting from half window size, each of them
	# pixels_per_cell[0] apart
	y_coordinates = range(int(round(y_half)), \
		int(round(np.shape(image)[0]-y_half)), \
		pixels_per_cell[0])
	# Do the same for X coordinates
	x_half = pixels_per_cell[1]/2
	x_coordinates = range(int(round(x_half)), \
		int(round(np.shape(image)[1]-x_half)), \
		pixels_per_cell[0])
	# Concatenate both coordinate sets in a list containing all coordinates
	coordinates = []
	for y in y_coordinates:
		for x in x_coordinates:
			# Vertical direction goes first in numpy 
			coordinates.append((y, x))
	return coordinates


def read_image_from_annotation_file(input_folder, annotated_image, book):
	""" Given the path of an annotation .py file, finds the corresponding
	image and returns it as returned by misc.imread """
	base = os.path.basename(annotated_image)
	name = os.path.splitext(base)[0]
	image_name = input_folder + os.sep + book + os.sep + \
		'raw' + os.sep + name + '.png'
	return misc.imread(image_name)


def get_bounding_boxes_from_page(page_path, model, number_of_blocks):
	""" reads 1 image from path, calcualtes hog features for that image,
	uses model to label feature, calls get_bounding_boxes_from_labels to
	obtain the predicted bounding boxes"""

	image = misc.imread(page_path)
	hog = calculate_hog(image, number_of_blocks)

	# reshape: have the 0'th index horizontal and the 1'th index vertical
	hog.shape = number_of_blocks[1], number_of_blocks[0], 8
	# transpose: fromg x-y to y-x coordinates
	hog = hog.transpose((1, 0, 2))

	labels = np.array(model.predict(hog))

	get_bounding_boxes_from_labels(labels, image, number_of_blocks)

def get_bounding_boxes_from_labels(labels, image, number_of_blocks):
	""" obtains the bounding boxes for 1 image from labeled hog features """
	# get the coordinates
	coordinates = calculate_hog_locations(image, number_of_blocks)


if __name__ == '__main__':
	logger = SaveLogger('../models/logc1.py')
	model = logger.load()
	get_bounding_boxes_from_page('../data/aanzienlykeScheepsTogt/raw/500_0001.png',
		model, [20, 10])


