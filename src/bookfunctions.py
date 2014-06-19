""" This file contains functions that are used by both the page classifier and
image localizer """

# find . -name *.py -exec sed -i 's/(2, 2))/(1, 1))/g' {} \;


from skimage import color
from skimage.feature import hog
from scipy import misc
import os, glob
import numpy as np
from pystruct.utils import SaveLogger
from numpy import array

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

def get_data(f):
	""" returns the data dict as string, or None in case there is an error"""
	try:
		data = eval(f.read())
	except Exception, e:
		print "WARNING!!! Error in reading data from %s: error: %s" % \
			(f.name, e.message)
		return
	return data

def get_hog_features_page(f, data, page_path, number_of_blocks):
	""" reads a vector of descriptors for an image. f is the file handle of the
	annotated data file, data is the data from that file, page_path the path
	to the image"""

	block_and_cells = (number_of_blocks, (1, 1))
	descriptor = None

	if data.has_key('hog_features') and \
		data['hog_features'].has_key(block_and_cells):
		descriptor = data['hog_features'][block_and_cells]
	else:
		print 'calculating hog features for %s' % str(f.name)
		# Get the image nparray
		image = misc.imread(page_path)
		descriptor = calculate_hog(image, number_of_blocks)

		# If needed, create the dictionary in 'hog_features'
		if not(data.has_key('hog_features')) or \
				type(data['hog_features']) != dict:
			data['hog_features'] = {}

		data['hog_features'][block_and_cells] = descriptor

		# write the new data to the original file:
		f.seek(0)
		f.write(str(data))
		# Remove any remaining text from the previous file contents
		f.truncate()

	assert descriptor != None, 'descriptor cant be None'


	# Reshape the descriptors to the pystruct desired shape
	# We now have the 0'th index horizontal and the 1'th index
	# vertical
	descriptor.shape = (number_of_blocks[1], number_of_blocks[0], 8)
	# Transpose the first two axes, in order to get from x-y
	# coordinates to y-x coordinates
	descriptor = descriptor.transpose((1, 0, 2))

	return descriptor

def get_data_path(annotated_image, input_folder, book):
	""" construct the path to an annotated data file from annotated image, input
	folder and a book """
	base = os.path.basename(annotated_image)
	name = os.path.splitext(base)[0]
	return os.path.join(input_folder, book, 'annotated', name, '.py')

def get_page_path(annotated_image, input_folder, book):
	""" construct the path to an image page from annotated image, input
	folder and a book """
	base = os.path.basename(annotated_image)
	name = os.path.splitext(base)[0]
	return os.path.join(input_folder, book, 'raw', name, '.png')

def get_hog_features(f, data, annotated_image, input_folder, book,
		number_of_blocks):
	""" gets the hog features from the annotated image, or calculates them, and
		them to file, if that is not possible """

	page_path = get_page_path(annotated_image, input_folder, book)
	descriptor = get_hog_features_page(f, data, page_path, number_of_blocks)

	return descriptor

def get_hog_locations_path(f, data, page_path, number_of_blocks):
	""" gets the hog locations from the annotated image, or calculates them, and
		them to file, if that is not possible

	Note: this is only necessary when some hog features are classified as image """

	# If there are images in this page, get their location:
	if data.has_key('hog_locations') and \
		data['hog_locations'].has_key(number_of_blocks):
		return data['hog_locations'][number_of_blocks]

	print 'calculating hog locations'

	# otherwise, calculate em, and save em
	image = misc.imread(page_path)

	hog_locations = calculate_hog_locations(image, number_of_blocks)

	# If needed, create the dictionary in 'hog_features'
	if not(data.has_key('hog_locations')) or \
		type(data['hog_locations']) != dict:
		data['hog_locations'] = {}
	data['hog_locations'][number_of_blocks] = hog_locations

	# write the new data to the original file:
	f.seek(0)
	f.write(str(data))
	# Remove any remaining text from the previous file contents
	f.truncate()

	return hog_locations

def get_hog_locations(f, data, annotated_image, input_folder, book, \
		number_of_blocks):
	""" constructs the path from annotated_image, input_folder and book and
	calls get_hog_locations_path """

	page_path = get_page_path(annotated, input_folder, book)
	return get_hog_locations_path(f, data, page_path, number_of_blocks)

def get_labels_path(f, data, data_path, page_path, number_of_blocks):
	""" calculates labels for the features of one page """

	label = np.ones((number_of_blocks[0], number_of_blocks[1]), dtype=np.int8)
	# if the type is text or bagger, all ones (as above) is okay, and we can stop

	if data['type'] != 'containing':
		return label

	# otherwise we need the hog locations to calculate which features are a 0 (image)
	hog_locations = get_hog_locations_path(f, data, page_path, number_of_blocks)

	# so we loop over the locations of the feature
	for j, coordinate in enumerate(hog_locations):
		# and check if they are in an annotated rectangle (bounding box)
		for pos, size in data['rectangles']:
			# Bounding boxes are x, y, w, h coordinates are y, x.
			# Let's go:
			x,y = pos
			w, h = size
			if coordinate[0] > y and \
					coordinate[0] < y + h and \
					coordinate[1] > x and \
					coordinate[1] < x + w:
				# Set the label to 'image' in stead of 'text'
				label.itemset(j, 0)

	return label

def get_labels(f, data, annotated_image, input_folder, book, number_of_blocks):
	""" constructs the path from annotated_image, input_folder and book and
	calls get_labels_path"""
	page_path = get_page_path(annotated_image, input_folder, book)
	data_path = get_data_path(annotated_image, input_folder, book)
	return get_labels_path(f, data, data_path, page_path, number_of_blocks)

def get_all_labels(pages_data, number_of_blocks):
	""" returns all labels for the pages stored in pages_data. pages_data is
	a tuple with a list of all the paths to all the image pages, and a list of
	all the paths to the annotated data files"""
	labels = [] #np.array([])
	for page_path, data_path in pages_data:
		with open(data_path) as f:
			data = get_data(f)
			#labels = np.append(labels,\
			#	get_labels_path(f, data, data_path, page_path, number_of_blocks))
			labels.append(get_labels_path(f, data, data_path, page_path, number_of_blocks))
	return np.array(labels)

def get_pages_and_data_from_folder(folder):
	"""finds all paths to the pages, and all the paths to the annotated data
	files in the folder. It searches to the folder for book folders."""
	books = os.listdir(folder)
	books = remove_unannotated_books(folder, books)

	pages_data = []

	for book_path in books:
		#pages += glob.glob(os.path.join(folder, book_path, 'raw', '*.png'))
		#data += glob.glob(os.path.join(folder, book_path, 'annotated', '*.py'))
		pages_data += zip(glob.glob(os.path.join(folder, book_path, 'raw',\
			'*.png')),glob.glob(os.path.join(folder, book_path, 'annotated', '*.py')))

	#return pages, data
	return pages_data



if __name__ == '__main__':
	logger = SaveLogger('../models/logc1.py')
	model = logger.load()
	get_bounding_boxes_from_page('../data/aanzienlykeScheepsTogt/raw/500_0001.png',
		model, [20, 10])


