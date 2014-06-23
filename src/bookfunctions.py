""" This file contains functions that are used by both the page classifier and
image localizer """

# find . -name *.py -exec sed -i 's/(2, 2))/(1, 1))/g' {} \;


from skimage import color
from sklearn.externals import joblib
from skimage.feature import hog
from scipy import misc
import os, glob
import numpy as np
import pickle
from pystruct.utils import SaveLogger
from numpy import array

np.set_printoptions(threshold=np.nan)

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
	""" calculates the hog features from an image (np array image) given
	the number of bocks"""
	pixels_per_cell = calculate_pixels_per_cell(image, number_of_blocks)
	image = color.rgb2gray(image)
	# The number of orientations should be kept fixed as long as we don't
	# save them to the metafiles
	return hog(image, orientations=8, pixels_per_cell=pixels_per_cell,
		cells_per_block=(1, 1))


def calculate_pixels_per_cell(image, number_of_blocks):
	""" calculates the amount of pixels in a hog feature cell given a number
	of blocks and an image"""
	s = np.shape(image)
	pixels_vertical = int(s[0]/number_of_blocks[0])
	pixels_horizontal = int(s[1]/number_of_blocks[1])
	return (pixels_vertical, pixels_horizontal)

def calculate_hog_locations(image, number_of_blocks):
	""" Calculates the center of all hogs, given the image (used for the
	size) Returns a list of tuples of size np.prod(number_of_blocks)
	"""
	height, width = calculate_pixels_per_cell(image, number_of_blocks)
	y_half = height/2.
	# All Y coordinates are starting from half window size, each of them
	# pixels_per_cell[0] apart
	y_coordinates = range(int(round(y_half)), \
		int(round(np.shape(image)[0])), \
		height)
	# Do the same for X coordinates
	x_half = width/2.
	x_coordinates = range(int(round(x_half)), \
		int(round(np.shape(image)[1])), \
		width)
	# Concatenate both coordinate sets in a list containing all coordinates
	coordinates = []
	for y in y_coordinates:
		for x in x_coordinates:
			# Vertical direction goes first in numpy 
			coordinates.append((y, x))
	return coordinates


def read_image_from_annotation_file(input_folder, annotated_image, book):
	""" Given the path of an annotation .py file, finds the corresponding
	image and returns it as returned by misc.imread

	OBSOLETE: could be done with imp = get_page_path(...), misc.imread(imp) """
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
	#try:
	data = eval(f.read())
	#except Exception, e:
	#	print "WARNING!!! Error in reading data from %s: error: %s" % \
	#		(f.name, e.message)
	#	return
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
	return os.path.join(input_folder, book, 'raw', name + '.png')

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
	# TODO: Uncommenting this can cause problems due to faulty saved hog
	# locations. This should be fixed sometime.
	# if data.has_key('hog_locations') and \
	# 	data['hog_locations'].has_key(number_of_blocks):
	# 	return data['hog_locations'][number_of_blocks]

	print 'calculating hog locations for %s in mode %s' % (f.name, f.mode)

	# otherwise, calculate em, and save em
	image = misc.imread(page_path)

	hog_locations = calculate_hog_locations(image, number_of_blocks)

	# If needed, create the dictionary in 'hog_features'
	if not(data.has_key('hog_locations')):
		data['hog_locations'] = {}
		data['hog_locations'][number_of_blocks] = hog_locations

	# write the new data to the original file:
	f.seek(0)
	f.write(str(data))
	# Remove any remaining text from the previous file contents
	f.truncate()

	return hog_locations

def get_hog_locations_pages_data(pages_data, number_of_blocks):
	"""returns a list of all hog location extracted  from pages_data. pages_data
	is a tuple with a list of all the paths to all the image pages, and a list of
	all the paths to the annotated data files

	calls get_hog_locations_path"""

	hog_locations = []
	for page_path, data_path in pages_data:
		with open(data_path, 'r+') as f:
			data = get_data(f)
			hog_locations.append(get_hog_locations_path(f, data, \
				page_path, number_of_blocks))
	return hog_locations

def get_hog_locations(f, data, annotated_image, input_folder, book, \
		number_of_blocks):
	""" constructs the path from annotated_image, input_folder and book and
	calls get_hog_locations_path """
	page_path = get_page_path(annotated, input_folder, book)
	return get_hog_locations_path(f, data, page_path, number_of_blocks)


def get_labels_path(f, data, data_path, page_path, number_of_blocks, \
		overlap=False):
	""" calculates labels for the features of one page """

	if overlap:
		label = np.ones((number_of_blocks[0]-1, number_of_blocks[1]-1), dtype=np.int8)
	else:
		label = np.ones((number_of_blocks[0], number_of_blocks[1]), dtype=np.int8)
	# if the type is text or bagger, all ones (as above) is okay, and we can stop

	if data['type'] != 'containing':
		return label

	# otherwise we need the hog locations to calculate which features are a 0 (image)
	hog_locations = get_hog_locations_path(f, data, page_path, number_of_blocks)
	if overlap:
		hog_locations = concatenate_hog_locations(hog_locations, \
			number_of_blocks)

	# so we loop over the locations of the feature
	for j, coordinate in enumerate(hog_locations):
		# and check if they are in an annotated rectangle (bounding box)
		for ((x, y), (w, h)) in data['rectangles']:
			# Bounding boxes are x, y, w, h coordinates are y, x.
			# Let's go:
			if coordinate[0] > y and \
					coordinate[0] < y + h and \
					coordinate[1] > x and \
					coordinate[1] < x + w:
				# Set the label to 'image' in stead of 'text'
				label.itemset(j, 0)
	return label

def concatenate_hog_locations(locations, number_of_blocks):
	locations = np.array(locations).reshape(number_of_blocks + (2,))
	new_locations = np.zeros((locations.shape[0]-1, locations.shape[1]-1,
		locations.shape[2]))
	for i in range(locations.shape[0]-1):
		for j in range(locations.shape[1]-1):
			new_locations[i][j] = np.array([np.mean([locations[i][j][0], \
				locations[i+1][j][0]]), \
				np.mean([locations[i][j][1], \
				locations[i][j+1][1]])])
	new_locations.shape = ((number_of_blocks[0]-1) * (number_of_blocks[1]-1), 2)
	return new_locations

def get_labels(f, data, annotated_image, input_folder, book, number_of_blocks):
	""" constructs the path from annotated_image, input_folder and book and
	calls get_labels_path"""
	page_path = get_page_path(annotated_image, input_folder, book)
	data_path = get_data_path(annotated_image, input_folder, book)
	return get_labels_path(f, data, data_path, page_path, number_of_blocks)

def get_all_labels(pages_data, number_of_blocks, overlap=False):
	""" returns all labels for the pages stored in pages_data. pages_data is
	a tuple with a list of all the paths to all the image pages, and a list of
	all the paths to the annotated data files"""
	labels = [] #np.array([])
	for page_path, data_path in pages_data:
		with open(data_path, 'r+') as f:
			data = get_data(f)
			#labels = np.append(labels,\
			#	get_labels_path(f, data, data_path, page_path, number_of_blocks))
			labels.append(get_labels_path(f, data, data_path, page_path,
				number_of_blocks, overlap=overlap))
	return np.array(labels)

def get_all_features(pages_data, number_of_blocks):
	""" returns all features for the pages stored in page_data. pages_date is
	a tuple with a list of all the paths to all the images pages, and a list of
	all the paths to the annotated data files"""
	features = [] #np.array([])
	for page, data_path in pages_data:
		with open(data_path, 'r+') as f:
			data = get_data(f)
			new_features = get_hog_features_page(f, data, page, number_of_blocks)
			#features = np.append(features, new_features)
			features.append(new_features)

	features = np.array(features)
	return features

def get_pages_and_data_from_book(book_path):
	raw = glob.glob(os.path.join(book_path, 'raw', '*.png'))
	annotated = glob.glob(os.path.join(book_path, 'annotated', '*.py'))
	return zip(sorted(raw),sorted(annotated))

def get_pages_and_data_from_folder(folder):
	"""finds all paths to the pages, and all the paths to the annotated data
	files in the folder. It searches to the folder for book folders."""
	books = os.listdir(folder)
	books = remove_unannotated_books(folder, books)

	pages_data = []

	for book_path in books:
		#pages += glob.glob(os.path.join(folder, book_path, 'raw', '*.png'))
		#data += glob.glob(os.path.join(folder, book_path, 'annotated', '*.py'))
		raw = glob.glob(os.path.join(folder, book_path, 'raw', '*.png'))
		annotated = glob.glob(os.path.join(folder, book_path, 'annotated', '*.py'))
		pages_data += zip(sorted(raw),sorted(annotated))
	# print str(pages_data)
	#return pages, data
	return pages_data

def concatenate_features(features):
	new_features = np.zeros((features.shape[0], features.shape[1]-1,
		features.shape[2]-1, features.shape[3] * 4))
	for i in range(features.shape[0]-1):
		for j in range(features.shape[1]-1):
			for k in range(features.shape[2]-1):
				new_feature = np.append(features[i][j][k], \
					[features[i][j+1][k], \
					features[i][j][k+1], \
					features[i][j+1][k+1]])
				new_features[i][j][k] = new_feature
	return new_features

# def even_labels(labels, features)
# 	""" Removes features and labels of the bigger set, until the number of
# 	features of label n is as big as the number of features for label /n """
# 	# Count the number of features. The n'th index of number_of_features is the
# 	# number of times n occurs in labels
# 	number_of_features = np.bincount(labels.flatten())
# 	
# 	

def prepare_data(input_folder):
	pages_data = get_pages_and_data_from_folder(input_folder)
	pages_data = pages_data
	#print pages_data

	# TODO throwaway x % of the pages containing only text

	train_pages_data = []
	validate_pages_data = []

	# take 20% for validate set
	for i, p_d in enumerate(pages_data):
		if i % 5 == 0:
			validate_pages_data.append(p_d)
		else:
			train_pages_data.append(p_d)

	return train_pages_data, validate_pages_data

def mcp(predicted_labels, true_labels):
	""" Creates a confusion matrix and the mean class precision per class
	"""
	confusion_matrix = {}
	# Create confusion matrix
	for label in ['text', 'containing']:
		confusion_matrix[label] = {'correct': 0, 'wrong': 0}

	# fill confusion matrix
	for i in range(1, len(true_labels)):
		if true_labels[i] == predicted_labels[i]:
			confusion_matrix[true_labels[i]]['correct'] += 1
		else:
			confusion_matrix[true_labels[i]]['wrong'] += 1

	# Create dictionary that will contain the mcp per class label
	cp = {}
	for label in confusion_matrix.keys():
		correct = confusion_matrix[label]['correct']
		wrong = confusion_matrix[label]['wrong']
		cp[label] = float(correct)/(float(correct+wrong))
	mcp = sum(cp.values())/float(len(confusion_matrix.keys()))
	return confusion_matrix, cp, mcp

def read_svm_data(path):
	with open(path, 'r') as f:
		return pickle.load(f)
	# 	n_folded_features = eval(f.read())
	# 	features = np.array([])
	# 	for n_fold in n_folded_features:
	# 		features = np.append(features, n_fold)
	# 	print features.shape, 'fdjakfajlf'
	# 	return features
	# print 'Something went terribly wrong in read svm data'
	# exit()

def get_features_from_pages_data(pages_data, number_of_blocks, overlap, svm_path):
	""" calculates (or reads from cache) all the hog features with the selected
	parameters. The svm_path should be None if no svm should be used as
	preliminary step for the ssvm is used. Otherwise it is the path to the
	joblib stored file.
	"""
	# get (or read from cache) all hog features
	features = get_all_features(pages_data, number_of_blocks)
	# and depending on overlap, concatenate 'em or not
	if overlap:
		features = concatenate_features(features)

	# if we use an svm we load it and get the features from its decision_function
	if svm_path:
		original_shape = features.shape
		original_shape = original_shape[0:3] + (1, )
		features = np.reshape(features, (features.shape[0] * features.shape[1] * \
			features.shape[2], features.shape[3]))
		svm = joblib.load(svm_path)
		features = np.array(svm.decision_function(features))
		features.shape = original_shape
	return features
