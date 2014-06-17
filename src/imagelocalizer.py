import bookfunctions
import argparse
import os, glob
import numpy as np
import random 
from numpy import array
import argparse

import bookfunctions

from scipy import misc

from skimage.feature import hog
from skimage import data, color, exposure

from pystruct.models import GridCRF
import pystruct.learners as ssvm


np.set_printoptions(threshold=np.nan)

class ImageLocalizer:
	""" Localizes images in books, given some training data """

	def __init__(self, input_folder, number_of_blocks, cells_per_block):
		""" The input folder is the folder containing all the books. The
		number_of_blocks are used for the hog features. cells_per_block denotes
		how many cells are in a hog
		"""
		self.input_folder = input_folder
		self.number_of_blocks = number_of_blocks
		books = os.listdir(self.input_folder)
		books = bookfunctions.remove_unannotated_books(input_folder, books)
		# Randomize result!
		random.shuffle(books)
		# Take 80 percent as train set:
		train_end = int(len(books)*.7)
		validation_end = int(len(books)*.8)
		# Check if train_end is not the same as len(books)-1: Then we wouldn't
		# have a test set
		if train_end == len(books) - 1:
			train_end = train_end - 2
		if validation_end == len(books) - 1:
			train_end = train_end - 1
		self.train_set = books[0:train_end]
		# For now, validate with 1 book. Yolo.
		self.validation_set = books[train_end:validation_end]
		self.test_set = books[validation_end:len(books)]
		#self.train_set = ['naauwKeurigeAanteekeningen']
		#self.test_set = ['journaelOfDaghRegister']
		self.all_descriptors = []
		self.all_labels = []


	def train(self):
		""" Trains the svm. self.train_set is used as the training set """
		for book in self.train_set:
			# read its descriptors and labels
			print "Calculating data for book %s" % (book)
			descriptors, labels = self.read_book_data(book)
			#print 'descriptors',  descriptors
			self.all_descriptors.extend(descriptors)
			self.all_labels.extend(labels)
		self.crf = GridCRF(neighborhood=4)
		self.clf = ssvm.OneSlackSSVM(model=self.crf, C=100, n_jobs=-1, inference_cache=100,
								tol=.1)
		self.clf.fit(self.all_descriptors, self.all_labels)
		#predicted_labels = np.array(clf.predict(self.all_descriptors))
		#print 'overal accuracy', clf.score(self.all_descriptors, self.all_labels)

	def validate(self):
		pass

	def test(self):
		""" Tests the trained svm on the test set in self.test_set. train (or
		some kind of "load" function  in later versions) has to be run first!
		"""
		test_descriptors = []
		test_real_labels = []
		for book in self.test_set:

			# read its descriptors and labels
			print "Calculating data for book %s" % (book)
			descriptors, labels = self.read_book_data(book)
			#print 'descriptors',  descriptors
			test_descriptors.extend(descriptors)
			test_real_labels.extend(labels)

		test_predicted_labels = np.array(self.clf.predict(test_descriptors))
		#print 'test accuracy', self.clf.score(test_descriptors, test_real_labels)

		#test_predicted_labels = self.classifier.predict(test_descriptors)
		print confusion_matrix(test_real_labels, test_predicted_labels)
		prfs = precision_recall_fscore_support(test_real_labels, \
			test_predicted_labels)
		print """
			Precision:
				Image: %f
				Text: %f
			Recall:
				Image: %f
				Text: %f
			Fscore:
				Image: %f
				Text: %f
			Support:
				Image: %f
				Text: %f
			""" % tuple(np.ndarray.flatten(np.array(prfs)))


	def read_book_data(self, book):
		""" Read the hog features from the image files, and the class from the
		python files corresponding to the book pages in the directory 'book'. A
		folder named 'annotated' needs to be present, with 500_<page>.py as
		annotated data for the image raw/500_<page>.png 
		""" 
		annotated_images = glob.glob(self.input_folder + os.sep + book + os.sep + 'annotated' + os.sep + '*.py')
		#annotated_images = glob.glob(os.path.join(self.input_folder, book,
		#	'annotated', '*.py'))

		# This array will hold the HOG feature descriptors for each class
		descriptors = []
		# This array will hold the labels of the images. Each index in this array
		# corresponds to the same index in descriptors. The two of them together
		# form the input for a classifier
		labels = []
		for annotated_image in annotated_images:
			with open(annotated_image, 'r+') as f:

				data = eval(f.read())

				# A two-tuple of blocks and cells will be used for saving and
				# loading this configuration's data
				block_and_cells = (self.number_of_blocks, cells_per_block)
				# Check if the needed hog features are already saved:
				if data.has_key('hog_features') and \
					data['hog_features'].has_key(block_and_cells):
					current_descriptor = data['hog_features'][block_and_cells]
				else:
					# Get the image nparray
					image = \
						self.read_image_from_annotation_file(annotated_image, \
						book)
					current_descriptor = self.calculate_hog(image)
					# If needed, create the dictionary in 'hog_features'
					if not(data.has_key('hog_features')) or \
						type(data['hog_features']) != dict:
						data['hog_features'] = {}
					data['hog_features'][block_and_cells] = current_descriptor
					# write the new data to the original file:
					f.seek(0)
					f.write(str(data))
					# Remove any remaining text from the previous file contents
					f.truncate()
				# First, label everything as text.
				# 8 = number of orientations
				current_label = np.ones((self.number_of_blocks[0], \
					self.number_of_blocks[1]), dtype=np.int8)
				# Reshape the descriptors to the pystruct desired shape
				# We now have the 0'th index horizontal and the 1'th index
				# vertical
				current_descriptor.shape = (self.number_of_blocks[1], \
					self.number_of_blocks[0], 8)
				# Transpose the first two axes, in order to get from x-y
				# coordinates to y-x coordinates
				current_descriptor = current_descriptor.transpose((1, 0, 2))
				# If there are images in this page, get their location:
				if data['type'] == 'containing':
					if data.has_key('hog_locations') and \
						data['hog_locations'].has_key(self.number_of_blocks):
						current_hog_locations = \
							data['hog_locations'][self.number_of_blocks]
					else:
						image = \
							self.read_image_from_annotation_file(annotated_image,
							book)
						current_hog_locations = \
							self.calculate_hog_locations(image)
						# If needed, create the dictionary in 'hog_features'
						if not(data.has_key('hog_locations')) or \
							type(data['hog_locations']) != dict:
							data['hog_locations'] = {}
						data['hog_locations'][self.number_of_blocks] = \
							current_hog_locations
						# write the new data to the original file:
						f.seek(0)
						f.write(str(data))
						# Remove any remaining text from the previous file contents
						f.truncate()
					for i, coordinate in enumerate(current_hog_locations):
						for pos, size in data['rectangles']:
							# Bounding boxes are x, y, w, h coordinates are y, x.
							# Let's go:
							x,y = pos
							w, h = size
							if (coordinate[0] > y and \
									coordinate[0] < y + h and \
									coordinate[1] > x and \
									coordinate[1] < x + w):
								# Set the label to 'image' in stead of 'text'
								current_label.itemset(i, -1)
				# Loop through all hog features and save them 
				descriptors.append(current_descriptor)
				labels.append(current_label)
		return descriptors, labels

	def calculate_hog_locations(self, image):
		""" Calculates the center of all hogs, given the image (used for the
		size) Returns a list of tuples of size np.prod(self.number_of_blocks) """
		pixels_per_cell = self.calculate_pixels_per_cell(image)
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

	def read_image_from_annotation_file(self, annotated_image, book):
		""" Given the path of an annotation .py file, finds the corresponding
		image and returns it as returned by misc.imread """
		base = os.path.basename(annotated_image)
		name = os.path.splitext(base)[0]
		image_name = self.input_folder + os.sep + book + os.sep + \
			'raw' + os.sep + name + '.png'
		return misc.imread(image_name)

	def calculate_hog(self, image):
		pixels_per_cell = self.calculate_pixels_per_cell(image)
		image = color.rgb2gray(image)
		# The number of orientations should be kept fixed as long as we don't
		# save them to the metafiles
		return hog(image, orientations=8, pixels_per_cell=pixels_per_cell,
			cells_per_block=(1, 1))

		# Replace 'bagger' tags to 'text':
		for i in range(len(labels)):
			if labels[i] == 'bagger':
				labels[i] = 'text'
		return descriptors, labels

	def calculate_pixels_per_cell(self, image):
		s = np.shape(image)
		pixels_vertical = int(s[0]/self.number_of_blocks[0])
		pixels_horizontal = int(s[1]/self.number_of_blocks[1])
		return (pixels_horizontal, pixels_vertical)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input_folder", type=str,
		help="""The folder containing (annotated) books.""")
	parser.add_argument('-n', "--number-of-blocks", type=str, default='20x10', required=False,
		help="""The number of hogs that are created per page. A vertical and
		horizontal dimension should be given as follows: VxH, where V is the
		vertical number of cells, and H the horizontal number of cells.
		The default is 20x10""")
	parser.add_argument('-b', "--cells-per-block", type=str, default='2x2', required=False,
		help="""The number of cells each block is built up from. Format is again
		VxH
		The default is 2x2""")

	args = vars(parser.parse_args())

	number_of_blocks = tuple([int(a) for a in args['number_of_blocks'].split('x')])
	cells_per_block = tuple([int(a) for a in args['cells_per_block'].split('x')])

	learner = ImageLocalizer(args['input_folder'], number_of_blocks, cells_per_block)
	learner.train()
	learner.validate()
	learner.test()
