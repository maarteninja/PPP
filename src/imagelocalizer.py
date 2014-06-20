import bookfunctions
import argparse
import os, glob
import numpy as np
import random
from numpy import array
import argparse

import bookfunctions
from weightedgridcrf import WeightedGridCRF

from scipy import misc

from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.metrics import confusion_matrix

from pystruct.models import GridCRF
from pystruct.utils import SaveLogger
import pystruct.learners as ssvm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


np.set_printoptions(threshold=np.nan)

class ImageLocalizer:
	""" Localizes images in books, given some training data """

	def __init__(self, input_folder, number_of_blocks):
		""" The input folder is the folder containing all the books. The
		number_of_blocks are used for the hog features. cells_per_block denotes
		how many cells are in a hog (cells_per_block is removed as parameter now)
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
		self.logger = SaveLogger('../logs/log%d.py', save_every=1)

	def train(self):
		""" Trains the svm. self.train_set is used as the training set """
		for book in self.train_set:
			# read its descriptors and labels
			print "Calculating data for book %s" % (book)
			descriptors, labels, ignore = self.read_book_data(book)
			#print 'descriptors',  descriptors
			self.all_descriptors.extend(descriptors)
			self.all_labels.extend(labels)
		self.all_descriptors = np.array(self.all_descriptors)
		self.all_labels = np.array(self.all_labels)

	def validate(self):
		""" Tweaks C for the svc. self.validation_set is used for validating """
		best_mcp = 0
		validation_descriptors = []
		validation_real_labels = []
		# Count the number of class labels in order to set the class weights
		class_weights = 1. / np.bincount(self.all_labels.flatten())
		# * n_states / sum, like in pystruct's image segmentation example
		class_weights *= 8. / np.sum(class_weights)
		# print(class_weights)
		self.crf = WeightedGridCRF(neighborhood=4, class_weight=class_weights)
		for book in self.validation_set:
			# read its descriptors and labels
			print "Calculating data for book %s" % (book)
			descriptors, labels, paths = self.read_book_data(book)
			validation_descriptors.extend(descriptors)
			validation_real_labels.extend(labels)

		for i in range(2, 3):
			c = 10**i
			self.logger = SaveLogger('logc%d.py' % c, save_every=1)
			print "validating with c = " + str(c)
			temp_classifier = ssvm.OneSlackSSVM(model=self.crf, C=c, n_jobs=-1,
				verbose=4, logger=self.logger, tol=.1)
			# temp_classifier.fit(self.all_descriptors, self.all_labels)
			# Fit the classifier:
			temp_classifier.fit(self.all_descriptors, self.all_labels)
			# Write the svm parameters!
			with open("paramsc%d.py" % c, 'w') as f:
				f.write(str(temp_classifier.get_params()))
			validation_predicted_labels = temp_classifier.predict(validation_descriptors)
			confusion_matrix, cp, mcp = self.mcp(validation_predicted_labels, \
				validation_real_labels)
			if mcp > best_mcp:
				best_mcp = mcp
				best_c = c
				self.classifier = temp_classifier
		print "Mean class presision for best c: %s" % str(best_mcp)
		return best_c

	def mcp(self, predicted_labels, true_labels):
		""" Creates a confusion matrix and the mean class precision per class
		"""
		confusion_matrix = {}
		# Create confusion matrix
		for label in ['text', 'containing']:
			confusion_matrix[label] = {'correct': 0, 'wrong': 0}
		# Specify an order, in order to be sure we loop through both arrays in
		# the same order (otherwise the memory order is used)
		true_labels = np.array(true_labels)
		predicted_labels = np.array(predicted_labels)
		for labels in zip(np.nditer(true_labels, order='C'), \
			 np.nditer(predicted_labels, order='C')):
			if(labels[0] == labels[1]):
				label = 'text' if labels[0] == 1 else 'containing'
				confusion_matrix[label]['correct'] += 1
			else:
				confusion_matrix[label]['wrong'] += 1
		# Create dictionary that will contain the mcp per class label
		print confusion_matrix
		cp = {}
		for label in confusion_matrix.keys():
			correct = confusion_matrix[label]['correct']
			wrong = confusion_matrix[label]['wrong']
			# Some times, this label is not predicted at all
			if correct+wrong != 0:
				cp[label] = float(correct)/(float(correct+wrong))
			else:
				cp[label] = 0
		mcp = sum(cp.values())/float(len(confusion_matrix.keys()))
		return confusion_matrix, cp, mcp

	def test(self):
		""" Tests the trained svm on the test set in self.test_set. train (or
		some kind of "load" function  in later versions) has to be run first!
		"""
		test_descriptors = []
		test_real_labels = []
		complete_page_paths = []
		for book in self.test_set:

			# read its descriptors and labels
			print "Calculating data for book %s" % (book)
			descriptors, labels, page_paths = self.read_book_data(book)
			#print 'descriptors',  descriptors
			test_descriptors.extend(descriptors)
			test_real_labels.extend(labels)
			complete_page_paths.extend(page_paths)
		test_real_labels = np.array(test_real_labels)
		test_predicted_labels = np.array(self.classifier.predict(test_descriptors))

		print confusion_matrix(test_real_labels.flatten(),
			test_predicted_labels.flatten())
		prfs = precision_recall_fscore_support(test_real_labels.flatten(), \
			test_predicted_labels.flatten())
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
		# bookfunctions.get_bounding_boxes_from_labels(test_predicted_labels, \
		# 	page_paths)

	def read_book_data(self, book):
		""" Read the hog features from the image files, and the class from the
		python files corresponding to the book pages in the directory 'book'. A
		folder named 'annotated' needs to be present, with 500_<page>.py as
		annotated data for the image raw/500_<page>.png 
		""" 

		# get all the paths to the annotated files
		annotated_images = glob.glob(os.path.join(self.input_folder, book,
			'annotated', '*.py'))
		# get all the paths to the images
		complete_page_paths = glob.glob(os.path.join(self.input_folder, book,
			'raw', '*.png'))

		# This array will hold the HOG feature descriptors for each class
		descriptors = []

		# This array will hold the labels of the images. Each index in this array
		# corresponds to the same index in descriptors. The two of them together
		# form the input for a classifier
		labels = []

		# this array will hold all the paths to the pages
		page_paths = []

		for i, annotated_image in enumerate(annotated_images):
			with open(annotated_image, 'r+') as f:
				data = bookfunctions.get_data(f)

				# get descriptors
				current_descriptors = bookfunctions.get_hog_features(f, data,
					annotated_image, self.input_folder, book,
					self.number_of_blocks)

				# get labels
				current_labels = bookfunctions.get_labels(f, data, annotated_image,
					self.input_folder, book, self.number_of_blocks)

				# store descriptors, labels and the page path
				descriptors.append(current_descriptors)
				labels.append(current_labels)
				page_paths.append(complete_page_paths[i])

		return descriptors, labels, page_paths

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input_folder", type=str,
		help="""The folder containing (annotated) books.""")
	parser.add_argument('-n', "--number-of-blocks", type=str, default='20x10', required=False,
		help="""The number of hogs that are created per page. A vertical and
		horizontal dimension should be given as follows: VxH, where V is the
		vertical number of cells, and H the horizontal number of cells.
		The default is 20x10""")

	# TODO: fix cells per block as argument, EVERYWHERE
	#parser.add_argument('-b', "--cells-per-block", type=str, default='2x2', required=False,
	#	help="""The number of cells each block is built up from. Format is again
	#	VxH
	#	The default is 2x2""")

	args = vars(parser.parse_args())

	number_of_blocks = tuple([int(a) for a in args['number_of_blocks'].split('x')])
	#cells_per_block = tuple([int(a) for a in args['cells_per_block'].split('x')])

	learner = ImageLocalizer(args['input_folder'], number_of_blocks)
	learner.train()
	learner.validate()
	learner.test()
