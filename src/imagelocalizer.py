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

	def __init__(self, input_folder, number_of_blocks, overlap=False, use_svm=False):
		""" The input folder is the folder containing all the books. The
		number_of_blocks are used for the hog features. cells_per_block denotes
		how many cells are in a hog (cells_per_block is removed as parameter now)
		"""
		self.input_folder = input_folder
		self.number_of_blocks = number_of_blocks

		self.overlap = overlap
		self.use_svm = use_svm

		self.train_set, self.validation_set = prepare_data(self.input_folder)

		self.train_features = bookfunctions.get_all_features(self.train_set, \
			self.number_of_blocks)

		if overlap:
			self.train_features = bookfunctions.concatenate_features(self.train_features)

		self.train_labels = bookfunctions.get_all_labels(self.train_set, \
			self.number_of_blocks, overlap=overlap)

		if use_svm:
			# TODO
			pass


	def validate(self):
		""" Tweaks C for the svc. self.validation_set is used for validating """

		validation_features = \
			bookfunctions.get_all_features(self.validation_set, \
			self.number_of_blocks)

		validation_labels = bookfunctions.get_all_labels(self.validation_set, \
			self.number_of_blocks, overlap=self.overlap)

		best_mcp = 0

		# Count the number of class labels in order to set the class weights
		class_weights = 1. / np.bincount(self.train_labels.flatten())
		self.crf = WeightedGridCRF(neighborhood=4, class_weight=class_weights)

		for i in range(3, 5):
			c = 10**i
			self.logger = SaveLogger(get_log_path('model', c, self.use_svm, \
				self.overlap), save_every=1)

			print "validating with c = " + str(c)
			temp_classifier = ssvm.OneSlackSSVM(model=self.crf, C=c, n_jobs=-1,
				verbose=4, logger=self.logger, tol=.1)

			# Fit the classifier:
			temp_classifier.fit(self.train_features, self.train_labels)

			# Write the svm parameters!
			with open(get_log_path('param', c, self.use_svm, self.overlap), 'w')\
					as f:
				f.write(str(temp_classifier.get_params()))

			validation_predicted_labels = np.array(\
				temp_classifier.predict(validation_features))

			prfs = precision_recall_fscore_support(validation_labels.flatten(), \
				validation_predicted_labels.flatten())

			mcp = (prfs[0][0] + prfs[0][1]) / 2

			if mcp > best_mcp:
				best_mcp = mcp
				best_c = c
				self.classifier = temp_classifier

		print "Mean class presision for best c: %s" % str(best_mcp)
		return best_c


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


def prepare_data(input_folder):
	pages_data = bookfunctions.get_pages_and_data_from_folder(input_folder)
	pages_data = pages_data[:10]
	print pages_data

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

def get_log_path(name, c, use_svm, overlap):
	return os.path.join('..', 'models', '%s_c_%d_svm_%d_overlap_%d.py' % \
		(name, c, int(use_svm), int(overlap)))


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
	learner.validate()
