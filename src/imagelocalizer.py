import bookfunctions
import argparse
import os, glob
import numpy as np
import random
from numpy import array
import argparse
import copy

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

	def __init__(self, input_folder, number_of_blocks, overlap=True,
		use_svm=False, use_page_classifier=False):
		""" The input folder is the folder containing all the books. The
		number_of_blocks are used for the hog features. cells_per_block denotes
		how many cells are in a hog (cells_per_block is removed as parameter now)
		"""
		self.input_folder = input_folder
		self.number_of_blocks = number_of_blocks
		self.overlap = overlap
		self.use_svm = use_svm
		self.use_page_classifier = use_page_classifier
		if use_svm:
			self.svm_path = os.path.join('..', 'models', 
				'svm_params_overlap_%d.py' % int(overlap))
		else:
			self.svm_path = False

		self.train_set, self.validation_set = bookfunctions.prepare_data(self.input_folder)
		print "train set size %s, validation set size %s" % \
			(str(np.shape(self.train_set)), str(np.shape(self.validation_set)))

		if use_svm:
			self.train_labels = \
				bookfunctions.read_svm_data('../models/svm_output_overlap_%d_labels.py'\
					% int(overlap))
			self.train_features = \
				bookfunctions.read_svm_data('../models/svm_output_overlap_%d.py'\
					% int(overlap))
			print "train set features size %s" % (str(np.shape(self.train_features)))
		else:
			self.train_features = bookfunctions.get_all_features(self.train_set, \
				self.number_of_blocks)

			if overlap:
				self.train_features = bookfunctions.concatenate_features(self.train_features)
			print "train set features size after concatenate %s" % \
				(str(np.shape(self.train_features)))

			self.train_labels = bookfunctions.get_all_labels(self.train_set, \
				self.number_of_blocks, overlap=overlap)
		if use_page_classifier:
			# If on, use a page classifier in order to toss out the text pages
			self.page_classifier = \
				bookfunctions.load_page_classifier(os.path.join('..', 'models',
					'classifier_svm_params.py'))

	def validate(self):
		""" Tweaks C for the svc. self.validation_set is used for validating """
		validation_features = \
			bookfunctions.get_features_from_pages_data(self.validation_set,
			self.number_of_blocks, self.overlap, self.svm_path)

		if self.use_page_classifier:
			# FIXME: number of blocks is fixed to what the page classifier has
			# learned on in my test case, for now.
			page_validation_features = bookfunctions.get_all_features(self.validation_set, \
				(5,5))
			s = page_validation_features.shape
			print "SHAPE", str(s)
			# Reshape all features to 1 feature vector
			page_validation_features.shape = (s[0], s[1] * s[2] * s[3])
			print page_validation_features.shape


		validation_labels = bookfunctions.get_all_labels(self.validation_set, \
			self.number_of_blocks, overlap=self.overlap)

		print """validation set features size after concatenate %s. validation
		labels size: %s""" % (str(np.shape(validation_features)), \
			str(np.shape(validation_labels)))
		best_f = 0

		# Count the number of class labels in order to set the class weights
		class_weights = 1. / np.bincount(self.train_labels.flatten())
		# Normalize class weights in order to have a scalable tolerance
		# parameter
		class_weights *= float(np.shape(self.train_features)[3]) / np.sum(class_weights)
		print "class weights: %s" % str(class_weights)
		self.crf = WeightedGridCRF(neighborhood=4, class_weight=class_weights)

		for i in range(1, 5):
			c = 10**i
			self.logger = SaveLogger(get_log_path('model', c, self.use_svm, \
				self.overlap, self.use_page_classifier), save_every=15)

			print "validating with c = " + str(c)
			temp_classifier = ssvm.OneSlackSSVM(model=self.crf, C=c, n_jobs=-1,
				verbose=2, logger=self.logger, tol=10)

			# Fit the classifier:
			temp_classifier.fit(self.train_features, self.train_labels)

			# Write the ssvm parameters!
			with open(get_log_path('param', c, self.use_svm, self.overlap,
				self.use_page_classifier), 'w') as f:
				f.write(str(temp_classifier.get_params()))

			print "validation features shape: %s" + str(np.shape(validation_features))

			validation_predicted_labels = temp_classifier.predict(validation_features)
			validation_predicted_labels = np.array(validation_predicted_labels)
			
			if self.use_page_classifier:
				# Get the page predictions, which have pretttyy high accuracy
				validation_predicted_pages = self.page_classifier.predict( \
					page_validation_features)
				for i, page in enumerate(validation_predicted_pages):
					print page
					if page != 0:
						# Replace any page that has no images according to the
						# page classifier, with a page that is fully classified
						# as 1.
						validation_predicted_labels[i] = \
							np.ones((validation_predicted_labels.shape[1],
								validation_predicted_labels.shape[2]))

			print "C = %d" % (c)
			prfs = precision_recall_fscore_support(validation_labels.flatten(), \
				validation_predicted_labels.flatten())
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

			f = prfs[2][0]

			if f > best_f:
				best_f = f
				self.classifier = temp_classifier

		print "F-score for best c: %s" % str(best_f)
		return best_f


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

def get_log_path(name, c, use_svm, overlap, page_classifier):
	return os.path.join('..', 'models', '%s_c_%d_svm_%d_overlap_%d_page_%d.py' % \
		(name, c, int(use_svm), int(overlap), int(page_classifier)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input_folder", type=str,
		help="""The folder containing (annotated) books.""")
	parser.add_argument('-n', "--number-of-blocks", type=str, default='20x10', required=False,
		help="""The number of hogs that are created per page. A vertical and
		horizontal dimension should be given as follows: VxH, where V is the
		vertical number of cells, and H the horizontal number of cells.
		The default is 20x10""")

	args = vars(parser.parse_args())

	number_of_blocks = tuple([int(a) for a in args['number_of_blocks'].split('x')])
	#cells_per_block = tuple([int(a) for a in args['cells_per_block'].split('x')])

	learner = ImageLocalizer(args['input_folder'], number_of_blocks,
		overlap=True, use_svm=True, use_page_classifier=True)
	learner.validate()
