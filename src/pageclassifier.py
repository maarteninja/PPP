import matplotlib.pyplot as plt
import os, glob
import numpy as np
import random 
from numpy import array
import argparse

import bookfunctions

import Image

from scipy import misc

from skimage.feature import hog
from skimage import data, color, exposure

from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.externals import joblib

np.set_printoptions(threshold=np.nan)

class PageClassifier:

	def __init__(self, input_folder, number_of_blocks):
		""" The input folder is the folder containing all the books. The
		number_of_blocks are used for the hog features. cells_per_block denotes
		how many cells are in a hog """
		self.input_folder = input_folder
		self.number_of_blocks = number_of_blocks
		self.train_set, self.validation_set = bookfunctions.prepare_data(self.input_folder)
		print "train set size %s, validation set size %s" % \
			(str(np.shape(self.train_set)), str(np.shape(self.validation_set)))

		self.train_features = bookfunctions.get_all_features(self.train_set, \
			self.number_of_blocks)
		s = self.train_features.shape
		# Reshape all features to 1 feature vector
		self.train_features.shape = (s[0], s[1] * s[2] * s[3])
		self.train_labels = bookfunctions.get_all_page_labels(self.train_set, \
			self.number_of_blocks)
		self.validation_features = bookfunctions.get_all_features(self.validation_set, \
			self.number_of_blocks)
		s = self.validation_features.shape
		# Reshape all features to 1 feature vector
		self.validation_features.shape = (s[0], s[1] * s[2] * s[3])
		self.validation_labels = bookfunctions.get_all_page_labels(self.validation_set, \
			self.number_of_blocks)

	def validate(self):
		""" Tweaks C for the svc. self.validation_set is used for validating """
		best_f2 = 0
		for c in range(1, 6):
			print "validating with c = " + str(10**c)
			temp_classifier = svm.SVC(C=10**c, probability=1, class_weight='auto')
			# Fit the classifier:
			temp_classifier.fit(self.train_features, self.train_labels)
			validation_predicted_labels = temp_classifier.predict(self.validation_features)
			prfs = precision_recall_fscore_support(self.validation_labels, \
				validation_predicted_labels, beta=2)
			print """
				Precision:
					Image: %f
					Text: %f
					Bagger: %f
				Recall:
					Image: %f
					Text: %f
					Bagger: %f
				F2-score:
					Image: %f
					Text: %f
					Bagger: %f
				Support:
					Image: %f
					Text: %f
					Bagger: %f
				""" % tuple(np.ndarray.flatten(np.array(prfs)))
			f2 = prfs[2][0]
			if f2 > best_f2:
				best_f2 = f2
				best_c = 10**c
				self.classifier = temp_classifier
		print "F score for best c %d: %f2" % (best_c, best_f2)
		joblib.dump(self.classifier, 
			os.path.join('..', 'models', 'classifier_svm_params.py'))
		return best_c

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
			test_descriptors.extend(descriptors)
			test_real_labels.extend(labels)
		test_predicted_labels = self.classifier.predict(test_descriptors)
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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input_folder", type=str,
		help="""The folder containing (annotated) books.""")
	parser.add_argument('-n', "--number-of-blocks", type=str, default='5x5', required=False,
		help="""The number of hogs that are created per page. A vertical and
		horizontal dimension should be given as follows: VxH, where V is the
		vertical number of cells, and H the horizontal number of cells.
		The default is 5x5""")
	
	args = vars(parser.parse_args())
	
	number_of_blocks = tuple([int(a) for a in args['number_of_blocks'].split('x')])

	learner = PageClassifier(args['input_folder'], number_of_blocks)
	learner.validate()
