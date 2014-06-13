import matplotlib.pyplot as plt
import os, glob
import numpy as np
from numpy import array
import argparse

import Image

from scipy import misc

from skimage.feature import hog
from skimage import data, color, exposure

from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

class BookLearner:

	def __init__(self, input_folder, number_of_blocks, cells_per_block):
		""" The input folder is the folder containing all the books. The
		number_of_blocks are used for the hog features. This tuple is (62,50) by
		default, which will result in 10x10 hog features per page
		"""
		self.input_folder = input_folder
		self.number_of_blocks = number_of_blocks
		books = os.listdir(self.input_folder)
		books = self.remove_unannotated_books(books)
		# Take 80 percent as train set:
		train_end = int(len(books)*.8)
		validation_end = int(len(books)*.9)
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

	def remove_unannotated_books(self, books):
		""" Removes the books from the array 'books' that do not have a
		subfolder called 'annotated' """
		return_books = []
		for book in books:
			path = self.input_folder + os.sep + book + os.sep + 'annotated'
			if os.path.exists(path) and os.path.isdir(path):
				return_books.append(book)
		return return_books

	def train(self):
		""" Trains the svm. self.train_set is used as the training set """
		for book in self.train_set:
			# read its descriptors and labels
			print "Calculating data for book %s" % (book)
			descriptors, labels = self.read_book_data(book)
			self.all_descriptors.extend(descriptors)
			self.all_labels.extend(labels)

	def validate(self):
		""" Tweaks C for the svc. self.validation_set is used for validating """
		best_mcp = 0
		validation_descriptors = []
		validation_real_labels = []
		for c in range(-1, 6):
			print "validating with c = " + str(10**c)
			temp_classifier = svm.SVC(C=10**c, probability=1, class_weight='auto')
			# Fit the classifier:
			temp_classifier.fit(self.all_descriptors, self.all_labels)
			for book in self.validation_set:
				# read its descriptors and labels
				print "Calculating data for book %s" % (book)
				descriptors, labels = self.read_book_data(book)
				validation_descriptors.extend(descriptors)
				validation_real_labels.extend(labels)
			validation_predicted_labels = temp_classifier.predict(validation_descriptors)
			confusion_matrix, cp, mcp = self.mcp(validation_predicted_labels, \
				validation_real_labels)
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
		for book in self.test_set:
			# read its descriptors and labels
			print "Calculating data for book %s" % (book)
			descriptors, labels = self.read_book_data(book)
			test_descriptors.extend(descriptors)
			test_real_labels.extend(labels)
		test_predicted_labels = self.classifier.predict(test_descriptors)
		print test_predicted_labels
		# For convenient counting, convert back to list
		print_labels = list(test_predicted_labels)
		print """
			Number of text predictions: %d
			Number of image predictions: %d
			Number of bagger predictions: %d
			""" % (print_labels.count('text'), 
				print_labels.count('containing'),
				print_labels.count('bagger'))
		# correct = wrong = 0
		# for i in range(1, len(test_real_labels)):
		# 	if(test_real_labels[i] == test_predicted_labels[i]):
		# 		correct += 1
		# 	else:
		# 		wrong += 1
		# print "Correct: %d, Wrong: %d" % (correct, wrong)
		print confusion_matrix(test_real_labels, test_predicted_labels)
		print precision_recall_fscore_support(test_real_labels, \
			test_predicted_labels)


	def mcp(self, predicted_labels, true_labels):
		""" Creates a confusion matrix and the mean class precision per class
		"""
		confusion_matrix = {}
		# Create confusion matrix
		for label in ['text', 'containing', 'bagger']:
			confusion_matrix[label] = {'correct': 0, 'wrong': 0}
		# fill confusion matrix
		for i in range(1, len(true_labels)):
			if(true_labels[i] == predicted_labels[i]):
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

	def calculate_and_show_hog(self, image):
		image = color.rgb2gray(image)
		pixels_per_cell = self.calculate_pixels_per_cell(image)
		fd, hog_image = hog(image, orientations=8,
			pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1),
			visualise=True)

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

		ax1.axis('off')
		ax1.imshow(image, cmap=plt.cm.gray)
		ax1.set_title('Input image')

		# Rescale histogram for better display
		hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,
			0.02))

		ax2.axis('off')
		ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
		ax2.set_title('Histogram of Oriented Gradients')
		plt.show()
		return fd

	def calculate_hog(self, image):
		pixels_per_cell = self.calculate_pixels_per_cell(image)
		image = color.rgb2gray(image)
		return hog(image, orientations=8, pixels_per_cell=pixels_per_cell,
			cells_per_block=(1, 1))

	def calculate_pixels_per_cell(self, image):
		s = np.shape(image)
		pixels_vertical = int(s[0]/self.number_of_blocks[0])
		pixels_horizontal = int(s[1]/self.number_of_blocks[1])
		return (pixels_horizontal, pixels_vertical)

	def read_book_data(self, book):
		""" Read the hog features from the image files, and the class from the
		python files corresponding to the book pages in the directory 'book'. A
		folder named 'annotated' needs to be present, with 500_<page>.py as
		annotated data for the image raw/500_<page>.png 
		""" 
		annotated_images = glob.glob(self.input_folder + os.sep + book + os.sep + 'annotated' + os.sep + '*.py')
		# This array will hold the HOG feature descriptors for each class
		descriptors= []
		# This array will hold the labels of the images. Each index in this array
		# corresponds to the same index in descriptors. The two of them together
		# form the input for a classifier
		labels = []

		for annotated_image in annotated_images:
			with open(annotated_image, 'r+') as f:
				data = eval(f.read())
				labels.append(data['type']);
				# A two-tuple of blocks and cells will be used for saving and
				# loading this configuration's data
				block_and_cells = (self.number_of_blocks, cells_per_block)
				# Check if the needed hog features are already saved:
				if data.has_key('hog_features') and \
					data['hog_features'].has_key(block_and_cells):
					current_descriptor = data['hog_features'][block_and_cells]
				else:
					# Find the image file
					base = os.path.basename(annotated_image)
					name = os.path.splitext(base)[0]
					image_name = self.input_folder + os.sep + book + os.sep + \
						'raw' + os.sep + name + '.png'
					image = misc.imread(image_name)
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
				descriptors.append(current_descriptor)
		return descriptors, labels

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input_folder", type=str,
		help="""The folder containing (annotated) books.""")
	parser.add_argument('-n', "--number-of-blocks", type=str, default='5x5', required=False,
		help="""The number of hogs that are created per page. A vertical and
		horizontal dimension should be given as follows: VxH, where V is the
		vertical number of cells, and H the horizontal number of cells.
		The default is 5x5""")
	parser.add_argument('-b', "--cells-per-block", type=str, default='2x2', required=False,
		help="""The number of cells each block is built up from. Format is again
		VxH
		The default is 2x2""")
	
	args = vars(parser.parse_args())
	
	number_of_blocks = tuple([int(a) for a in args['number_of_blocks'].split('x')])
	cells_per_block = tuple([int(a) for a in args['cells_per_block'].split('x')])

	learner = BookLearner(args['input_folder'], number_of_blocks, cells_per_block)
	learner.train()
	learner.validate()
	learner.test()

