import matplotlib.pyplot as plt
import os, glob
import numpy as np
import argparse

import Image

from scipy import misc

from skimage.feature import hog
from skimage import data, color, exposure

from sklearn import svm

class BookLearner:

	def __init__(self, input_folder, number_of_cells=(2,2)):
		""" The input folder is the folder containing all the books. The
		number_of_cells are used for the hog features. This tuple is (62,50) by
		default, which will result in 10x10 hog features per page
		"""
		self.input_folder = input_folder
		self.number_of_cells = number_of_cells
		self.classifier = svm.SVC(probability=1)
		books = os.listdir(self.input_folder)
		# Take 80 percent as train set:
		train_end = int(len(books)*.8)
		# Check if train_end is not the same as len(books)-1: Then we wouldn't
		# have a test set
		if train_end == len(books) - 1:
			train_end = train_end - 1
		self.train_set = books[0:train_end]
		self.test_set = books[train_end:len(books)]
		# self.train_set = ['naauwKeurigeAanteekeningen']
		# self.test_set = ['journaelOfDaghRegister']
		self.all_descriptors = []
		self.all_labels = []


	def train(self):
		""" Trains the svm. self.train_set is used as the training set """
		for book in self.train_set:
			# If the book is annotated, this directory should exist:
			if(os.path.exists(self.input_folder + os.sep + book + os.sep + 'annotated') and 
				os.path.isdir(self.input_folder + os.sep + book + os.sep + 'annotated')):
				# Then read its descriptors and labels
				print "reading book %s" % (book)
				descriptors, labels = self.read_book_data(book)
				self.all_descriptors.extend(descriptors)
				self.all_labels.extend(labels)
			else:
				print 'no data for book %s' % (book)
		# self.all_descriptors = np.array(self.all_descriptors)

		# Fit the classifier:
		self.classifier.fit(self.all_descriptors, self.all_labels)

	def test(self):
		""" Tests the trained svm on the test set in self.test_set. train (or
		some kind of "load" function  in later versions) has to be run first!
		"""
		test_descriptors = []
		test_real_labels = []
		for book in self.test_set:
			if(os.path.exists(self.input_folder + os.sep + book + os.sep + 'annotated') and 
				os.path.isdir(self.input_folder + os.sep + book + os.sep + 'annotated')):
				# Then read its descriptors and labels
				print "reading book %s" % (book)
				descriptors, labels = self.read_book_data(book)
				test_descriptors.extend(descriptors)
				test_real_labels.extend(labels)
			else:
				print 'no data for book %s' % (book)
		# print test_descriptors
		#test_descriptors = np.array(test_descriptors)
		test_predicted_labels = self.classifier.predict(test_descriptors)
		correct = wrong = 0
		for i in range(1, len(test_real_labels)):
			if(test_real_labels[i] == test_predicted_labels[i]):
				correct += 1
			else:
				wrong += 1
		print "Correct: %d, Wrong: %d" % (correct, wrong)


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
		pixels_vertical = int(s[0]/self.number_of_cells[0])
		pixels_horizontal = int(s[1]/self.number_of_cells[1])
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
			with open(annotated_image) as f:
				data = eval(f.read())
				labels.append(data['type']);
				print 'Reading %s' % (annotated_image)
				# Find the image file
				base = os.path.basename(annotated_image)
				name = os.path.splitext(base)[0]
				image_name = self.input_folder + os.sep + book + os.sep + \
					'raw' + os.sep + name + '.png'
				image = misc.imread(image_name);
				descriptors.append(self.calculate_hog(image));

		return descriptors, labels

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("input_folder", metavar='input_folder', type=str,
		help="""The folder containing (annotated) books.""")
	# parser.add_argument("number-of-cells", type=str,
	# 	help="""The number of hogs that are created per page. A vertical and
	# 	horizontal dimension should be given as follows: VxH, where V is the
	# 	vertical number of cells, and H the horizontal number of cells"""
	
	args = vars(parser.parse_args())

	learner = BookLearner(args['input_folder'])
	learner.train()
	learner.test()

