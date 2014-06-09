import matplotlib.pyplot as plt
import os, glob
import numpy as np

import Image

from scipy import misc

from skimage.feature import hog
from skimage import data, color, exposure

from sklearn import svm

PIXELS_PER_CELL = (60,60)

def calculate_and_show_hog(image):
	image = color.rgb2gray(image)

	fd, hog_image = hog(image, orientations=8, pixels_per_cell=PIXELS_PER_CELL,
						cells_per_block=(1, 1), visualise=True)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

	ax1.axis('off')
	ax1.imshow(image, cmap=plt.cm.gray)
	ax1.set_title('Input image')

	# Rescale histogram for better display
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

	ax2.axis('off')
	ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
	ax2.set_title('Histogram of Oriented Gradients')
	plt.show()
	return fd

def calculate_hog(image):
	image = color.rgb2gray(image)
	return hog(image, orientations=8, pixels_per_cell=PIXELS_PER_CELL,
						cells_per_block=(1, 1))

if __name__ == '__main__':
	folder = '../data/atlas/raw'
	# folder = '../data/atlas/raw/'
	images = glob.glob(folder + os.sep + '*.png')
	# This array will hold the HOG feature descriptors for each class
	descriptors= []
	# This array will hold the labels of the images. Each index in this array
	# corresponds to the same index in descriptors. The two of them together
	# form the input for a classifier
	labels = []
	for image in images[1:10]:
		print 'Reading %s' % (image)
		im = misc.imread(image);
		fd = calculate_hog(im)
		descriptors.append(fd);
		# Let's act as if all images are text, for the sake of testing
		labels.append('text');
	for image in images[11:20]:
		print 'Reading %s' % (image)
		im = misc.imread(image);
		fd = calculate_hog(im)
		descriptors.append(fd);
		# Let's act as if all images are text, for the sake of testing
		labels.append('pic');
	print descriptors
	print labels

	# Start the classifier:
	classifier = svm.SVC(probability=1)
	classifier.fit(descriptors, labels)
	im = misc.imread(images[21]);
	fd = calculate_hog(im)
	print classifier.predict_log_proba(fd)
