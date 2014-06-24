import bookfunctions
from sklearn.externals import joblib
from pystruct.utils import SaveLogger
import argparse
import numpy as np
from pygame.locals import *
from PIL import Image
import os
import pygame
from annotater import Annotater

def process_image_boxes(image_boxes, page_paths, out_folder):
	for n, page_path in enumerate(page_paths):
		boxes = image_boxes.get(n, None)
		if not boxes:
			continue

		image = pygame.image.load(page_path)
		for i, (a, b, c, d) in enumerate(boxes):
			pos, size = Annotater.get_rectangle_pos_size((b, a), (d, c))
			sub_surface = image.subsurface(pos, size)

			out = os.path.join(out_folder, 'pic_%d_box_%d.png' % (n, i))
			pygame.image.save(sub_surface, out)

def get_image_boxes_from_prediction(labels, boxes, output_folder, overlap):
	boxes_shape = np.shape(boxes)
	labels_shape = np.shape(labels)
	boxes = np.reshape(boxes, (boxes_shape[0], labels_shape[1], labels_shape[2], 4))

	image_boxes = {}

	for n in range(labels_shape[0]):

		for i in range(labels_shape[1]):
			for j in range(labels_shape[2]):

				# current index is an image and current index not already extracted
				if labels[n][i][j] == 0 and \
						not(box_in_boxes(image_boxes.get(n, []), (boxes[n][i][j]))):
					# find largest rectangle
					new_box = find_box(n, i, j, boxes, labels)
					if new_box:
						l = image_boxes.get(n, [])
						print 'adding ', new_box, ' to ', l
						l.append(new_box)
						image_boxes[n] = l
	print image_boxes
	return image_boxes

def find_box(n, i_start, j_start, boxes, labels):

	column_success = True
	row_success = True

	i_s = []
	j_s = []
	i_current = i_start
	j_current = j_start

	while column_success or row_success:
		# try to add a column
		if column_success:
			j_s += [j_current]
			j_current += 1

			if j_current >= len(labels[n][0]):
				column_success = False

			if column_success:
				for i in i_s:
					if i >= len(labels[n]) or labels[n][i][j_current] != 0:
						# one is not an image, then stop!
						column_success = False
						break

		# try to add a row
		if row_success:
			i_s += [i_current]
			i_current += 1

			if i_current >= len(labels[n]):
				row_success = False

			if row_success:
				for j in j_s:
					if j >= len(labels[n][i_current]) or labels[n][i_current][j] != 0:
						# one is not an image, then stop
						row_success = False
						break

	j_max = max(j_s)
	i_max = max(i_s)
	if j_max == j_start and i_max == i_start:
		return None
	top_left = boxes[n][i_start][j_start]
	bottom_right = boxes[n][i_max][j_max]

	return top_left[0], top_left[1], bottom_right[2], bottom_right[3]

def box_in_boxes(boxes, box):
	if len(boxes) == 0:
		return False
	assert type(boxes) == list
	#print 'in boxes:', boxes
	#print 'checking for:', box
	y1, x1, y2, x2 = box
	for found_box in boxes:
		if point_in_rectangle(found_box, y1, x1) and \
				point_in_rectangle(found_box, y2, x2):
			#print 'TRUEEEEEEEEEEEEEE point ', y1, x1, y2, x2, ' is in ', found_box
			return True
	#print 'RETURNING FALSSSEEEEEEEE point ', y1, x1, y2, x2, ' is in ', found_box
	return False

def point_in_rectangle(box, y, x):
	y1, x1, y2, x2 = box
	return y1 <= y and y <= y2 and x1 <= x and x <= x2


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("input_folder", type=str,
		help="""The folder containing 1 book.""")
	parser.add_argument('-n', "--number-of-blocks", type=str, default='20x10', required=False,
		help="""The number of hogs that are created per page. A vertical and
		horizontal dimension should be given as follows: VxH, where V is the
		vertical number of cells, and H the horizontal number of cells.
		The default is 20x10""")
	args = vars(parser.parse_args())


	# input parameters
	input_folder = args['input_folder']
	number_of_blocks = tuple([int(a) for a in args['number_of_blocks'].split('x')])
	#svm_path = '../models/svm_params_overlap_1.py' # if None or '', then nu svm is used
	svm_path = False
	ssvm_path = '../models/model_c_10_svm_0_overlap_1.py'
	overlap = True
	output_folder = '../output'

	# read all images from input folder
	pages_data = bookfunctions.get_pages_and_data_from_book(input_folder)
	#pages_data = pages_data[:10]

	features = bookfunctions.get_features_from_pages_data(pages_data, \
		number_of_blocks, overlap, svm_path)

	# put features in ssvm
	logger = SaveLogger(ssvm_path)
	ssvm = logger.load()
	print features.shape
	predicted_labels = ssvm.predict(features)

	### now actually obtain the image from the classified features

	# get (or read from cache) all hog locations
	# depending on concatenate_features, recalculate locations or not
	hog_locations = bookfunctions.get_hog_locations_pages_data(pages_data, \
		number_of_blocks, overlap=overlap)

	hog_boxes = bookfunctions.transform_locations(hog_locations, \
		zip(*pages_data)[0], number_of_blocks, overlap=overlap)

	#hog_boxes = bookfunctions.add_labels_to_boxes(hog_boxes, predicted_labels)

	image_boxes = get_image_boxes_from_prediction(predicted_labels, hog_boxes, \
		output_folder, overlap)

	process_image_boxes(image_boxes, zip(*pages_data)[0], output_folder)


