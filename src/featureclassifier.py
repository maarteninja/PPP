""" Learns a classifier on features """

import bookfunctions
import random
import argparse
import numpy as np
import os

from sklearn import svm

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def main(pages_data, number_of_blocks, overlap=True):
	# get all the page info from all books
	#pages_data = bookfunctions.get_pages_and_data_from_folder(folder)
	#pages_data = pages_data[:1000]
	#random.shuffle(pages_data)

	# get the features and the labels
	features = bookfunctions.get_all_features(pages_data, number_of_blocks)
	if overlap:
		features = bookfunctions.concatenate_features(features)
	labels = bookfunctions.get_all_labels(pages_data, number_of_blocks, \
		overlap=overlap)

	# reshape for the SCV
	#features = np.reshape(features, (features.shape[0] * features.shape[1] * \
	#	features.shape[2], features.shape[3])) # <- changed from 32 to f.shape[3]
	#labels = np.reshape(labels, labels.shape[0] * labels.shape[1] * labels.shape[2])

	# This will contain the ssvm features in image-order
	ssvm_features = np.array([])
	# This will contain the corresponding labels to the ssvm features
	original_labels = np.array([])

	for i in range(4):
		train_features, validate_features = train_validation_split(features, i)
		train_labels, validate_labels = train_validation_split(labels, i)
		original_shape_validate = validate_labels.shape
		# Save the validation labels order for the SSVM to train on
		original_labels = np.append(original_labels, validate_labels)

		# Reshape the features for the SVC
		train_features = np.reshape(train_features, (train_features.shape[0] * train_features.shape[1] * \
			train_features.shape[2], train_features.shape[3])) # <- changed from 32 to f.shape[3]
		validate_features = np.reshape(validate_features, (validate_features.shape[0] * validate_features.shape[1] * \
			validate_features.shape[2], validate_features.shape[3])) # <- changed from 32 to f.shape[3]
		train_labels = np.reshape(train_labels, train_labels.shape[0] * train_labels.shape[1] * train_labels.shape[2])
		validate_labels = np.reshape(validate_labels, validate_labels.shape[0] * validate_labels.shape[1] * validate_labels.shape[2])

		print 'features shape:', validate_features.shape
		print 'labels shape:', validate_labels.shape

		classifiers = []

		# try out some values for c
		for c in range(-1, 3):
			f, classifier = validate(c, train_features, train_labels, validate_features,
				validate_labels)
			classifiers.append((f, classifier))
		best_classifier = sorted(classifiers, key=lambda x: x[0], reverse=True)[0][1]
		print "data for best classifier in iteration %d:" % (i)
		print classifier.get_params(deep=True)
		predictions = best_classifier.decision_function(validate_features)
		# Reshape predictions to the shape of labels (TODO: Check if this
		# works!)
		predictions.shape = original_shape_validate + (1,)
		ssvm_features = np.append(ssvm_features, predictions)
	with open(os.path.join('..', 'models', 'svm_output_overlap_%d.py' % \
			int(overlap)), 'w') as f:
		f.write(str(ssvm_features))
	with open(os.path.join('..', 'models', 'svm_output_overlap_%d_labels.py' % \
			int(overlap)), 'w') as f:
		f.write(str(original_labels))

	# Now evaluate one last time on the entire set:
	for c in range(1, 6):
		f, classifier = validate(c, train_features, train_labels, validate_features,
			validate_labels)
		classifiers.append((f, classifier))
	best_classifier = sorted(classifiers, key=lambda x: x[0], reverse=True)[0][1]
	joblib.dump(best_classifier, os.path.join('..', 'models', 'svm_params_overlap_%d.py' % \
		int(overlap)), 'w')
	return ssvm_features

def validate(c, train_features, train_labels, validate_features, validate_labels):
	# I know this runs, but I do not know exactly how well .. 
	# (training took too long)
	print "Learning classifier for C = %d" % 10**c
	# Dual is false, because the documentation advises us to set it like that
	classifier = svm.LinearSVC(C=10**c, class_weight='auto',
		verbose=10, dual=False)
	classifier.fit(train_features, train_labels)
	print "Predicting validation set"
	predicted_labels = classifier.predict(validate_features)
	cm = confusion_matrix(validate_labels.flatten(),
		predicted_labels.flatten())
	print "For c = %d, %s" % (10**c, str(cm))
	prfs = precision_recall_fscore_support(validate_labels.flatten(), \
		predicted_labels.flatten())
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
	#mcp = (prfs[0][0] + prfs[0][1]) / 2.
	# We want the one with the best F-score for Image 
	f = prfs[2][0]
	print "f = %f" % f
	print "Params: "
	print classifier.get_params(deep=True)
	return f, classifier

def train_validation_split(data, iteration, n=4):
	""" Creates an n-fold split of train and validation set. If iteration > n,
	the train_set is data and validation_set is empty"""
	# take 20% for validation set
	validation_data = []
	train_data = []
	for i, p_d in enumerate(data):
		if i % n == iteration:
			validation_data.append(p_d)
		else:
			train_data.append(p_d)
	return np.array(train_data), np.array(validation_data)

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

	input_folder = args['input_folder']
	number_of_blocks = tuple([int(a) for a in args['number_of_blocks'].split('x')])
	train_pages, validation_pages = bookfunctions.prepare_data(input_folder)
	main(train_pages, number_of_blocks, overlap=False)
