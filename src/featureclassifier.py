""" Learns a classifier on features """

import bookfunctions
import random
import argparse
import numpy as np

from sklearn import svm

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support



def main(folder, pages_data, number_of_blocks):
	# get all the page info from all books
	#pages_data = bookfunctions.get_pages_and_data_from_folder(folder)
	#pages_data = pages_data[:1000]
	random.shuffle(pages_data)

	# get the features and the labels
	features = bookfunctions.get_all_features(pages_data, number_of_blocks)
	features = bookfunctions.concatenate_features(features)
	labels = bookfunctions.get_all_labels(pages_data, number_of_blocks, \
		overlap=True)

	# reshape for the SCV
	# size of hog features = 4*8
	features = np.reshape(features, (features.shape[0] * features.shape[1] *
		features.shape[2], 32))
	labels = np.reshape(labels, labels.shape[0] * labels.shape[1] * labels.shape[2])

	# split up in train-validate sets
	cut_off = len(features) * 0.8
	train_features = features[:cut_off]
	train_labels = labels[:cut_off]
	validate_features = features[cut_off:]
	validate_labels = labels[cut_off:]
	print 'features shape:', validate_features.shape
	print 'labels shape:', validate_labels.shape

	# try out some values for c
	for c in range(1, 6):
		# I know this runs, but I do not know exactly how well .. 
		# (training took too long)
		print "Learning classifier for C = %d" % c**10
		classifier = svm.LinearSVC(C=10**c, class_weight='auto',
			verbose=10)
		classifier.fit(train_features, train_labels)
		print "Predicting validation set"
		predicted_labels = classifier.predict(validate_features)
		#confusion_matrix, cp, mcp = bookfunctions.mcp(predicted_labels, \
		#	validate_labels)
		cm = confusion_matrix(validate_labels.flatten(),
			predicted_labels.flatten())
		#print "For c = %d, %s, %s, %s" % (10**c, str(confusion_matrix), str(cp), \
		#	str(mcp))
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
		mcp = (prfs[0][0] + prfs[0][1]) / 2
		print "MCP = %f" % mcp
		print "Params: "
		print classifier.get_params(deep=True)

		# TODO: store the best SVC so we can read and use it later!



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

	main(input_folder, number_of_blocks)
