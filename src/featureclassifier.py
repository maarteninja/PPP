""" Learns a classifier on features """

import bookfunctions
import random
import argparse
import numpy as np

from sklearn import svm



def get_features_pages_data(pages_data, number_of_blocks):
	features = []#np.array([])
	for page, data_path in pages_data:
		with open(data_path, 'r+') as f:
			data = bookfunctions.get_data(f)
			new_features = bookfunctions.get_hog_features_page(f, data, page,\
				number_of_blocks)
			#features = np.append(features, new_features)
			features.append(new_features)

	features = np.array(features)
	return features

def main(folder, number_of_blocks):
	pages_data = bookfunctions.get_pages_and_data_from_folder(folder)
	pages_data = pages_data[:300]

	random.shuffle(pages_data)

	features = get_features_pages_data(pages_data, number_of_blocks)
	labels = bookfunctions.get_all_labels(pages_data, number_of_blocks)

	features = np.reshape(features, (labels.shape[0] * labels.shape[1] * labels.shape[2], 8))
	labels = np.reshape(labels, labels.shape[0] * labels.shape[1] * labels.shape[2])

	cut_off = len(features) * 0.8

	train_features = features[:cut_off]
	train_labels = labels[:cut_off]
	validate_features = features[cut_off:]
	validate_labels = labels[cut_off:]
	print 'features shape:', validate_features.shape
	print 'labels shape:', validate_labels.shape
	#print len(train_features), len(train_labels), len(validate_features), \
	#	len(validate_labels)

	

	for c in range(-1, 6):
		classifier = svm.SVC(C=10**c, probability=1, class_weight='auto')
		classifier.fit(train_features, train_labels)
		predicted_labels = classifier.predict(validate_features)
		confusion_matrix, cp, mcp = self.mcp(predicted_labels, validate_labels)
		print "For c = %d, %s, %s, %s" % (c, str(confusion_matrix), str(cp), \
			str(mcp))


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
