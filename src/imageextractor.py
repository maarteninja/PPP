import bookfunctions
from sklearn.externals import joblib
from pystruct.utils import SaveLogger
import argparse

def process_prediction(page_labels, hog_locations, output_folder):
	for label, locations in zip(page_labels, hog_locations):
		print label
		print '-----'
		print locations

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
	svm_path = '../models/svm_params_overlap_1.py' # if None or '', then nu svm is used
	ssvm_path = ''
	overlap = True
	output_folder = '../output'

	# read all images from input folder
	pages_data = bookfunctions.get_pages_and_data_from_book(input_folder)

	features = bookfunctions.get_features_from_pages_data(pages_data, \
		number_of_blocks, overlap, svm_path)

	# put features in ssvm
	logger = SaveLogger(ssvm_path)
	ssvm = logger.load()
	predicted_labels = ssvm.predict(features)

	### now actually obtain the image from the classified features

	# get (or read from cache) all hog locations
	# depending on concatenate_features, recalculate locations or not
	hog_locations = bookfunctions.get_hog_locations_pages_data(pages_data, \
		number_of_blocks)

	process_predictions(predicted_labels, hog_locations, output_folder)


