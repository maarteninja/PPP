import bookfunctions
from sklearn.externals import joblib
from pystruct.utils import SaveLogger

def process_prediction(page_labels, hog_locations, output_folder):
	for label, locations in zip(page_labels, hog_locations):
		print label
		print '-----'
		print locations

if __name__ == '__main__':

	# input parameters
	input_folder = '../test_data2/lesSixVoyagesDeJeanBaptisteTaverni'
	number_of_blocks = [20, 10]
	svm_path = '../models/svm_params_overlap_1.py' # if None or '', then nu svm is used
	overlap = True
	output_folder = '../output'

	# read all images from input folder
	pages_data = bookfunctions.get_pages_and_data_from_folder(input_folder)

	features = bookfunctions.get_features_from_pages_data(pages_data, \
		number_of_blocks, overlap, svm_path)

	# put features in ssvm
	logger = SaveLogger(get_log_path(ssvm_path))
	ssvm = logger.load()
	predicted_labels = ssvm.predict(features)

	### now actually obtain the image from the classified features

	# get (or read from cache) all hog locations
	# depending on concatenate_features, recalculate locations or not
	hog_locations = bookfunctions.get_hog_locations_pages_data(pages_data, \
		number_of_blocks)

	process_predictions(predicted_labels, hog_locations, output_folder)


