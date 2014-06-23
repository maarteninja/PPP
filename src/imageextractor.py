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
	svm_path = '../models/svm.pickle' # if None or '', then nu svm is used
	overlap = False
	output_folder = '../output'

	# read all images from input folder
	pages_data = bookfunctions.get_pages_and_data_from_folder(input_folder)

	# get (or read from cache) all hog features
	features = bookfunctions.get_all_features(pages_data, number_of_blocks)
	# and depending on overlap, concatenate 'em or not
	if overlap:
		features = bookfunctions.concatenate_features(features)

	# if we use an svm we load it and get the features from its decision_function
	if svm_path:
		svm = joblib.load(svm_path)
		features =- svm.decision_function(features)

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


