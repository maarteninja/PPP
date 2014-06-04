import matplotlib.pyplot as plt

from scipy import misc

from skimage.feature import hog
from skimage import data, color, exposure


def calculate_hog(image):
	image = color.rgb2gray(image)

	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(32, 32),
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

if __name__ == '__main__':
	folder = '../data/atlas/annotated'
	image = misc.imread(folder . '/0057.png');
	calculate_hog(image)
