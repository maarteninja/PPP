import bookfunctions
import os, glob
import numpy as np


if __name__ == '__main__':
	input_folder = '../testset'
	books = os.listdir(input_folder)

	pages_count = []
	images_count = []

	for book in books:
		if '.md' in book:
			continue
		pages = glob.glob(os.path.join(input_folder, book, 'raw', '*.png'))
		images = glob.glob(os.path.join(input_folder, book, 'annotated', 'pic*.png'))
		print book, len(pages), len(images)
		pages_count.append(len(pages))
		images_count.append(len(images))

	print 'number of pages', np.sum(pages_count)
	print 'pages mean', np.mean(pages_count)
	print 'pages std', np.std(pages_count)
	print 'number of images', np.sum(images_count)
	print 'images mean', np.mean(images_count)
	print 'images std', np.std(images_count)
