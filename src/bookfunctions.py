""" This file contains functions that are used by both the page classifier and
image localizer """

import os

def remove_unannotated_books(input_folder, books):
	""" Removes the books from the array 'books' that do not have a
	subfolder called 'annotated' """
	return_books = []
	for book in books:
		path = input_folder + os.sep + book + os.sep + 'annotated'
		if os.path.exists(path) and os.path.isdir(path):
			return_books.append(book)
	return return_books
