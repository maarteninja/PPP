#!/bin/bash

# require input pdf file, book_name (as output dir)
if [ $# -lt 2 ]; then
	echo 'Usage: ./convert.sh input_pdf book_name. Writes output to book_name/raw'
	exit 1
fi

# Get the number of pages from the input pdf
numPages=`pdftk $1 dump_data | grep NumberOfPages | sed 's/[^0-9]*//'`
if [ "$numPages" = '' ]; then
	echo 'Error reading pdf'
	exit 2
fi

# obtain extension
f_ext=$(echo $1 | awk -F . '{print $NF}')

# to only process pdf files!
if [ "$f_ext" != 'pdf' ]; then
	echo 'input is not a pdf file'
	exit 3
fi

# Write to raw data directory with the book name as root
directory="../data/$2/raw"

# Create the output directory, if needed
if [ ! -d "$directory" ]; then
	echo "Creating directory $directory"
	mkdir -p "$directory" || { echo 'ERROR'; exit; }
fi

# loop over all pages in the pdf. Skip the first 2 pages, since they contain
# Google information
for (( i=2; i<$numPages; i++ ))
do
	f="$directory/500_`printf '%04d' $i`.png"
	echo "Exporting file $f"

	# to check if the output file already exists, we'll just stop
	if [ -f $f ]; then
		echo 'converted file already existed: '$f
		exit
	fi

	# convert all images to 500 pixels in width
	convert "$1[$i]" -scale 500 $f
done
