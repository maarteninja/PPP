#!/bin/bash

mkdir -p ../data/new/

# Usage: download-book number_of_pages
for (( i=1; i<=$1; i++ ))
do
	# Dit werkt iig voor 1 pagina:
	wget "http://boeken.delpher.nl/nl/api/resource?coll=boeken&id=dpo:11394:mpeg21:`printf '%04d' $i`&operation=download&type=image" --output-document="../data/new/`printf '%04d' $i`.jpg"
done
