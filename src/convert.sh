# !/bin/bash

# require input directory with the jpg files to convert
if [ $# -lt 1 ]; then
    echo 'need input dir as first argument'
    exit
fi


#curdir=$(pwd)

# change dir to input dir
cd $1

# loop over all files in the folder
for f in $(ls)
do

    # only process jpg files!
    if [ $f == *'.jpg' ]; then

        # if the output file already exists, well just stop
        if [ -f 500_$f ]; then
            echo 'converted file already existed: '$f
            exit
        fi


        # convert all images to 500 pixels in width
        convert $f -scale 500 500_$fsd
    fi
done


#cd $curdir
