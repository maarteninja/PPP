# !/bin/bash

# require input directory with the png files to convert
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

    # obtain extension
    f_ext=$(echo $f | awk -F . '{print $NF}')

    # to only process png files!
    if [ $f_ext != 'png' ]; then
        echo 'contains a file that is not png file'
        exit
    fi

    # obtain something before a '_'
    f_500=$(echo $f | awk -F _ '{print $1; }')

    # to check if the output file already exists, we'll just stop
    if [ $f_500 == '500' ]; then
    #if [ -f 500_$f ]; then
        echo 'converted file already existed: '$f
        exit
    fi

    # convert all images to 500 pixels in width
    convert $f -scale 500 500_$f
done


#cd $curdir
