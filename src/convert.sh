# !/bin/bash

# require input directory with the jpg files to convert
if [ $# -lt 1 ]; then
    echo 'need input dir as first argument'
    exit
fi


#curdir=$(pwd)

# change dir to input dir
cd $1
#cd $curdir
