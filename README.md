PPP
===  

Plucking Pictures from Publications

Segmentate images/figures from text. The source: old books from the Dutch golden age era.

===  

## Known dependencies:  
* scikit-image
* six
* cython
* pystruct
* cvxopt

One could install those with pip, thats what we did! We probably missed a lot of
packages, just install them if you find out modules are missing.

## Data  
The data requires special a special folder hierarchy:  
* BOOK-FOLDER
    - raw (contains the raw images of a book)
    - annotated (contains the extracted bounding box images and python files
        with meta on each file)
If the scripts used in the source folder are used, keeping to this hierarchy
should not be a problem.  

## Source  
* conver.sh, shell script to export and prepare a pdf file to a data folder. Use
  this after downloading a book from google books  
* annotator.py, program to annotate exported books. This is used in the second
  step. The help message is actually really good so one can just read that.  
* pageclassifier.py, classify features using an SVM. Proof of concept and can be
  incorperated into the annotator  
* featureclassifier.py, classify features using a linear SVM  
* imageextractor.py, extracts images from classified features, either using a
  maximum bounding box or maximum rectangle finding method.  
* bookfunctions.py, library with all sorts of usefull functions used throughout
  all files.  
* statistics.py, quickly obtain some statistics on some data sets  
* weightedgridcrf.py, contains a class that enables class weights for the CRFs  

old (ignore):  
* download-book.sh, script to download books from delpher.  
* convert-old.sh, old version that converts books differently  
