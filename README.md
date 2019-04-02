# RainRemover

Download the rainy-image-dataset
Split into train and test sets such that there is not overlap between images in train and test
For example move first 200 images from ground truth to testing's ground truth and first 200 * 14 images from rainy images ti testing's rainy images

The file structure looks like this:
rainy-image-dataset/
--------------------/testing/
----------------------------/ground truth Containing first 200 images
----------------------------/rainy images Contains the corresponding images to the first 200 images total size 200*14
--------------------training/
----------------------------/ground truth
----------------------------/rainy images


Running python conv_auto.py
