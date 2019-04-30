import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
from skimage.measure import compare_ssim
import re

def getitem(idx,files_clean,files_rain):
        # image_clean = cv2.imread(self.files_clean[idx%14])
        image_rain = cv2.imread(files_rain[idx])
        for image_name in files_clean:
            image_name_ =  os.path.split(image_name)[1]
            name = os.path.split(files_rain[idx])[1]
            grd = name[:name.find("_")]
            if(image_name_==grd+".jpg"):
                image_clean = cv2.imread(image_name)
                break

        # print(idx,self.files_clean[idx%14],self.files_rain[idx])

        # print(image_clean..shape,self.files_clean[idx%14])

        sample = (image_clean,image_rain)
        return sample

def bilateralFiltering_ssim(clean_img,rainy_img):
	bf = cv2.bilateralFilter(rainy_img,30,70,10)
	if(clean_img.shape!=rainy_img.shape):
		
		return -1
	cv2.imshow("Original",clean_img)
	cv2.imshow("Rainy",rainy_img)
	cv2.imshow("Bilateral Filter",bf)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return compare_ssim(clean_img,bf,multichannel=True)

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)



# img=cv2.imread("test1.jpg")
# dst = cv2.bilateralFilter(img,30,70,10)

# # img=img.astype(float)
# # dst=dst.astype(float)

# # diff=np.subtract(img,dst)

# #print(compare_ssim(img,dst))

# cv2.imshow("Bilateral Filter",dst)
# cv2.waitKey(0)
root_folder='rainy-image-dataset/training'

files_clean = glob.glob(root_folder+"/ground truth/*")
files_rain = glob.glob(root_folder+"/rainy image/*")
sort_nicely(files_clean)
#print(len(files_rain))
s=0.0
l=len(files_rain)
for i in range(l):
	(c,r)=getitem(i,files_clean,files_rain)
	#print(c.shape,r.shape)
	x=bilateralFiltering_ssim(c,r)
	if(x!=-1):
		s+=x
	else:
		l-=1

print(l)
s/=l
print("Average SSIM using Bilateral Filtering: ",s)


