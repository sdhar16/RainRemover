import os
import glob
from torch.utils.data import  Dataset
import cv2
import re
import numpy as np 
from torchvision import transforms

import numpy as np
import cv2
from matplotlib import pyplot as plt

def fourier(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f) ## shift for centering 0.0 (x,y)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    ## removing low frequency contents by applying a 60x60 rectangle window (for masking)
    rows = np.size(img, 0) 
    cols = np.size(img, 1)
    crow, ccol = int(rows/2), int(cols/2)

    radius = 10
    fshift[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0
    f_ishift= np.fft.ifftshift(fshift)

    img_back = np.fft.ifft2(f_ishift) ## shift for centering 0.0 (x,y)
    img_back = np.abs(img_back)

    return img_back
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

class RainyDataset(Dataset):
    def __init__(self,root_folder,transform=None):
        self.files_clean = glob.glob(root_folder+"/ground truth/*")
        self.files_rain = glob.glob(root_folder+"/rainy image/*")
        
        assert (len(self.files_clean)*14==len(self.files_rain))
        
        self.transform = transform
        sort_nicely(self.files_clean)

    def __len__(self):
        # return 1000
        return len(self.files_rain)

    def __getitem__(self,idx):
        # image_clean = cv2.imread(self.files_clean[idx%14])
        image_rain = cv2.imread(self.files_rain[idx])
        for image_name in self.files_clean:
            image_name_ =  os.path.split(image_name)[1]
            name = os.path.split(self.files_rain[idx])[1]
            grd = name[:name.find("_")]
            if(image_name_==grd+".jpg"):
                image_clean = cv2.imread(image_name)
                break

        foul = np.zeros_like(image_rain)
        # print(fourier(image_rain[:,:,]))
        # foul = np.zeros((fourier(image_rain[:,:,0].shape[0]),fourier(image_rain[:,:,0]).shape[1],3))
        foul[:,:,0] = fourier(image_rain[:,:,0])[:image_rain.shape[0],:image_rain.shape[1]]
        foul[:,:,1] = fourier(image_rain[:,:,1])[:image_rain.shape[0],:image_rain.shape[1]]
        foul[:,:,2] = fourier(image_rain[:,:,2])[:image_rain.shape[0],:image_rain.shape[1]]
        cv2.imwrite("zz.jpg",foul)
        sample = {"clean":image_clean,"rain":image_rain,"fourier_rain":foul}
        
        if(self.transform):
            sample["clean"] = self.transform(sample["clean"])
            sample["rain"] = self.transform(sample["rain"])
            sample["fourier_rain"] = self.transform(sample["fourier_rain"])
        return sample

