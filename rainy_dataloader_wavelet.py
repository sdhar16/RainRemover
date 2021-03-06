import os
import glob
from torch.utils.data import  Dataset
import cv2
import re
import numpy as np
import pywt 
def give_high_freq(img):
    coeffs = pywt.wavedec2(img, 'haar',level=2)
    coeffs[0]=np.zeros_like(coeffs[0])
    new_img=pywt.waverec2(coeffs, 'haar')
    new_img=new_img.astype('uint8')

    return new_img[:,:,:-1]
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

        # print(idx,self.files_clean[idx%14],self.files_rain[idx])

        # print(image_clean..shape,self.files_clean[idx%14])

        sample = {"clean":image_clean,"rain":image_rain,"wavelet":give_high_freq(image_rain)}

        # print(sample["wavelet"].shape)
        if(self.transform):
            sample["clean"] = self.transform(sample["clean"])
            sample["rain"] = self.transform(sample["rain"])
            sample["wavelet"] = self.transform(sample["wavelet"])

        return sample

