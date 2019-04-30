import os
import glob
from torch.utils.data import  Dataset
import cv2
import re
import numpy as np 

def tryint(s):
    try:
        return int(s)
    except:
        return s

def fftimg(img):    
    dim=(128,128)
    img=cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
    channels = cv2.split(img)
    fin=[]
    for i in range(len(channels)):
        f = np.fft.fft2(channels[i])
        r=np.real(f)
        m=np.imag(f)
        fin.append(r)
        fin.append(m)
        
    #fshift = np.fft.fftshift(f)    
    
    #magnitude_spectrum_real = np.log(np.abs(np.real(fshift)))
    #magnitude_spectrum_imag = np.log(np.abs(np.imag(fshift)))
    
    needed_multi_channel_img = np.zeros((len(fin),img.shape[0],img.shape[1]))
    for i in range(len(fin)):
        needed_multi_channel_img[i,:,:]= fin[i]
    #needed_multi_channel_img [:,:,1]= magnitude_spectrum_imag
    #print(needed_multi_channel_img.shape)
    return needed_multi_channel_img #.transpose(2,1,0)

"""
def ifftimg(img):  
    #fin = np.zeros((img.shape[0],img.shape[1],3), dtype=np.complex)
    img = img.transpose(1,2,0)
    real_b=img[:,:,0]
    imag_b=img[:,:,1]
    real_g=img[:,:,2]
    imag_g=img[:,:,3]
    real_r=img[:,:,4]
    imag_r=img[:,:,5]
    channel_b=real_b+(imag_b)*1j
    channel_g=real_g+(imag_g)*1j
    channel_r=real_r+(imag_r)*1j
    fin_b=np.abs(np.fft.ifft2(channel_b))
    fin_g=np.abs(np.fft.ifft2(channel_g))
    fin_r=np.abs(np.fft.ifft2(channel_r))
    #print(fin_b.shape)
    
    fin=cv2.merge([fin_b,fin_g,fin_r])
    
    #print(fin)
    
    
    return fin.transpose(2,1,0)
"""

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
    def __init__(self,root_folder,transform=None,bilateralFilter=False,fourier=False):
        self.files_clean = glob.glob(root_folder+"/ground truth/*")
        self.files_rain = glob.glob(root_folder+"/rainy image/*")
        
        assert (len(self.files_clean)*14==len(self.files_rain))
        
        self.transform = transform
        sort_nicely(self.files_clean)
        # sort_nicely(self.files_rain)
        self.bilateral = bilateralFilter
        self.fourier = fourier

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

        sample = {"clean":image_clean,"rain":image_rain}

        if(self.bilateral==True):
            bilateral = cv2.bilateralFilter(image_rain,25,70,10)
            sample = {"clean":image_clean,"rain":image_rain,"bilateral":bilateral}
        elif(self.fourier):
            sample = {"clean":image_clean,"rain":image_rain,"clean_ff":fftimg(image_clean),"rain_ff":fftimg(image_rain)}
        if(self.transform):
            sample["clean"] = self.transform(sample["clean"])
            sample["rain"] = self.transform(sample["rain"])
            if(self.bilateral):
                sample["bilateral"] = self.transform(sample["bilateral"])
            # if(self.fourier):
            #     sample["clean_ff"] = self.transform(sample["clean_ff"])
            #     sample["rain_ff"] = self.transform(sample["rain_ff"])

        return sample

