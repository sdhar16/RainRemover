import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np 

import cv2
def to_img(x,image_size):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size, image_size)
    return x
def to_img_fourier(x,image_size):
    x = 0.5 * (x + 1)
    x = x.view(x.size(0), 6, image_size, image_size)
    return x

def ifftimg(img):
    # print(img.shape)
    img = img.numpy().transpose(1,2,0)
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
    
    fin= cv2.merge([fin_r,fin_g,fin_b])
    return torch.from_numpy(fin.transpose(2,0,1))
def ifftimg_batch(img):
    x = np.zeros((img.detach().numpy().shape[0],3,img.detach().numpy().shape[2],img.detach().numpy().shape[3]))
    for i in range(len(x)):
        x[i] = ifftimg(img[i])
    # from torchvision.utils import save_image
    # save_image(torch.from_numpy(x/255),"zx.jpg")
    # print(torch.from_numpy(np.uint8(x)))
    return torch.from_numpy(x/255)