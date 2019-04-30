import cv2
import numpy as np

def ifftimg(img):  
    #fin = np.zeros((img.shape[0],img.shape[1],3), dtype=np.complex)
    #img = img[0]
    print(img.shape)

    real_b=img[0,:,:]
    imag_b=img[1,:,:]
    real_g=img[2,:,:]
    imag_g=img[3,:,:]
    real_r=img[4,:,:]
    imag_r=img[5,:,:]

    #print(real_b.shape)

    channel_b=real_b+(imag_b)*1j
    channel_g=real_g+(imag_g)*1j
    channel_r=real_r+(imag_r)*1j
    fin_b=np.abs(np.fft.ifft2(channel_b))
    fin_g=np.abs(np.fft.ifft2(channel_g))
    fin_r=np.abs(np.fft.ifft2(channel_r))
    #print(fin_b.shape)
    
    fin=cv2.merge([fin_b,fin_g,fin_r])
    
    print(fin.shape)
    
    #cv2.imshow("out",fin)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return fin


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

image_rain = cv2.imread('./rainy-image-dataset/testing/ground truth/1.jpg')
finfin=ifftimg(fftimg(image_rain))
#print(finfin.shape)
cv2.imshow('tt',np.uint8(finfin))
cv2.waitKey(0)
cv2.destroyAllWindows()
