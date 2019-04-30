__author__ = 'SherlockLiao'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
from rainy_dataloader_wavelet import  RainyDataset
import cv2
from model import autoencoder
from skimage.measure import compare_ssim
from utils import *
import statistics

image_dirs = "./dc_wavelet"
model_dirs = "./models_wavelet"
os.makedirs(image_dirs,exist_ok=True)
os.makedirs(model_dirs,exist_ok=True)

image_size = 256

    
num_epochs = 20
batch_size = 8
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset_training = RainyDataset('rainy-image-dataset/training', transform=img_transform)
total_train = len(dataset_training)
dataloader_training = DataLoader(dataset_training, batch_size=batch_size, shuffle=True,num_workers=4)

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

model.train()
print("Training model, total samples %d"%total_train)

for epoch in range(num_epochs):
    std=[]
    epoch_loss = 0
    ssim = 0
    os.makedirs('%s/epoch_%d'%(image_dirs,epoch),exist_ok=True)
    for index,data in enumerate(dataloader_training):
        clean_img = data["clean"]
        rainy_img = data["rain"]
        wavelet = data["wavelet"]

        clean_img = Variable(clean_img).cuda()
        rainy_img = Variable(rainy_img).cuda()
        wave_img = Variable(wavelet).cuda()
        # ===================forward=====================
        output = model(wave_img)
        loss = criterion(output, clean_img-rainy_img)
        epoch_loss += loss.data.item()
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
        if (index % 20)== 0:
            residual = -output.cpu().data
            output = rainy_img + output
            pic = to_img(output.cpu().data,image_size)
            # print(pic.shape)
            original = to_img(clean_img.cpu().data,image_size)
            # print(original.shape)
            rainy = to_img(rainy_img.cpu().data,image_size)

            #BGR to RGB

            permute = [2, 1, 0] 

            pic=pic[:, permute]
            original=original[:, permute]
            rainy=rainy[:, permute]

            save_image(torch.cat((pic,residual,original,rainy)), '%s/epoch_%d/image_%d.png'%(image_dirs,epoch,index))

        clean_img = clean_img.cpu().detach().numpy()
        output = rainy_img + output
        output = output.cpu().detach().numpy()

        for i in range(batch_size):
            bigpeepee=compare_ssim(clean_img[i].transpose(1,2,0),output[i].transpose(1,2,0),data_range = output[i].max() - output[i].min(),multichannel = True)
            std.append(bigpeepee)
            ssim += bigpeepee

        
    print('epoch [{}/{}], loss:{:.5f}'
          .format(epoch, num_epochs-1, epoch_loss/total_train))
    print("SSIM: %f"%(ssim/total_train))
    print(statistics.stdev(std))
    torch.save(model.state_dict(), '%s/conv_autoencoder_%d.pth'%(model_dirs,epoch))

dataset_testing = RainyDataset('rainy-image-dataset/testing', transform=img_transform)
total_test = len(dataset_testing)
dataloader_testing = DataLoader(dataset_testing, batch_size=batch_size, shuffle=True,num_workers=4)


model.load_state_dict(torch.load("%s/conv_autoencoder_19.pth"%model_dirs))

print("Validating model, total samples %d"%total_test)
model.eval()
test_loss = 0

os.makedirs('%s/testing'%image_dirs,exist_ok=True)


ssim = 0
std=[]
for index,data in enumerate(dataloader_testing):
    clean_img = data["clean"]
    rainy_img = data["rain"]
    wave_img = data["clean"]

    clean_img = Variable(clean_img).cuda()
    rainy_img = Variable(rainy_img).cuda()
    wave_img = Variable(wave_img).cuda()
    # ===================forward=====================
    output = model(wave_img)
    loss = criterion(output, clean_img-rainy_img)
    test_loss += loss.data.item()
    # ===================log========================
    residual = -output.cpu().data
    output = rainy_img + output
    permute = [2, 1, 0]
    pic = to_img(output.cpu().data,image_size)[:,permute]
    original = to_img(clean_img.cpu().data,image_size)[:,permute]
    rainy = to_img(rainy_img.cpu().data,image_size)[:,permute]
    save_image(torch.cat((pic,residual,original,rainy)), '%s/testing/image_%d.png'%(image_dirs,index))

    output = output.cpu().detach().numpy()
    clean_img = clean_img.cpu().detach().numpy()

    for i in range(batch_size):
        bigpeepee=compare_ssim(clean_img[i].transpose(1,2,0),output[i].transpose(1,2,0),data_range = output[i].max() - output[i].min(),multichannel = True)
        std.append(bigpeepee)
        ssim += bigpeepee

    
    
print("Test loss",test_loss/total_test)
print("SSIM: %f"%(ssim/total_test))
print(statistics.stdev(std))