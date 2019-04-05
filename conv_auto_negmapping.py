__author__ = 'SherlockLiao'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
from rainy_dataloader import  RainyDataset
import cv2
from model import autoencoder
from skimage.measure import compare_ssim
from utils import *

img_dirs = "./dc_img_neg"
model_dirs = "./models_neg"
os.makedirs(img_dirs,exist_ok=True)
os.makedirs(model_dirs,exist_ok=True)

image_size = 128
    
num_epochs = 10
batch_size = 16
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
    epoch_loss = 0
    os.makedirs('%s/epoch_%d'%(img_dirs,epoch),exist_ok=True)
    ssim = 0
    for index,data in enumerate(dataloader_training):
        clean_img = data["clean"]
        rainy_img = data["rain"]

        clean_img = Variable(clean_img).cuda()
        rainy_img = Variable(rainy_img).cuda()
        # ===================forward=====================
        output = model(rainy_img)
        loss = criterion(output, clean_img - rainy_img)
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

            save_image(torch.cat((pic,residual,original,rainy)), '%s/epoch_%d/image_%d.png'%(img_dirs,epoch,index))
        
        output = rainy_img + output
        clean_img = clean_img.cpu().detach().numpy()
        output = output.cpu().detach().numpy()

        for i in range(batch_size):
            ssim += compare_ssim(clean_img[i].transpose(1,2,0),output[i].transpose(1,2,0),data_range = output[i].max() - output[i].min(),multichannel = True)

    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, num_epochs-1, epoch_loss/total_train))
    print("SSIM: %f"%(ssim/total_train))
    torch.save(model.state_dict(), '%s/conv_autoencoder_%d.pth'%(model_dirs,epoch))

dataset_testing = RainyDataset('rainy-image-dataset/testing', transform=img_transform)
total_test = len(dataset_testing)
dataloader_testing = DataLoader(dataset_testing, batch_size=batch_size, shuffle=True,num_workers=4)

# model.load_state_dict(torch.load("%s/conv_autoencoder_9.pth"%model_dirs))

print("Validating model, total samples %d"%total_test)
model.eval()
test_loss = 0

os.makedirs('%s/testing'%img_dirs,exist_ok=True)
ssim = 0
for index,data in enumerate(dataloader_testing):
    clean_img = data["clean"]
    rainy_img = data["rain"]

    clean_img = Variable(clean_img).cuda()
    rainy_img = Variable(rainy_img).cuda()
    # ===================forward=====================
    output = model(rainy_img)
    loss = criterion(output, clean_img - rainy_img)
    test_loss += loss.data.item()
    # ===================log========================
    residual = -output.cpu().data
    output = rainy_img + output

    permute = [2,1,0]
    pic = to_img(output.cpu().data,image_size)[:,permute]
    original = to_img(clean_img.cpu().data,image_size)[:,permute]
    rainy = to_img(rainy_img.cpu().data,image_size)[:,permute]
    save_image(torch.cat((pic,residual,original,rainy)), '%s/testing/image_%d.png'%(img_dirs,index))

    output = output.cpu().detach().numpy()
    clean_img = clean_img.cpu().detach().numpy()

    for i in range(batch_size):
        ssim += compare_ssim(clean_img[i].transpose(1,2,0),output[i].transpose(1,2,0),data_range = output[i].max() - output[i].min(),multichannel = True)

    
    
print("Test loss",test_loss/total_test)
print("SSIM: %f"%(ssim/total_test))
