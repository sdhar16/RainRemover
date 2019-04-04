__author__ = 'SherlockLiao'

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from rainy_dataloader import  RainyDataset
import cv2
from model import autoencoder

os.makedirs('./dc_img',exist_ok=True)
os.makedirs('./models',exist_ok=True)

image_size = 128
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size, image_size)
    return x
    
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
    os.makedirs('./dc_img/epoch_%d'%(epoch),exist_ok=True)
    for index,data in enumerate(dataloader_training):
        clean_img = data["clean"]
        rainy_img = data["rain"]

        clean_img = Variable(clean_img).cuda()
        rainy_img = Variable(rainy_img).cuda()
        # ===================forward=====================
        output = model(rainy_img)
        loss = criterion(output, clean_img)
        epoch_loss += loss.data.item()
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
        if (index % 20)== 0:
            pic = to_img(output.cpu().data)
            # print(pic.shape)
            original = to_img(clean_img.cpu().data)
            # print(original.shape)
            rainy = to_img(rainy_img.cpu().data)

            #BGR to RGB

            permute = [2, 1, 0] 

            pic=pic[:, permute]
            original=original[:, permute]
            rainy=rainy[:, permute]

            save_image(torch.cat((pic,original,rainy)), './dc_img/epoch_%d/image_%d.png'%(epoch,index))
    print('epoch [{}/{}], loss:{:.5f}'
          .format(epoch, num_epochs-1, epoch_loss/total_train))
    torch.save(model.state_dict(), './models/conv_autoencoder_%d.pth'%epoch)

dataset_testing = RainyDataset('rainy-image-dataset/testing', transform=img_transform)
total_test = len(dataset_testing)
dataloader_testing = DataLoader(dataset_testing, batch_size=batch_size, shuffle=True,num_workers=4)


# model = autoencoder().cuda()
# model.load_state_dict(torch.load("./models/conv_autoencoder_9.pth"))

print("Validating model, total samples %d"%total_test)
model.eval()
test_loss = 0

os.makedirs('./dc_img/testing',exist_ok=True)
for index,data in enumerate(dataloader_testing):
    clean_img = data["clean"]
    rainy_img = data["rain"]

    clean_img = Variable(clean_img).cuda()
    rainy_img = Variable(rainy_img).cuda()
    # ===================forward=====================
    output = model(rainy_img)
    loss = criterion(output, clean_img)
    test_loss += loss.data.item()
    # ===================log========================
    permute = [2, 1, 0]
    pic = to_img(output.cpu().data)[:,permute]
    original = to_img(clean_img.cpu().data)[:,permute]
    rainy = to_img(rainy_img.cpu().data)[:,permute]
    save_image(torch.cat((pic,original,rainy)), './dc_img/testing/image_%d.png'%(index))
print("Test loss",test_loss/total_test)
