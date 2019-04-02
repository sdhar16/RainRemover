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
dataloader_training = DataLoader(dataset_training, batch_size=batch_size, shuffle=True,num_workers=4)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.LeakyReLU(),
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1,padding = 1),  # b, 16, 5, 5
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),  # b, 8, 15, 15
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  # b, 8, 15, 15
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

model.train()

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
            save_image(torch.cat((pic,original,rainy)), './dc_img/epoch_%d/image_%d.png'%(epoch,index))
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, epoch_loss/batch_size))
    torch.save(model.state_dict(), './models/conv_autoencoder_%d.pth'%epoch)

dataset_testing = RainyDataset('rainy-image-dataset/testing', transform=img_transform)
dataloader_testing = DataLoader(dataset_testing, batch_size=batch_size, shuffle=True,num_workers=4)

model = autoencoder().cuda()
model.load_state_dict(torch.load("./models/conv_autoencoder_9.pth"))

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
    pic = to_img(output.cpu().data)
    original = to_img(clean_img.cpu().data)
    rainy = to_img(rainy_img.cpu().data)
    save_image(torch.cat((pic,original,rainy)), './dc_img/testing/image_%d.png'%(index))
print("Test loss",test_loss/batch_size)