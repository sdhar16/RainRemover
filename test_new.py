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

image_size = 128
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size, image_size)
    return x

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


test_img = cv2.imread("rain.jpg")
test_img = Variable(img_transform(test_img).unsqueeze(0)).cuda()


model = autoencoder().cuda()
model.load_state_dict(torch.load("./models/conv_autoencoder_9.pth"))

model.eval()
    # ===================forward=====================
output = model(test_img)

pic = to_img(output.cpu().data)
save_image(pic, "pic.jpg")
# print("Test loss",test_loss/total_test)