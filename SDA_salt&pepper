# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 09:45:37 2019

@author: BlaKe
"""
import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import random

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_epochs = 11
batch_size = 100
learning_rate = 0.009 
prob=0.05
thres= 0.95

def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x
    
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = torch.zeros(image.shape,dtype=torch.float32) 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
           rdn = random.random()
           if rdn < prob:
               output[i][j] = 0
           elif rdn > thres:
                output[i][j] = 1
           else:
                output[i][j] = image[i][j]
    return output

def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))
    


img_transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.FashionMNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)




class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(500, 28*28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        noisy_img = sp_noise(img,prob).float()
        noisy_img = Variable(noisy_img)
        img = Variable(img)
        # ===================forward=====================
        output = model(noisy_img)
        loss = criterion(output, img)
        MSE_loss = nn.MSELoss()(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

 

model.encoder.add_module('New_Encoder_Layer',nn.Sequential(nn.Linear(500,256),nn.Tanh()))
model.encoder.add_module('New_Decoder_Layer',nn.Sequential(nn.Linear(256,500),nn.Tanh()))


for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        noisy_img =sp_noise(img,prob).float()
        noisy_img = Variable(noisy_img)
        img = Variable(img)
        # ===================forward=====================
        output = model(noisy_img)
        loss = criterion(output, img)
        MSE_loss = nn.MSELoss()(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data, MSE_loss.data))
    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(output.cpu().data)
        x_noisy = to_img(noisy_img.cpu().data)
        weights = to_img(model.encoder[0].weight.cpu().data)
        save_image(x, './mlp_img/x_{}.png'.format(epoch))
        save_image(x_hat, './mlp_img/x_hat_{}.png'.format(epoch))
        save_image(x_noisy, './mlp_img/x_noisy_{}.png'.format(epoch))
        save_image(weights, './filters/epoch_{}.png'.format(epoch))
