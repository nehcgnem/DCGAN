# Using boilerplate code from https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
# for testing, generation, and training

import os
import shutil
import random
import torch
from PIL import Image
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utils import *

image_info = pd.read_csv("dataset/train_info.csv") # not sure I need this metadata but whatever
imgdir = "dataset/train_1/"
preprocesseddir = "dataset/processed/"

img_height = 64
img_width = img_height
channels = 3

cuda = torch.device('cuda')

def checkForReplacements():
    """Replace images in imgdir folder with uncorrupted version from replacements
    folder if provided by Kaggle. Run manually when needed."""
    main_contents = os.listdir(imgdir)
    replacements = os.listdir("dataset/replacements/train/")
    to_replace = [img for img in main_contents if img in replacements]
    for img in to_replace:
        shutil.copyfile(("dataset/replacements/train/"+img),(imgdir+img))

def preprocessData():
    transform = transforms.Compose(
        [transforms.Resize((img_width, img_height)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    images = os.listdir(imgdir)

    for img_name in images:
        img = Image.open(imgdir + img_name)
        img = img.convert(mode="RGB")
        img = transform(img)

        torch.save(img, preprocesseddir + img_name + ".pt")


class paintingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.root_dir + self.img_list[idx]
        img = torch.load(img_name)

        return img

class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = img_width*img_height*channels
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)

        return x

def images_to_vectors(images):
    return images.view(images.size(0), img_width*img_height*channels)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), channels, img_width, img_height)

class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = img_width*img_height*channels

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

def noise(size):
    n = torch.randn(size, 100, device=cuda)
    return n

def ones_target(size):
    data = torch.ones(size, 1, device=cuda)
    return data

def zeros_target(size):
    data = torch.zeros(size, 1, device=cuda)
    return data

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    optimizer.zero_grad()

    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)

    optimizer.zero_grad()
    
    prediction = discriminator(fake_data)

    error = loss(prediction, ones_target(N))
    error.backward()

    optimizer.step()

    return error

if __name__ == "__main__":


    painting_dataset = paintingDataset(preprocesseddir)
    data_loader = DataLoader(
        painting_dataset,
        batch_size=100,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    num_batches = len(data_loader)

    discriminator = DiscriminatorNet()
    generator = GeneratorNet()

    discriminator.cuda()
    generator.cuda()

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    loss = nn.BCELoss()

    num_test_samples = 16
    test_noise = noise(num_test_samples)

    logger = Logger(model_name='VanillaGAN', data_name='PainterTraining1')

    num_epochs = 100

    for epoch in range(num_epochs):
        for n_batch, real_batch in enumerate(data_loader):
            real_batch = real_batch.cuda()

            N = real_batch.size(0)

            real_data = images_to_vectors(real_batch)

            fake_data = generator(noise(N).detach())

            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

            fake_data = generator(noise(N))
            g_error = train_generator(g_optimizer, fake_data)

            logger.log(d_error, g_error, epoch, n_batch, num_batches)

            if n_batch == 0:
                test_images = vectors_to_images(generator(test_noise))
                test_images = test_images.data.cpu()

                logger.log_images(
                    test_images, num_test_samples,
                    epoch, n_batch, num_batches
                )
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake
                )
