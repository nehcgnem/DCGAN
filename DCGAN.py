# Using boilerplate code from https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
# for testing, generation, and training

# This implementation is of an example DCGAN on our dataset

import os
import shutil
import random
import torch
import argparse
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

# Just copy-pasting the example for now...
# https://github.com/pytorch/examples/blob/master/dcgan/main.py
nz = 100
ngf = ndf = 64
nc = channels

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

        self.structure = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.structure(x)

        return x

# def images_to_vectors(images):
#     return images.view(images.size(0), img_width*img_height*channels)

# def vectors_to_images(vectors):
#     return vectors.view(vectors.size(0), channels, img_width, img_height)

class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = img_width*img_height*channels

        self.structure = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        x = self.structure(x)
        return x



def load_models(model_dir, model_name):
    input_dir = '{}/{}'.format(model_dir, model_name)
    the_model = GeneratorNet()
    the_model.load_state_dict(torch.load(input_dir))
    return the_model

def noise(size):
    n = torch.randn(size, 100, 1, 1, device=cuda)
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
    # optinal pretrained model load command 
    parser = argparse.ArgumentParser(description='Optional pretrained model path and name')
    parser.add_argument('--path', nargs='?', default = '',
        help='directory of the generator and discriminator')
    parser.add_argument('--gen', nargs='?',default = '',
        help='the name of the generator ')
    parser.add_argument('--dis', nargs='?',default = '',
        help='the name of the discriminator ')
    args = parser.parse_args()




    painting_dataset = paintingDataset(preprocesseddir)
    data_loader = DataLoader(
        painting_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    num_batches = len(data_loader)
    if args.gen != '':
        generator = load_models(args.path, args.gen)
    else:
        generator = GeneratorNet()
        generator.cuda()


    if args.dis != '':
        discriminator = load_models(args.path, args.dis)
    else:
        discriminator = DiscriminatorNet()
        discriminator.cuda()

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    loss = nn.BCELoss()

    num_test_samples = 16
    test_noise = noise(num_test_samples)

    logger = Logger(model_name='DCGAN', data_name='PainterTraining1')

    num_epochs = 100

    for epoch in range(num_epochs):
        for n_batch, real_batch in enumerate(data_loader):
            real_batch = real_batch.cuda()

            N = real_batch.size(0)

            # real_data = images_to_vectors(real_batch)
            real_data = real_batch

            fake_data = generator(noise(N).detach())

            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

            fake_data = generator(noise(N))
            g_error = train_generator(g_optimizer, fake_data)

            logger.log(d_error, g_error, epoch, n_batch, num_batches)

            if n_batch == 0:
                # test_images = vectors_to_images(generator(test_noise))
                test_images = generator(test_noise)
                test_images = test_images.data.cpu()

                logger.log_images(
                    test_images, num_test_samples,
                    epoch, n_batch, num_batches
                )
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake
                )
        logger.save_models(generator, discriminator, epoch)

