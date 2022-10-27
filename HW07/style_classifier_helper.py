import numpy as np
import torch
import torch.nn as nn
from BitVector import *
import os
import cv2

'''VGG19(nn.Module)
Input: (256 x 256) image tensor
Output: (512 x 16 x 16) image tensor
Purpose: given an image tensor, extract feature map'''
class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # encode 1-1
            nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 1-1
            # encode 2-1
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 2-1
            # encoder 3-1
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 3-1
            # encoder 4-1
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 4-1
            # rest of vgg not used
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(inplace=True),  # relu 5-1
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
            # nn.ReLU(inplace=True)
        )

    def load_weights(self, path_to_weights):
        vgg_model = torch.load(path_to_weights)
        # Don't care about the extra weights
        self.model.load_state_dict(vgg_model, strict=False)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        # Input is numpy array of shape (H, W, 3)
        # Output is numpy array of shape (N_l, H_l, W_l)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
        out = self.model(x)
        out = out.squeeze(0).numpy()
        return out


'''loadImages
Input: file directory
Output: 2 lists
Purpose: read all train and test images'''
def loadImages(dir_path):
    img_list = list()
    label_list = list()
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        if os.path.isfile(filepath) and ('.jpg' in filepath or '.jpeg' in filepath):
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            resize_img = cv2.resize(img, (256, 256), cv2.INTER_AREA)
            assert(resize_img.shape == (256, 256, 3))
            img_label = -1
            if 'cloudy' in filename:
                img_label = 0
            elif 'rain' in filename:
                img_label = 1
            elif 'shine' in filename:
                img_label = 2
            elif 'sunrise' in filename:
                img_label = 3
            try:
                assert (img_label >= 0)
                img_list.append(resize_img)
                label_list.append(img_label)
            except AssertionError:
                pass
    return img_list, label_list


'''loadImages
Input: file directory
Output: 2 lists
Purpose: read all train and test images'''
def loadGrayImages(dir_path):
    img_list = list()
    label_list = list()
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        if os.path.isfile(filepath) and ('.jpg' in filepath or '.jpeg' in filepath):
            img = cv2.imread(filepath, 0)
            resize_img = cv2.resize(img, (64, 64), cv2.INTER_AREA)
            resize_img = np.pad(resize_img, 1)
            assert(resize_img.shape == (66, 66))
            img_label = -1
            if 'cloudy' in filename:
                img_label = 0
            elif 'rain' in filename:
                img_label = 1
            elif 'shine' in filename:
                img_label = 2
            elif 'sunrise' in filename:
                img_label = 3
            try:
                assert (img_label >= 0)
                img_list.append(resize_img)
                label_list.append(img_label)
            except AssertionError:
                pass
    return img_list, label_list


'''lbp_encode
Input: grey scale image
Output: histogram of lbp encodings
Purpose: Given an image, extract the lbp feature descriptor'''
def lbp_encode(img):
    encoded_image = np.zeros((64,64))
    lbp_hist = {t:0 for t in range(10)}
    k = 0.707
    l = 0.707
    img = np.transpose(img, (1,0))
    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[0]-1):
            p0 = img[x,y+1]
            p2 = img[x+1,y]
            p4 = img[x,y-1]
            p6 = img[x-1,y]
            p1 = (1-k) * (1-l) * img[x,y] \
                 + (1-k) * l * img[x+1,y] \
                 + k * (1-l) * img[x, y+1] \
                 + k * l * img[x+1, y+1]
            p3 = (1-k) * (1-l) * img[x,y] \
                 + (1-k) * l * img[x,y-1] \
                 + k * (1-l) * img[x+1, y] \
                 + k * l * img[x+1, y-1]
            p5 = (1-k) * (1-l) * img[x,y] \
                 + (1-k) * l * img[x-1,y] \
                 + k * (1-l) * img[x, y-1] \
                 + k * l * img[x-1, y-1]
            p7 = (1-k) * (1-l) * img[x,y] \
                 + (1-k) * l * img[x,y+1] \
                 + k * (1-l) * img[x-1, y] \
                 + k * l * img[x-1, y+1]

            p0 = 1 if p0 >= img[x,y] else 0
            p1 = 1 if p1 >= img[x,y] else 0
            p2 = 1 if p2 >= img[x,y] else 0
            p3 = 1 if p3 >= img[x,y] else 0
            p4 = 1 if p4 >= img[x,y] else 0
            p5 = 1 if p5 >= img[x,y] else 0
            p6 = 1 if p6 >= img[x,y] else 0
            p7 = 1 if p7 >= img[x,y] else 0

            pattern = [p0, p1, p2, p3, p4, p5, p6, p7]

            bv = BitVector(bitlist=pattern)
            intvals_for_circular_shifts = [int(bv << 1) for _ in range(8)]
            minbv = BitVector(intVal=min(intvals_for_circular_shifts), size=8)

            bvruns = minbv.runs()
            encoding = None
            if len(bvruns) > 2:
                lbp_hist[9] += 1
            elif len(bvruns) == 1 and bvruns[0][0] == '1':
                lbp_hist[8] += 1
            elif len(bvruns) == 1 and bvruns[0][0] == '0':
                lbp_hist[0] += 1
            else:
                lbp_hist[len(bvruns[1])] += 1

    return lbp_hist

