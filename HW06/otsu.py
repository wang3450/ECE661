#!/usr/bin/env python
# coding: utf-8

# Import Statements

# In[166]:


import cv2
from otsu_helper import *

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Load the Image Set

# In[167]:


imageSet = "miku2"
raw_input_image, grey_input_image = loadImages(imageSet)
raw_input_image.shape


# Perform Image Segmentation Using RGB values

# In[168]:


# split the raw input image into 3 individual channels
blueChannel, greenChannel, redChannel = cv2.split(raw_input_image)

# apply erosion then dilation of the image
# kernel = np.ones((5, 5), np.uint8)
# blueChannel_mask = cv2.erode(blueChannel, kernel, iterations=5)
# greenChannel_mask = cv2.erode(greenChannel, kernel, iterations=5)
# redChannel_mask = cv2.erode(redChannel, kernel, iterations=5)
#
# blueChannel_mask = cv2.dilate(blueChannel, kernel, iterations=5)
# greenChannel_mask = cv2.dilate(greenChannel, kernel, iterations=5)
# redChannel_mask = cv2.dilate(redChannel, kernel, iterations=5)

# apply segmentation on each individual channel
blueChannel_mask = minWithinClassVariance(blueChannel)
greenChannel_mask = minWithinClassVariance(greenChannel)
redChannel_mask = minWithinClassVariance(redChannel)

# # combine all masks together


# In[169]:


mask_all = np.ones(blueChannel.shape)
mask_all = np.logical_and(mask_all, blueChannel_mask)
mask_all = np.logical_and(mask_all, redChannel_mask)
mask_all = np.logical_and(mask_all, greenChannel_mask).astype(np.uint8) * 255

all_channels_plot = np.hstack((blueChannel_mask, greenChannel_mask))
all_channels_plot = np.hstack((all_channels_plot, redChannel_mask))
all_channels_plot = np.hstack((all_channels_plot, mask_all))

cv2.imwrite(f'/home/jo_wang/Desktop/ECE661/HW06/rgb_suplots/{imageSet}_rgb_subplot.jpg', all_channels_plot)
cv2.imwrite(f"/home/jo_wang/Desktop/ECE661/HW06/rgb_segmentation/{imageSet}_rgb_seg.jpg", mask_all)


# Perform Texture Based Segmentation

# In[ ]:


ch1 = performTexture(grey_input_image, 3)
ch2 = performTexture(grey_input_image, 5)
ch3 = performTexture(grey_input_image, 7)

ch1_otsu = minWithinClassVariance(ch1)
ch2_otsu = minWithinClassVariance(ch2)
ch3_otsu = minWithinClassVariance(ch3)

texture_all = np.ones(ch1_otsu.shape)
texture_all = np.logical_and(texture_all, ch1_otsu)
texture_all = np.logical_and(texture_all, ch2_otsu)

all_texture_subplot = np.hstack((np.logical_not(ch1_otsu).astype(np.uint8) * 255, np.logical_not(ch2_otsu).astype(np.uint8) * 255))
all_texture_subplot = np.hstack((all_texture_subplot, np.logical_not(ch3_otsu).astype(np.uint8) * 255))
all_texture_subplot = np.hstack((all_texture_subplot, np.logical_not(texture_all).astype(np.uint8) * 255))
cv2.imwrite(f'/home/jo_wang/Desktop/ECE661/HW06/texture_subplot/{imageSet}_texture_subplot.jpg', all_texture_subplot)
cv2.imwrite(f"/home/jo_wang/Desktop/ECE661/HW06/texture_segmentation/{imageSet}_texture_seg.jpg", np.logical_not(texture_all).astype(np.uint8) * 255)


# Extract Contours

# In[ ]:


texture_all = np.logical_not(texture_all).astype(np.uint8) * 255

# kernel = np.ones((5, 5), np.uint8)
# texture_all = cv2.erode(texture_all, kernel, iterations=1)
# texture_all = cv2.dilate(texture_all, kernel, iterations=1)

contourImage = getContour(np.logical_not(texture_all // 255).astype(np.uint8))
cv2.imwrite(f"/home/jo_wang/Desktop/ECE661/HW06/contour_extraction/{imageSet}_contour_seg.jpg", np.logical_not(contourImage // 255).astype(np.uint8) * 255)


# In[ ]:


print(texture_all.shape)
print(contourImage.shape)
og_bgr = np.hstack((raw_input_image, cv2.cvtColor(mask_all, cv2.COLOR_GRAY2RGB)))
texture_contour = np.hstack((cv2.cvtColor(texture_all, cv2.COLOR_GRAY2RGB), cv2.cvtColor(np.logical_not(contourImage // 255).astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB) ) )
final_image = np.vstack((og_bgr, texture_contour))

# original       bgr segmentation
# texture        contour extraction
cv2.imwrite(f"/home/jo_wang/Desktop/ECE661/HW06/final_subplots/{imageSet}_final_subplots.jpg", final_image)


# In[ ]:




