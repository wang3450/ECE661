import sys
import cv2
from copy import deepcopy
import numpy as np
from tqdm import tqdm

MAX_ITERATIONS = 10

'''loadImage(imageSet:str) -> list
Input: imageSet (str)
Output: 2 list of ndarrays, one raw and one grey
Purpose: given imageSet return a list of corresponding images'''
def loadImages(imageSet: str) -> list:
    input_image_raw = cv2.imread(f"/home/jo_wang/Desktop/ECE661/HW06/input_images/{imageSet}.jpg",
                                 cv2.IMREAD_UNCHANGED)
    input_image_grey = cv2.cvtColor(input_image_raw, cv2.COLOR_BGR2GRAY)
    return input_image_raw, input_image_grey



'''minWithinClassVariance(img)
Input: single channel img
Output: threshold
Purpose: Given a single channel image, return threshold
that minimizes the within class variance'''
def minWithinClassVariance(img):
    p_i, bins = np.histogram(img, bins=np.arange(np.min(img), np.max(img)+1, 1), density=False)
    # assert (len(p_i) == len(bins) - 1)
    data = img.flatten()
    data.sort()

    bestThreshold = 99999
    bestSigma = 99999

    for t in tqdm(bins):
        w0 = np.sum(p_i[:t+1]) / np.sum(p_i) # w for class 0
        w1 = np.sum(p_i[t+1:]) / np.sum(p_i) # w for class 1

        if t > np.max(img):
            break
        try:
            last_occurence_of_t = max(idx for idx, val in enumerate(data) if val == t)
            var0 = np.var(data[:last_occurence_of_t])
            var1 = np.var(data[last_occurence_of_t+1:])

            sigma = w0 * var0 + w1 * var1

            if sigma < bestSigma:
                bestSigma = sigma
                bestThreshold = t
        except ValueError:
            pass



    _, image_mask = cv2.threshold(img, bestThreshold, 255, cv2.THRESH_BINARY_INV)
    print(f"best threshold:{bestThreshold}")
    return image_mask


def performTexture(img, window):
    texture = np.zeros_like(img).astype(np.uint8)
    height, width = img.shape
    window_size = window // 2
    for x in range(window_size, height - window_size +1):
        for y in range(window_size, width - window_size + 1):
            texture[x,y] = np.var(img[x-window_size: x+window_size+1,y-window_size: y+window_size+1])
    return texture

def getContour(img):
    contour = np.zeros_like(img).astype(np.uint8)
    height, width = img.shape
    for x in range(1, height -1):
        for y in range(1, width -  1):
            if img[x,y] == 1 and np.sum(img[x-1:x+2,y-1: y+2]) < 9:
                contour[x,y] = 1
    return contour * 255







