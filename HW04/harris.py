import cv2
import numpy as np
import math
import sys
from tqdm import tqdm
from copy import deepcopy
import random

'''getImage(imageSet:str, imageNum:int) -> tuple
Input: imageSet (str), imageNum (int)
Output: raw_image (ndarray), grey_scale_image (ndarray)
Purpose: given imageSet and imageNum return the proper raw and grey-scaled images'''
def getImage(imageSet:str) -> tuple:
    if imageSet == "book":
        raw_input_image1 = cv2.imread("/Users/wang3450/Desktop/ECE661/HW04/input_images/books_1.jpeg",
                                      cv2.IMREAD_UNCHANGED)
        raw_input_image2 = cv2.imread("/Users/wang3450/Desktop/ECE661/HW04/input_images/books_2.jpeg",
                                      cv2.IMREAD_UNCHANGED)

        h1, w1, _ = raw_input_image1.shape
        raw_input_image2 = cv2.resize(raw_input_image2, (w1,h1), cv2.INTER_AREA)

        grey_input_image1 = cv2.cvtColor(raw_input_image1, cv2.COLOR_BGR2GRAY)
        grey_input_image2 = cv2.cvtColor(raw_input_image2, cv2.COLOR_BGR2GRAY)
        return raw_input_image1, grey_input_image1, raw_input_image2, grey_input_image2
    elif imageSet == "fountain":
        raw_input_image1 = cv2.imread("/Users/wang3450/Desktop/ECE661/HW04/input_images/fountain_1.jpg",
                                      cv2.IMREAD_UNCHANGED)
        raw_input_image2 = cv2.imread("/Users/wang3450/Desktop/ECE661/HW04/input_images/fountain_2.jpg",
                                      cv2.IMREAD_UNCHANGED)

        h1, w1, _ = raw_input_image1.shape
        raw_input_image2 = cv2.resize(raw_input_image2, (w1, h1), cv2.INTER_AREA)

        grey_input_image1 = cv2.cvtColor(raw_input_image1, cv2.COLOR_BGR2GRAY)
        grey_input_image2 = cv2.cvtColor(raw_input_image2, cv2.COLOR_BGR2GRAY)
        return raw_input_image1, grey_input_image1, raw_input_image2, grey_input_image2
    elif imageSet == "checkerboard":
        raw_input_image1 = cv2.imread("/Users/wang3450/Desktop/ECE661/HW04/input_images/checkerboard_1.jpg",
                                      cv2.IMREAD_UNCHANGED)
        raw_input_image2 = cv2.imread("/Users/wang3450/Desktop/ECE661/HW04/input_images/checkerboard_2.jpg",
                                      cv2.IMREAD_UNCHANGED)

        h1, w1, _ = raw_input_image1.shape
        raw_input_image2 = cv2.resize(raw_input_image2, (w1, h1), cv2.INTER_AREA)

        grey_input_image1 = cv2.cvtColor(raw_input_image1, cv2.COLOR_BGR2GRAY)
        grey_input_image2 = cv2.cvtColor(raw_input_image2, cv2.COLOR_BGR2GRAY)
        return raw_input_image1, grey_input_image1, raw_input_image2, grey_input_image2


'''getHaarFilters(sigma: float) -> tuple
Input: sigma (float)
Output: Tuple consisting of h_x, and h_y
Purpose: Given sigma, compute the Haar Wavelet Filters'''
def getHaarFilters(sigma: float) -> tuple:
    # Base X: [-1, 1]
    # Base Y: [1, -1] ^T

    filter_dim = math.ceil(sigma * 4)
    if filter_dim % 2 == 1:
        filter_dim += 1

    filter_dim_midpoint = int(filter_dim / 2)

    h_x = np.zeros((filter_dim, filter_dim))
    h_y = np.zeros((filter_dim, filter_dim))

    h_x[:, 0:filter_dim_midpoint] = -1
    h_x[:,filter_dim_midpoint:] = 1

    h_y[0:filter_dim_midpoint,:] = 1
    h_y[filter_dim_midpoint:, :] = -1

    return h_x, h_y


'''getHarris(grey_input_image, sigma)
Input: normalized grey-scale image, sigma
Output: list of interest points, image plotted with interest points'''
def getHarris(raw_input_image, grey_input_image, sigma):
    '''Compute dx, dy'''
    dx = cv2.filter2D(grey_input_image, ddepth=-1, kernel=h_x)
    dy = cv2.filter2D(grey_input_image, ddepth=-1, kernel=h_y)

    '''Compute dx2, dy2, dxdy'''
    dx2 = dx * dx
    dy2 = dy * dy
    dxdy = dx * dy

    '''Create Window with dim = 5 * sigma'''
    window_dim = math.ceil(5 * sigma)
    window_dim = int(window_dim)
    if window_dim % 2 == 1:
        window_dim += 1
    window = np.ones((window_dim, window_dim))

    '''Compute sigma_dx2, sigma_dy2, sigma_dxdy'''
    sigma_dx2 = cv2.filter2D(dx2, ddepth=-1, kernel=window)
    sigma_dy2 = cv2.filter2D(dy2, ddepth=-1, kernel=window)
    sigma_dxdy = cv2.filter2D(dxdy, ddepth=-1, kernel=window)

    '''Compute Tr(c) and Det(c)'''
    trace_c = sigma_dx2 + sigma_dy2
    det_c = sigma_dx2 * sigma_dy2 - (sigma_dxdy ** 2)

    '''Compute the Harris Ratio'''
    k = 0.04  # empirically defined constant
    r = det_c - k * (trace_c ** 2)

    r_thresh = np.sort(r.flatten())[-1000]
    threshold = list()
    corner = list()

    for x in tqdm(range(5, grey_input_image.shape[1] - 5, 1)):
        for y in (range(5, grey_input_image.shape[0] - 5, 1)):
            region = r[y - 5:y + 5, x - 5:x + 5]
            r_max = np.max(region)
            if r[y, x] == r_max and r_max >= r_thresh:
                threshold.append(r_max)
                corner.append([x, y])
                cv2.circle(raw_input_image, (x, y), 4, (10, 240, 10), -1)

    return corner, raw_input_image


def getDistance(grey_img1, grey_img2, p1, p2, mode):
    window_size = 21
    window1 = grey_img1[p1[1]: p1[1] + window_size, p1[0]: p1[0] + window_size].flatten()
    window2 = grey_img2[p2[1]: p2[1] + window_size, p2[0]: p2[0] + window_size].flatten()

    if mode == 'SSD':
        return np.sum((window2 - window1) ** 2)
    elif mode == 'NCC':
        mean1 = np.mean(window1)
        mean2 = np.mean(window2)
        numerator = np.sum((window1 - mean1) * (window2 - mean2))
        denom = np.sqrt((np.sum((window1 - mean1) ** 2) * np.sum((window1 - mean1) ** 2)))
        return numerator / denom


def getPointCorrespondence(img1, img2, Points1, Points2, mode='NCC'):
    grey_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    grey_img1 = grey_img1 / 255
    grey_img2 = grey_img2 / 255

    width = img1.shape[1]
    cat_raw_img = np.concatenate((img1, img2), axis=1)
    rainbow = [(211, 0, 148), (130, 0, 75), (255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 127, 255), (0, 0, 255)]

    for p1 in Points1:
        distanceList = list()
        for p2 in Points2:
            try:
                distance = getDistance(grey_img1, grey_img2, p1, p2, mode)
                distanceList.append(distance)
            except ValueError:
                pass
        bestPoint = Points2[np.argsort(distanceList)[0]]
        if np.min(distanceList) < 25:
            plotP1 = p1
            plotP2 = (bestPoint[0] + width, bestPoint[1])
            color = random.choice(rainbow)
            cv2.circle(cat_raw_img, plotP1, 4, color, -1)
            cv2.circle(cat_raw_img, plotP2, 4, color, -1)
            cv2.line(cat_raw_img, plotP1, plotP2, color, 1)

    # cv2.imwrite("book_ncc_sigma_0.8.jpg", cat_raw_img)
    cv2.imshow("cast", cat_raw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    '''Proper Execution Checker'''
    if len(sys.argv) != 3:
        print("Incorrect Usage")
        print("Try: python3 harris.py <imageSet> <sigma>")

    '''Load Images'''
    imageSet = sys.argv[1]
    raw_input_image1, grey_input_image1, raw_input_image2, grey_input_image2 = getImage(imageSet)
    copy_raw_input_image1 = deepcopy(raw_input_image1)
    copy_raw_input_image2 = deepcopy(raw_input_image2)

    '''Normalize the Grey Image'''
    grey_input_image1 = grey_input_image1 / 255
    grey_input_image2 = grey_input_image2 / 255

    '''Get the Haar Wavelet Filters'''
    sigma = float(sys.argv[2])
    h_x, h_y = getHaarFilters(sigma)

    '''Perform Harris Corner Detection On Image Set'''
    Points1, pointImage1 = getHarris(raw_input_image1, grey_input_image1, sigma)
    Points2, pointImage2 = getHarris(raw_input_image2, grey_input_image2, sigma)

    '''Perform Point Correspondences'''
    getPointCorrespondence(copy_raw_input_image1, copy_raw_input_image2, Points1, Points2, mode='SSD')

    '''display image to console'''
    # cv2.imshow(f"{imageSet}1", pointImage1)
    # cv2.imshow(f"{imageSet}2", pointImage2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

