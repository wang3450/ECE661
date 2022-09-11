import cv2
import numpy as np
import sys
import copy
from tqdm import tqdm
import math


'''loadImages
input: imageSet (str), imageNum (int)
output: cv2 image handler
purpose: given the imageSet and imageNum return cv2 image handler'''
def loadImage(imageSet: str, imageNum: int)-> np.ndarray:
    if imageSet == 'given':
        if imageNum == 1:
            return cv2.imread('building.jpg', cv2.IMREAD_UNCHANGED)
        elif imageNum == 2:
            return cv2.imread('nighthawks.jpg', cv2.IMREAD_UNCHANGED)


'''loadPoints
input: imageSet (str), imageNum (int)
output: X, X_prime 2 4x2 ndarray
purpose: given the imageSet and imageNum return PQRS points for both distorted and undistorted image'''
def loadPoints(imageSet: str, imageNum: int):
    if imageSet == 'given':
        if imageNum == 1:
            p1 = [240, 122,1]
            p2 = [717, 290,1]

            p3 = [240, 193,1]
            p4 = [719, 325,1]

            p5 = [156, 462,1]
            p6 = [167, 154,1]

            p7 = [104, 459,1]
            p8 = [112, 217,1]

            return p1, p2, p3, p4, p5, p6, p7, p8

        elif imageNum == 2:
            p1 = [74, 177, 1]
            p2 = [77, 654, 1]

            p3 = [805, 217, 1]
            p4 = [806, 622, 1]

            p5 = [77, 655, 1]
            p6 = [805, 622, 1]

            p7 = [74, 177, 1]
            p8 = [804, 218, 1]

            return p1, p2, p3, p4, p5, p6, p7, p8


'''drawBoundingBox
Input: 4x2 ndarray, cv2 image file
Output: cv2 image file
Purpose: drawBoundingBox draws a bounding box around the subject in the img'''
def drawBoundingBox(PQRS, img):
    img = cv2.line(img, (PQRS[0][0], PQRS[0][1]),(PQRS[1][0], PQRS[1][1]), (0,0,0), 2)
    img = cv2.line(img, (PQRS[2][0], PQRS[2][1]), (PQRS[1][0], PQRS[1][1]), (0, 0, 0), 2)
    img = cv2.line(img, (PQRS[2][0], PQRS[2][1]), (PQRS[3][0], PQRS[3][1]), (0, 0, 0), 2)
    img = cv2.line(img, (PQRS[0][0], PQRS[0][1]), (PQRS[3][0], PQRS[3][1]), (0, 0, 0), 2)
    return img


'''computeProjHomography
Input: 8 lists of length 3 ndarray
Output: 3x3 ndarray
Purpose: Given 8 points, compute the projective homography'''
def computeProjHomography(p1, p2, p3, p4, p5, p6, p7, p8):
    x1 = np.array(p1)
    x2 = np.array(p2)
    x3 = np.array(p3)
    x4 = np.array(p4)
    x5 = np.array(p5)
    x6 = np.array(p6)
    x7 = np.array(p7)
    x8 = np.array(p8)

    l1 = np.cross(x1, x2)
    l2 = np.cross(x3, x4)
    l3 = np.cross(x5, x6)
    l4 = np.cross(x7, x8)

    vp1 = np.cross(l1, l2)
    vp1 = vp1 / vp1[2]
    vp2 = np.cross(l3, l4)
    vp2 = vp2 / vp2[2]

    vl = np.cross(vp1, vp2)
    vl = vl / vl[2]

    H = np.zeros((3, 3))
    H[0][0] = 1
    H[1][1] = 1
    H[2] = vl

    return H

'''computeAffineHomography
Input: 8 lists of length 3 ndarray
Output: 3x3 ndarray
Purpose: Given 8 points, compute the affine homography'''

def computeAffineHomography(p1, p2, p3, p4, p5, p6, p7, p8):


    x1 = np.array(p1)
    x2 = np.array(p2)
    x3 = np.array(p3)
    x4 = np.array(p4)
    x5 = np.array(p5)
    x6 = np.array(p6)
    x7 = np.array(p7)
    x8 = np.array(p8)

    l1 = np.cross(x1, x2)
    m1 = np.cross(x7, x8)

    l2 = np.cross(x3, x4)
    m2 = np.cross(x5, x6)

    a = np.zeros((2, 2))

    a[0][0] = m1[0] * l1[0]
    a[0][1] = (m1[0] * l1[1]) + (m1[1] * l1[0])
    a[1][0] = m2[0] * l2[0]
    a[1][1] = (m2[0] * l2[1]) + (m2[1] * l2[0])
    b = np.array([[-1 * m1[1] * l1[1]], [-1 * m2[1] * l2[1]]])

    x = np.dot(np.linalg.inv(a), b)

    s = np.zeros((2,2))
    s[0][0] = x[0]
    s[0][1] = x[1]
    s[1][0] = x[1]
    s[1][1] = 1

    u, d_square, v = np.linalg.svd(s)
    d = np.sqrt(d_square)
    D = np.diag(d)
    A = np.dot(np.dot(u, D), np.transpose(u))
    H = np.append(A[0], (0, A[1][0], A[1][1], 0, 0, 0, 1))
    H = np.reshape(H, (3, 3))

    return np.linalg.inv(H)


'''homogenizePoints
Input: H (3x3), key = unhomogenized point, ROI[key] = color
Ouput: HOMO_KEY = new point, HOMO_COLOR = color
Purpose: homogenizePoints calculates the new points to transform'''
def homogenizePoints(H, key):
    x_list = list()
    for i in key:
        x_list.append([i])
    x_list.append([1])

    temp = copy.deepcopy(x_list[0])
    x_list[0] = x_list[1]
    x_list[1] = temp
    x_prime = np.dot(H,np.array(x_list))
    x1 = float(x_prime[1])
    x2 = float(x_prime[0])
    x3 = float(x_prime[2])

    newPoint = (x1/x3, x2/x3)
    # newPoint = (y,x)
    return newPoint


'''interpolatePixels
Input: single point and its rgb value
Output: pixel value
Purpose: Given a point where each dim is a float, compute its pixel value'''
def interpolatePixels(homoPoint, image):
    if homoPoint[0].is_integer() and homoPoint[1].is_integer():
        return (image[int(homoPoint[0]), (int(homoPoint[1]))])

    else:
        try:
            x_prime = homoPoint[1]
            y_prime = homoPoint[0]

            p1 = (math.floor(y_prime), math.floor(x_prime))
            p2 = (math.ceil(y_prime), math.floor(x_prime))
            p3 = (math.ceil(y_prime), math.ceil(x_prime))
            p4 = (math.floor(y_prime), math.ceil(x_prime))

            d1 = getDistance(homoPoint, p1)
            d2 = getDistance(homoPoint, p2)
            d3 = getDistance(homoPoint, p3)
            d4 = getDistance(homoPoint, p4)

            pv1 = image[p1]
            pv2 = image[p2]
            pv3 = image[p3]
            pv4 = image[p4]


            denom = d1 + d2 + d3 + d4

            d1_pv1 = [pv1[0] * d1, pv1[1] * d1, pv1[2] * d1]
            d2_pv2 = [pv2[0] * d2, pv2[1] * d2, pv2[2] * d2]

            d3_pv3 = [pv3[0] * d3, pv3[1] * d3, pv3[2] * d3]
            d4_pv4 = [pv4[0] * d4, pv4[1] * d4, pv4[2] * d4]

            numer = [d1_pv1[0] + d2_pv2[0] + d3_pv3[0] + d4_pv4[0], d1_pv1[1] + d2_pv2[1] + d3_pv3[1] + d4_pv4[1], d1_pv1[2] + d2_pv2[2] + d3_pv3[2] + d4_pv4[2]]

            return (int(numer[0] / denom), int(numer[1] / denom), int(numer[2] / denom))
        except IndexError:
            return (0,0,0)

'''getDistance
Input: 2 points
Output: Euclidean Distance
Purpose: Used in interpolatePixels'''
def getDistance(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    return math.sqrt(math.pow(x1-x2,2) + math.pow(y1-y2,2))


if __name__ == "__main__":
    '''Execute: python3 p2p.py <set> <1/2/3>'''
    if len(sys.argv) != 3:
        print("Incorrect Usage: python3 p2p.py <set> <1/2/3>")

    '''data loaders:
    -imageSet (str): which set to load {given, custom}
    -imageNum (int): which image from set {1,2}
    -distortedImage (cv2): distorted image
    -p1-p8 (list): 8 points which help identify vanishing line'''
    imageSet = sys.argv[1];
    imageNum = int(sys.argv[2])
    distorted_image = loadImage(imageSet, imageNum)
    p1, p2, p3, p4, p5, p6, p7, p8 = loadPoints(imageSet, imageNum)


    '''Estimate Projective Homography projH'''
    projH = computeProjHomography(p1, p2, p3, p4, p5, p6, p7, p8)
    if imageSet == 'given':
        if imageNum == 1:
            projH[2] = projH[2] * 5
        elif imageNum == 2:
            projH[2] = projH[2] * 2
    projH_inverse = np.linalg.inv(projH)


    '''Map New Points With Projective Homography'''
    undistorted_image = np.ones((distorted_image.shape[0], distorted_image.shape[1], 3), dtype=np.uint8)
    copy_distorted = copy.deepcopy(distorted_image)
    for y in range(distorted_image.shape[0]):
         for x in range(distorted_image.shape[1]):
             try:
                 HOMO_point = homogenizePoints(projH_inverse, (y,x))
                 color = interpolatePixels(HOMO_point, copy_distorted)
                 distorted_image[y,x] = color
             except IndexError:
                 distorted_image[y, x] = [0,0,0]



    affineH = computeAffineHomography(p1, p2, p3, p4, p5, p6, p7, p8)
    copy_distorted = copy.deepcopy(distorted_image)
    for y in range(distorted_image.shape[0]):
         for x in range(distorted_image.shape[1]):
             try:
                 HOMO_point = homogenizePoints(projH_inverse, (y,x))
                 color = interpolatePixels(HOMO_point, copy_distorted)
                 distorted_image[y,x] = color
             except IndexError:
                 distorted_image[y, x] = [0,0,0]

    '''display code'''
    # cv2.imshow("fin", distorted_image)
    cv2.imshow("input", distorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''Write Image to File'''
    # cv2.imwrite("proj_rect_building.jpg", undistorted_image)