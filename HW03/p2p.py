import cv2
import numpy as np
import sys
import copy
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
    elif imageSet == 'custom':
        if imageNum == 1:
            return cv2.imread('custom1.jpg', cv2.IMREAD_UNCHANGED)
        elif imageNum == 2:
            return cv2.imread('custom2.jpg', cv2.IMREAD_UNCHANGED)


'''loadPoints
input: imageSet (str), imageNum (int)
output: X, X_prime 2 4x2 ndarray
purpose: given the imageSet and imageNum return PQRS points for both distorted and undistorted image'''
def loadPoints(imageSet: str, imageNum: int):
    if imageSet == 'given':
        if imageNum == 1:
            t = 100
            c = 10

            P = [0 * c + t, 0 * c + t]
            Q = [0 * c + t, 9 * c + t]
            R = [3 * c + t, 9 * c + t]
            S = [3 * c + t, 0 * c + t]

            P_prime = [241, 201]
            Q_prime = [235, 368]
            R_prime = [295, 373]
            S_prime = [297, 216]

            return np.array([P, Q, R, S]), np.array([P_prime, Q_prime, R_prime, S_prime])
        elif imageNum == 2:
            t = 100
            c = 5

            P = [0 * c + t, 0 * c + t]
            Q = [0 * c + t, 85 * c + t]
            R = [150 * c + t, 85 * c + t]
            S = [150 * c + t, 0 * c + t]

            P_prime = [76, 180]
            Q_prime = [78, 654]
            R_prime = [805, 621]
            S_prime = [803, 220]

            return np.array([P, Q, R, S]), np.array([P_prime, Q_prime, R_prime, S_prime])
    if imageSet == 'custom':
        if imageNum == 1:
            t = 100
            c = 200

            P = [0 * c + t, 0 * c + t]
            Q = [0 * c + t, 10 * c + t]
            R = [10 * c + t, 10 * c + t]
            S = [10 * c + t, 0 * c + t]


            P_prime = [475, 133]
            Q_prime = [478, 374]
            R_prime = [652, 386]
            S_prime = [638, 192]

            return np.array([P, Q, R, S]), np.array([P_prime, Q_prime, R_prime, S_prime])
        elif imageNum == 2:
            t = 100
            c = 200

            P = [0 * c + t, 0 * c + t]
            Q = [0 * c + t, 5 * c + t]
            R = [20 * c + t, 5 * c + t]
            S = [20 * c + t, 0 * c + t]

            P_prime = [163, 323]
            Q_prime = [163, 409]
            R_prime = [696, 424]
            S_prime = [691, 362]

            return np.array([P, Q, R, S]), np.array([P_prime, Q_prime, R_prime, S_prime])


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


'''computeHomography
Input: 2 4x2 ndarray
Output: 3x3 ndarray
Purpose: Given X and X_prime, compute H'''
def computeHomography(X, X_prime):
    ''' ax = b
    a = 8x8 matrix
    b = 8x1 matrix X_prime
    x = 8x1 matrix homography '''
    a = np.zeros((8, 8))
    b_list = list()
    for i in X_prime:
        b_list.append([i[0]])
        b_list.append([i[1]])
    b = np.array(b_list)
    H = np.zeros((3,3))

    '''Row 1'''
    a[0][0] = X[0][0]
    a[0][1] = X[0][1]
    a[0][2] = 1
    a[0][6] = -1 * X[0][0] * X_prime[0][0]
    a[0][7] = -1 * X[0][1] * X_prime[0][0]

    '''Row 2'''
    a[1][3] = X[0][0]
    a[1][4] = X[0][1]
    a[1][5] = 1
    a[1][6] = -1 * X[0][0] * X_prime[0][1]
    a[1][7] = -1 * X[0][1] * X_prime[0][1]

    '''Row 3'''
    a[2][0] = X[1][0]
    a[2][1] = X[1][1]
    a[2][2] = 1
    a[2][6] = -1 * X[1][0] * X_prime[1][0]
    a[2][7] = -1 * X[1][1] * X_prime[1][0]

    '''Row 4'''
    a[3][3] = X[1][0]
    a[3][4] = X[1][1]
    a[3][5] = 1
    a[3][6] = -1 * X[1][0] * X_prime[1][1]
    a[3][7] = -1 * X[1][1] * X_prime[1][1]

    '''Row 5'''
    a[4][0] = X[2][0]
    a[4][1] = X[2][1]
    a[4][2] = 1
    a[4][6] = -1 * X[2][0] * X_prime[2][0]
    a[4][7] = -1 * X[2][1] * X_prime[2][0]

    '''Row 6'''
    a[5][3] = X[2][0]
    a[5][4] = X[2][1]
    a[5][5] = 1
    a[5][6] = -1 * X[2][0] * X_prime[2][1]
    a[5][7] = -1 * X[2][1] * X_prime[2][1]

    '''Row 7'''
    a[6][0] = X[3][0]
    a[6][1] = X[3][1]
    a[6][2] = 1
    a[6][6] = -1 * X[3][0] * X_prime[3][0]
    a[6][7] = -1 * X[3][1] * X_prime[3][0]

    '''Row 8'''
    a[7][3] = X[3][0]
    a[7][4] = X[3][1]
    a[7][5] = 1
    a[7][6] = -1 * X[3][0] * X_prime[3][1]
    a[7][7] = -1 * X[3][1] * X_prime[3][1]

    x = np.dot(np.linalg.inv(a), b)

    H[0][0] = x[0]
    H[0][1] = x[1]
    H[0][2] = x[2]
    H[1][0] = x[3]
    H[1][1] = x[4]
    H[1][2] = x[5]
    H[2][0] = x[6]
    H[2][1] = x[7]
    H[2][2] = 1

    # np.set_printoptions(suppress=True)

    return H


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
    -X_PQRS (ndarray): PQRS points in undistorted
    -X_prime_PQRS (ndarray): PQRS points in distorted'''
    imageSet = sys.argv[1];
    imageNum = int(sys.argv[2])
    distorted_image = loadImage(imageSet, imageNum)
    X_PQRS, X_prime_PQRS = loadPoints(imageSet, imageNum)

    '''Estimate Homography H'''
    H = (computeHomography(X_PQRS, X_prime_PQRS))
    H_inverse = np.linalg.inv(H)
    np.set_printoptions(suppress=True)

    print(H)

    '''Map New Points with the estimated homography'''
    undistorted_image = np.ones((distorted_image.shape[0], distorted_image.shape[1], 3), dtype=np.uint8)
    for y in range(undistorted_image.shape[0]):
        for x in range(undistorted_image.shape[1]):
            try:
                HOMO_point = homogenizePoints(H, (y,x))
                color = interpolatePixels(HOMO_point, distorted_image)
                undistorted_image[y,x] = color
            except IndexError:
                undistorted_image[y, x] = [0,0,0]

    '''Write Image to File'''
    # cv2.imwrite("rectified_custom2.jpg", undistorted_image)

    '''display code'''
    cv2.imshow("test bb", undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



