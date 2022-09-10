import cv2
import numpy as np
import sys
import copy

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
            P = [0, 0]
            Q = [0, 9]
            R = [3, 9]
            S = [3, 0]

            P_prime = [241, 201]
            Q_prime = [235, 368]
            R_prime = [295, 373]
            S_prime = [297, 216]

            return np.array([P, Q, R, S]), np.array([P_prime, Q_prime, R_prime, S_prime])
        elif imageNum == 2:
            P = [0, 0]
            Q = [0, 150]
            R = [150, 85]
            S = [150, 0]

            P_prime = [76, 180]
            Q_prime = [78, 654]
            R_prime = [805, 621]
            S_prime = [803, 220]

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
    newPoint = (int(x_prime[1]/x_prime[2]), int(x_prime[0]/x_prime[2]))

    return newPoint


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
    H = np.linalg.inv(computeHomography(X_PQRS, X_prime_PQRS))

    '''Map New Points'''
    test = {}
    pt = list()
    undistorted_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    for y in range(distorted_image.shape[0]):
        for x in range(distorted_image.shape[1]):
            color = distorted_image[y,x];
            newPoint = homogenizePoints(H, (y,x))
            test[newPoint] = color
            pt.append(newPoint)


    newTest = list()
    for i in pt:
        if i not in newTest:
            newTest.append(i)
    print(len(newTest))



    '''display code'''
    # cv2.imshow("test bb", undistorted_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
