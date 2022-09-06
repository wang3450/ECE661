import cv2
import numpy as np
import sys
import copy

'''getCardPoints:
Input: int \in [1,3]
Return: (4,2) ndarray
Purpose: getCardPoints accepts an num between 1 and 3 and returns a 4x2 array
if PQRS points from the respective card'''
def getCardPoints(n):
    if n == 1:
        P = [489,252]
        Q = [611, 1114]
        R = [1221, 799]
        S = [1241, 177]
        return np.array([P, Q, R, S])
    elif n == 2:
        P = [318, 230]
        Q = [212, 860]
        R = [872, 1129]
        S = [1043, 232]
        return np.array([P, Q, R, S])
    else:
        P = [586, 46]
        Q = [62, 592]
        R = [702, 1214]
        S = [1230, 675]
        return np.array([P, Q, R, S])


'''getCardImage
Input: int \in [1,3]
Output: image pointer
Purpose: getCardImage returns a cardImage cv2 image pointer corresponding to n '''
def getCardImage(n):
    if n == 1:
        return cv2.imread('card1.jpeg', cv2.IMREAD_UNCHANGED)
    elif n == 2:
        return cv2.imread('card2.jpeg', cv2.IMREAD_UNCHANGED)
    else:
        return cv2.imread('card3.jpeg', cv2.IMREAD_UNCHANGED)


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


'''getROI
Input: cv2 image, 4x2 PQRS points
Output: dictionary[points] = pixel color
Purpose: Given an image and PQRS, extract ROI as a dictionary'''
def getROI(img, PQRS):
    P = PQRS[0]
    Q = PQRS[1]
    R = PQRS[2]
    S = PQRS[3]
    ROI = {}

    count = 0
    for i in range(P[1]-1, Q[1]):
        for j in range(P[0]-1, S[0]):
            count = count + 1
            ROI[(i,j)] = img[i,j]
    return ROI


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
    '''Data Loading: ie load the PQRS points and image files'''
    X_prime = getCardPoints(int(sys.argv[1]))                   # X_prime is a 4x2 ndarray of PQRS points on the cards
    X = np.array([[29,22], [29,509], [725,509], [725,22]])      # X is a 4x2 ndarray of ROI on the rb18
    card_image = getCardImage(int(sys.argv[1]))                 # card_image stores the cv2 image pointer of the cardImage
    car_image = cv2.imread('car.jpg', cv2.IMREAD_UNCHANGED)     # car_image stores the cv2 image pointer of the rb18

    '''Computation of Homography'''
    H = computeHomography(X, X_prime)
    np.set_printoptions(suppress=True)
    print(H)

    '''
    -Get a dict of points that make up the ROI and plot it
    -Apply Homography to the points and store them in HOMO_ROI dict'''
    ROI = getROI(car_image, X)
    ROI_image = np.ones((1000, 1000, 3), dtype=np.uint8)
    HOMO_ROI = {}
    for key in ROI:
        ROI_image[key] = ROI[key]
        HOMO_KEY = homogenizePoints(H, key)
        print(key)
        HOMO_ROI[HOMO_KEY] = ROI[key]

    '''Plot the new homogenous coordinates on the card!'''
    for key in HOMO_ROI:
        try:
            card_image[key] = HOMO_ROI[key]
        except(IndexError):
            pass

    '''Write the image to the file'''
    filename = 'card{num}_car.jpeg'.format(num=sys.argv[1])
    cv2.imwrite(filename,card_image)

    '''display image test code'''
    # cv2.imshow("test Car Image", card_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

