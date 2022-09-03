import cv2
import numpy as np
import sys

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
    print(PQRS[0])
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


    np.set_printoptions(suppress=True)
    print(a)

    x = np.dot(np.linalg.inv(a), b)

    print(x)




if __name__ == "__main__":
    '''Data Loading: ie load the PQRS points and image files'''
    X_prime = getCardPoints(int(sys.argv[1]))                   # X_prime is a 4x2 ndarray of PQRS points on the cards
    X = np.array([[29,22], [29,509], [725,509], [725,22]])      # X is a 4x2 ndarray of ROI on the rb18
    card_image = getCardImage(int(sys.argv[1]))                 # card_image stores the cv2 image pointer of the cardImage
    car_image = cv2.imread('car.jpg', cv2.IMREAD_UNCHANGED)     # car_image stores the cv2 image pointer of the rb18

    '''Computation of Homography'''
    H = computeHomography(X, X_prime)

    # cv2.imshow("Car Image", drawBoundingBox(X, car_image))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

