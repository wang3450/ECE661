import cv2
import numpy as np
import sys
import copy

'''getRangePlane
Input: int \in [1,3], whichImageSet
Output: image pointer
Purpose: getRangePlane returns a range plane cv2 image pointer corresponding to n'''
def getRangePlane(n, img):
    if img == "card":
        if n == 1:
            print("card 1")
            return cv2.imread('card1.jpeg', cv2.IMREAD_UNCHANGED), cv2.imread('car.jpg', cv2.IMREAD_UNCHANGED)
        elif n == 2:
            print("card 2")
            return cv2.imread('card2.jpeg', cv2.IMREAD_UNCHANGED), cv2.imread('car.jpg', cv2.IMREAD_UNCHANGED)
        else:
            print("card 3")
            return cv2.imread('card3.jpeg', cv2.IMREAD_UNCHANGED), cv2.imread('car.jpg', cv2.IMREAD_UNCHANGED)
    elif img == "custom":
        if n == 1:
            print("tv 1")
            return cv2.imread('tv1.jpeg', cv2.IMREAD_UNCHANGED), cv2.imread('w13.jpg', cv2.IMREAD_UNCHANGED)
        elif n == 2:
            print("tv 2")
            return cv2.imread('tv2.jpeg', cv2.IMREAD_UNCHANGED), cv2.imread('w13.jpg', cv2.IMREAD_UNCHANGED)
        else:
            print("tv 3")
            return cv2.imread('tv3.jpeg', cv2.IMREAD_UNCHANGED), cv2.imread('w13.jpg', cv2.IMREAD_UNCHANGED)


'''getPoints:
Input: int \in [1,3], whichImageSet
Return: 2 (4,2) ndarray
Purpose: getPoints accepts a num between 1 and 3 and the imageSet and returns 2 4x2 array
of PQRS points from the respective card'''
def getPoints(n, img):
    if img == "card":
        if n == 1:
            P = [489,252]
            Q = [611, 1114]
            R = [1221, 799]
            S = [1241, 177]
            return np.array([P, Q, R, S]), np.array([[29,22], [29,509], [725,509], [725,22]])
        elif n == 2:
            P = [318, 230]
            Q = [212, 860]
            R = [872, 1129]
            S = [1043, 232]
            return np.array([P, Q, R, S]), np.array([[29,22], [29,509], [725,509], [725,22]])
        else:
            P = [586, 46]
            Q = [62, 592]
            R = [702, 1214]
            S = [1230, 675]
            return np.array([P, Q, R, S]), np.array([[29,22], [29,509], [725,509], [725,22]])
    elif img == "custom":
        if n == 1:
            P = [366, 498]
            Q = [478, 2369]
            R = [2309, 1833]
            S = [2286, 814]
            return np.array([P, Q, R, S]), np.array([[556,1090], [556,2070], [4030,2070], [4030,1090]])
        elif n == 2:
            P = [1021, 1040]
            Q = [1192, 1715]
            R = [2837, 1735]
            S = [3083, 1083]
            return np.array([P, Q, R, S]), np.array([[556,1090], [556,2070], [4030,2070], [4030,1090]])
        else:
            P = [1418, 910]
            Q = [1568, 1882]
            R = [3544, 2264]
            S = [3752, 990]
            return np.array([P, Q, R, S]), np.array([[556,1090], [556,2070], [4030,2070], [4030,1090]])


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
    '''Execute: python3 homography.py <card/custom> <1/2/3>'''
    if len(sys.argv) != 3:
        print("Incorrect Usage: python3 homography.py <card/custom> <1/2/3>")

    '''Data Loader:
    -loads the domain and range plane image: <car/tv>
    -loads the domain and range ROI points'''
    whichSet = sys.argv[1]
    imageNum = sys.argv[2]
    X_prime_image, X_image = getRangePlane(int(imageNum), whichSet)
    X_prime_PQRS, X_PQRS = getPoints(int(imageNum), whichSet)

    '''Compute Homography H; x' = Hx'''
    H = computeHomography(X_PQRS, X_prime_PQRS)
    np.set_printoptions(suppress=True)
    print(f'H: {H}')


    X_ROI = getROI(X_image, X_PQRS)
    X_prime_ROI = {}
    for point in X_ROI:
        newPoint = homogenizePoints(H, point)
        X_prime_ROI[newPoint] = X_ROI[point]

    for point in X_prime_ROI:
        try:
            X_prime_image[point] = X_prime_ROI[point]
        except IndexError:
            pass

    '''Write the image data'''
    if whichSet == 'card':
        filename = f'card{imageNum}_car.jpeg'
    elif whichSet == 'custom':
        filename = f'tv{imageNum}_w13.jpeg'
    cv2.imwrite(filename, X_prime_image)

    # cv2.imshow("X_image", X_image)
    # cv2.imshow("X__prime_image", X_prime_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


