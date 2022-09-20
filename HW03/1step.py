import numpy as np
import cv2
import copy
import sys
import math
from transformImage import transformInputImage


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
output: ndarray of points
purpose: given the imageSet and imageNum return PQRS points for both distorted and undistorted image'''
def loadPoints(imageSet: str, imageNum: int):
    if imageSet == 'given':
        if imageNum == 1:
            box1 = np.array([[317, 215, 1], [315, 378, 1], [460, 390, 1], [460, 254, 1]])
            box2 = np.array([[554, 282, 1], [555, 397, 1], [626, 402, 1], [625, 301, 1]])
            return box1, box2
        elif imageNum == 2:
            box1 = np.array([[12, 101, 1], [12, 729, 1], [865, 677, 1], [862, 162, 1]])
            box2 = np.array([[76, 179, 1], [78, 653, 1], [806, 620, 1], [804, 220, 1]])
            return box1, box2
    elif imageSet == 'custom':
        if imageNum == 1:
            box1 = np.array([[637, 186, 1], [422, 385, 1], [704, 399, 1], [704, 390, 1]])
            box2 = np.array([[475, 133, 1], [478, 374, 1], [652, 386, 1], [638, 192, 1]])
            return box1, box2
        elif imageNum == 2:
            box1 = np.array([[163, 323, 1], [163, 409, 1], [696, 424, 1], [691, 362, 1]])
            box2 = np.array([[341, 258, 1], [343, 328, 1], [397, 332, 1], [396, 264, 1]])
            return box1, box2


'''getLines
input: ndarray of points
output: ndarray of lines
purpose: given points, return the lines that connect them

Structure of Box:
P------------S
|            |
|            |
|            |
Q------------R
'''
def getLines(box):
    P = box[0]
    Q = box[1]
    R = box[2]
    S = box[3]

    PS = np.cross(P,S)
    QR = np.cross(Q,R)
    PQ = np.cross(P,Q)
    SR = np.cross(S,R)

    return PS / PS[2], QR / QR[2], PQ / PQ[2], SR / SR[2]

'''computeHomography
Input: Orthogonal Lines
Output: one shot H
Purpose: Given orthogonal lines, compute one shot H'''
def computeHomography(ps1, qr1, pq1, sr1, ps2, qr2, pq2, sr2):

    '''(li, mi) for i in [1,5] are pairs perpendicular lines'''
    l1 = ps1
    m1 = sr1
    l2 = m1
    m2 = qr1
    l3 = m2
    m3 = pq1
    l4 = m3
    m4 = l1
    l5 = ps2
    m5 = sr2

    a = np.zeros((5, 5))
    b = np.zeros((5,1))

    '''Populate the a matrix'''
    a[0][0] = m1[0] * l1[0]
    a[0][1] = (m1[1] * l1[0]) + (m1[0] * l1[1])
    a[0][2] = m1[1] * l1[1]
    a[0][3] = l1[0] * m1[2] + l1[2] * m1[0]
    a[0][4] = l1[1] * m1[2] + l1[2] * m1[1]

    a[1][0] = m2[0] * l2[0]
    a[1][1] = (m2[1] * l2[0]) + (m2[0] * l2[1])
    a[1][2] = m2[1] * l2[1]
    a[1][3] = l2[0] * m2[2] + l2[2] * m2[0]
    a[1][4] = l2[1] * m2[2] + l2[2] * m2[1]

    a[2][0] = m3[0] * l3[0]
    a[2][1] = (m3[1] * l3[0]) + (m3[0] * l3[1])
    a[2][2] = m3[1] * l3[1]
    a[2][3] = l3[0] * m3[2] + l3[2] * m3[0]
    a[2][4] = l3[1] * m3[2] + l3[2] * m3[1]

    a[3][0] = m4[0] * l4[0]
    a[3][1] = (m4[1] * l4[0]) + (m4[0] * l4[1])
    a[3][2] = m4[1] * l4[1]
    a[3][3] = l4[0] * m4[2] + l4[2] * m4[0]
    a[3][4] = l4[1] * m4[2] + l4[2] * m4[1]

    a[4][0] = m5[0] * l5[0]
    a[4][1] = (m5[1] * l5[0]) + (m5[0] * l5[1])
    a[4][2] = m5[1] * l5[1]
    a[4][3] = l5[0] * m5[2] + l5[2] * m5[0]
    a[4][4] = l5[1] * m5[2] + l5[2] * m5[1]

    '''Populate the b matrix'''
    b[0] = -1
    b[1] = -1
    b[2] = -1
    b[3] = -1
    b[4] = -1

    x = np.dot(np.linalg.inv(a), b)
    x = x / np.max(x)

    aa_transpose = np.zeros((2, 2))
    aa_transpose[0][0] = x[0]
    aa_transpose[0][1] = x[1]
    aa_transpose[1][0] = x[1]
    aa_transpose[1][1] = x[2]

    u, d, v = np.linalg.svd(aa_transpose)
    d= np.sqrt(d)
    d = np.diag(d)

    A = np.dot(np.dot(u,d), u.T)

    temp = list()
    temp.append(x[3])
    temp.append(x[4])
    temp = np.array(temp)

    v = np.dot(np.linalg.inv(A), temp)

    H = np.zeros((3,3))
    H[0][0] = A[0][0]
    H[0][1] = A[0][1]
    H[1][0] = A[1][0]
    H[1][1] = A[1][1]
    H[2][0] = v[0]
    H[2][1] = v[1]
    H[2][2] = 1

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
    '''Execution Checker'''
    if len(sys.argv) != 3:
        print("Incorrect Usage: python3 oneStep.py <imageSet> <imageNum>")

    '''Data Loaders
    -imageSet (str): which image set {given, custom}
    -imageNum (int): which image in the set {1,2}
    -distortedImage (np.ndarray): input image
    -box1/box2 (ndarray): points that form a box on the distorted image'''
    imageSet = sys.argv[1]
    imageNum = int(sys.argv[2])
    distorted_image = loadImage(imageSet, imageNum)
    box1, box2 = loadPoints(imageSet, imageNum)

    print(box1)
    print("\n")
    print(box2)
    print("\n")

    '''Compute the lines that build the box around the points'''
    ps1, qr1, pq1, sr1 = getLines(box2)
    ps2, qr2, pq2, sr2 = getLines(box1)

    np.printoptions(suppress=True)
    print(ps1, qr1, pq1, sr1)
    print("\n")
    print(ps2, qr2, pq2, sr2)
    print("\n")
    '''Compute the Homography and Apply it to the Distorted Image'''
    oneStepH = computeHomography(ps1, qr1, pq1, sr1, ps2, qr2, pq2, sr2 )
    print(f'H: {oneStepH}')
    rectified_image = transformInputImage(distorted_image, oneStepH)
    '''Map New Points with the estimated homography'''

    '''display image to console'''
    cv2.imshow("custom1.jpg", rectified_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
