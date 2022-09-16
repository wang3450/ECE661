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
        if imageNum == 1:       #p              #q              #r          #s
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


def computeProjH(ps1, qr1, pq1, sr1, ps2, qr2, pq2, sr2):
    vp1 = np.cross(ps1,qr1)
    vp2 = np.cross(pq1, sr1)
    vp1 = vp1 / vp1[2]
    vp2 = vp2 / vp2[2]

    vl = np.cross(vp1, vp2)
    vl = vl / vl[2]
    H = np.zeros((3, 3))
    H[0][0] = 1
    H[1][1] = 1
    H[2] = vl

    return H


def computeAffineH(ps1, qr1, pq1, sr1, ps2, qr2, pq2, sr2):
    l1 = np.cross(pq1, ps1)
    m1 = np.cross(pq1, qr1)

    l2 = np.cross(pq2, ps2)
    m2 = np.cross(pq2, qr2)

    l1 = l1 / l1[2]
    l2 = l2 / l2[2]
    m1 = m1 / m1[2]
    m2 = m2 / m2[2]
    a = np.zeros((2, 2))

    a[0][0] = m1[0] * l1[0]
    a[0][1] = (m1[0] * l1[1]) + (m1[1] * l1[0])
    a[1][0] = m2[0] * l2[0]
    a[1][1] = (m2[0] * l2[1]) + (m2[1] * l2[0])
    b = np.array([[-1 * m1[1] * l1[1]], [-1 * m2[1] * l2[1]]])

    x = np.dot(np.linalg.inv(a), b)

    s = np.zeros((2, 2))
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


if __name__ == "__main__":
    '''Execution Checker'''
    if len(sys.argv) != 3:
        print("Incorrect Usage: python3 2step.py <imageSet> <imageNum>")

    '''Data Loaders
    -imageSet (str): which image set {given, custom}
    -imageNum (int): which image in the set {1,2}
    -distortedImage (np.ndarray): input image
    -box1/box2 (ndarray): points that form a box on the distorted image'''
    imageSet = sys.argv[1]
    imageNum = int(sys.argv[2])
    distorted_image = loadImage(imageSet, imageNum)
    box1, box2 = loadPoints(imageSet, imageNum)

    ps1, qr1, pq1, sr1 = getLines(box2)
    ps2, qr2, pq2, sr2 = getLines(box1)

    projH = computeProjH(ps1, qr1, pq1, sr1, ps2, qr2, pq2, sr2)
    x = np.linalg.inv(projH)

    affineH = computeAffineH(ps1, qr1, pq1, sr1, ps2, qr2, pq2, sr2)

    H = np.dot(projH,np.linalg.inv(affineH))
    print(H)


