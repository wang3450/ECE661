import cv2
import numpy as np
import math
import sys
from tqdm import tqdm
from copy import deepcopy
import random

'''loadImage(imageSet:str) -> list
Input: imageSet (str)
Output: 2 list of ndarrays, one raw and one grey
Purpose: given imageSet return a list of corresponding images'''
def loadImages(imageSet:str)->list:
    input_image_list_raw = list()
    input_image_list_grey = list()
    if imageSet == "given":
        for i in range(0,5):
            image = cv2.imread(f"/Users/wang3450/Desktop/ECE661/HW05/input_images/{i}.jpg",cv2.IMREAD_UNCHANGED)
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            input_image_list_raw.append(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            input_image_list_grey.append(image)
    elif imageSet == "custom":
        for i in range(0,5):
            image = cv2.imread(f"/Users/wang3450/Desktop/ECE661/HW05/input_images/custom_{i}.jpg",cv2.IMREAD_UNCHANGED)
            #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            input_image_list_raw.append(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            input_image_list_grey.append(image)
    return input_image_list_raw, input_image_list_grey


'''getFeatures(img1, img2)
Input: 2 grey-scale images (ndarray)
Output: 2 lists of keypoints, and 2 lists of descriptors
Purpose: given two grey-scale images, extract the features'''
def getFeatures(img1:np.ndarray, img2:np.ndarray) -> tuple:
    siftObject = cv2.xfeatures2d.SIFT_create()
    kp1, descriptor1 = siftObject.detectAndCompute(img1, None)
    kp2, descriptor2 = siftObject.detectAndCompute(img2, None)

    return kp1, kp2, descriptor1, descriptor2


'''getMatches(descriptor1, descriptor2)
Input: 2 lists of features
Output: list of matches
Purpose: extract the matches from the descriptors'''
def getMatches(descriptor1, descriptor2):
    featureMapper = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = featureMapper.match(descriptor1, descriptor2)
    matches = sorted(matches, key=lambda x:x.distance)
    return matches


'''getCandidateMatches(grey_image1, grey_image2)
Input: 2 adjacent grey-scale images
Output: list of matches, float and int
Purpose: given two grey-scale images, find matching interest points'''
def getCandidateMatches(grey_image1:np.ndarray, grey_image2:np.ndarray)->list:
    kp1, kp2, descriptor1, descriptor2 = getFeatures(grey_image1, grey_image2)
    matches = getMatches(descriptor1, descriptor2)

    candidate_matches_float = list()
    candidate_matches_int = list()

    for match in matches:
        float_p1 = kp1[match.queryIdx].pt
        float_p2 = kp2[match.trainIdx].pt
        candidate_matches_float.append([float_p1, float_p2])
        int_p1 = (int(float_p1[0]), int(float_p1[1]))
        int_p2 = (int(float_p2[0]), int(float_p2[1]))
        candidate_matches_int.append([int_p1, int_p2])

    return candidate_matches_float, candidate_matches_int


'''performRANSAC(candidate_matches_float, candidate_matches_int)
Input: 2 lists of candidate matches
Output: Inlier Set
Purpose: outlier rejection mechanism'''
def performRANSAC(candidate_matches_float:list, candidate_matches_int:list):
    NUM_TRIALS = 1000
    NUM_POINTS = 7
    DELTA = 3

    best_inlier_set = 0
    largest_inlier_size = 0
    bestHomography = np.ones((3,3))
    for i in tqdm(range(NUM_TRIALS)):
        correspondences = list()
        inlier_set = list()
        for i in range(NUM_POINTS):
            correspondences.append(random.choice(candidate_matches_float))
        assert(len(correspondences) == NUM_POINTS)

        homography = estimateHomography(correspondences)
        for candidate in candidate_matches_float:
            x = candidate[0]
            x_prime = candidate[1]
            computed_x_prime = homogenizePoint(homography, x)
            if getDistance(x_prime, computed_x_prime) <= DELTA:
                inlier_set.append(candidate)
        if len(inlier_set) > largest_inlier_size:
            best_inlier_set = deepcopy(inlier_set)
            bestHomography = deepcopy(homography)
            largest_inlier_size = len(inlier_set)

    return bestHomography, best_inlier_set


'''getDistance(x_prime, computed_x_prime)
Input: two points
Output: euclidean distance
Purpose: compute the distance between two points'''
def getDistance(x_prime, computed_x_prime):
    x1 = x_prime[0]
    x2 = x_prime[1]
    x_prime1 = computed_x_prime[0]
    x_prime2 = computed_x_prime[1]
    distance = np.sqrt((x1 - x_prime1) ** 2 + (x2 - x_prime2) ** 2)
    return distance


'''homogenizePoint(h, x)
Input: h (3x3 ndarray), x (2x1 tuple)
Output: computed_x_prime (2x1 tuple)
Purpose: given x and h, return x' = hx'''
def homogenizePoint(h, x):
    homo_x = [x[0], x[1], 1]
    homo_x = np.asarray(homo_x, order='F')
    computed_x_prime = h@homo_x
    return (computed_x_prime[0] / computed_x_prime[2], computed_x_prime[1] / computed_x_prime[2])


'''estiamteHomography(correspondences)
Input: List of correspondences, (X, X')
Ouput: H: (3x3 ndarray)
Purpose: Given 7 point correspondences, estimate H'''
def estimateHomography(correspondences):
    pair0 = correspondences[0]
    pair1 = correspondences[1]
    pair2 = correspondences[2]
    pair3 = correspondences[3]
    pair4 = correspondences[4]
    pair5 = correspondences[5]
    pair6 = correspondences[6]

    x0 = pair0[0][0]
    y0 = pair0[0][1]
    x_prime_0 = pair0[1][0]
    y_prime_0 = pair0[1][1]

    x1 = pair1[0][0]
    y1 = pair1[0][1]
    x_prime_1 = pair1[1][0]
    y_prime_1 = pair1[1][1]

    x2 = pair2[0][0]
    y2 = pair2[0][1]
    x_prime_2 = pair2[1][0]
    y_prime_2 = pair2[1][1]

    x3 = pair3[0][0]
    y3 = pair3[0][1]
    x_prime_3 = pair3[1][0]
    y_prime_3 = pair3[1][1]

    x4 = pair4[0][0]
    y4 = pair4[0][1]
    x_prime_4 = pair4[1][0]
    y_prime_4 = pair4[1][1]

    x5 = pair5[0][0]
    y5 = pair5[0][1]
    x_prime_5 = pair5[1][0]
    y_prime_5 = pair5[1][1]

    x6 = pair6[0][0]
    y6 = pair6[0][1]
    x_prime_6 = pair6[1][0]
    y_prime_6 = pair6[1][1]

    A = np.zeros((14,8))

    '''Row 1'''
    A[0][0] = x0
    A[0][1] = y0
    A[0][2] = 1
    A[0][6] = -1 * x0 * x_prime_0
    A[0][7] = -1 * y0 * x_prime_0

    '''Row 2'''
    A[1][3] = x0
    A[1][4] = y0
    A[1][5] = 1
    A[1][6] = -1 * x0 * y_prime_0
    A[1][7] = -1 * y0 * y_prime_0

    '''Row 3'''
    A[2][0] = x1
    A[2][1] = y1
    A[2][2] = 1
    A[2][6] = -1 * x1 * x_prime_1
    A[2][7] = -1 * y1 * x_prime_1

    '''Row 4'''
    A[3][3] = x1
    A[3][4] = y1
    A[3][5] = 1
    A[3][6] = -1 * x1 * y_prime_1
    A[3][7] = -1 * y1 * y_prime_1

    '''Row 5'''
    A[4][0] = x2
    A[4][1] = y2
    A[4][2] = 1
    A[4][6] = -1 * x2 * x_prime_2
    A[4][7] = -1 * y2 * x_prime_2

    '''Row 6'''
    A[5][3] = x2
    A[5][4] = y2
    A[5][5] = 1
    A[5][6] = -1 * x2 * y_prime_2
    A[5][7] = -1 * y2 * y_prime_2

    '''Row 7'''
    A[6][0] = x3
    A[6][1] = y3
    A[6][2] = 1
    A[6][6] = -1 * x3 * x_prime_3
    A[6][7] = -1 * y3 * x_prime_3

    '''Row 8'''
    A[7][3] = x3
    A[7][4] = y3
    A[7][5] = 1
    A[7][6] = -1 * x3 * y_prime_3
    A[7][7] = -1 * y3 * y_prime_3

    '''Row 9'''
    A[8][0] = x4
    A[8][1] = y4
    A[8][2] = 1
    A[8][6] = -1 * x4 * x_prime_4
    A[8][7] = -1 * y4 * x_prime_4

    '''Row 10'''
    A[9][3] = x4
    A[9][4] = y4
    A[9][5] = 1
    A[9][6] = -1 * x4 * y_prime_4
    A[9][7] = -1 * y4 * y_prime_4

    '''Row 11'''
    A[10][0] = x5
    A[10][1] = y5
    A[10][2] = 1
    A[10][6] = -1 * x5 * x_prime_5
    A[10][7] = -1 * y5 * x_prime_5

    '''Row 12'''
    A[11][3] = x5
    A[11][4] = y5
    A[11][5] = 1
    A[11][6] = -1 * x5 * y_prime_5
    A[11][7] = -1 * y5 * y_prime_5

    '''Row 13'''
    A[12][0] = x6
    A[12][1] = y6
    A[12][2] = 1
    A[12][6] = -1 * x6 * x_prime_6
    A[12][7] = -1 * y6 * x_prime_6

    '''Row 14'''
    A[13][3] = x6
    A[13][4] = y6
    A[13][5] = 1
    A[13][6] = -1 * x6 * y_prime_6
    A[13][7] = -1 * y6 * y_prime_6

    b = [x_prime_0, y_prime_0,
         x_prime_1, y_prime_1,
         x_prime_2, y_prime_2,
         x_prime_3, y_prime_3,
         x_prime_4, y_prime_4,
         x_prime_5, y_prime_5,
         x_prime_6, y_prime_6]
    b = np.asarray(b, order='F')

    h = np.linalg.inv(A.T@A)@A.T@b

    homography = np.ones((3,3))
    homography[0][0] = h[0]
    homography[0][1] = h[1]
    homography[0][2] = h[2]
    homography[1][0] = h[3]
    homography[1][1] = h[4]
    homography[1][2] = h[5]
    homography[2][0] = h[6]
    homography[2][1] = h[7]
    homography[2][2] = 1

    return homography


'''getValidPoints(pts, Hpts, w, h)
Input: pts, homogoneized points, width, height
Output: valid pts and homogenized points
Purpose: Given a pts and hpts, compute valid versions of both'''
def getValidPoints(pts, Hpts, w, h):
    xmin = Hpts[:, 0] >= 0
    Hpts = Hpts[xmin, :]
    pts = pts[xmin, :]

    xmax = Hpts[:, 0] <= w
    Hpts = Hpts[xmax, :]
    pts = pts[xmax, :]

    ymin = Hpts[:, 1] >= 0
    Hpts = Hpts[ymin, :]
    pts = pts[ymin, :]

    ymax = Hpts[:, 1] <= h
    Hpts = Hpts[ymax, :]
    pts = pts[ymax, :]
    return pts, Hpts


'''map_pixel(panorama, img, H)
Input: panorama mask, an image we want to stitch into panorama, homography H
Output: panorama stitched with img
Purpose: Given a panorama, img, and h, stitch image onto panorama'''
def map_pixel(panorama, img, H):
    h = img.shape[0]
    w = img.shape[1]

    h_panorama = panorama.shape[0]
    w_panorama = panorama.shape[1]

    H = np.linalg.inv(H)

    X,Y = np.meshgrid(np.arange(0, w_panorama, 1), np.arange(0, h_panorama, 1))
    pts = np.vstack((X.ravel(), Y.ravel())).T
    pts = np.hstack((pts[:, 0:2], pts[:, 0:1] * 0 + 1))

    Hpts = H@pts.T

    Hpts = Hpts / Hpts[2,:]
    Hpts = Hpts.T[:, 0:2].astype('int')

    valid_pts, valid_Hpts = getValidPoints(pts, Hpts, w-1, h-1)

    for i in range(valid_pts.shape[0]):
        if not (panorama[valid_pts[i, 1], valid_pts[i, 0]] != 0).all():
            panorama[valid_pts[i, 1], valid_pts[i, 0]] = img[valid_Hpts[i, 1], valid_Hpts[i, 0]]
    return panorama


'''getPanorama(list_best_homography)
Input: List of homographies and images
Output: final panorama
Purpose: Given list of all images and homographies, stitch images together'''
def getPanorama(list_best_homography, img_list):
    H_to_mid = np.eye(3)
    for i in range(2, 4):
        H_to_mid = H_to_mid@np.linalg.inv(list_best_homography[i])
        list_best_homography[i] = H_to_mid
    H_to_mid = np.eye(3)
    for i in range(1, -1, -1):
        H_to_mid = H_to_mid@list_best_homography[i]
        list_best_homography[i] = H_to_mid

    list_best_homography.insert(2, np.eye(3))
    correction = 0
    for i in range(2):
        correction += img_list[i].shape[1]
    H_correction = np.array([[1, 0, correction] ,[0, 1, 0], [0, 0, 1]],dtype=float)

    height = 0
    width = 0
    for i in range(5):
        height = max(height, img_list[i].shape[0])
        width += img_list[i].shape[1]

    panorama = np.zeros((height, width, 3), np.uint8)

    for i in range(5):
        H = H_correction@list_best_homography[i]
        panorama = map_pixel(panorama, img_list[i], H)
    return panorama


'''getJacobian(H, correspondences)
Input: H and correspondence points
Output: Jacobian Matrix
Purpse: Given H and the correspondences from RANSAC, compute Jacobian'''
def getJacobian(H, correspondences):
    numPoints = len(correspondences)
    J = np.zeros((2*numPoints, 9))
    for i in range(0, len(correspondences)):
        single_correspondence = correspondences[i]
        domain_point = single_correspondence[0]
        x = domain_point[0]
        y = domain_point[1]

        denom = (H[2][0] * x) + (H[2][1] * y) + H[2][2]
        num1 = (H[0][0] * x) + (H[0][1] * y) + H[0][2]
        num2 = (H[1][0] * x) + (H[1][1] * y) + H[1][2]

        J[2*i][0] = np.divide(x, denom)
        J[2*i][1] = np.divide(y, denom)
        J[2*i][2] = np.divide(1, denom)
        J[2*i][6] = np.divide(-1 * x * num1, denom ** 2)
        J[2*i][7] = np.divide(-1 * y * num1, denom ** 2)
        J[2*i][8] = np.divide(-1 * num1, denom ** 2)

        J[2*i + 1][3] = np.divide(x, denom)
        J[2*i + 1][4] = np.divide(y, denom)
        J[2*i + 1][5] = np.divide(1, denom)
        J[2*i + 1][6] = np.divide(-1 * x * num2, denom ** 2)
        J[2*i + 1][7] = np.divide(-1 * y * num2, denom ** 2)
        J[2*i + 1][8] = np.divide(-1 * num2, denom ** 2)

    return J


'''performLM(H, correspondences)
Input: initial homography, list of correspondences
Output: refined homography
Purpose: Given H and a list of valid correspondences, refine H with LM'''
def performLM(H, correspondences):
    k = 0
    k_max = 50
    v = 2
    J = getJacobian(H, correspondences)
    A = J.T@J
    tau = 0.5
    rho = -10000000

    x = list()
    fp = list()
    p = deepcopy(H.flatten())

    #build x and fp
    for correspondence in correspondences:
        domain_pt = correspondence[0]
        range_pt  = correspondence[1]

        # homogenize the domain points to get fp
        fp_point = homogenizePoint(H, domain_pt)
        fp.append(fp_point[0])
        fp.append(fp_point[1])

        #range points just stay range points
        x.append(range_pt[0])
        x.append(range_pt[1])

    x = np.asarray(x, order='F')
    fp = np.asarray(fp, order='F')
    assert(x.shape == fp.shape)

    # error associated with "ground truth" from ransac range points
    # and applying estimated H to domain points
    error_p = np.subtract(x, fp)

    g = J.T@error_p

    mu = np.max(np.diag(A)) * tau


    while(k < k_max):
        k += 1

        while (rho < 0):

            mu_vector = mu * np.eye(9)
            a_ui = A + mu_vector
            delta_p = np.linalg.pinv(a_ui)@g
            p_new = deepcopy(p) + delta_p

            fp = list()
            for correspondence in correspondences:
                domain_pt = correspondence[0]
                range_pt  = correspondence[1]

                # homogenize the domain points to get fp
                fp_point = homogenizePoint(np.reshape(p_new, (3,3)), domain_pt)
                fp.append(fp_point[0])
                fp.append(fp_point[1])

            fp = np.asarray(fp, order='F')

            rho_num = (np.linalg.norm(error_p) - np.linalg.norm(x - fp))
            rho_denom = (delta_p.T@(mu*delta_p + g))

            rho = rho_num / rho_denom
            print(rho)
            if rho > 0:
                p = deepcopy(p_new)
                J = getJacobian(np.reshape(p, (3,3)), correspondences)
                A = J.T@J
                error_p = np.subtract(x, fp)
                mu = mu * np.max([1/3, 1-(2*rho-1) ** 3])
                v = 2
            else:
                mu = mu * v
                v = v * 2
    return np.reshape(p, (3,3))








