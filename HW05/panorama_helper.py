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


'''getNewFrame(baseImg, branchImage, H)
Input: 2 images and the h that relates them
Output: new frame
Purpose: Given two images we wish to stitch together, compute the new frame'''
def getNewFrame(baseImage, branchImage, H):
    branch_height, branch_width = branchImage.shape[:2]
    base_height, base_width = baseImage.shape[:2]

    initMatrix = np.array([[0, branch_width-1, branch_width-1, 0],
                           [0, 0, branch_height-1, branch_height-1],
                           [1, 1, 1, 1]])
    [x, y, z] = H@initMatrix
    x = np.divide(x,z)
    y = np.divide(y,z)

    x_min = int(round(min(x)))
    x_max = int(round(max(x)))
    y_min = int(round(min(y)))
    y_max = int(round(max(x)))

    new_width = x_max
    new_height = y_max
    correction_xy = [0, 0]

    if x_min < 0:
        new_width = new_width - x_min
        correction_xy[0] = abs(x_min)
    if new_width < base_width + correction_xy[0]:
        new_width = base_width + correction_xy[0]
    if y_min < 0:
        new_height = new_height - y_min
        correction_xy[1] = abs(y_min)
    if new_height < base_height + correction_xy[1]:
        new_height = base_height + correction_xy[1]

    x = np.add(x, correction_xy[0])
    y = np.add(y, correction_xy[1])

    OldInitialPoints = np.float32([[0, 0],
                                   [branch_width - 1, 0],
                                   [branch_width - 1, branch_height - 1],
                                   [0, branch_height - 1]])
    NewFinalPonts = np.float32(np.array([x, y]).transpose())

    newH = cv2.getPerspectiveTransform(OldInitialPoints, NewFinalPonts)

    return [new_height, new_width], correction_xy, newH


'''stitchImage(baseImage, branchImage)
Input: 2 input images and homography relating them
Output: stitched image
Purpose: stitch branchImage onto baseImage'''
def stitchImage(baseImage, branchImage):
    # Applying Cylindrical projection on SecImage
    branchImage_cylinder, mask_x, mask_y = ProjectOntoCylinder(branchImage)

    # Getting SecImage Mask
    branchImage_mask = np.zeros(branchImage_cylinder.shape, dtype=np.uint8)
    branchImage_mask[mask_y, mask_x, :] = 255

    Matches, BaseImage_kp, SecImage_kp = FindMatches(baseImage, branchImage_cylinder)
    HomographyMatrix, Status = FindHomography(Matches, BaseImage_kp, SecImage_kp)


    newFrame, correction, improvedHomography = getNewFrame(baseImage, branchImage, HomographyMatrix)

    SecImage_Transformed = cv2.warpPerspective(branchImage_cylinder, improvedHomography, (newFrame[1], newFrame[0]))
    SecImage_Transformed_Mask = cv2.warpPerspective(branchImage_mask, improvedHomography, (newFrame[1], newFrame[0]))
    BaseImage_Transformed = np.zeros((newFrame[0], newFrame[1], 3), dtype=np.uint8)
    BaseImage_Transformed[correction[1]:correction[1] + baseImage.shape[0], correction[0]:correction[0] + baseImage.shape[1]] = baseImage

    StitchedImage = cv2.bitwise_or(SecImage_Transformed,cv2.bitwise_and(BaseImage_Transformed, cv2.bitwise_not(SecImage_Transformed_Mask)))

    return StitchedImage


def Convert_xy(x, y):
    global center, f

    xt = (f * np.tan((x - center[0]) / f)) + center[0]
    yt = ((y - center[1]) / np.cos((x - center[0]) / f)) + center[1]

    return xt, yt


def ProjectOntoCylinder(InitialImage):
    global w, h, center, f
    h, w = InitialImage.shape[:2]
    center = [w // 2, h // 2]
    f = 1100  # 1100 field; 1000 Sun; 1500 Rainier; 1050 Helens

    # Creating a blank transformed image
    TransformedImage = np.zeros(InitialImage.shape, dtype=np.uint8)

    # Storing all coordinates of the transformed image in 2 arrays (x and y coordinates)
    AllCoordinates_of_ti = np.array([np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x = AllCoordinates_of_ti[:, 0]
    ti_y = AllCoordinates_of_ti[:, 1]

    # Finding corresponding coordinates of the transformed image in the initial image
    ii_x, ii_y = Convert_xy(ti_x, ti_y)

    # Rounding off the coordinate values to get exact pixel values (top-left corner)
    ii_tl_x = ii_x.astype(int)
    ii_tl_y = ii_y.astype(int)

    # Finding transformed image points whose corresponding
    # initial image points lies inside the initial image
    GoodIndices = (ii_tl_x >= 0) * (ii_tl_x <= (w - 2)) * \
                  (ii_tl_y >= 0) * (ii_tl_y <= (h - 2))

    # Removing all the outside points from everywhere
    ti_x = ti_x[GoodIndices]
    ti_y = ti_y[GoodIndices]

    ii_x = ii_x[GoodIndices]
    ii_y = ii_y[GoodIndices]

    ii_tl_x = ii_tl_x[GoodIndices]
    ii_tl_y = ii_tl_y[GoodIndices]

    # Bilinear interpolation
    dx = ii_x - ii_tl_x
    dy = ii_y - ii_tl_y

    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx) * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx) * (dy)

    TransformedImage[ti_y, ti_x, :] = (weight_tl[:, None] * InitialImage[ii_tl_y, ii_tl_x, :]) + \
                                      (weight_tr[:, None] * InitialImage[ii_tl_y, ii_tl_x + 1, :]) + \
                                      (weight_bl[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x, :]) + \
                                      (weight_br[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x + 1, :])

    # Getting x coorinate to remove black region from right and left in the transformed image
    min_x = min(ti_x)

    # Cropping out the black region from both sides (using symmetricity)
    TransformedImage = TransformedImage[:, min_x: -min_x, :]

    return TransformedImage, ti_x - min_x, ti_y


def FindHomography(Matches, BaseImage_kp, SecImage_kp):
    # If less than 4 matches found, exit the code.
    if len(Matches) < 4:
        print("\nNot enough matches found between the images.\n")
        exit(0)

    # Storing coordinates of points corresponding to the matches found in both the images
    BaseImage_pts = []
    SecImage_pts = []
    for Match in Matches:
        BaseImage_pts.append(BaseImage_kp[Match[0].queryIdx].pt)
        SecImage_pts.append(SecImage_kp[Match[0].trainIdx].pt)

    # Changing the datatype to "float32" for finding homography
    BaseImage_pts = np.float32(BaseImage_pts)
    SecImage_pts = np.float32(SecImage_pts)

    # Finding the homography matrix(transformation matrix).
    (HomographyMatrix, Status) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)

    return HomographyMatrix, Status


def FindMatches(BaseImage, SecImage):
    # Using SIFT to find the keypoints and decriptors in the images
    Sift = cv2.SIFT_create()
    BaseImage_kp, BaseImage_des = Sift.detectAndCompute(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
    SecImage_kp, SecImage_des = Sift.detectAndCompute(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)

    # Using Brute Force matcher to find matches.
    BF_Matcher = cv2.BFMatcher()
    InitialMatches = BF_Matcher.knnMatch(BaseImage_des, SecImage_des, k=2)

    # Applytng ratio test and filtering out the good matches.
    GoodMatches = []
    for m, n in InitialMatches:
        if m.distance < 0.75 * n.distance:
            GoodMatches.append([m])

    return GoodMatches, BaseImage_kp, SecImage_kp

if __name__ == "__main__":
    '''Proper Execution Checker'''
    if len(sys.argv) != 2:
        print("Incorrect Usage")
        print("Try: python3 panorama.py <imageSet>")

    '''get the input images'''
    imageSet = sys.argv[1]
    input_image_list_raw, input_image_list_grey = loadImages(imageSet)

    '''Build list of Candidate matches between images
    -each index in the list is one candidate match,
    i.e. a list of two tuples'''

    '''candidate matches between img0, img1'''
    candidate_matches_float_01, candidate_matches_int_01 = getCandidateMatches(input_image_list_grey[0], input_image_list_grey[1])
    assert(len(candidate_matches_float_01) == len(candidate_matches_int_01))

    '''candidate matches between img1, img2'''
    candidate_matches_float_12, candidate_matches_int_12 = getCandidateMatches(input_image_list_grey[1], input_image_list_grey[2])
    assert (len(candidate_matches_float_12) == len(candidate_matches_int_12))

    '''candidate matches between img2, img3'''
    candidate_matches_float_23, candidate_matches_int_23 = getCandidateMatches(input_image_list_grey[2], input_image_list_grey[3])
    assert (len(candidate_matches_float_23) == len(candidate_matches_int_23))

    '''candidate matches between img3, img4'''
    candidate_matches_float_34, candidate_matches_int_34 = getCandidateMatches(input_image_list_grey[3], input_image_list_grey[4])
    assert (len(candidate_matches_float_34) == len(candidate_matches_int_34))

    '''perform RANSAC'''
    # performRANSAC(candidate_matches_float_01, candidate_matches_int_01)
    '''Display Image to Console'''
    # cv2.imshow("0", input_image_list_raw[0])
    # cv2.imshow("2", input_image_list_raw[1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()