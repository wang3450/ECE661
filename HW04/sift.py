import cv2
import sys
import numpy as np

'''getImage(imageSet:str, imageNum:int) -> tuple
Input: imageSet (str), imageNum (int)
Output: raw_image (ndarray), grey_scale_image (ndarray)
Purpose: given imageSet and imageNum return the proper raw and grey-scaled images'''
def getImage(imageSet:str) -> tuple:
    if imageSet == "book":
        raw_input_image1 = cv2.imread("/Users/wang3450/Desktop/ECE661/HW04/input_images/books_1.jpeg",
                                      cv2.IMREAD_UNCHANGED)
        raw_input_image2 = cv2.imread("/Users/wang3450/Desktop/ECE661/HW04/input_images/books_2.jpeg",
                                      cv2.IMREAD_UNCHANGED)

        h1, w1, _ = raw_input_image1.shape
        raw_input_image2 = cv2.resize(raw_input_image2, (w1,h1), cv2.INTER_AREA)

        grey_input_image1 = cv2.cvtColor(raw_input_image1, cv2.COLOR_BGR2GRAY)
        grey_input_image2 = cv2.cvtColor(raw_input_image2, cv2.COLOR_BGR2GRAY)
        return raw_input_image1, grey_input_image1, raw_input_image2, grey_input_image2
    elif imageSet == "fountain":
        raw_input_image1 = cv2.imread("/Users/wang3450/Desktop/ECE661/HW04/input_images/fountain_1.jpg",
                                      cv2.IMREAD_UNCHANGED)
        raw_input_image2 = cv2.imread("/Users/wang3450/Desktop/ECE661/HW04/input_images/fountain_2.jpg",
                                      cv2.IMREAD_UNCHANGED)

        h1, w1, _ = raw_input_image1.shape
        raw_input_image2 = cv2.resize(raw_input_image2, (w1, h1), cv2.INTER_AREA)

        grey_input_image1 = cv2.cvtColor(raw_input_image1, cv2.COLOR_BGR2GRAY)
        grey_input_image2 = cv2.cvtColor(raw_input_image2, cv2.COLOR_BGR2GRAY)
        return raw_input_image1, grey_input_image1, raw_input_image2, grey_input_image2
    elif imageSet == "checkerboard":
        raw_input_image1 = cv2.imread("/Users/wang3450/Desktop/ECE661/HW04/input_images/checkerboard_1.jpg",
                                      cv2.IMREAD_UNCHANGED)
        raw_input_image2 = cv2.imread("/Users/wang3450/Desktop/ECE661/HW04/input_images/checkerboard_2.jpg",
                                      cv2.IMREAD_UNCHANGED)

        h1, w1, _ = raw_input_image1.shape
        raw_input_image2 = cv2.resize(raw_input_image2, (w1, h1), cv2.INTER_AREA)

        grey_input_image1 = cv2.cvtColor(raw_input_image1, cv2.COLOR_BGR2GRAY)
        grey_input_image2 = cv2.cvtColor(raw_input_image2, cv2.COLOR_BGR2GRAY)
        return raw_input_image1, grey_input_image1, raw_input_image2, grey_input_image2



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


if __name__ == "__main__":
    '''Proper Execution Checker'''
    if len(sys.argv) != 2:
        print("Incorrect Usage")
        print("Try: python3 sift.py <imageSet>")

    '''Load Images'''
    imageSet = sys.argv[1]
    raw_input_image1, grey_input_image1, raw_input_image2, grey_input_image2 = getImage(imageSet)

    kp1, kp2, descriptor1, descriptor2 = getFeatures(grey_input_image1, grey_input_image2)
    matches = getMatches(descriptor1, descriptor2)
    match_img = cv2.drawMatches(raw_input_image1, kp1, raw_input_image2, kp2, matches, raw_input_image2, flags=2 )

    cv2.imwrite(f'{imageSet}_sift.jpg', match_img)
    cv2.imshow("test", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
