import os
import cv2
import numpy as np

"""loadImages(dir_path)
Input: directory path
Output: list of grey scale images and labels
Purpose: given directory path, load images and labels"""
def loadImages(dir_path):
    raw_img_list = list()
    grey_img_list = list()
    for filename in sorted(os.listdir(dir_path)):
        filepath = os.path.join(dir_path, filename)
        if os.path.isfile(filepath) and ('.jpg' in filepath or '.jpeg' in filepath):
            raw_img = cv2.imread(filepath)
            grey_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            raw_img_list.append(raw_img)
            grey_img_list.append(grey_img)
    return raw_img_list, grey_img_list


"""performCanny(grey_img_list)
Input: list of grey-scale images
Output: list of canny edge maps
Purpose: Given a list of grey scale images, apply canny on them"""
def performCanny(grey_img_list):
    edge_img_list = list()
    for img in grey_img_list:
        edge = cv2.Canny(img, 300, 300)
        edge_img_list.append(edge)
    return edge_img_list


"""performHoughTransform(edge_img_list)
Input: list of edge maps
Output: list of hough lines
Purpose: Given a list of edge maps, return a list of hough lines"""
def performHoughTransform(edge_img_list):
    hough_lines_list = list()
    for img in edge_img_list:
        line = cv2.HoughLines(img, 1, np.pi/180, 52)
        hough_lines_list.append(line)
    return hough_lines_list


"""get_Horizontal_Vert_Lines(lines)
Input: Hough lines for a single image as a list
Output: list of hor and vert lines
Purpose: separate hor and vert hough lines"""
def get_Horizontal_Vert_Lines(lines):
    h_lines = list()
    v_lines = list()
    for l in lines:
        for rho, theta in l:
            theta += -np.pi/2
            if np.abs(theta) < np.pi/4:
                h_lines.append(l)
            else:
                v_lines.append(l)
    return h_lines, v_lines
