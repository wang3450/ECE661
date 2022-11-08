import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import sys

"""loadImages(dir_path)
Input: directory path
Output: list of grey scale images and labels
Purpose: given directory path, load images and labels"""
def loadImages(dir_path):
    raw_img_list = list()
    grey_img_list = list()
    img_labels = list()
    for filename in sorted(os.listdir(dir_path)):
        filepath = os.path.join(dir_path, filename)
        if os.path.isfile(filepath) and ('.jpg' in filepath or '.jpeg' in filepath):
            raw_img = cv2.imread(filepath)
            grey_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            raw_img_list.append(raw_img)
            grey_img_list.append(grey_img)
            img_labels.append(filename)
    return raw_img_list, grey_img_list, img_labels


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
        line = cv2.HoughLines(img, 1, np.pi / 180, 52)
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
            theta += -np.pi / 2
            if np.abs(theta) < np.pi / 4:
                h_lines.append(l)
            else:
                v_lines.append(l)
    return h_lines, v_lines


"""getCorners(v_lines, h_lines)
Input: horizontal and vertical lines as ndarrays
Output: list of 80 corners
Purpose: Given horizontal and vertical hough lines, find the corners"""
def getCorners(v_lines, h_lines):
    """y-intercept = horizontal line cross y-axis"""
    x_intercept = list()
    for i in range(v_lines.shape[0]):
        rho, theta = v_lines[i]
        x_intercept.append(np.divide(rho, np.cos(theta)))

    """y-intercept = horizontal line cross y-axis"""
    y_intercept = list()
    for i in range(h_lines.shape[0]):
        rho, theta = h_lines[i]
        y_intercept.append(np.divide(rho, np.sin(theta)))
    assert (len(x_intercept) == len(v_lines))
    assert (len(y_intercept) == len(h_lines))

    kmeans_v_lines = KMeans(n_clusters=8, random_state=0).fit(np.array(x_intercept).reshape(-1, 1))
    kmeans_h_lines = KMeans(n_clusters=10, random_state=0).fit(np.array(y_intercept).reshape(-1, 1))

    v_clustered_lines = list()
    h_clustered_lines = list()

    for i in range(8):
        v_clustered_lines.append(list(np.mean(v_lines[kmeans_v_lines.labels_ == i], axis=0)))

    for i in range(10):
        h_clustered_lines.append(list(np.mean(h_lines[kmeans_h_lines.labels_ == i], axis=0)))

    v_lines_sorted = sorted(v_clustered_lines, key=lambda x: np.abs(x[0] / np.cos(x[1])))
    h_lines_sorted = sorted(h_clustered_lines, key=lambda x: np.abs(x[0] / np.sin(x[1])))


    corner_points = list()
    for v_line in v_lines_sorted:
        v_rho, v_theta = v_line
        v_HC = np.array([np.cos(v_theta), np.sin(v_theta), -v_rho])
        v_HC = v_HC / v_HC[-1]
        for h_line in h_lines_sorted:
            h_rho, h_theta = h_line
            h_HC = np.array([np.cos(h_theta), np.sin(h_theta), -h_rho])
            h_HC = h_HC / h_HC[-1]
            point = np.cross(h_HC, v_HC)
            # print(f'v_HC: {v_HC}')
            # print(f'h_HC: {h_HC}')
            # print(f'point: {point}')
            print('\n')
            if point[-1] == 0:
                continue
            point = point / point[-1]
            corner_points.append(tuple(point[:2].astype('int')))
    return corner_points






