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
        edge = cv2.Canny(img, 450, 450)
        edge_img_list.append(edge)
    return edge_img_list


"""performHoughTransform(edge_img_list)
Input: list of edge maps
Output: list of hough lines
Purpose: Given a list of edge maps, return a list of hough lines"""
def performHoughTransform(edge_img_list):
    hough_lines_list = list()
    for img in edge_img_list:
        line = cv2.HoughLines(img, 1, np.pi / 180, 60)
        hough_lines_list.append(line)
    return hough_lines_list


'''draw_hough_lines(line, img)
Input: line (list), img (np.ndarray)
Output: img (np.ndarray)
Purpose: Given a list of hough lines, draw them'''
def draw_hough_lines(line, img):
    for l in line:
        for rho, theta in l:
            L = 1000
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + L * (-b))
            y1 = int(y0 + L * (a))
            x2 = int(x0 - L * (-b))
            y2 = int(y0 - L * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img


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


'''get_Ab(r2_points, projected_points)
Input: world points, corners as lists
Output: A, b matrices
Purpose: Given x and x' determne a and b'''
def get_Ab(r2_points, projected_points):
    A = list()
    for i, j in zip(r2_points, projected_points):
        r1 = i + [1] + [0, 0, 0] + [-i[0] * j[0], -i[1] * j[0]]
        r2 = [0, 0, 0] + i + [1] + [-i[0] * j[1], -i[1] * j[1]]
        A.append([r1, r2])
    b = np.array(projected_points).reshape(-1, 1)
    return np.array(A).reshape(-1, 8), b


'''get_H(world_points, corners)
Input: x and x'
Output: h
Purpose: Given x and x', find h'''
def get_H(world_points, corners):
    A, b = get_Ab(world_points, corners)
    H = list(np.linalg.solve(A.T @ A, A.T @ b).reshape(-1))
    H.append(1)
    return np.array(H).reshape(3, 3)


'''get_V(i,j,h)
Input: index i, j and homography h
Output: 6x1 matrix
Purpose: Given i, j, h, compute Vij'''
def get_V(i, j, h):
    v = np.zeros((6, 1))
    i -= 1
    j -= 1

    v[0][0] = h[0][i] * h[0][j]
    v[1][0] = (h[0][i] * h[1][j]) + (h[1][i] * h[0][j])
    v[2][0] = h[1][i] * h[1][j]
    v[3][0] = (h[2][i] * h[0][j]) + (h[0][i] * h[2][j])
    v[4][0] = (h[2][i] * h[1][j]) + (h[1][i] * h[2][j])
    v[5][0] = h[2][i] * h[2][j]

    return v


'''ReprojectPoints(img, world_coord, corner, k, r, t)
Input: img: raw colored image
       world_cord: list of world coords
       corners: list of identified corners
       k: intrinsic parameters
       r: rotation matrix
       t: translation vector
Output: img with points, mean error, var error
Purpose: Reproject world coords onto img'''
def ReprojectPoints(img, world_coord, Corners, K, R, t):
    X_hc = np.ones((len(world_coord), 3))
    X_hc[:, :-1] = np.array(world_coord)
    X_hc = X_hc.T
    P = np.concatenate((R[:, :2], t), axis=1)
    P = K @ P
    rep_pt_hc = P @ X_hc

    rep_pt_hc = rep_pt_hc / rep_pt_hc[-1]
    rep_pt = rep_pt_hc[0:2]
    e = np.array(Corners).T - rep_pt
    e = np.linalg.norm(e, axis=0)
    mean_e = np.mean(e)
    var_e = np.var(e)

    rep_img = np.copy(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(world_coord)):
        rep_img = cv2.circle(img, (int(rep_pt[0, i]), int(rep_pt[1, i])), 2, (0, 255, 0), -1)
        rep_img = cv2.circle(img, (int(Corners[i][0]), int(Corners[i][1])), 2, (0, 0, 255), -1)
        rep_img = cv2.putText(img, str(i), (int(rep_pt[0, i]), int(rep_pt[1, i])), font, 0.5, (255, 0, 0), 1,
                              cv2.LINE_AA)
    return rep_img, mean_e, var_e


'''get_extrinsic(k,h)
Input: k: 3x3, h: 3x3
Output: R: 3x3, t: 3x1
Purpose: Given h and k (intrinsic/homo) compute extrinsic'''
def get_extrinsic(k, h):
    zeta = 1 / np.linalg.norm(np.linalg.inv(k) @ h[:, 0])

    r1 = zeta * np.linalg.inv(k) @ h[:, 0]
    r2 = zeta * np.linalg.inv(k) @ h[:, 1]
    r3 = zeta * np.cross(r1, r2)
    t = zeta * np.linalg.inv(k) @ h[:, 2]

    r1 = np.reshape(r1, (3, 1))
    r2 = np.reshape(r2, (3, 1))
    r3 = np.reshape(r3, (3, 1))
    t = np.reshape(t, (3, 1))

    R = np.hstack((r1, r2))
    R = np.hstack((R, r3))
    R = np.reshape(R, (3, 3))

    u, _, vh = np.linalg.svd(R)

    R = u @ vh

    return R, t


'''rotation2rod(R)
Input: 3x3 R rotation
Output: 3 vector rodriguez matrix
Purpose: Convert 9 dof to 3 dof rep of rotation matrix'''
def rotation2rod(R):
    phi = np.arccos((np.trace(R) - 1) / 2)
    w = (phi / (2 * np.sin(phi))) * np.array([(R[2, 1] - R[1, 2]),
                                              (R[0, 2] - R[2, 0]),
                                              (R[1, 0] - R[0, 1])])
    return (-w)


'''rod2rotation(w)
Input: 3 vector rodriguez matrix
Output: 3x3 R rotation
Purpose: Convert from 3 dof rep to 9 dof Rep'''
def rod2rotation(w):
    # make Wx from w
    Wx = np.array([[0, -1 * w[2], w[1]],
                   [w[2], 0, -1 * w[0]],
                   [-1 * w[1], w[0], 0]])
    phi = np.linalg.norm(w)
    R = np.eye(3) + (np.sin(phi) / phi) * (Wx) + ((1 - np.cos(phi)) / phi ** 2) * (Wx @ Wx)
    return (R)


'''cost_function_no_rad(p,x,x_m)
Input: p = [K,w1,t1,w2,t2,...wn,tn]
       x: list of list of corners for all images
       x_m: list of real world coordinates
Output: sum of square errors (scalar)
Purpose: cost function with no radial distortion'''
def cost_function_no_rad(p, x, x_m):
    # make K: intrinsic matrix
    a_x = p[0];
    a_y = p[1];
    s = p[2]
    x0 = p[3];
    y0 = p[4]
    K = np.array([[a_x, s, x0],
                  [0, a_y, y0],
                  [0, 0, 1]])

    num_img = int((len(p) - 5) / 6)
    N = len(x_m)
    cost = np.zeros(2 * num_img * N)
    for i in range(num_img):
        iw = p[6 * i + 5:6 * i + 8]
        it = p[6 * i + 8:6 * i + 11]
        iR = rod2rotation(iw)
        est_map = np.array([iR[:, 0].T, iR[:, 1].T, it.T])
        est_map = K @ (est_map.T)
        xij = np.array(x[i]);
        xij = xij.T
        x_m_hc = np.ones((len(x_m), 3));
        x_m_hc[:, :-1] = np.array(x_m)
        x_m_hc = x_m_hc.T
        x_hat_hc = est_map @ x_m_hc
        x_hat = np.linalg.inv(np.diag(x_hat_hc[-1, :])) @ x_hat_hc.T
        x_hat = x_hat.T
        x_hat = x_hat[:-1, :]
        temp = xij - x_hat
        cost[i * 2 * N:(i + 1) * 2 * N] = np.hstack((temp[0, :], temp[1, :]))
    return cost


'''cost_function_yes_rad
Input: p = [K,w1,t1,w2,t2,...wn,tn, k1,k2]
       x: list of list of corners for all images
       x_m: list of real world coordinates
Output: sum of square errors (scalar)
Purpose: cost function with radial distortion'''
def cost_function_yes_rad(p, x, x_m):
    a_x = p[0];
    a_y = p[1];
    s = p[2]
    x0 = p[3];
    y0 = p[4];
    k1 = p[-2];
    k2 = p[-1]
    K = np.array([[a_x, s, x0],
                  [0, a_y, y0],
                  [0, 0, 1]])
    num_img = int((len(p) - 7) / 6)
    N = len(x_m)
    cost = np.zeros(2 * num_img * N)
    for i in range(num_img):
        iw = p[6 * i + 5:6 * i + 8]
        it = p[6 * i + 8:6 * i + 11]
        iR = rod2rotation(iw)
        est_map = np.array([iR[:, 0].T, iR[:, 1].T, it.T])
        est_map = K @ est_map.T
        xij = np.array(x[i])
        xij = xij.T
        x_m_hc = np.ones((len(x_m), 3));
        x_m_hc[:, :-1] = np.array(x_m)
        x_m_hc = x_m_hc.T
        x_hat_hc = est_map @ x_m_hc
        x_hat = np.linalg.inv(np.diag(x_hat_hc[-1, :])) @ x_hat_hc.T
        x_hat = x_hat.T
        x_hat = x_hat[:-1, :]
        diff = x_hat - (np.kron(np.array([x0, y0]), np.ones((N, 1)))).T
        r_2 = np.sum(np.square(diff), axis=0)
        m = k1 * r_2 + k2 * np.square(r_2)
        m = np.vstack((m, m))
        x_hat_rad = x_hat + np.multiply(m, diff)
        temp = xij - x_hat_rad
        cost[i * 2 * N:(i + 1) * 2 * N] = np.hstack((temp[0, :], temp[1, :]))
    return cost
