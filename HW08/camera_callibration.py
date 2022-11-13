#!/usr/bin/env python
# coding: utf-8

# # Zhang's Algorithm For Camera Calibration

# ### Import Statements

# In[16]:


from camera_callibration_helper import *
import cv2
import numpy as np
from copy import deepcopy
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings('ignore')


# ### Load the Images
# * raw_img_list (list): list of 40 BGR input images
# * grey_img_list (list): list of 40 grey scale input images
# * img_labels (list): list of 40 image filenames (mainly for debugging)

# In[17]:


# given_data_path = 'C:\\Users\jo_wang\Desktop\ECE661\HW08\Dataset1'
#given_data_path = "/Users/wang3450/Desktop/ECE661/HW08/Dataset1"

# given_data_path = "/home/jo_wang/Desktop/ECE661/HW08/Dataset1"
given_data_path = "/home/jo_wang/Desktop/ECE661/HW08/Dataset2"
raw_img_list, grey_img_list, img_labels = loadImages(given_data_path)
assert(len(grey_img_list) == 4)
assert(len(raw_img_list) == 4)
assert(len(img_labels) == 4)

# x = img_labels.index('Pic_1.jpg')
# y = img_labels.index('Pic_5.jpg')
# z = img_labels.index('Pic_10.jpg')
# w = img_labels.index('Pic_34.jpg')
#
# print(x,y,z,w)


# ### Apply Canny Edge Detector On Grey Scale Images
# * edge_img_list (list): list of edge maps from Canny

# In[18]:


edge_img_list = performCanny(grey_img_list)
assert(len(edge_img_list) == 4)
cv2.imwrite('canny_custom1.jpg', edge_img_list[0])
cv2.imwrite('canny_custom2.jpg', edge_img_list[1])
cv2.imwrite('canny_custom3.jpg', edge_img_list[2])
cv2.imwrite('canny_custom4.jpg', edge_img_list[3])


# ### Apply Hough Transform To all the Images
# * hough_lines_list (list): list of 40 images after applying hough transform

# In[19]:


hough_lines_list = performHoughTransform(edge_img_list)
assert(len(hough_lines_list) == len(edge_img_list))

cv2.imwrite('hough_lines_custom1.jpg', draw_hough_lines(hough_lines_list[0], deepcopy(raw_img_list[0])))
cv2.imwrite('hough_lines_custom2.jpg', draw_hough_lines(hough_lines_list[1], deepcopy(raw_img_list[1])))
cv2.imwrite('hough_lines_custom3.jpg', draw_hough_lines(hough_lines_list[2], deepcopy(raw_img_list[2])))
cv2.imwrite('hough_lines_custom4.jpg', draw_hough_lines(hough_lines_list[3], deepcopy(raw_img_list[3])))


# ### Get the corner points from selected images
# * all_corners (list): at each index, list of 80 corner points
# * the_chosen_one (list): index of images to use

# In[20]:


# the_chosen_one = [0, 35, 1, 27]
the_chosen_one = [0, 1, 2, 3]


all_corners = list()
for i in the_chosen_one:
    h_lines, v_lines = get_Horizontal_Vert_Lines(hough_lines_list[i])

    v_lines = np.array(v_lines).reshape(-1,2)
    h_lines = np.array(h_lines).reshape(-1,2)

    img = deepcopy(raw_img_list[i])
    corner_points = getCorners(v_lines, h_lines)
    if len(corner_points) == 80:
        all_corners.append(corner_points)

    for j, point in enumerate(corner_points):
        try:
            img = cv2.circle(img, point, 3, (0, 0, 255), -1)
            cv2.putText(img, str(j), (point[0]+5, point[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        except OverflowError:
            pass

    cv2.imwrite(f'points_{i+1}.jpg', img)


# ### Get world point coordinates
# * world_points (list): list of 80 world point coordinates in sorted order
# * Assumption made: squares are 20 pixels apart

# In[21]:


world_points = list()
for i in range(0, 160, 20):
    for j in range(0, 200, 20):
        world_points.append([i,j])


# ### Estimate Homographies between world points and all corners
# * all_homographies (list): list of 3x3 homographies relating world points to each image
# * DON'T DELETE THIS ONE CUZ IT WORKS FOR NOW!!!!!!

# In[22]:


all_homographies = list()
for corners in all_corners:
    h = get_H(world_points, corners)
    all_homographies.append(h)


# ### Compute W
# * W is a 3x3 matrix
# * Derived from the solution of Vb = 0
# * Use svd to solve Vb=0

# In[23]:


Big_V = np.zeros((1,6))
for h in all_homographies:
    r1 = get_V(i=1, j=2, h=h).T
    r2 = get_V(i=1,j=1,h=h).T - get_V(i=2,j=2,h=h).T
    Big_V = np.vstack((Big_V, r1))
    Big_V = np.vstack((Big_V, r2))

Big_V = Big_V[1:, :]

u, s, vh = np.linalg.svd(Big_V)
b = vh[-1]

w = np.zeros((3,3))
w[0][0] = b[0]
w[0][1] = b[1]
w[0][2] = b[3]
w[1][0] = b[1]
w[1][1] = b[2]
w[1][2] = b[4]
w[2][0] = b[3]
w[2][1] = b[4]
w[2][2] = b[5]


# 

# ### Compute Intrinsic Camera Parameters Matrix k
# * k is 3x3 matrix
# * k is based on y0, a_x, a_y, skew, x0, lambda
# 

# In[24]:


y0 = ((w[0][1] * w[0][2]) - (w[0][0] * w[1][2])) / (w[0][0] * w[1][1] - w[0][1] ** 2)
scale_lambda = w[2][2] - (w[0][2] ** 2 + y0 * (w[0][1] * w[0][2] - w[0][0] * w[1][2])) / w[0][0]
a_x = np.sqrt(np.abs((scale_lambda / w[0][0])))
a_y = np.sqrt(np.abs((scale_lambda * w[0][0]) / (w[0][0] * w[1][1] - w[0][1] **2)))
skew = (-1 * w[0][1] * (a_x ** 2) * a_y) / scale_lambda
x0 = (skew * y0) / a_y - (w[0][2] * (a_x ** 2)) / scale_lambda

k = np.zeros((3,3))
k[0][0] = a_x
k[0][1] = skew
k[0][2] = x0
k[1][1] = a_y
k[1][2] = y0
k[2][2] = 1

print(k)


# ### Compute Extrinsic Parameters

# In[25]:


all_rotations = list()
all_translations = list()

for homographies in all_homographies:
    R, t = get_extrinsic(k, homographies)
    all_rotations.append(R)
    all_translations.append(t)

print(len(all_rotations))
print(len(all_translations))
assert(len(all_rotations) == len(all_translations))
assert(len(all_rotations) == len(the_chosen_one))

print("Pic 1")
print(f'Rotation Matrix: \n{all_rotations[0]}')
print(f'Translation Matrix: \n {all_translations[0]}')
print("\n")
print("Pic 5")
print(f'Rotation Matrix: \n{all_rotations[1]}')
print(f'Translation Matrix: \n {all_translations[1]}')
print("\n")
print("Pic 10")
print(f'Rotation Matrix: \n{all_rotations[2]}')
print(f'Translation Matrix: \n {all_translations[2]}')
print("\n")
print("Pic 34")
print(f'Rotation Matrix: \n{all_rotations[3]}')
print(f'Translation Matrix: \n {all_translations[3]}')
print("\n")


# ### Reproject the World Coordinates

# In[26]:


#the_chosen_one = [0, 35, 1, 27]
corner0 = [list(i) for i in all_corners[0]]
corner1 = [list(i) for i in all_corners[1]]
corner2 = [list(i) for i in all_corners[2]]
corner3 = [list(i) for i in all_corners[3]]

all_corners_list = [corner0, corner1, corner2, corner3]

rep_img0, rep_img0_mean_e, rep_img0_var_e = ReprojectPoints(deepcopy(raw_img_list[0]),world_points,corner0,k,all_rotations[0],all_translations[0])

rep_img1, rep_img1_mean_e, rep_img1_var_e = ReprojectPoints(deepcopy(raw_img_list[1]),world_points,corner1,k,all_rotations[1],all_translations[1])

rep_img2, rep_img2_mean_e, rep_img2_var_e = ReprojectPoints(deepcopy(raw_img_list[2]),world_points,corner2,k,all_rotations[2],all_translations[2])

rep_img3, rep_img3_mean_e, rep_img3_var_e = ReprojectPoints(deepcopy(raw_img_list[3]),world_points,corner3,k,all_rotations[3],all_translations[3])

cv2.imwrite('rep_custom1.jpg', rep_img0)
cv2.imwrite('rep_custom2.jpg', rep_img1)
cv2.imwrite('rep_custom3.jpg', rep_img2)
cv2.imwrite('rep_custom4.jpg', rep_img3)

print('Pic #     Mean Error             Error Variance')
print(f'Pic_1    {rep_img0_mean_e}      {rep_img0_var_e}')
print(f'Pic_5    {rep_img1_mean_e}      {rep_img1_var_e}')
print(f'Pic_10   {rep_img2_mean_e}      {rep_img2_var_e}')
print(f'Pic_34   {rep_img3_mean_e}      {rep_img3_var_e}')


# ### Refinement of Calibration Parameters

# 1). Prepare p0 depending on whether we want to consider radial distortion
# 2). Express R as rodriguez form
# 3). Resize translations (3,1) -> (3,)
# 
# p0 is constituted by the intrinsic and extrinsic parameters
# * pack k = [a_x, a_y, s, x0, y0] into first 5 index of p
# * pack the linear least squares estimated rotational and translational matrices for each view thereafter

# In[27]:


rodrigues_rotation = list()
for R in all_rotations:
    rodrigues_rotation.append(rotation2rod(R))

translations_for_refine = [np.resize(translation, (3,)) for translation in all_translations]

'''Create p0 to be optimized (no radial distortion)'''
rad_dist = False
if rad_dist:
    k1,k2 = np.zeros(2)
    p0=np.zeros(7+6*len(the_chosen_one))
    p0[:5]=np.array([a_x,a_y,skew,x0,y0])
    for i in range(len(the_chosen_one)):
        p0[6*i+5:6*i+8]=rodrigues_rotation[i]
        p0[6*i+8:6*i+11]=translations_for_refine[i]
    p0[-2]=k1;  p0[-1]=k2
else:
    p0=np.zeros(5+6*len(the_chosen_one))
    p0[:5]=np.array([a_x,a_y,skew,x0,y0])
    for i in range(len(the_chosen_one)):
        p0[6*i+5:6*i+8]=rodrigues_rotation[i]
        p0[6*i+8:6*i+11]=translations_for_refine[i]


# Call the optimizer with:
#     * cost_function
#     * parameter to be optimized (p0)
#     * method = "lm"
#     * args = (all_corners_list, world_point)
# 
# Note: all_corners_list = [corners0, corners1, corners,2]
# where [cornersX] = [[x1,y1], [x2,y2], ..., [xn,yn]]
# 
# Optimum p_star = optim['x']
# p_star is same shape as p0

# In[28]:


if rad_dist:
    optim=least_squares(cost_function_yes_rad,p0,method='lm',args=(all_corners_list,world_points))
else:
    optim=least_squares(cost_function_no_rad,p0, method='lm',args=(all_corners_list,world_points))

p_star=optim['x']


# Unpack the intrinsic and extrinsic parameters from p_star
# * k = [a_x, a_y, s, x0, y0] located in first 5 indexes of p_star
# * unpack the refined rotational and translational matrices for each view.

# In[29]:


a_x=p_star[0]
a_y=p_star[1]
skew=p_star[2]
x0=p_star[3]
y0=p_star[4]

K_ref = np.zeros((3,3))
K_ref[0][0] = a_x
K_ref[0][1] = skew
K_ref[0][2] = x0
K_ref[1][1] = a_y
K_ref[1][2] = y0
K_ref[2][2] = 1

if rad_dist:
    k1=p_star[-2]; k2=p_star[-1]
    print('Radial Distortion parameters: k1='+str(k1)+' k2='+str(k2))

R_ref=[]
t_ref=[]
for i in range(len(the_chosen_one)):
    iw=p_star[6*i+5:6*i+8]
    it=p_star[6*i+8:6*i+11]
    iR=rod2rotation(iw)
    R_ref.append(iR)
    t_ref.append(it)


# In[30]:


t_ref[0] = np.reshape(t_ref[0], (3,1))
t_ref[1] = np.reshape(t_ref[1], (3,1))
t_ref[2] = np.reshape(t_ref[2], (3,1))
t_ref[3] = np.reshape(t_ref[3], (3,1))
refine_img0, refine_img0_mean_e, refine_img0_var_e = ReprojectPoints(raw_img_list[0],world_points,corner0,K_ref,R_ref[0],t_ref[0])
refine_img1, refine_img1_mean_e, refine_img1_var_e = ReprojectPoints(raw_img_list[1],world_points,corner1,K_ref,R_ref[1],t_ref[1])
refine_img2, refine_img2_mean_e, refine_img2_var_e = ReprojectPoints(raw_img_list[2],world_points,corner2,K_ref,R_ref[2],t_ref[2])
refine_img3, refine_img3_mean_e, refine_img3_var_e = ReprojectPoints(raw_img_list[3],world_points,corner3,K_ref,R_ref[3],t_ref[3])

cv2.imwrite('refine_no_rad_pic1.jpg', refine_img0)
cv2.imwrite('refine_no_rad_pic5.jpg', refine_img1)
cv2.imwrite('refine_no_rad_pic10.jpg', refine_img2)
cv2.imwrite('refine_no_rad_pic34.jpg', refine_img3)

print('Pic #     Mean Error             Error Variance')
print(f'Pic_1    {refine_img0_mean_e}      {refine_img0_var_e}')
print(f'Pic_5    {refine_img1_mean_e}      {refine_img1_var_e}')
print(f'Pic_10   {refine_img2_mean_e}      {refine_img2_var_e}')
print(f'Pic_34   {refine_img3_mean_e}      {refine_img3_var_e}')

