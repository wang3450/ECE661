#!/usr/bin/python
import numpy as np
from pylab import *
import cv2
from rectification import *
from features import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, ConnectionPatch

def pick_points(image):
	'''
		Prompt user to click points on image.
	'''
	points = []
	fig = plt.figure()
	def onclick(event):
		if len(points) == 8:
			return
		x = int(event.xdata)
		y = int(event.ydata)
		points.append((x,y))
		print ("Mouse clicked at (x,y)...", (x,y), "; Total number of points clicked...", len(points))
		print (points)
	imshow(image)
	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	show()

def main():
	nfeatures = 1000 # Number of SIFT features
	w,h = 2000, 1500 # Size of the canvas
	image1 = imread('/Users/wang3450/Desktop/ECE661/test_hw09/left.jpg')
	image2 = imread('/Users/wang3450/Desktop/ECE661/test_hw09/right.jpg')

	KPL = np.array([[188, 429, 444, 257, 575, 786, 736, 551],  # X cord. (--)
					[348, 534, 653, 494, 77, 164, 311, 251],  # Y cord. (|)
					[1, 1, 1, 1, 1, 1, 1, 1]])  # HC: all ones
	KPR = np.array([[181, 328, 356, 235, 626, 806, 742, 544],  # X cord. (--)
					[287, 495, 628, 441, 117, 231, 377, 269],  # Y cord. (|)
					[1, 1, 1, 1, 1, 1, 1, 1]])  # HC: all ones

	pts1_manual = [(188, 348), (429, 534), (444, 653), (257, 494), (575, 77), (786, 164), (736, 311), (551, 251)]
	pts2_manual = [(181, 287), (328, 495), (356, 628), (235, 441), (626, 117), (806, 231), (742, 377), (544, 269)]

	figure()
	subplot(1,2,1)
	imshow(image1)
	subplot(1,2,2)
	imshow(image2)
	# Uncomment to pick points if necessary
	# pick_points(image1)
	# pick_points(image2)
	# Plot the manual correspondences
	figure()
	fig, axes = subplots(1,2)
	axes[0].set_aspect('equal')
	axes[0].imshow(image1)
	axes[1].set_aspect('equal')
	axes[1].imshow(image2)
	for i in range(8):
		# color = np.random.rand(3,1)
		color = (1., 0, 0)
		pt1 = pts1_manual[i]
		pt2 = pts2_manual[i]
		axes[0].add_patch( Circle(pt1, 5, fill=False, color=color, clip_on=False) )
		axes[1].add_patch( Circle(pt2, 5, fill=False, color=color, clip_on=False) )
		# Draw lines for matching pairs
		line1 = ConnectionPatch(xyA=pt1, xyB=pt2, coordsA='data', coordsB='data', axesA=axes[0], axesB=axes[1], color=color)
		line2 = ConnectionPatch(xyA=pt2, xyB=pt1, coordsA='data', coordsB='data', axesA=axes[1], axesB=axes[0], color=color)
		axes[0].add_patch(line1)
		axes[1].add_patch(line2)
	F = get_fundamental_matrix(pts1_manual, pts2_manual)
	print ("F = ", F)
	e, ep = get_epipoles(F)
	print ("e = ", e)
	print ("ep = ", ep)
	P, Pp = get_canonical_projection_matrices(F, ep)
	print ("P = ", P)
	print ("Pp = ", Pp)
	Pp_reformated = np.zeros((3,4))
	for i in range(3):
		for j in range(4):
			if j == 3:
				Pp_reformated[i][j] = Pp[i][j]
			else:
				Pp_reformated[i][j] = float(Pp[i][j][0])
	print(Pp_reformated)
	print ("======== First Nonlinear Optimization ========")
	P_refined, Pp_refined = nonlinear_optimization(pts1_manual, pts2_manual, P, Pp_reformated)
	print ("P_refined = ", P_refined)
	print ("Pp_refined = ", Pp_refined)
	F_refined = get_fundamental_matrix_from_projection(P_refined, Pp_refined)
	F_refined_reformat = np.zeros((3,3))
	# for i in range(3):
	# 	for j in range(3):
	# 		F_refined_reformat[i][j] = float(F_refined[i][j][0])

	print ("F_refined = ", F_refined)
	e_refined, ep_refined = get_epipoles(F_refined)
	print ("e_refined = ", e_refined)
	print ("ep_refined = ", ep_refined)
	print ("======== Rectification ========")
	H, Hp =	get_rectification_homographies(image1, image2, pts1_manual, pts2_manual, e_refined, ep_refined, P_refined, Pp_refined)
	print ("H = ", H)
	print ("Hp = ", Hp)
	print ("e = ", dot(H, e_refined))
	print ("ep = ", dot(Hp, ep_refined))
	rectified1 = cv2.warpPerspective(image1, H, (w,h))
	rectified2 = cv2.warpPerspective(image2, Hp, (w,h))
	figure()
	subplot(1,2,1)
	imshow(rectified1)
	subplot(1,2,2)
	imshow(rectified2)
	print ("======== Featusre Matching ========")
	kp1, des1 = get_sift_kp_des(rectified1, nfeatures=nfeatures)
	kp2, des2 = get_sift_kp_des(rectified2, nfeatures=nfeatures)
	pts1_ft, pts2_ft, good = get_sift_matchings(kp1, des1, kp2, des2)
	for i,pt in enumerate(pts1_ft):
		pts1_ft[i] = (pt[1],pt[0])
	for i,pt in enumerate(pts2_ft):
		pts2_ft[i] = (pt[1],pt[0])
	matchings = cv2.drawMatchesKnn(rectified1,kp1,rectified2,kp2,good,array([]),flags=2)
	figure()
	imshow(matchings)
	print ("======== Final Nonlinear Optimization ========")
	P_final, Pp_final = nonlinear_optimization(pts1_ft, pts2_ft, P_refined, Pp_refined)
	print ("P_final = ", P_final)
	print ("Pp_final = ", Pp_final)
	F_final = get_fundamental_matrix_from_projection(P_final, Pp_final)
	print ("F_final = ", F_final)
	e_final, ep_final = get_epipoles(F_final)
	print ("e_final = ", e_final)
	print ("ep_final = ", ep_final)
	print ("======== Final Triangulation ========")
	pts_world = triangulate_points(P_final, Pp_final, pts1_ft, pts2_ft)
	pts_world = array(pts_world)
	fig = figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(pts_world[:,1], pts_world[:,0], pts_world[:,2], c='b', depthshade=False)
	# Also plot the original manual points
	pts1_manual_rect = apply_transformation_on_points(pts1_manual, H)
	pts2_manual_rect = apply_transformation_on_points(pts2_manual, Hp)
	pts_manual_world = triangulate_points(P_final, Pp_final, pts1_manual_rect, pts2_manual_rect)
	pts = pts_manual_world
	# 9 lines in total
	for s,e in [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (0,4), (3,5), (2,6)]:
		ax.plot([pts[s][0], pts[e][0]], [pts[s][1], pts[e][1]], zs=[pts[s][2], pts[e][2]])
	# pts_manual_world = array(pts_manual_world)
	for i,c in enumerate(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']):
		pt = pts_manual_world[i]
		ax.scatter(pt[0], pt[1], pt[2], c=c, s=80, depthshade=False)
	show()

if __name__ == '__main__':
	main()