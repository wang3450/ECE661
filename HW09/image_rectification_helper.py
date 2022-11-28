import numpy as np
from copy import deepcopy
from tqdm import tqdm
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


'''mat_mul(T, x)
Input: T (3x3 ndarray), x (list 2)
Output: list[x1/x3, x2/x3]
Purpose: tx'''
def T_mul(T:np.ndarray, x:list):
    temp = deepcopy(x)
    temp.append(1)
    temp = np.array(temp).reshape((3,1))
    x_hat = T@temp
    assert(x_hat.shape == (3,1))
    return [float(x_hat[0] / x_hat[2]), float(x_hat[1] / x_hat[2])]


'''compute_F(x, x_prime)
Input: list of x and x_prime points
Output: F (3x3 ndarray)
Purpose: Given 8 point corr, compute F'''
def compute_F(x, x_prime):
    '''computes mean(x) mean(y) mean(d) s and T for x'''
    x_mean = np.mean([point[0] for point in x])
    y_mean = np.mean([point[1] for point in x])
    d_bar = np.mean([getDistance(point, [x_mean, y_mean]) for point in x])
    s = np.sqrt(2) / d_bar
    T = np.array([s, 0, -s*x_mean, 0, s, -s*y_mean, 0, 0, 1]).reshape((3,3))

    '''computes mean(x) mean(y) mean(d) s and T for x_prime'''
    x_prime_mean = np.mean([point[0] for point in x_prime])
    y_prime_mean = np.mean([point[1] for point in x_prime])
    d_prime_bar = np.mean([getDistance(point, [x_prime_mean, y_prime_mean]) for point in x_prime])
    s_prime = np.sqrt(2) / d_prime_bar
    T_prime = np.array([s_prime, 0, -s_prime*x_prime_mean, 0, s_prime, -s_prime*y_prime_mean, 0, 0, 1]).reshape((3,3))

    '''compute x_hat = Tx and x_prime_hat = T_prime x_prime'''
    x_hat = [T_mul(T, point) for point in x]
    x_prime_hat = [T_mul(T_prime, point) for point in x_prime]
    assert(len(x_hat) == len(x_prime_hat))
    assert(len(x_hat) == 8)

    '''compute F_hat using linear least squares'''
    A = np.zeros((8,9))
    for i in range(len(x)):
        A[i] = [x_prime[i][0] * x[i][0],
                x_prime[i][0] * x[i][1],
                x_prime[i][0],
                x_prime[i][1] * x[i][0],
                x_prime[i][1] * x[i][1],
                x_prime[i][1],
                x[i][0],
                x[i][1],
                1]

    _, _, vh = np.linalg.svd(A)
    f_hat = np.reshape(vh[-1], (3,3))

    '''condition f_hat into f_prime_hat'''
    u, d, newVh = np.linalg.svd(f_hat)
    d[2] = 0
    d = np.diag(d)
    f_prime_hat = u@d@newVh

    '''denormalize f_prime_hat into f'''
    f = T_prime.T@f_prime_hat@T
    return f/f[-1,-1]


'''compute_e(F)
Input: F (3x3 ndarray)
Output: e, e'
Purpose: Given F, compute epipoles'''
def compute_e(F:np.ndarray):
    u, _, vh = np.linalg.svd(F)
    e = np.transpose(vh[-1,:])
    e_prime = u[:, -1]
    e = e / e[2]
    e_prime = e_prime / e_prime[2]
    return e, e_prime


'''compute_P(e_prime, F)
Input: e_prime (3,) ndarray
       F (3,3) ndarray
Output: P, Prime (3,4) ndarray
Purpose: Estimate the Projection Matrices'''
def compute_P(e_prime, F):
    P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    e_prime_matrix = np.array([[0, -e_prime[2], e_prime[1]], [e_prime[2], 0, -e_prime[0]], [-e_prime[1], e_prime[0], 0]])
    e_prime = np.reshape(e_prime, (3, 1))
    P_prime = np.hstack((e_prime_matrix@F, e_prime))

    assert(np.linalg.matrix_rank(P) == 3)
    assert(np.linalg.matrix_rank(P_prime) == 3)

    return P, P_prime


'''triangulate(P, P_prrime, x_points, x_prime)
Input: P (3x4 ndarray)
       P' (3x4 ndarray)
       x_points (list of points in left image)
       x_prime (list of points in right image)
Output: World point X in physical form (3 vector)
Purpose: Given P, P', find world point of (x,x') '''
def triangulate(P, P_prime, x_points, x_prime):
    A = np.zeros((4,4))
    ''' get the rows of p and p' '''
    p1 = P[0,:]
    p2 = P[1,:]
    p3 = P[2,:]
    p_1_prime = P_prime[0,:]
    p_2_prime = P_prime[1,:]
    p_3_prime = P_prime[2:,]

    '''populate the A matrix'''
    A[0] = (x_points[0] * p3) - p1
    A[1] = (x_points[1] * p3) - p2
    A[2] = (x_prime[0] * p_3_prime) - p_1_prime
    A[3] = (x_prime[1] * p_3_prime) - p_2_prime

    '''X is smallest eigenvector of A^TA'''
    _, _, vh = np.linalg.svd(A)
    world_point = vh[-1]
    world_point = world_point / world_point[-1]

    return [float(world_point[0]), float(world_point[1]), float(world_point[2])]


'''world2image(p,x)
Input: p (3x4) projection matrix
       x (length 3 iteratable)
Output: len(x) = 2 , world point in physical
Purpose: Project world point into image space'''
def world2image(p, x):
    temp = list(x)
    temp.append(1)
    temp = np.array(temp)
    temp = np.reshape(temp, (4,1))
    img_point = p@temp
    return [float(img_point[0]/img_point[2]), float(img_point[1] / img_point[2])]


'''cost_function(params, x_points, x_prime_points)
Input: params: list of parameters
       x_points: [[x1,y1], ... [x8,y8]]
       x_prime_points: [[x'1,y'1], ... [x'8,y'8]]
Output: d^2_geom
Purpose: Cost Function for LM'''
def cost_function(params, x_points, x_prime_points):
    '''unpacking parameter '''
    P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P_prime = np.reshape(params[0:12], (3,4))
    world_points = list()
    for i in range(len(x_points)):
        world_points.append(params[12+3*i:12+3*(i+1)])

    x_hat = [world2image(P, i) for i in world_points]
    x_prime_hat = [world2image(P_prime, i) for i in world_points]

    diff_x_hat = np.subtract(x_points, x_hat)
    diff_x_prime_hat = np.subtract(x_prime_points, x_prime_hat)

    cost = np.hstack((diff_x_hat[:,0], diff_x_hat[:,1], diff_x_prime_hat[:,0], diff_x_prime_hat[:,1]))

    return cost


'''def get_H_prime(img, e_prime)
Input: img (ndarray): right image
       e_prime (3, ndaray): refined right epipole
Output: H_prime (3x3 ndarray)
Purpose: Compute H that rectifies right image'''
def get_H_prime(img, e_prime):
    h, w, _ = img.shape

    '''Compute T1'''
    T1= np.array([[1.0, 0.0, -(w/2)],
                  [0.0, 1.0, -(h/2)],
                  [0.0, 0.0, 1.0]])

    '''Compute T2'''
    T2= np.array([[1.0, 0.0, (w/2)],
                  [0.0, 1.0, (h/2)],
                  [0.0, 0.0, 1.0]])

    '''Compute R, f'''
    x0=w/2
    y0=h/2
    e_prime = e_prime/e_prime[-1]
    ex = e_prime[0]
    ey = e_prime[1]
    t = np.arctan(-(ey-y0)/(ex-x0)) # theta
    R=np.array([[np.cos(t), -1*np.sin(t),    0],
                [np.sin(t),   np.cos(t),    0],
                [0        ,   0        ,    1]])
    f=np.abs((ex-x0)*np.cos(t)-(ey-y0)*np.sin(t))

    '''Compute G'''
    G= np.array([[1   , 0, 0],
                 [0   , 1, 0],
                 [-1/f, 0, 1]])

    H_prime = T2@G@R@T1

    return H_prime, f


'''def get_H(P, P_prime, H_prime, x_points, x_prime_points)
Input: P: [3x4] matrix
       P_prime: refined p'
       H_prime: right rectifying H
       x_points: pts in left image
       x_prime_points: pts in right image
Output: H 
Purpose: Compute H to rectify left image'''
def get_H(P, P_prime, H_prime, x_points, x_prime_points):
    '''compute pseudo inverse of P'''
    P_pinv = P.T@(np.linalg.inv(P@P.T))

    '''compute H_0'''
    M = P_prime @ P_pinv
    H_0 = H_prime@M
    '''Transform Points'''
    x_hat = np.array([T_mul(H_0, i) for i in x_points])
    x_prime_hat = np.array([T_mul(H_prime, i) for i in x_prime_points])

    '''Solve For H_a'''
    A = np.ones((x_hat.shape[0],3))
    A[:,:-1]= x_hat
    b = x_prime_hat[:,0]
    # solve Ax-b = 0 using left Psuedo inverse
    a = (np.linalg.inv((A.T)@A)@A.T)@b
    Ha = np.eye(3)
    Ha[0,:]=a
    print(f'Ha: \n{Ha}')
    H = Ha@H_0
    return H, H_0


def transformInputImage(source_image, H):
    input_image = np.transpose(source_image, (1, 0, 2))
    new_image_shape_width = input_image.shape[0]
    new_image_shape_height = input_image.shape[1]
    grid = np.array(np.meshgrid(np.arange(new_image_shape_width), np.arange(new_image_shape_height), np.arange(1, 2)))
    combinations = np.transpose(grid.T.reshape(-1, 3))
    homography_inverse = np.linalg.inv(H)
    new_homogenous_coord = homography_inverse.dot(combinations)
    new_homogenous_coord = new_homogenous_coord / new_homogenous_coord[2]

    min_values = np.amin(new_homogenous_coord, axis=1)
    min_values = np.round(min_values).astype(int)
    max_values = np.amax(new_homogenous_coord, axis=1)
    max_values = np.round(max_values).astype(int)
    required_image_size = max_values - min_values
    required_image_width = int(required_image_size[0])
    required_image_height = int(required_image_size[1])

    aspect_ratio = required_image_width / required_image_height
    scaled_height = 3000
    scaled_width = scaled_height * aspect_ratio
    scaled_width = int(round(scaled_width))
    s = required_image_height / scaled_height
    # Initialize the new frame
    img_replication = np.zeros((scaled_width, scaled_height, 3), np.uint8) * 255
    # Look at all pixels in the new frame: does the homography map them into the original image frame?
    x_values_in_target = np.arange(scaled_width)
    y_values_in_target = np.arange(scaled_height)
    target_grid = np.array(np.meshgrid(x_values_in_target, y_values_in_target, np.arange(1, 2)))
    combinations_in_target = np.transpose(target_grid.T.reshape(-1, 3))
    target_combinations_wo_offset = deepcopy(combinations_in_target)
    combinations_in_target[0] = combinations_in_target[0] * s + min_values[0]
    combinations_in_target[1] = combinations_in_target[1] * s + min_values[1]
    target_new_homogenous_coord = H.dot(combinations_in_target)
    target_new_homogenous_coord = target_new_homogenous_coord / target_new_homogenous_coord[2]

    x_coordinates_rounded = np.round(target_new_homogenous_coord[0]).astype(int)
    y_coordinates_rounded = np.round(target_new_homogenous_coord[1]).astype(int)
    # Check conditions on x
    x_coordinates_greater_zero = x_coordinates_rounded > 0
    x_coordinates_smaller_x_shape = x_coordinates_rounded < new_image_shape_width
    x_coordinates_valid = x_coordinates_greater_zero * x_coordinates_smaller_x_shape
    # Check conditions on y
    y_coordinates_greater_zero = y_coordinates_rounded > 0
    y_coordinates_smaller_y_shape = y_coordinates_rounded < new_image_shape_height
    y_coordinates_valid = y_coordinates_greater_zero * y_coordinates_smaller_y_shape

    valid_coordinates = x_coordinates_valid * y_coordinates_valid
    target_valid_x = target_combinations_wo_offset[0][valid_coordinates == True]
    target_valid_y = target_combinations_wo_offset[1][valid_coordinates == True]

    list_of_x_values_target = list(target_valid_x)
    list_of_y_values_target = list(target_valid_y)
    list_of_valid_coordinate_pairs = list()

    for i in tqdm(range(len(list_of_x_values_target))):
        list_of_valid_coordinate_pairs.append([list_of_x_values_target[i], list_of_y_values_target[i]])
    valid_x = x_coordinates_rounded[valid_coordinates == True]
    valid_y = y_coordinates_rounded[valid_coordinates == True]

    list_of_x_values = list(valid_x)
    list_of_y_values = list(valid_y)

    list_of_original_coords_for_mapping = list()
    for i in tqdm(range(len(list_of_x_values))):
        list_of_original_coords_for_mapping.append([list_of_x_values[i], list_of_y_values[i]])

    j = 0
    for pair in tqdm(list_of_valid_coordinate_pairs):
        img_replication[pair[0], pair[1], :] = input_image[list_of_original_coords_for_mapping[j][0],
                                               list_of_original_coords_for_mapping[j][1], :]
        j = j + 1

    img_replication = np.transpose(img_replication, (1, 0, 2))
    return img_replication