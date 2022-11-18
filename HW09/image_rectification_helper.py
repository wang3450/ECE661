import numpy as np
from copy import deepcopy

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

    return [float(world_point[0]), float(world_point[1]), float(world_point[2]), float(world_point[3])]


'''world2image(p,x)
Input: p (3x4) projection matrix
Output: len(x) = 4, world point in HC
Purpose: Project world point into image space'''
def world2image(p, x):
    temp = np.array(x)
    temp = np.reshape(temp, (4,1))
    img_point = p@temp
    return [float(img_point[0]/img_point[2]), float(img_point[1] / img_point[2])]




def cross_rep_mat(w):
    """
    Input:w 3x1 vector
    Output: W 3x3 matrix
    """
    # make Wx from w
    Wx=np.array([[0     , -1*w[2],    w[1]],
                [w[2]   , 0      , -1*w[0]],
                [-1*w[1], w[0]   ,      0]])
    return(Wx)