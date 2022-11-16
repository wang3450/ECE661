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
    return f
