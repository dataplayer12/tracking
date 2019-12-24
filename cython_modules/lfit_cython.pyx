#cython.wraparound=False
#cython.boundscheck=False

# Build command: python setup.py build_ext --inplace

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand

DTYPE = np.float32

ctypedef np.float32_t DTYPE_t
# np.ndarray[DTYPE_t, ndim=1] #this is deprecated


def fit_linear(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
    #assert x.dtype==DTYPE and y.dtype==DTYPE
    assert x.shape[0] == y.shape[0]
    cdef:
        DTYPE_t slope
        DTYPE_t intercept
        DTYPE_t x_bar = 0.0
        DTYPE_t y_bar = 0.0
        DTYPE_t x2_bar = 0.0
        DTYPE_t y2_bar = 0.0
        DTYPE_t xy_bar = 0.0
        int i

    for i in range(x.shape[0]):
        x_bar += x[i]
        x2_bar += x[i] * x[i]
        y_bar += y[i]
        y2_bar += y[i] * y[i]
        xy_bar += x[i] * y[i]

    x_bar /= (i + 1)
    y_bar /= (i + 1)
    x2_bar /= (i + 1)
    y2_bar /= (i + 1)
    xy_bar /= (i + 1)

    slope = (xy_bar - x_bar * y_bar) / (x2_bar - x_bar**2)
    intercept = y_bar - slope * x_bar
    return (slope, intercept)


def linear_ransac1D(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y, int n_iter, float threshold, float good_frac, int check_idx):
    """
    Warning: you MUST make sure that x and y are np.float32
    x,y: a set of observations
        model: always linear here
        min_: 2, here. minimum number of data points required to estimate model parameters
        n_iter: maximum number of iterations allowed in the algorithm
        threshold: threshold value to determine data points that are fit well by model 
        good_frac: number of close data points required to assert that a model fits well to data
    """
    assert x.shape[0] == y.shape[0]
    best_fit = None

    cdef:
        int d = int(x.shape[0] * good_frac)
        int _iter = 0
        int ridx1, ridx2
        DTYPE_t _slope
        DTYPE_t _intercept
        DTYPE_t best_slope
        DTYPE_t best_intercept
        DTYPE_t out
        float best_err = 1e4
        float this_err = 0.0
        float _pt_err = 0.0
        int n_inliers = 0
        int pointer = 0
        int idx
        np.ndarray[DTYPE_t, ndim = 1] maybeinliers_x
        np.ndarray[DTYPE_t, ndim = 1] maybeinliers_y

    while(_iter < n_iter):
        n_inliers = 0
        ridx1 = rand() % x.shape[0]  # np.random.randint(0, x.shape[0])
        ridx2 = rand() % x.shape[0]  # np.random.randint(0, x.shape[0])
        if ridx1 == ridx2:
            continue

        _slope = -0.3 #(y[ridx2] - y[ridx1]) / (x[ridx2] - x[ridx1])

        # if _slope > 0.0: continue

        _intercept = y[ridx2] - _slope * x[ridx2]

        for idx in range(x.shape[0]):
            _pt_err = abs(_intercept + _slope * x[idx] - y[idx])
            if _pt_err < threshold:
                n_inliers += 1

        if n_inliers - 2 >= d:
            maybeinliers_x = np.zeros(n_inliers, dtype=DTYPE)
            maybeinliers_y = np.zeros(n_inliers, dtype=DTYPE)
            pointer = 0
            for idx in range(x.shape[0]):
                _pt_err = abs(_intercept + _slope * x[idx] - y[idx])
                if _pt_err < threshold:
                    maybeinliers_x[pointer] = x[idx]
                    maybeinliers_y[pointer] = y[idx]
                    pointer += 1
            _slope, _intercept = fit_linear(maybeinliers_x, maybeinliers_y)

            this_err = 0.0
            for idx in range(n_inliers):
                _pt_err = abs(_intercept + _slope *
                              maybeinliers_x[idx] - maybeinliers_y[idx])
                this_err += _pt_err

            if this_err < best_err:
                best_slope = _slope
                best_intercept = _intercept
                best_err = this_err
        _iter += 1

        out=abs(best_slope*x[check_idx]+best_intercept-y[check_idx])

    return out,out>=threshold

def computepairwise(np.ndarray[DTYPE_t, ndim=2] matrix1,np.ndarray[DTYPE_t, ndim=2] matrix2):
    #assert len(matrix1.shape)==2, 'First argument is not 2D'
    #assert len(matrix2.shape)==2, 'Second argument is not 2D'
    assert matrix1.shape[1]==matrix2.shape[1], 'Matrices have different number of features'
    cdef:
        int feature
        int idx
        np.ndarray[DTYPE_t, ndim = 2] result
        np.ndarray[DTYPE_t, ndim = 2] diff

    result=np.zeros((matrix1.shape[0],matrix2.shape[0]),dtype=DTYPE)
    for feature in range(matrix1.shape[1]):
        diff=(np.repeat(matrix1[:,feature][:,None],matrix2.shape[0],axis=1)-matrix2[:,feature][None,:])
        assert diff.shape[0]==matrix1.shape[0]
        assert diff.shape[1]==matrix2.shape[0]

        result += diff**2
    return result