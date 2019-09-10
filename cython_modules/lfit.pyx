cimport cython
import numpy as np
cimport numpy as np

DTYPE=np.float

ctypedef np.float_t DTYPE_t

def fit_linear(np.ndarray[DTYPE_t, ndim=1] x,np.ndarray[DTYPE_t, ndim=1] y):
	assert x.dtype==DTYPE and y.dtype==DTYPE
	assert x.shape[0]==y.shape[0]
	cdef float slope
	cdef float intercept
	cdef float x_bar = 0.0
	cdef float y_bar = 0.0
	cdef float x2_bar = 0.0
	cdef float y2_bar = 0.0
	cdef float xy_bar = 0.0

	for i in range(len(x)):
		x_bar+=x[i]
		x2_bar+=x[i]*x[i]
		y_bar+=y[i]
		y2_bar+=y[i]*y[i]
		xy_bar+=x[i]*y[i]

	x_bar/=(i+1)
	y_bar/=(i+1)
	x2_bar/=(i+1)
	y2_bar/=(i+1)
	xy_bar/=(i+1)

	slope=(xy_bar-x_bar*y_bar)/(x2_bar-x_bar**2)
	intercept=y_bar-slope*x_bar
	return slope, intercept

