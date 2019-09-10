import numpy as np

def fit_linear(x,y):
	x_bar = np.mean(x)
	y_bar = np.mean(y)
	x2_bar = np.mean(x*x)
	y2_bar = np.mean(y*y)
	xy_bar = np.mean(x*y)

	# for i in range(len(x)):
	# 	x_bar+=x[i]
	# 	x2_bar+=x[i]*x[i]
	# 	y_bar+=y[i]
	# 	y2_bar+=y[i]*y[i]
	# 	xy_bar+=x[i]*y[i]

	# x_bar/=(i+1)
	# y_bar/=(i+1)
	# x2_bar/=(i+1)
	# y2_bar/=(i+1)
	# xy_bar/=(i+1)

	slope=(xy_bar-x_bar*y_bar)/(x2_bar-x_bar**2)
	intercept=y_bar-slope*x_bar
	return slope, intercept