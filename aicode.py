import numpy as np
from scipy import optimize
from autodiff import function, gradient
import math
import matplotlib.pyplot as plt


def F(x, b0, w1, b1, w2, b2):

	"""
	This function defines a 3 layer preceptron model
	x 	-- A length N vector of inputs
	b0 	-- A length N vector of input bias
	w1  -- An IxN matrix of weights from the first layer to the second layer
	b1 	-- A length I vector of bias
	w2  -- An OxI matrix of weights from the second layer to the third layer
	b2 	-- A length O vector of bias
	"""
	return np.transpose(np.dot(np.tanh(np.dot((np.transpose(x)+np.transpose(b0)),w1))+b1,w2)+b2)


def expandParams(params):
	b0 = params[0:N]
	w1 = (params[N:N+N*I]).reshape(N, I)
	b1 = params[N+N*I:N+N*I+I]
	w2 = (params[N+N*I+I:N+N*I+I+O*I]).reshape(I, O)
	b2 = (params[N+N*I+I+O*I:N+N*I+I+O*I+O])
	return (b0, w1, b1, w2, b2)

def Cost(params, X, Y):

	"""
	This is the cost function
	params = (b0, w1, b1, w2, b2)  -- Parameters for the NN
	X 	--  Input
	Y 	-- 	Output
	"""
	(b0, w1, b1, w2, b2) = expandParams(params)

	C = 0.01
	return np.mean(np.square(F(X, b0, w1, b1, w2, b2) - Y))+C*np.sum(np.abs(w2)**2)







def plot_model_output(params):
	(b0_0, w1_0, b1_0, w2_0, b2_0) = expandParams(params)

	f, axarr = plt.subplots(2, 4)



	model_output = np.array([F(np.array([inp/100.0]*N), b0_0, w1_0, b1_0, w2_0, b2_0) for inp in range(1000)])
	
	actual_output = np.array([p**2 for p in np.arange(0, 10, .01)])
	axarr[0,0].plot(model_output[:,0], label='Model Output')
	axarr[0,0].plot(actual_output, label= "Actual Output")

	actual_output = np.array([p**3-10*p**2+p-1 for p in np.arange(0, 10, .01)])
	axarr[0,1].plot(model_output[:,1], label='Model Output')
	axarr[0,1].plot(actual_output, label= "Actual Output")

	actual_output = np.array([p**(3/2)-20*p**(0.5)+2*p+2 for p in np.arange(0, 10, .01)])
	axarr[0,2].plot(model_output[:,2], label='Model Output')
	axarr[0,2].plot(actual_output, label= "Actual Output")

	actual_output = np.array([3*p**(5/2)-20*p**(0.3)-10*p+5 for p in np.arange(0, 10, .01)])
	axarr[0,3].plot(model_output[:,3], label='Model Output')
	axarr[0,3].plot(actual_output, label= "Actual Output")

	actual_output = np.array([math.sin(math.pi*p) for p in np.arange(0, 10, .01)])
	axarr[1,0].plot(model_output[:,4], label='Model Output')
	axarr[1,0].plot(actual_output, label= "Actual Output")

	actual_output = np.array([math.cos(math.pi*p) for p in np.arange(0, 10, .01)])
	axarr[1,1].plot(model_output[:,5], label='Model Output')
	axarr[1,1].plot(actual_output, label= "Actual Output")

	actual_output = np.array([math.sin(2*math.pi*p) for p in np.arange(0, 10, .01)])
	axarr[1,2].plot(model_output[:,6], label='Model Output')
	axarr[1,2].plot(actual_output, label= "Actual Output")

	actual_output = np.array([math.tan(math.pi * (p+0.5)) for p in np.arange(0, 10, .01)])
	axarr[1,3].plot(model_output[:,7], label='Model Output')
	axarr[1,3].plot(actual_output, label= "Actual Output")

	plt.show()


def calculate_model_mse(params):

	tempy = np.array([[p**2 for p in np.arange(0, 10, .01)],
	[p**3-10*p**2+p-1 for p in np.arange(0, 10, .01)],
	[p**(3/2)-20*p**(0.5)+2*p+2 for p in np.arange(0, 10, .01)],
	[3*p**(5/2)-20*p**(0.3)-10*p+5 for p in np.arange(0, 10, .01)],
	[math.sin(math.pi*p) for p in np.arange(0, 10, .01)],
	[math.cos(math.pi*p) for p in np.arange(0, 10, .01)],
	[math.sin(2*math.pi*p) for p in np.arange(0, 10, .01)],
	[math.tan(math.pi * (p+0.5)) for p in np.arange(0, 10, .01)]])

	tempy = np.clip(tempy, -1000, 1000)


	(b0_0, w1_0, b1_0, w2_0, b2_0) = expandParams(params)
	model_output = np.array([F(np.array([inp/100.0]*N), b0_0, w1_0, b1_0, w2_0, b2_0) for inp in range(1000)])


	return np.mean((np.transpose(model_output)[0:6,:]-tempy[0:6, :])**2)

	


mse_inner = []

# for j in range(0, 200, 5):
# 	mse = 0
	# for jj in range(3):
N = 15
I = 500
O = 8

@gradient(wrt = 'params')
def dCost(params, X, Y):
	"""
	Processed through Theano, computes gradient wrt b0, w1, b1, w2, b2
	and returns in vector form 
	"""
	return Cost(params, X, Y)


b0_0 = np.random.uniform(-1,1,N)
w1_0 = np.random.uniform(-1,1,I*N)
b1_0 = np.random.uniform(-1,1,I)
w2_0 = np.random.uniform(-1,1,O*I)
b2_0 = np.random.uniform(-1,1,O)



tempx = np.tile(np.arange(0, 10, .1), (N, 1))
tempy = np.array([[p**2 for p in np.arange(0, 10, .1)],
	[p**3-10*p**2+p-1 for p in np.arange(0, 10, .1)],
	[p**(3/2)-20*p**(0.5)+2*p+2 for p in np.arange(0, 10, .1)],
	[3*p**(5/2)-20*p**(0.3)-10*p+5 for p in np.arange(0, 10, .1)],
	[math.sin(math.pi*p) for p in np.arange(0, 10, .1)],
	[math.cos(math.pi*p) for p in np.arange(0, 10, .1)],
	[math.sin(2*math.pi*p) for p in np.arange(0, 10, .1)],
	 np.array([math.tan(math.pi * (p+0.501)) for p in np.arange(0, 10, .1)])/1000.0])

tempy = np.clip(tempy, -1000, 1000)



valuesinit = np.concatenate((b0_0, w1_0, b1_0, w2_0, b2_0))
(b0_0, w1_0, b1_0, w2_0, b2_0) = expandParams(valuesinit)



result = optimize.fmin_l_bfgs_b(Cost, x0 = valuesinit, fprime = dCost,  args = (tempx, tempy))
(b0_0, w1_0, b1_0, w2_0, b2_0) = expandParams(result[0])

# print(F(np.array([1.2]*N), b0_0, w1_0, b1_0, w2_0, b2_0))


print(calculate_model_mse(result[0]))
plot_model_output(result[0])

	# mse = mse+calculate_model_mse(result[0])

# print(mse/3.0)
# mse_inner.append(mse/3.0)


# print(mse_inner)
# plt.plot(mse_inner)
# plt.ylabel('mse')
# plt.show()












