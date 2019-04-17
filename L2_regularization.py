#Ridge Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

class L2Regularization:

	#cost function
	def cost(self, expected, predicted):
		return expected - predicted


	#foward propagation
	def forward_propagation(self, x, weights, bias):
		return self.sigmoid((np.dot(x, weights) + bias), False)


	#sigmoid activation function
	def sigmoid(self, x, deriv):
		if deriv == True:
			return x * (1 - x)
		return 1 / (1 + np.exp(x))


	#compute the gradient
	def compute_gradient(self, error, weights):
		return error * self.sigmoid(weights, True)


	#visualize the error
	def visualize_error(self, epoch, error):
		plot.xlabel('epoch')
		plot.ylabel('error')
		plot.title('cost function')
		plot.scatter(np.arange(0, epoch), error, marker = '.', color = 'red')
		plot.show()

	#updating weights
	#implementing L2 Regularization or L2 Norm
	def update_weights(self, weights, layer, gradient):
		lamda = 10
		size = len(layer)

		#L2 norm or L2 regularization
		reg = lamda / (2 * size) + (weights**2 / size)
		
		#updating weights with regularization  
		weights += reg * np.dot(layer.T, gradient)
		return weights


if __name__ == "__main__":
	l2 = L2Regularization()

	#to store the error over iterations
	error = [ ]

	#number of iterations and iteration step
	epoch = 500
	step = 10

	#load the blood fat data
	data = pd.read_csv('blood_fat.csv')

	#leave all rows and slice the columns
	#first 2 columns features 
	x = data.iloc[:, 0:2].values

	#label
	y = data.iloc[:, 0:1].values

	#to generate the same number 
	np.random.seed(1)

	#defining weights
	weights1 = 2 * np.random.random(size = (2, 4)) - 1
	weights2 = 2 * np.random.random(size = (4, 1)) - 1

	#defining bias
	bias1 = 2 * np.random.random(size = (4))
	bias2 = 2 * np.random.random(size = (1))

	for i in range(epoch):
		#hidden layer 
		layer = l2.forward_propagation(x, weights1, bias1)

		#output
		output = l2.forward_propagation(layer, weights2, bias2)

		#calculate the error for output layer
		op_error = l2.cost(y, output)
		#gradient for output layer
		op_gradient = l2.compute_gradient(op_error, output)

		#calculate the error for hidden layer
		layer_error = op_gradient.dot(weights2.T)
		#gradient for hidden layer
		layer_gradient = l2.compute_gradient(layer_error, layer)

		#update weights
		weights1 = l2.update_weights(weights1, x, layer_gradient)
		weights2 = l2.update_weights(weights2, layer, op_gradient)

		#round off error to 4 decimal points
		k = round(np.mean(np.abs(op_error)), 4)
		
		#append errors for a later visualization
		error.append(k)

		#print out the error and epoch for every step number of iteration
		if i % step == 0:
			print("epoch : {0} error : {1}".format(i, k))

	#lets visualize the error 
	l2.visualize_error(epoch, error)