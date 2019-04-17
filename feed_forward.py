#single layer feed forward network

import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

class FeedForwardNetwork:

	#forward propagation
	def forward_propagation(self, x, weights, bias):
		return self.sigmoid((np.dot(x, weights) + bias), False)

	#sigmoid activation function
	def sigmoid(self, x, deriv):
		if deriv == True:
			return x * (1 - x)
		return 1 / (1 + np.exp(x))

	#cost function
	def cost(self, expected, predicted):
		return expected - predicted

	#compute gradient
	def compute_gradient(self, error, layer):
		return error * self.sigmoid(layer, True)

	#update weights
	def update_weights(self, weights, layer, gradient):
		weights += np.dot(layer.T, gradient)
		return weights

	#lets visualize the error
	def visualize_error(self, epoch, error):
		plot.xlabel('epoch')
		plot.ylabel('error')
		plot.title('Cost Function')
		plot.scatter(np.arange(0, epoch), error)
		plot.show()

if __name__ == "__main__":
	ffn = FeedForwardNetwork()

	#to store the error over number of iterations
	error = [ ]

	#number of iterations and iteration step
	epoch = 250
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
	#weights1 for hidden layer 1
	#weights2 for output layer
	weights1 = 2 * np.random.random((2, 4)) - 1
	weights2 = 2 * np.random.random((4, 1)) - 1

	#defining bias
	#bias1 for hidden layer 1
	#bias2 for output layer
	bias1 = 2 * np.random.random((4)) - 1
	bias2 = 2 * np.random.random((1)) - 1 

	#training
	for i in range(epoch):

		#forward propagation
		layer1 = ffn.forward_propagation(x, weights1, bias1)
		output = ffn.forward_propagation(layer1, weights2, bias2)
		
		#back propagation through number of iteration 
		#error of output layer
		op_error = ffn.cost(y, output)

		#calculating the gradient of output layer
		op_gradient = ffn.compute_gradient(op_error, output)

		#error of layer1
		l1_error = op_gradient.dot(weights2.T)

		#calculating the gradient of layer1
		l1_gradient = ffn.compute_gradient(l1_error, layer1)

		#updating weights
		weights1 = ffn.update_weights(weights1, x, l1_gradient)
		weights2 = ffn.update_weights(weights2, layer1, op_gradient)

		#round off the error to 4 decimal points
		k = round(np.mean(np.abs(op_error)), 4)

		#append errors for later visualization
		error.append(k)

		#print error for every 5 iteration
		if i % step == 0:		
			print("epoch: {0} error: {1}".format(i, k))

	#lets visualize it
	ffn.visualize_error(epoch, error)