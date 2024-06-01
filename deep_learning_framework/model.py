import random
from deep_learning_framework.variable import Variable


class Neuron:
	def __init__(self, num_neurons_in, activation):
		"""
		self.weights is a list of the number of neurons going in to a layer
		"""
		self.weights = [
			Variable(random.normalvariate(0, 1), name="w_{}".format(i+1)) for i in range(num_neurons_in)
		]
		self.bias = Variable(random.normalvariate(0, 1), name="b")
		self.activation = activation

	def __call__(self, X):
		Z = sum((w_i * x_i for w_i, x_i in zip(self.weights, X)), start=self.bias)
		A = self.activation(Z)
		return A

	def get_neuron_parameters(self):
		""" Returns all trainable parameters for back propagation """
		return self.weights + [self.bias]


class DenseLayer:
	def __init__(self, num_neurons_in, num_neurons_out, activation):
		"""
		Initializes the weight matrix of shape (num_neurons_in, num_neurons_out)
		"""
		self.neurons = [
			Neuron(num_neurons_in, activation) for _ in range(num_neurons_out)
		]

	def __call__(self, X):
		""" 
		Each neuron(X) does (length of input) calculations.
		The number of self.neurons determines how many neurons are coming out.
		"""
		layer_output = [neuron(X) for neuron in self.neurons]
		return layer_output[0] if len(layer_output) == 1 else layer_output

	def get_layer_weights(self):
		""" Gets the weights matrix of a layer """
		weights_matrix = [neuron.weights for neuron in self.neurons]
		return weights_matrix

	def get_layer_parameters(self):
		parameters = []
		for neuron in self.neurons:
			neuron_parameters = neuron.get_neuron_parameters()
			parameters.extend(neuron_parameters)

		return parameters


class Model:
	def __init__(self):
		self.layers = []
		self.history = {}
		self.loss = 0

	def __call__(self, X):
		for i, layer in enumerate(self.layers):
			X = layer(X)

		return X

	def add(self, layer):
		self.layers.append(layer)

	def binary_cross_entropy_loss(self, y_train_i, y_pred_i):
		if y_train_i == 1:
			loss = - 1.0 * y_pred_i.log()
		else:
			loss = - 1.0 * (Variable(1) - y_pred_i).log()

		return loss 

	def get_model_parameters(self):
		parameters = []
		for layer in self.layers:
			layer_parameters = layer.get_layer_parameters()
			parameters.extend(layer_parameters)

		return parameters

	def fit(self, x_train, y_train, epochs=3, learning_rate=0.001):
		assert len(x_train) == len(y_train), "x_train and y_train must have an equal number of examples"
		losses = []
		for epoch in range(epochs):
			epoch_loss = 0
			for i in range(len(x_train)):
				y_pred_i = self.__call__(x_train[i])
				loss = self.binary_cross_entropy_loss(y_train[i], y_pred_i)
				epoch_loss += loss

			epoch_loss /= len(x_train)

			print("EPOCH {} ======== ".format(epoch+1) + "training loss: {}".format(epoch_loss.value))
			losses.append(epoch_loss.value)

			# back propagation
			epoch_loss.run_back_propagation()
			model_parameters = self.get_model_parameters()

			# gradient descent
			for parameter in model_parameters:
				parameter.value = parameter.value - learning_rate * parameter.derivative

		self.history["train_losses"] = losses