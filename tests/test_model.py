from deep_learning_framework.model import Neuron, DenseLayer, Model
from deep_learning_framework.variable import SigmoidActivation

class TestNeuron:

	def test_call(self):

		num_neurons_in = 3
		activation = SigmoidActivation()
		X = [1.0, 2.0, 3.0]

		neurons = Neuron(num_neurons_in, activation)
		weights = neurons.weights
		bias = neurons.bias

		A = neurons(X)

		z = 0
		for w_i, x_i in zip(weights, X):
			z += w_i * x_i
		z += bias

		assert round(A.value, 4) == round(activation(z).value, 4)


class TestDenseLayer:

	def setup_class(self):
		self.num_neurons_in = 2
		self.num_neurons_out = 3
		self.activation = SigmoidActivation()

		self.layer = DenseLayer(self.num_neurons_in, 
								self.num_neurons_out, 
								self.activation)

	def test_get_layer_weights(self):

		weights_matrix = self.layer.get_layer_weights()
		weights_matrix_shape = (len(weights_matrix[0]), len(weights_matrix))

		assert weights_matrix_shape == (self.num_neurons_in, self.num_neurons_out)

	def test_call(self):
		X = [1.0, 2.0, 3.0]
		Y = self.layer(X)
		
		assert len(Y) == self.num_neurons_out

		for output_neuron in Y:
			assert output_neuron.value < 1


class TestModel():

	def test_fit(self):
		sigmoid = SigmoidActivation()
		dense_layer_1 = DenseLayer(2, 3, sigmoid)
		dense_layer_2 = DenseLayer(3, 1, sigmoid)

		X = [[2.3, -1.8],
			 [0.11, -4.3],
			 [4.5, -1.4]]

		Y = [1, 0, 1]

		model = Model()
		model.add(dense_layer_1)
		model.add(dense_layer_2)

		model.fit(X, Y, epochs=10, learning_rate=0.1)

		assert model.history["train_losses"][-1] < model.history["train_losses"][0]