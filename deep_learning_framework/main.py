from topological_sort import TopologicalSort
from variable import *
from model import DenseLayer, Model


if __name__ == "__main__":
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

	model.fit(X, Y, epochs=10, learning_rate=1)