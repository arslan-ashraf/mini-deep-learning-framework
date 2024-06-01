from deep_learning_framework.topological_sort import TopologicalSort
from deep_learning_framework.variable import Variable, SigmoidActivation


class TestTopologicalSort:

	def test_call(self):

		x1 = Variable(1.5, name="x1"); x2 = Variable(0.5, name="x2")
		w1 = Variable(3.0, name="w1"); w2 = Variable(-1.0, name="w2")
		b = Variable(-2.5, name="b")

		x1w1 = x1 * w1
		x2w2 = x2 * w2
		add_xs_ws = x1w1 + x2w2
		sigmoid_in = add_xs_ws + b

		sigmoid = SigmoidActivation()
		sigmoid_out = sigmoid(sigmoid_in)

		topological_sort = TopologicalSort(sigmoid_out)

		top_sort_stack = topological_sort.stack
		assert top_sort_stack[-1] == sigmoid_out
		assert top_sort_stack[-2] == sigmoid_in

		for i, node in enumerate(topological_sort.stack):
			if node == x1: x1_index = i
			if node == x2: x2_index = i
			if node == x1w1: x1w1_index = i
			if node == x2w2: x2w2_index = i
			if node == b: b_index = i
			if node == add_xs_ws: add_xs_ws_index = i
			if node == sigmoid_in: sigmoid_in_index = i

		assert x1_index < x1w1_index
		assert x2_index < x2w2_index
		assert x1w1_index < add_xs_ws_index
		assert x2w2_index < add_xs_ws_index
		assert add_xs_ws_index < sigmoid_in_index
		assert b_index < sigmoid_in_index