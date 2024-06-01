from deep_learning_framework.variable import Variable, SigmoidActivation
import math

class TestVariable:

	def test_add(self):
		x = Variable(3.5)
		w = Variable(-3.0)
		x_plus_w = x + w

		assert (x_plus_w).value == 0.5

		x_plus_w.derivative = 5.0

		x_plus_w.chain_rule_backwards()

		assert x.derivative == x_plus_w.derivative
		assert w.derivative == x_plus_w.derivative

	def test_mul(self):
		x = Variable(2.0)
		w = Variable(-3.0)
		x_times_w = x * w

		assert (x_times_w).value == -6.0

		x_times_w.derivative = 5.0

		x_times_w.chain_rule_backwards()
		
		assert x.derivative == (w.value * x_times_w.derivative)
		assert w.derivative == (x.value * x_times_w.derivative)

	def test_pow(self):
		x = Variable(2.0)
		exponent = 3
		x_raised_to_exponent = x ** exponent

		assert (x_raised_to_exponent).value == 8.0

		x_raised_to_exponent.derivative = 5.0

		x_raised_to_exponent.chain_rule_backwards()
		
		exponent_derivative = exponent * x.value**(exponent - 1)
		assert x.derivative == (exponent_derivative * x_raised_to_exponent.derivative)

	def test_log(self):
		x = Variable(3.0)

		log_of_x = x.log()

		assert (log_of_x).value == math.log(3.0)

		log_of_x.derivative = 5.0

		log_of_x.chain_rule_backwards()
		
		log_derivative = 1 / (x.value)
		assert x.derivative == (log_derivative * log_of_x.derivative)


class TestSigmoidActivation:

	def test_call(self):
		x = Variable(1.5)

		sigmoid = SigmoidActivation()
		sigmoid_of_x = sigmoid(x)

		assert (sigmoid_of_x).value == 1 / (1 + math.exp(-x.value))

		sigmoid_of_x.derivative = 5.0

		sigmoid_of_x.chain_rule_backwards()
		
		log_derivative = (1 - sigmoid_of_x.value) * sigmoid_of_x.value
		assert x.derivative == (log_derivative * sigmoid_of_x.derivative)