import math
import random
from deep_learning_framework.topological_sort import TopologicalSort


class Variable:
	"""
	The Variable class represents each weight or bias scalar value in the 
	neural network.  It has one or two parent nodes and every operation will 
	involve at most two terms, each term represented by this Variable class
	"""

	def __init__(self, value=None, parent_nodes=(), math_operation="", name=""):
		"""
		Each Variable instance will hold its value and it derivative as well,
		this is what makes back propagation efficient, as the chain rule calculated
		backwards involves only two derivatives at a time.  

		The chain_rule_backwards is a function which multiplies the current node's
		derivative with it's parents derivative.
		"""
		self.value = value
		self.parent_nodes = set(parent_nodes)
		self.math_operation = math_operation
		self.name = name
		self.derivative = 0.0
		self.chain_rule_backwards = lambda: None

	def __repr__(self):
		return "Variable(value={}, name={})".format(self.value, self.name)

	def __add__(self, second_var):
		second_var = second_var if isinstance(second_var, Variable) else Variable(second_var)
		_sum = self.value + second_var.value
		new_var = Variable(value=_sum, 
						   parent_nodes=(self, second_var), 
						   math_operation="+", 
						   name=self.name + "+" + second_var.name)

		def chain_rule_backwards():
			"""
			The reason why the derivatives are set to "+=" and not "=" is because if a variable
			is used more than once, then self and second_var are the same, and setting second_var.derivative
			overwrites the first sestting self.derivative, so x+x should have derivative 1+1=2
			"""
			self.derivative += 1.0 * new_var.derivative
			second_var.derivative += 1.0 * new_var.derivative

		new_var.chain_rule_backwards = chain_rule_backwards

		return new_var

	def __radd__(self, second_var):
		return self + second_var

	def __sub__(self, second_var):
		return self + (-second_var)

	def __neg__(self):
		return self * -1

	def __mul__(self, second_var):
		second_var = second_var if isinstance(second_var, Variable) else Variable(second_var)
		product = self.value * second_var.value
		new_var = Variable(value=product, 
						   parent_nodes=(self, second_var), 
						   math_operation="*", 
						   name=self.name + "*" + second_var.name)

		def chain_rule_backwards():
			"""
			The reason why the derivatives are set to "+=" and not "=" is because if a variable
			is used more than once, then self and second_var are the same, and setting second_var.derivative
			overwrites the first sestting self.derivative, so x*x should have derivative 2x
			"""
			self.derivative += second_var.value * new_var.derivative
			second_var.derivative += self.value * new_var.derivative

		new_var.chain_rule_backwards = chain_rule_backwards

		return new_var

	def __rmul__(self, second_var):
		return self * second_var

	def __pow__(self, exponent):
		assert isinstance(exponent, (int, float)), "exponent must be float or int"
		value_raised_to_power = self.value ** exponent
		new_var = Variable(value=value_raised_to_power, 
						   parent_nodes=(self,), 
						   math_operation="**", 
						   name="(" + self.name + ")**" + str(exponent))

		def chain_rule_backwards():
			self.derivative += exponent * self.value**(exponent - 1) * new_var.derivative

		new_var.chain_rule_backwards = chain_rule_backwards

		return new_var

	def __truediv__(self, second_var):
		return self * second_var**-1

	def log(self):
		_log = math.log(self.value)
		new_var = Variable(value=_log, 
						   parent_nodes=(self,), 
						   math_operation="log", 
						   name="log(" + self.name + ")")

		def chain_rule_backwards():
			self.derivative += 1.0 * (self.value**-1) * new_var.derivative

		new_var.chain_rule_backwards = chain_rule_backwards

		return new_var

	def run_back_propagation(self):
		topological_sort = TopologicalSort(self)

		# set base case df/df = 1
		self.derivative = 1
		for node in reversed(topological_sort.stack):
			node.chain_rule_backwards()


class SigmoidActivation(Variable):
	def __init__(self):
		super().__init__()

	def __call__(self, _variable):
		node_value = _variable.value
		sigmoid_output = 1 / (1 + math.exp(-node_value))
		new_var = Variable(value=sigmoid_output, 
						   parent_nodes=(_variable,), 
						   math_operation="sigmoid", 
						   name="sigmoid(" +str(node_value) + ")")

		def chain_rule_backwards():
			_variable.derivative += (1.0 - sigmoid_output) * sigmoid_output * new_var.derivative

		new_var.chain_rule_backwards = chain_rule_backwards

		return new_var