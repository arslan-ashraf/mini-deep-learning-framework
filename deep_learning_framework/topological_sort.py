class TopologicalSort:

	"""
	Performs topological sort, but specifically to build a graph for a neural
	network.  It starts on the given node and recursively moves backwards 
	through the graph so partial derivatives can easily be calculated
	"""

	def __init__(self, node):
		self.stack = []
		self.visited_set = set()
		self.node_set = set()
		self.__call__(node)

	def __call__(self, node):
		# get the list of the nodes in the graph, starting from the input node
		self.get_all_parent_nodes(node)

		# runs depth first search
		for node in self.node_set:
			if node not in self.visited_set:
				self.depth_first_search(node)

	def get_all_parent_nodes(self, node):
		""" Recursively gets all parent nodes and puts them in the node_set"""
		if node not in self.node_set:
			self.node_set.add(node)
			for parent_node in node.parent_nodes:
				self.get_all_parent_nodes(parent_node)

	def depth_first_search(self, node):
		self.visited_set.add(node)
		for parent_node in node.parent_nodes:
			if parent_node not in self.visited_set:
				self.depth_first_search(parent_node)

		self.stack.append(node)