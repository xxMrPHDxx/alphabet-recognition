import numpy as np

class Dense():
	def __init__(self, n_input, n_output, depth=1):
		self._n_input = n_input
		self._n_output = n_output
		self._depth = depth
		self._weights = np.random.randn(n_output * n_input).reshape(n_output, n_input)
		self._biases = np.random.randn(n_output * depth).reshape(n_output, depth)
	def forward(self, inputs):
		inputs = np.array(inputs).reshape(self._n_input, self._depth)
		self._output = np.dot(self._weights, inputs) + self._biases

if __name__ == '__main__':
	layer = Dense(3, 1)
	layer.forward(np.array([1, 2, 3]))
	print(layer._output)