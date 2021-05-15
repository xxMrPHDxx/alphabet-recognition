from functools import reduce
import numpy as np

def product(arr):
  return reduce(lambda a,b: a*b, arr, 1)

class Activation():
  def forward(self, inputs):
    raise RuntimeError('Unimplemented abstract function Activation::forward!')

class ReLU(Activation):
  def forward(self, inputs):
    inputs[inputs < 0] = 0.0
    return inputs

class SoftMax(Activation):
  def forward(self, inputs):
    _max = np.max(inputs)
    return inputs / _max

class Layer():
  @property
  def input_shape(self): return self._input_shape
  @property
  def output_shape(self): return self._output_shape
  def forward(self, inputs):
    if self._input_shape is None:
      raise RuntimeError('Please run Layer::build(input_shape) first!')
  
class Conv2D(Layer):
  def __init__(self, filter_size, kernel_shape, **kwargs):
    self._filter_size = filter_size
    self._kernel_shape = kernel_shape
    self._input_shape = None
    self._output_shape = None
    self._activation = None if 'activation' not in kwargs else kwargs['activation']
  def build(self, input_shape):
    self._depth_size = (input_shape[0])
    self._input_shape = tuple(input_shape[-2:])
    self._output_shape = (
      self._filter_size, 
      *[i-j+1 for i, j in zip(input_shape[1:], self._kernel_shape)]
    )
    self._weight_shape = (
      self._filter_size, 
      self._depth_size, 
      *self._kernel_shape
    )
    self._weights = np.random.randn(product(self._weight_shape)).reshape(self._weight_shape)
    self._biases = np.random.randn(self._filter_size)
  def forward(self, inputs):
    Layer.forward(self, inputs)
    out_depth, out_rows, out_cols = self._output_shape
    kernel_rows, kernel_cols = self._kernel_shape
    outputs = np.zeros(self._output_shape)
    for f in range(self._filter_size):
      for r in range(0, out_rows):
        for c in range(0, out_cols):
          cells  = inputs[:, r:, c:][:, :kernel_rows, :kernel_cols]
          weight = self._weights[f, :, :, :]
          total  = np.sum(np.cross(weight, cells))
          outputs[f, r, c] = total + self._biases[f]
    if self._activation is not None:
      outputs = self._activation.forward(outputs)
    return outputs

class MaxPooling2D(Layer):
  def __init__(self, kernel_shape):
    self._kernel_shape = kernel_shape
    self._input_shape = None
    self._output_shape = None
  def build(self, input_shape):
    self._input_shape = input_shape
    self._output_shape = (
      *input_shape[:-2],
      *[i//j for i, j in zip(input_shape[-2:], self._kernel_shape)]
    )
  def forward(self, inputs):
    Layer.forward(self, inputs)
    kr, kc = self._kernel_shape
    out_rows, out_cols = self._output_shape[-2:]
    outputs = np.zeros(self._output_shape)
    for r in range(out_rows):
      rr = r * kr
      for c in range(out_cols):
        cc = c * kc
        cells     = inputs[:, rr:, cc:][:, :kr, :kc]
        shape_len = len(cells.shape)
        outputs[:, r, c] = np.sum(
          cells,
          axis=tuple(i for i in range(shape_len) if i>=shape_len-2)
        )
    return outputs

class Flatten(Layer):
  def __init__(self):
    self._input_shape, self._output_shape = None, None
  def build(self, input_shape):
    self._input_shape = input_shape
    self._output_shape = product(input_shape)
  def forward(self, inputs):
    Layer.forward(self, inputs)
    return inputs.reshape(self._output_shape)

class Dense(Layer):
  def __init__(self, node_size, **kwargs):
    self._node_size = node_size
    self._input_shape, self._output_shape = None, None
    self._activation = None if 'activation' not in kwargs else kwargs['activation']
  def build(self, input_shape):
    self._input_shape = input_shape
    self._weight_shape = (self._node_size, self._input_shape)
    self._bias_shape = (self._node_size)
    self._output_shape = (
      self._node_size
    )
    self._weights = np.random.randn(product(self._weight_shape)).reshape(self._weight_shape)
    self._biases = np.random.randn(self._node_size)
  def forward(self, inputs):
    Layer.forward(self, inputs)
    outputs = np.dot(self._weights, inputs) + self._biases
    if self._activation is not None:
      outputs = self._activation.forward(outputs)
    return outputs

class Model():
  def __init__(self, **kwargs):
    self._layers = []
    self._input_shape = None if 'input_shape' not in kwargs else kwargs['input_shape']
    self._output_shape = None
  def add(self, layer):
    if not isinstance(layer, Layer):
      raise RuntimeError(f'Argument 1 is not an instance of Layer!')
    self._layers.append(layer)
  def build(self):
    if self._input_shape is None:
      raise RuntimeError(f'Model doesn\'t have input_shape specified. Add "input_shape=?" to the constructor!')
    self._layers[0].build(self._input_shape)
    for curr, prev in zip(self._layers[1:], self._layers[:-1]):
      curr.build(prev.output_shape)
    self._output_shape = self._layers[-1].output_shape
  def forward(self, inputs):
    if self._output_shape is None:
      raise RuntimeError('Please run Model::build() first!')
    outputs = None
    for layer in self._layers:
      outputs = layer.forward(inputs if outputs is None else outputs)
    return outputs

if __name__ == '__main__':
  model = Model(input_shape=(1, 28, 28))
  model.add(Conv2D(64, (3, 3), activation=ReLU()))
  model.add(Conv2D(32, (3, 3), activation=ReLU()))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(100, activation=ReLU()))
  model.add(Dense(26, activation=SoftMax()))
  model.build()

  inputs = np.random.randn(28**2).reshape((1, 28, 28))
  
  outputs = model.forward(inputs)

  print('Output shape: ', outputs.shape)
  print(outputs)
