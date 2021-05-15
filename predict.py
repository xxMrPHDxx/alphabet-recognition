from functools import reduce
from random import random
import numpy as np
import json

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 64)        640       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 32)        18464     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 32)        0         
_________________________________________________________________
flatten (Flatten)            (None, 4608)              0         
_________________________________________________________________
dense (Dense)                (None, 100)               460900    
_________________________________________________________________
dense_1 (Dense)              (None, 26)                2626      
=================================================================
Total params: 482,630
Trainable params: 482,630
Non-trainable params: 0
_________________________________________________________________

'''

def product(iterable):
  return reduce(lambda a, b: a*b, list(iterable), 1)

class Activation():
  @staticmethod
  def activate(inputs):
    raise RuntimeError('Unimplemented method Activation:activate!')
  @staticmethod
  def deactivate(outputs):
    raise RuntimeError('Unimplemented method Activation:deactivate!')

class ReLU(Activation):
  @staticmethod
  def activate(inputs):
    inputs[inputs < 0] = 0
    return inputs
  @staticmethod
  def deactivate(inputs):
    raise RuntimeError('Unimplemented method ReLU:deactivate!')

class SoftMax(Activation):
  @staticmethod
  def activate(inputs):
    return inputs / np.max(inputs, axis=0, keepdims=True)
  @staticmethod
  def deactivate(inputs):
    raise RuntimeError('Unimplemented method SoftMax:deactivate!')

class Layer():
  @property
  def weights(self): return self._weights
  @property
  def biases(self): return self._biases
  @property
  def output(self): return self._output
  def forward(self, inputs):
    raise RuntimeError('Unimplemented method Layer:forward!')
  def set_biases(self, biases):
    biases = np.array(biases)
    self._biases = biases
  def set_weights(self, weights):
    weights = np.array(weights).T
    self._weights = weights

class Conv2D(Layer):
  def __init__(self, filter, kernel_size, input_shape, **kwargs):
    self._input   = input_shape if len(input_shape) == 3 else (*input_shape, 1)
    self._kernel  = (filter, input_shape[-1], *kernel_size)
    self._output  = (*[i-j+1 for i, j in zip(input_shape[:2], kernel_size)], filter)
    self._weights = np.random.randn(product(self._kernel)).reshape(self._kernel)
    self._biases  = np.random.randn(filter)
    self._kwargs  = kwargs
  def forward(self, inputs):
    if len(inputs.shape) == 2:
      inputs = inputs.reshape((*inputs.shape, 1))
    inputs         = inputs.T
    fl, il, kw, kh = self._kernel
    ow, oh         = [i-2 for i in self._input[:2]]
    output         = np.zeros(self._output)

    for f in range(fl):
      for i in range(1, ow):
        for j in range(1, oh):
            try: 
              for d in range(il):
                cells = inputs[d, i-1:, j-1:][:kw, :kh]
                kernels = self._weights[f, d, :, :]
                self._weights[f, d, :, :] = np.dot(cells.T, kernels)
            except Exception as e: 
              print(cells.shape, kernels.shape, f, d, i-1, i+2, j-1, j+2)
              print(e)
              exit(-1)

    self._output = output + self._biases
    if 'activation' in self._kwargs:
      self._output = self._kwargs['activation'].activate(self._output)
    return self._output

class MaxPooling2D(Layer):
  def __init__(self, pool_size, input_shape):
    self._ps    = pool_size
    self._is    = input_shape
    self._shape = np.array(input_shape[:2]) // np.array(pool_size)
  def forward(self, inputs):
    w, h, *_ = inputs.shape
    self._output = np.zeros((*self._shape, *_))
    for i in range(0, w-1, 2):
      for j in range(0, h-1, 2):
        self._output[i//2, j//2, :] = np.max(inputs[i:i+self._ps[0], j:j+self._ps[1]])
    return self._output

class Flatten(Layer):
  def forward(self, inputs):
    self._output = inputs.reshape(product(inputs.shape))
    return self._output

class Dense(Layer):
  def __init__(self, n_nodes, input_shape, **kwargs):
    self._nn      = n_nodes
    self._weights = np.random.randn(input_shape*n_nodes).reshape(n_nodes, input_shape)
    self._biases  = np.random.randn(n_nodes)
    self._kwargs  = kwargs
  def forward(self, inputs):
    self._output = np.dot(self._weights, inputs) + self._biases
    if 'activation' in self._kwargs:
      self._output = self._kwargs['activation'].activate(self._output)
    return self._output

class Model():
  def __init__(self):
    self._layers = [
      Conv2D(64, (3, 3), (28, 28, 1), activation=ReLU),
      Conv2D(32, (3, 3), (26, 26, 64), activation=ReLU),
      MaxPooling2D((2, 2), (24, 24, 32)),
      Flatten(),
      Dense(100, 4608, activation=ReLU),
      Dense(26, 100, activation=SoftMax)
    ]
  @property
  def layers(self): return self._layers
  def forward(self, inputs):
    outputs = np.array(inputs).reshape((28, 28))
    for layer in self._layers:
      layer.forward(outputs)
      outputs = layer.output
    return outputs
  def load_weights(self, path):
    with open(path, 'r', encoding='utf-8') as f:
      obj = json.load(f)
      layers = [self._layers[i] for i in [0, 1, -2, -1]]
      for i in range(0, len(obj), 2):
        weights, biases = obj[i], obj[i+1]
        layer = layers[i//2]
        layer.set_weights(weights)
        layer.set_biases(biases)

if __name__ == '__main__':
  import pickle
  import re

  with open('weights.json', 'r') as f:
    obj = json.load(f)
    for i in range(len(obj)//2):
      weights, biases = np.array(obj[i*2]).tolist(), np.array(obj[i*2+1]).tolist()
      with open(f'weights_{i}.txt', 'w') as f2:
        f2.write(re.sub(r'[\[\],]', '', json.dumps(weights)))
      with open(f'biases_{i}.txt', 'w') as f2:
        f2.write(re.sub(r'[\[\],]', '', json.dumps(biases)))
  
  exit(-1)

  ALPHABETS = 'ABCDEFHIJKLMNOPQRSTUVWXYZ'

  with open('x_test.pickle', 'rb') as f:
    X = pickle.load(f)
  with open('y_test.pickle', 'rb') as f:
    Y = pickle.load(f)

  model = Model()
  model.load_weights('weights.json')

  output = model.forward(X[0])

  print('Expected', ALPHABETS[list(Y[0] == np.max(Y[0])).index(True)])
  print('Actual', ALPHABETS[list(output == np.max(output)).index(True)])
