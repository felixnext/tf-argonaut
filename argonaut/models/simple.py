'''
Defines some simple baseline models.

author: Felix Geilert
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as lyrs

class ConvNet(tf.keras.Model):
  '''Defines a simple convolutional model.

  Args:
    num_layers (int): Number of layers to use
    num_classes (int): Number of classes to use
  '''
  def __init__(self, num_layers, activation=tf.nn.relu):
    super(ConvNet, self).__init__()
    self.num_layers = num_layers

    # define the internal blocks
    self.layer_names = []
    filters = 32
    next = 1
    for i in range(num_layers - 1):
      layer_name = "conv_{}".format(i + 1)
      lyr = lyrs.Conv2D(filters, (3, 3) if (i % 3 == 0) or next == i else (1, 1), 2 if next == i else 1)
      self.__setattr__(layer_name, lyr)
      self.__setattr__("{}_act".format(layer_name), lyrs.ReLU())
      self.layer_names.append((layer_name, False))
      if next == i:
        filters = filters * 2
        next = next + i

  def call(self, input_tensor, training=False):
    x = input_tensor
    for (layer_name, train_only) in self.layer_names:
      if train_only is True and training is False:
        continue
      x = self.__getattribute__(layer_name)(x)
      x = self.__getattribute__("{}_act".format(layer_name))(x)
    return x

class MLP(tf.keras.Model):
  '''Defines fully connected layers (multi-layer perceptron).

  Args:
    num_layers (int): Number of layers to use
    num_classes (int): Number of classes to use
  '''
  def __init__(self, num_layers, activation=tf.nn.relu):
    super(MLP, self).__init__()
    self.num_layers = num_layers

    # define the internal blocks
    self.layer_names = []
    features = 32
    next = 1
    for i in range(num_layers - 1):
      layer_name = "dense_{}".format(i + 1)
      self.__setattr__(layer_name, lyrs.Dense(features))
      self.__setattr__("{}_act".format(layer_name), lyrs.ReLU())
      self.layer_names.append((layer_name, False))
      if next == i:
        features = features * 2
        next = next + i

  def call(self, input_tensor, training=False):
    x = input_tensor
    x = lyrs.Reshape((np.prod(x.shape[1:]), ))(x)
    for (layer_name, train_only) in self.layer_names:
      if train_only is True and training is False:
        continue
      x = self.__getattribute__(layer_name)(x)
      x = self.__getattribute__("{}_act".format(layer_name))(x)
    return x
