'''
Implementation of the DenseNets.
'''

import tensorflow as tf
from tensorflow.keras import layers as lyrs

conv_params = {
  "padding": "same",
  "kernel_initializer": tf.keras.initializers.he_normal(),
  "bias_initializer": "zeros",
  "kernel_regularizer": tf.keras.regularizers.l2(0.0001)
}

class DenseBlock(tf.keras.Model):
  def __init__(self, growth_rate, depth, use_dropout=False, name=None):
    super(DenseBlock, self).__init__()
    self.depth = depth
    self.use_dropout = use_dropout

    # define all layers
    for i in range(depth):
      self.__setattr__("bn{}_1".format(i + 1), lyrs.BatchNormalization())
      self.__setattr__("relu{}_2".format(i + 1), lyrs.ReLU())
      self.__setattr__("conv{}_3".format(i + 1), lyrs.Conv2D(growth_rate * 4, (1, 1), (1, 1), **conv_params))
      self.__setattr__("bn{}_4".format(i + 1), lyrs.BatchNormalization())
      self.__setattr__("relu{}_5".format(i + 1), lyrs.ReLU())
      self.__setattr__("conv{}_6".format(i + 1), lyrs.Conv2D(growth_rate, (3, 3), (1, 1), **conv_params))
      if self.use_dropout == True:
        self.__setattr__("do1_1", lyrs.Dropout(0.2))
        self.__setattr__("do1_2", lyrs.Dropout(0.2))

  def call(self, input_tensor, training=False):
    state = input_tensor
    for i in range(self.depth):
      x = self.__getattribute__("bn{}_1".format(i + 1))(state, training)
      x = self.__getattribute__("relu{}_2".format(i + 1))(x)
      x = self.__getattribute__("conv{}_3".format(i + 1))(x)
      if self.use_dropout == True:
        x = self.__getattribute__("do1_1")(x, training)
      x = self.__getattribute__("bn{}_4".format(i + 1))(x, training)
      x = self.__getattribute__("relu{}_5".format(i + 1))(x)
      x = self.__getattribute__("conv{}_6".format(i + 1))(x)
      if self.use_dropout == True:
        x = self.__getattribute__("do1_2")(x, training)
      state = lyrs.concatenate([state, x])
    return state

class DenseNet(tf.keras.Model):
  def __init__(self, in_feat, factors, growth_rate=32, compression=0.5, use_dropout=False):
    super(DenseNet, self).__init__()
    self.use_dropout = use_dropout
    self.conv1 = lyrs.Conv2D(in_feat, (7, 7), (2, 2), **conv_params, activation=tf.nn.relu)
    self.pool1 = lyrs.MaxPool2D((3, 3), (2, 2), padding="same")
    self.block1 = DenseBlock(growth_rate, factors[0], self.use_dropout)
    in_feat += factors[0] * growth_rate

    self.conv2 = lyrs.Conv2D(int(in_feat * compression), (1, 1), (1, 1), **conv_params)
    if self.use_dropout == True:
      self.do1 = lyrs.Dropout(0.2)
    self.pool2 = lyrs.MaxPool2D((2, 2), (2, 2), padding="same")
    self.block2 = DenseBlock(growth_rate, factors[1], self.use_dropout)
    in_feat = (in_feat * compression) + growth_rate * factors[1]

    self.conv3 = lyrs.Conv2D(int(in_feat * compression), (1, 1), (1, 1), **conv_params)
    if self.use_dropout == True:
      self.do2 = lyrs.Dropout(0.2)
    self.pool3 = lyrs.MaxPool2D((2, 2), (2, 2), padding="same")
    self.block3 = DenseBlock(growth_rate, factors[2], self.use_dropout)
    in_feat = (in_feat * compression) + growth_rate * factors[2]

    self.conv4 = lyrs.Conv2D(int(in_feat * compression), (1, 1), (1, 1), **conv_params)
    if self.use_dropout == True:
      self.do3 = lyrs.Dropout(0.2)
    self.pool4 = lyrs.MaxPool2D((2, 2), (2, 2), padding="same")
    self.block4 = DenseBlock(growth_rate, factors[3], self.use_dropout)

  def call(self, input_tensor, training):
    x = self.pool1(self.conv1(input_tensor))
    x = self.block1(x, training)
    x = self.conv2(x)
    if self.use_dropout == True:
      x = self.do1(x, training)
    x = self.pool2(x)
    x = self.block2(x, training)
    x = self.conv3(x)
    if self.use_dropout == True:
      x = self.do2(x, training)
    x = self.pool3(x)
    x = self.block3(x, training)
    x = self.conv4(x)
    if self.use_dropout == True:
      x = self.do3(x, training)
    x = self.pool4(x)
    x = self.block4(x, training)
    return x

class DenseNet121(DenseNet):
  def __init__(self, growth_rate=32, compression=0.5, use_dropout=False):
    super(DenseNet121, self).__init__(64, [6, 12, 24, 16], growth_rate, compression, use_dropout)

class DenseNet169(DenseNet):
  def __init__(self, growth_rate=32, compression=0.5, use_dropout=False):
    super(DenseNet169, self).__init__(64, [6, 12, 32, 32], growth_rate, compression, use_dropout)

class DenseNet201(DenseNet):
  def __init__(self, growth_rate=32, compression=0.5, use_dropout=False):
    super(DenseNet201, self).__init__(64, [6, 12, 48, 32], growth_rate, compression, use_dropout)

class DenseNet264(DenseNet):
  def __init__(self, growth_rate=32, compression=0.5, use_dropout=False):
    super(DenseNet264, self).__init__(64, [6, 12, 64, 48], growth_rate, compression, use_dropout)
