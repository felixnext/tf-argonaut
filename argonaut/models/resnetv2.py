'''Defines the resnet v2.'''

import tensorflow as tf
from tensorflow.keras import layers as lyrs

conv_params = {
  "padding": "same",
  "kernel_initializer": tf.keras.initializers.he_normal(),
  "bias_initializer": "zeros",
  "kernel_regularizer": tf.keras.regularizers.l2(0.0001)
}

class ResBlock_v2(tf.keras.Model):
  '''Main building block for residual networks.

  Args:
    bottleneck (bool): Defines if this layers is a bottleneck
    featrues (int): Defines the number of features to use
  '''
  def __init__(self, features, activation, is_transition=False, is_large=True):
    super(ResBlock_v2, self).__init__()
    # define data
    self.activation = activation
    self.is_trans = is_transition
    self.is_large = is_large

    # define the relevant layers
    self.bn_1 = lyrs.BatchNormalization()
    self.bn_2 = lyrs.BatchNormalization()
    self.add = lyrs.Add()

    if is_large == False:
      self.conv3_1 = lyrs.Conv2D(features, (3, 3), (1, 1) if self.is_trans == False else (2, 2), **conv_params, name="conv3_1")
      self.conv3_2 = lyrs.Conv2D(features, (3, 3), (1, 1), **conv_params, name="conv3_2")
    else:
      self.bn_3 = lyrs.BatchNormalization()
      self.conv1_1 = lyrs.Conv2D(int(features / 4), (1, 1), (1, 1), **conv_params, name="conv1_1")
      self.conv3_2 = lyrs.Conv2D(int(features / 4), (3, 3), (1, 1) if self.is_trans == False else (2, 2), **conv_params, name="conv3_2")
      self.conv1_3 = lyrs.Conv2D(features, (1, 1), (1, 1), **conv_params, name="conv1_3")

    # check for bottleneck
    if self.is_trans == True:
      self.conv_tn = lyrs.Conv2D(features, (1, 1), (2, 2), **conv_params, name="transition")

  def call(self, input_tensor, training=False):
    residual = input_tensor
    if self.is_large == False:
      residual = self.conv3_1(self.activation(self.bn_1(residual, training)))
      residual = self.conv3_2(self.activation(self.bn_2(residual, training)))
    else:
      residual = self.conv1_1(self.activation(self.bn_1(residual, training)))
      residual = self.conv3_2(self.activation(self.bn_2(residual, training)))
      residual = self.conv1_3(self.activation(self.bn_3(residual, training)))

    # check
    if self.is_trans == True:
      input_tensor = self.conv_tn(input_tensor)

    return self.add([input_tensor, residual])

class ResNet_v2_50(tf.keras.Model):
  def __init__(self):
    super(ResNet_v2_50, self).__init__()
    act = lyrs.ReLU
    self.conv1 = lyrs.Conv2D(64, (7, 7), (2, 2), **conv_params, name="conv_1")
    self.pool1 = lyrs.MaxPool2D((3, 3), (2, 2), padding="same")
    self.conv2_1 = ResBlock_v2(256, act(), True, True)
    for i in range(2):
      self.__setattr__("conv2_{}".format(i + 2), ResBlock_v2(256, act(), False, True))
    self.conv3_1 = ResBlock_v2(512, act(), True, True)
    for i in range(3):
      self.__setattr__("conv3_{}".format(i + 2), ResBlock_v2(512, act(), False, True))
    self.conv4_1 = ResBlock_v2(1024, act(), True, True)
    for i in range(5):
      self.__setattr__("conv4_{}".format(i + 2), ResBlock_v2(1024, act(), False, True))
    self.conv5_1 = ResBlock_v2(2048, act(), True, True)
    for i in range(2):
      self.__setattr__("conv5_{}".format(i + 2), ResBlock_v2(2048, act(), False, True))

  def call(self, input_tensor, training=False):
    x = self.pool1(self.conv1(input_tensor))
    for i in range(3):
      x = self.__getattribute__("conv2_{}".format(i + 1))(x, training)
    for i in range(4):
      x = self.__getattribute__("conv3_{}".format(i + 1))(x, training)
    for i in range(6):
      x = self.__getattribute__("conv4_{}".format(i + 1))(x, training)
    for i in range(3):
      x = self.__getattribute__("conv5_{}".format(i + 1))(x, training)
    return x

class ResNet_v2_101(tf.keras.Model):
  def __init__(self):
    super(ResNet_v2_101, self).__init__()
    act = lyrs.ReLU
    self.conv1 = lyrs.Conv2D(64, (7, 7), (2, 2), **conv_params, name="conv_1")
    self.pool1 = lyrs.MaxPool2D((3, 3), (2, 2), padding="same")
    self.conv2_1 = ResBlock_v2(256, act(), True, True)
    for i in range(2):
      self.__setattr__("conv2_{}".format(i + 2), ResBlock_v2(256, act(), False, True))
    self.conv3_1 = ResBlock_v2(512, act(), True, True)
    for i in range(3):
      self.__setattr__("conv3_{}".format(i + 2), ResBlock_v2(512, act(), False, True))
    self.conv4_1 = ResBlock_v2(1024, act(), True, True)
    for i in range(22):
      self.__setattr__("conv4_{}".format(i + 2), ResBlock_v2(1024, act(), False, True))
    self.conv5_1 = ResBlock_v2(2048, act(), True, True)
    for i in range(2):
      self.__setattr__("conv5_{}".format(i + 2), ResBlock_v2(2048, act(), False, True))

  def call(self, input_tensor, training=False):
    x = self.pool1(self.conv1(input_tensor))
    for i in range(3):
      x = self.__getattribute__("conv2_{}".format(i + 1))(x, training)
    for i in range(4):
      x = self.__getattribute__("conv3_{}".format(i + 1))(x, training)
    for i in range(23):
      x = self.__getattribute__("conv4_{}".format(i + 1))(x, training)
    for i in range(3):
      x = self.__getattribute__("conv5_{}".format(i + 1))(x, training)
    return x

class ResNet_v2_152(tf.keras.Model):
  def __init__(self):
    super(ResNet_v2_152, self).__init__()
    act = lyrs.ReLU
    self.conv1 = lyrs.Conv2D(64, (7, 7), (2, 2), **conv_params, name="conv_1")
    self.pool1 = lyrs.MaxPool2D((3, 3), (2, 2), padding="same")
    self.conv2_1 = ResBlock_v2(256, act(), True, True)
    for i in range(2):
      self.__setattr__("conv2_{}".format(i + 2), ResBlock_v2(256, act(), False, True))
    self.conv3_1 = ResBlock_v2(512, act(), True, True)
    for i in range(7):
      self.__setattr__("conv3_{}".format(i + 2), ResBlock_v2(512, act(), False, True))
    self.conv4_1 = ResBlock_v2(1024, act(), True, True)
    for i in range(35):
      self.__setattr__("conv4_{}".format(i + 2), ResBlock_v2(1024, act(), False, True))
    self.conv5_1 = ResBlock_v2(2048, act(), True, True)
    for i in range(2):
      self.__setattr__("conv5_{}".format(i + 2), ResBlock_v2(2048, act(), False, True))

  def call(self, input_tensor, training=False):
    x = self.pool1(self.conv1(input_tensor))
    for i in range(3):
      x = self.__getattribute__("conv2_{}".format(i + 1))(x, training)
    for i in range(8):
      x = self.__getattribute__("conv3_{}".format(i + 1))(x, training)
    for i in range(36):
      x = self.__getattribute__("conv4_{}".format(i + 1))(x, training)
    for i in range(3):
      x = self.__getattribute__("conv5_{}".format(i + 1))(x, training)
    return x
