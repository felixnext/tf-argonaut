'''
Create a AlexNet-Style network.
Exact network parameters are from (for sake of comparision): https://arxiv.org/abs/1801.01423

author: Felix Geilert
'''

import tensorflow as tf
from tensorflow.keras import layers as lyrs

conv_params = {
  "padding": "same",
  #"kernel_initializer": tf.keras.initializers.glorot_normal(),
  "kernel_initializer": tf.keras.initializers.he_normal(),
  "bias_initializer": "zeros"
}

dense_params = {
  "kernel_initializer": tf.keras.initializers.glorot_normal(),
  "bias_initializer": "zeros"
}

class AlexNet(tf.keras.Model):
  '''Defines AlexNet like model.

  Args:
    shape (tuple): Image input Size
  '''
  def __init__(self, shape=(32, 32, 3)):
    super(AlexNet, self).__init__()
    # generate all items
    comp = 1
    self.conv1 = lyrs.Conv2D(64, (4, 4), (1, 1), **conv_params, activation=tf.nn.relu)
    self.pool2 = lyrs.MaxPool2D((2, 2), padding='same')
    comp *= 2
    self.dropout3 = lyrs.Dropout(0.2)
    self.conv4 = lyrs.Conv2D(128, (3, 3), (1, 1), **conv_params, activation=tf.nn.relu)
    self.pool5 = lyrs.MaxPool2D((2, 2), padding='same')
    comp *= 2
    self.dropout6 = lyrs.Dropout(0.2)
    self.conv7 = lyrs.Conv2D(256, (2, 2), (1, 1), **conv_params, activation=tf.nn.relu)
    self.pool8 = lyrs.MaxPool2D((2, 2), padding='same')
    comp *= 2
    self.dropout9 = lyrs.Dropout(0.5)

    # calculate exact reshape
    self.reshape = lyrs.Reshape(( int((shape[0] / comp) * (shape[1] / comp) * 256), ))

    # create dense layers
    self.fc10 = lyrs.Dense(2048, **dense_params, activation=tf.nn.relu)
    self.dropout11 = lyrs.Dropout(0.5)
    self.fc12 = lyrs.Dense(2048, **dense_params, activation=tf.nn.relu)
    self.dropout13 = lyrs.Dropout(0.5)

  def call(self, input_tensor, training=None):
    # iterate through the network
    x = self.dropout3(self.pool2(self.conv1(input_tensor)), training=training)
    x = self.dropout6(self.pool5(self.conv4(x)), training=training)
    x = self.dropout9(self.pool8(self.conv7(x)), training=training)
    x = self.reshape(x)
    x = self.dropout11(self.fc10(x), training=training)
    x = self.dropout13(self.fc12(x), training=training)

    return x
