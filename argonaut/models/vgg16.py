'''
Defines VGG 16 Model
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as lyrs

conv_params = {
  "padding": "same",
  "kernel_initializer": tf.keras.initializers.glorot_normal(),
  "bias_initializer": "zeros",
  "kernel_regularizer": tf.keras.regularizers.l2(0.0001)
}

class VGG16(tf.keras.Model):
  '''Defines the VGG 16 Model.

  Note that the model expects a 224x224 Input Image.

  Args:
    input_size (tuple): Tuple of the input size
    activation (tf.activation): Activation function to use, default `tf.nn.relu`
  '''
  def __init__(self, input_size=(224, 224), activation=tf.nn.relu):
    super(VGG16, self).__init__()
    # define the relevant data
    self.conv3_1 = lyrs.Conv2D(64, (3, 3), 1, **conv_params, activation=activation, name="conv3_1")
    self.conv3_2 = lyrs.Conv2D(64, (3, 3), 1, **conv_params, activation=activation, name="conv3_2")
    self.pool2_1 = lyrs.MaxPool2D((2, 2), (2, 2), 'same', name="pool2_1")

    self.conv3_3 = lyrs.Conv2D(128, (3, 3), 1, **conv_params, activation=activation, name="conv3_3")
    self.conv3_4 = lyrs.Conv2D(128, (3, 3), 1, **conv_params, activation=activation, name="conv3_4")
    self.pool2_2 = lyrs.MaxPool2D((2, 2), (2, 2), 'same', name="pool2_2")

    self.conv3_5 = lyrs.Conv2D(256, (3, 3), 1, **conv_params, activation=activation, name="conv3_5")
    self.conv3_6 = lyrs.Conv2D(256, (3, 3), 1, **conv_params, activation=activation, name="conv3_6")
    self.conv1_7 = lyrs.Conv2D(256, (1, 1), 1, **conv_params, activation=activation, name="conv3_7")
    self.pool2_3 = lyrs.MaxPool2D((2, 2), (2, 2), 'same', name="pool2_3")

    self.conv3_8 = lyrs.Conv2D(512, (3, 3), 1, **conv_params, activation=activation, name="conv3_8")
    self.conv3_9 = lyrs.Conv2D(512, (3, 3), 1, **conv_params, activation=activation, name="conv3_9")
    self.conv1_10 = lyrs.Conv2D(512, (1, 1), 1, **conv_params, activation=activation, name="conv3_10")
    self.pool2_4 = lyrs.MaxPool2D((2, 2), (2, 2), 'same', name="pool2_4")

    self.conv3_11 = lyrs.Conv2D(512, (3, 3), 1, **conv_params, activation=activation, name="conv3_11")
    self.conv3_12 = lyrs.Conv2D(512, (3, 3), 1, **conv_params, activation=activation, name="conv3_12")
    self.conv1_13 = lyrs.Conv2D(512, (1, 1), 1, **conv_params, activation=activation, name="conv3_13")
    self.pool2_5 = lyrs.MaxPool2D((2, 2), (2, 2), 'same', name="pool2_5")

    self.reshape = lyrs.Reshape((np.ceil(input_size[0] / 32) * np.ceil(input_size[1] / 32) * 512, ))
    self.fc_14 = lyrs.Dense(4096, activation=activation, name="fc_14")
    self.fc_15 = lyrs.Dense(4096, activation=activation, name="fc_15")
    # NOTE: last layer will be created by the head (dense and softmax)

  def call(self, input_tensor, training=False):
    # build all the blocks accordingly
    b1 = self.pool2_1(self.conv3_2(self.conv3_1(input_tensor)))
    b2 = self.pool2_2(self.conv3_4(self.conv3_3(b1)))
    b3 = self.pool2_3(self.conv1_7(self.conv3_6(self.conv3_5(b2))))
    b4 = self.pool2_4(self.conv1_10(self.conv3_9(self.conv3_8(b3))))
    b5 = self.pool2_5(self.conv1_13(self.conv3_12(self.conv3_11(b4))))
    b5 = self.reshape(b5)
    out = self.fc_15(self.fc_14(b5))

    return out
