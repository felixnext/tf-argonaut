'''
Definition of heads for models in various settings.
'''

import tensorflow as tf
from tensorflow.keras import layers as lyrs

class ClassifyHeadDense(tf.keras.Model):
  '''Creates a classification head for the model.

  Args:
    dataset (dict): Information about the dataset use
    use_pre_layer (bool): Defines if there should be a cooldown dense layer with tanh activation (might stabilize training)
  '''
  def __init__(self, dataset, use_pre_layer=False, name=None):
    super(ClassifyHeadDense, self).__init__()
    self.num_classes = dataset["num_classes"]
    self._name = name
    self.pre_layer = use_pre_layer
    with tf.name_scope(name if name is not None else "classify-head"):
      if use_pre_layer == True:
        self.pre_dense = lyrs.Dense(self.num_classes * 2, activation=tf.nn.tanh)
      self.dense = lyrs.Dense(self.num_classes)
      self.softmax = lyrs.Softmax()

  def call_logits(self, input_tensor):
    x = input_tensor
    if self.pre_layer == True:
      x = self.pre_dense(x)
    return self.dense(x)

  def call(self, input_tensor):
    x = self.call_logits(input_tensor)
    return self.softmax(x)

class ClassifyHeadConv(tf.keras.Model):
  '''Creates a classification head for the model.

  Args:
    dataset (dict): Information about the dataset use
  '''
  def __init__(self, dataset, name=None):
    super(ClassifyHeadConv, self).__init__()
    self.num_classes = dataset["num_classes"]
    self._name = name
    with tf.name_scope(name if name is not None else "classify-head"):
      self.conv = lyrs.Conv2D(self.num_classes, (1, 1), 1)
      self.pool = lyrs.GlobalAveragePooling2D()
      self.act = lyrs.Softmax()

  def call_logits(self, input_tensor):
    return self.pool(self.conv(input_tensor))

  def call(self, input_tensor):
    x = self.call_logits(input_tensor)
    return self.act(x)
