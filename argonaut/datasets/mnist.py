'''
Various functions to load and permute the mnist dataset.

author: Felix Geilert
'''


import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import utils
from bunch import Bunch

from hormones.utils import Summary
#from . import utils as ds_utils

tf.executing_eagerly()

def _mnist_load(channels=1, norm=True, one_hot=True):
  '''Loads the mnist data.'''
  # load through keras
  (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

  # define base parameters
  img_width = 28
  img_height = 28

  # normalize data
  if norm == True:
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255.0
    x_test /= 255.0
  else:
    x_train = x_train.astype("int32")
    x_test = x_test.astype("int32")

  # reshape input data
  x_train = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
  x_test = x_test.reshape(x_test.shape[0], img_width, img_height, 1)
  if channels == 3:
    x_train = np.tile(x_train, (1, 1, 1, 3))
    x_test = np.tile(x_test, (1, 1, 1, 3))

  # one hot encode outputs
  if one_hot == True:
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
  else:
    num_classes = np.unique(y_train).shape[0]

  return x_train, y_train, x_test, y_test, num_classes

def mnist(channels=1, norm=True, one_hot=True, summary=None, name=None):
  '''Loads classic mnist.

  Args:
    norm (bool): Defines if the values should be normalized between 0 and 1 (floats)
    channels (int): Defines the number of channels to use (options: 1 or 3)
    one_hot (bool): Defines if output should be encoded as one_hot categoricals
    summary (Summary): Summary item to insert the operation
    name (str): Name of the operation (default: exclude)

  Returns:
    Bunch object that contains the following keywords:
    - test: contains `x` and `y` for the output values
    - train: contains `x` and `y` for the output values
    - size: size of the images
    - num_classes: Integer of the number of different classes
    - class_names: Names of the classes if available
  '''
  if summary is None:
    summary = Summary()
  # load through keras
  x_train, y_train, x_test, y_test, num_classes = _mnist_load(channels, norm, one_hot)

  # add to summary
  summary.add("data loading", "loaded mnist dataset", {"channels": channels, "one-hot": one_hot, "normalized": norm})

  # create bunch package
  return Bunch({
    'name': 'mnist',
    'test': Bunch({'x': x_test, 'y': y_test}),
    'train': Bunch({'x': x_train, 'y': y_train}),
    'type': 'preloaded',
    'size': (28, 28, x_train.shape[-1]),
    'num_classes': num_classes,
    'class_names': None,
    'unique_classes': np.unique(y_train, axis=0)
  }), summary


# define variables in global scope
g_x_train = None
g_y_train = None
g_x_test = None
g_y_test = None

def _mnist_gen(mode, channels, norm, one_hot, load_once=True):
  '''Defines generator around data.'''
  if load_once == False:
    x_train, y_train, x_test, y_test, num_classes = _mnist_load(channels, norm, one_hot)
  else:
    # set variables in global scope to avoid reloading
    global g_x_train, g_y_train, g_x_test, g_y_test
    if g_x_train is None:
      g_x_train, g_y_train, g_x_test, g_y_test, num_classes = _mnist_load(channels, norm, one_hot)
    x_train, y_train, x_test, y_test = (g_x_train, g_y_train, g_x_test, g_y_test)
  if not isinstance(mode, str):
    mode = mode.decode('UTF-8')
  i = 0
  if mode == "train":
    while i < x_train.shape[0]:
      img = x_train[i, ...]
      lbl = y_train[i, ...]
      yield img, lbl
      i += 1
  if mode == "test":
    while i < x_test.shape[0]:
      img = x_test[i, ...]
      lbl = y_test[i, ...]
      yield img, lbl
      i += 1

def mnist_generator(channels=1, norm=True, one_hot=True, load_once=True, summary=None):
  '''Creates generators for the relevant data.'''
  if summary is None:
    summary = Summary()

  # add to summary
  summary.add("data loading", "loaded mnist (generator) dataset", {"channels": channels, "one-hot": one_hot, "normalized": norm, "load_once": load_once})

  # retrieve data
  gen = _mnist_gen("train", channels, norm, one_hot, load_once)
  ucls = {}
  for x, y in gen:
    if str(y) not in ucls:
      ucls[str(y)] = y

  # create bunch output
  return Bunch({
    'name': 'mnist',
    'train': lambda: _mnist_gen("train", channels, norm, one_hot, load_once),
    'test': lambda: _mnist_gen("test", channels, norm, one_hot, load_once),
    'type': 'generator',
    'size': (28, 28, channels),
    'num_classes': 10,
    'class_names': None,
    'unique_classes': np.array(list(ucls.values()))
  }), summary

def mnist_tfdata(channels=1, norm=True, one_hot=True, shuffle=None, summary=None):
  '''Loads mnist as a tf.data dataset.'''
  if summary is None:
    summary = Summary()

  # define size
  img_width = 28
  img_height = 28

  # create generators
  shape = () if one_hot == False else (10,)
  ds_train = tf.data.Dataset.from_generator(_mnist_gen, args=["train", channels, norm, one_hot], output_types=(tf.float32, tf.float32),
    output_shapes=([img_width, img_height, channels], shape))
  ds_test  = tf.data.Dataset.from_generator(_mnist_gen, args=["test",  channels, norm, one_hot], output_types=(tf.float32, tf.float32),
    output_shapes=([img_width, img_height, channels], shape))

  # check for shuffle
  if shuffle is not None:
    ds_train = ds_train.shuffle(shuffle)
    ds_test  = ds_test.shuffle(shuffle)

  # add to summary
  summary.add("data loading", "loaded mnist (tfdata) dataset", {"channels": channels, "one-hot": one_hot, "normalized": norm, "shuffle": shuffle})

  # create bunch package
  return Bunch({
    'name': 'mnist',
    'train': ds_train,
    'test': ds_test,
    'type': 'tfdata',
    'size': (img_width, img_height, channels),
    #'steps': Bunch({'train': None, 'test': None}),
    'num_classes': 10,
    'class_names': None,
    'unique_classes': [c.numpy() for c in ds_train.map(lambda x, y: tf.cast(y, tf.int32)).apply(tf.data.experimental.unique())]
  }), summary