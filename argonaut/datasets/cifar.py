'''
Various functions to load and permute the cifar datasets.

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

def _cifar_load(mode=10, norm=True, one_hot=True):
  '''Loads the cifar data.'''
  # load through keras
  (x_train, y_train), (x_test, y_test) = ((None, None), (None, None))
  if mode == 10:
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
  elif mode == 100:
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
  else:
    raise ValueError("Unkown Cifar Type ({})".format(mode))

  # define base parameters
  img_width = 32
  img_height = 32

  # normalize data
  y_train = np.squeeze(y_train)
  y_test = np.squeeze(y_test)
  if norm == True:
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255.0
    x_test /= 255.0
  else:
    x_train = x_train.astype("int32")
    x_test = x_test.astype("int32")

  # reshape input data
  x_train = x_train.reshape(x_train.shape[0], img_width, img_height, 3)
  x_test = x_test.reshape(x_test.shape[0], img_width, img_height, 3)

  # one hot encode outputs
  if one_hot == True:
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
  else:
    num_classes = np.unique(y_train).shape[0]

  return x_train, y_train, x_test, y_test, num_classes

def cifar(mode=10, norm=True, one_hot=True, summary=None, name=None):
  '''Loads cifar dataset.

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
  x_train, y_train, x_test, y_test, num_classes = _cifar_load(mode, norm, one_hot)

  # add to summary
  summary.add("data loading", "loaded mnist dataset", {"mode": mode, "one-hot": one_hot, "normalized": norm})

  # create bunch package
  return Bunch({
    'name': 'cifar{}'.format(mode),
    'test': Bunch({'x': x_test, 'y': y_test}),
    'train': Bunch({'x': x_train, 'y': y_train}),
    'type': 'preloaded',
    'size': (32, 32, x_train.shape[-1]),
    'num_classes': num_classes,
    'class_names': None,
    'unique_classes': np.unique(y_train, axis=0)
  }), summary

def cifar10(norm=True, one_hot=True, summary=None, name=None):
  return cifar(10, norm, one_hot, summary, name)

def cifar100(norm=True, one_hot=True, summary=None, name=None):
  return cifar(100, norm, one_hot, summary, name)


# define variables in global scope
g_x_train = [None, None]
g_y_train = [None, None]
g_x_test = [None, None]
g_y_test = [None, None]

def _cifar_gen(mode, cifar_mode, norm, one_hot, load_once=True):
  '''Defines generator around data.'''
  if load_once == False:
    x_train, y_train, x_test, y_test, num_classes = _cifar_load(cifar_mode, norm, one_hot)
  else:
    # set variables in global scope to avoid reloading
    gid = 0 if cifar_mode == 10 else 1
    global g_x_train, g_y_train, g_x_test, g_y_test
    if g_x_train[gid] is None:
      g_x_train[gid], g_y_train[gid], g_x_test[gid], g_y_test[gid], num_classes = _cifar_load(cifar_mode, norm, one_hot)
    x_train, y_train, x_test, y_test = (g_x_train[gid], g_y_train[gid], g_x_test[gid], g_y_test[gid])
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

def cifar_generator(mode=10, norm=True, one_hot=True, load_once=True, summary=None):
  '''Creates generators for the relevant data.'''
  if summary is None:
    summary = Summary()

  # add to summary
  summary.add("data loading", "loaded cifar (generator) dataset", {"mode": mode, "one-hot": one_hot, "normalized": norm, "load_once": load_once})

  # retrieve data
  gen = _cifar_gen("train", mode, norm, one_hot, load_once)
  ucls = {}
  for x, y in gen:
    if str(y) not in ucls:
      ucls[str(y)] = y

  # create bunch output
  return Bunch({
    'name': 'cifar{}'.format(mode),
    'train': lambda: _cifar_gen("train", mode, norm, one_hot, load_once),
    'test': lambda: _cifar_gen("test", mode, norm, one_hot, load_once),
    'type': 'generator',
    'size': (32, 32, 3),
    'num_classes': mode,
    'class_names': None,
    'unique_classes': np.array(list(ucls.values()))
  }), summary

def cifar10_generator(norm=True, one_hot=True, load_once=True, summary=None):
  return cifar_generator(10, norm, one_hot, load_once, summary)

def cifar100_generator(norm=True, one_hot=True, load_once=True, summary=None):
  return cifar_generator(100, norm, one_hot, load_once, summary)
