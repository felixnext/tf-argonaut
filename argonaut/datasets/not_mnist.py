'''
Various functions to load and permute the not-mnist dataset.
More Infos: https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html

author: Felix Geilert
'''


import os
import glob
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from bunch import Bunch

from argonaut.utils import Summary
from argonaut.datasets import utils as ds_util

def not_mnist_download():
  '''Downloads the dataset to the specified folder.'''
  # try to load data
  data_link = "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz"
  train_folder = ds_util.download_dataset(data_link)
  return train_folder

def _load_gen(folder, mode, one_hot, seed):
  # iterate through data
  folder = os.path.join(folder, "notMNIST_small")
  classes = [x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]
  eye = np.eye(len(classes))

  # create list of all data
  datapoints = []

  # iterate through relevant classes
  for i, c in enumerate(classes):
    imgs = glob.glob(os.path.join(folder, c, "*.png"))
    for img_path in imgs:
      datapoints.append((img_path, c))

  # shuffle the dataset list
  random.Random(seed).shuffle(datapoints)
  #shuffle(datapoints)

  # iterate data
  rand = random.Random(seed)
  for img_path, c in datapoints:
    # do selection
    r = rand.random()
    if mode == "train" and r > 0.85:
      continue
    elif mode == "test" and r <= 0.85:
      continue

    # load the image
    img = cv2.imread(img_path, 1)
    lbl = classes.index(c)
    if img is None:
      continue
    img = img.astype("float32")

    # update the image
    img /= 255.0
    # update label
    if one_hot == True:
      lbl = np.copy(eye[[lbl]][0])

    yield img, lbl

def not_mnist_generator(one_hot=True, seed=200, summary=None):
  '''Loads the not Mnist Dataset as a generator.

  Args:
    size (tuple): Rescaling size of the images
    pad_mode (str): Defines the padding mode (see `utils.resize_and_pad`)
    seed (int): Seed for split with train data
    summary (Summary): Summary that contains processing information
  '''
  # load the metadata
  train_folder = not_mnist_download()

  # setup seed
  if seed is None:
    seed = np.random.randint(0, 1000)

  # write to summary
  summary.add("data loading", "loaded notMnist (generator) dataset", {"one_hot": one_hot})

  # create final bunch
  return Bunch({
    'name': 'notmnist',
    "train": lambda: _load_gen(train_folder, "train", one_hot, seed),
    "test": lambda: _load_gen(train_folder, "test", one_hot, seed),
    "type": "generator",
    "size": (28, 28, 3),
    "num_classes": 10,
    "class_names": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'J'],
    "unique_classes": np.eye(10) if one_hot is True else np.arange(10)
  }), summary
