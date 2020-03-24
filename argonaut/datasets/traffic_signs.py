'''
Loads the RUB TrafficSigns Datasets.

Note: this uses the 2011 version

Paper: https://authors.library.caltech.edu/27452/
Website: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
'''

import os
import glob
import random
import cv2
import numpy as np
import pandas as pd
from bunch import Bunch

from argonaut.utils import Summary
from argonaut.datasets import utils as ds_util


def traffic_signs_download():
  '''Downloads the dataset to the specified folder.'''
  # try to load data
  train_link = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
  #test_link = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
  train_folder = ds_util.download_dataset(train_link)
  #test_folder = ds_util.download_dataset(test_link)
  #return train_folder, test_folder
  return train_folder

def _traffic_signs_gen(folder, mode, size, pad_mode, num_classes, one_hot, seed):
  '''Creates a generator from the given image_list.

  Args:
    mode (str): mode in which to use the dataset
    size (tuple): Size the images should be scaled to
    pad_mode (str): See `resize_and_pad` function
  '''
  # create one_hot labels
  eye = np.eye(num_classes)
  # load the relevant csv files (find in folder)
  ls = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
  items = None
  for csv_path in ls:
    # read files
    df = pd.read_csv(csv_path, sep=';')
    df["Folder"] = os.path.dirname(csv_path)
    # concat data
    if items is None:
      items = df
    else:
      items = pd.concat([items, df])

  # create generator for random split?
  rand = random.Random(seed)
  # shuffle the dataset
  items = items.sample(frac=1, random_state=seed)

  # iterate through image_list
  for idx, row in items.iterrows():
    # do selection
    r = rand.random()
    if mode == "train" and r > 0.85:
      continue
    elif mode == "test" and r <= 0.85:
      continue
    # load data
    img = cv2.imread(os.path.join(row["Folder"], row["Filename"]), 1)
    lbl = int(row["ClassId"])
    img, scale = ds_util.resize_and_pad(img, size, pad_mode)
    img /= 255.0
    # update label
    if one_hot == True:
      lbl = np.copy(eye[[lbl]][0])
    yield img, lbl

def traffic_signs_generator(size=(250, 250), pad_mode='fit_center', one_hot=True, seed=200, summary=None):
  '''Loads the RUB Traffic Signs Dataset as a generator.

  Args:
    size (tuple): Rescaling size of the images
    pad_mode (str): Defines the padding mode (see `utils.resize_and_pad`)
    seed (int): Seed for split with train data
    summary (Summary): Summary that contains processing information
  '''
  # load the metadata
  #train_folder, test_folder = traffic_signs_download()
  train_folder = traffic_signs_download()
  train_folder_img = os.path.join(train_folder, "GTSRB", "Final_Training", "Images")
  #test_folder_img = os.path.join(train_folder, "GTSRB", "Final_Test", "Images")
  # count folders
  num_classes = len([name for name in os.listdir(train_folder_img) if os.path.isdir(os.path.join(train_folder_img, name))])

  # setup seed
  if seed is None:
    seed = np.random.randint(0, 1000)

  # write to summary
  summary.add("data loading", "loaded traffic-signs (generator) dataset", {"size": size, "pad_mode": pad_mode, "one_hot": one_hot})

  # create final bunch
  return Bunch({
    'name': 'trafficsigns',
    "train": lambda: _traffic_signs_gen(train_folder_img, "train", size, pad_mode, num_classes, one_hot, seed),
    "test": lambda: _traffic_signs_gen(train_folder_img, "test", size, pad_mode, num_classes, one_hot, seed),
    "type": "generator",
    "size": (size[0], size[1], 3),
    "num_classes": num_classes,
    "class_names": None,
    "unique_classes": np.eye(num_classes) if one_hot == True else np.arange(num_classes)
  }), summary
