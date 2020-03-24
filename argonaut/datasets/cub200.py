'''
Loads the Caltech CUB-200 Datasets.

Note: this uses the 2011 version

Paper: https://authors.library.caltech.edu/27452/
Website: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
'''

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import utils
from bunch import Bunch

from argonaut.utils import Summary
from argonaut.datasets import utils as ds_util


def cub200_download():
  '''Downloads the dataset to the specified folder.'''
  # try to load data
  link = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
  return ds_util.download_dataset(link)

def _cub200_gen(folder, df_images, size, pad_mode, num_classes, one_hot):
  '''Creates a generator from the given image_list.

  Args:
    df_images (DataFrame): Dataframe that contains at least `class_id` and `path`
    size (tuple): Size the images should be scaled to
    pad_mode (str): See `resize_and_pad` function
  '''
  # create one_hot labels
  eye = np.eye(num_classes)
  # iterate through image_list
  for idx, row in df_images.iterrows():
    # load the image
    img = cv2.imread(os.path.join(folder, 'CUB_200_2011', 'images', row['path']), 1) / 255
    img /= 255.0
    img, scale = ds_util.resize_and_pad(img, size, pad_mode)
    # load the label
    cls_lbl = row['class_id']
    if one_hot == True:
      cls_lbl = np.copy(eye[[cls_lbl]][0])

    # return
    yield img, cls_lbl

def cub200_generator(size=(500, 500), pad_mode='fit_center', one_hot=True, seed=None, summary=None):
  '''Loads the Caltech CUB-200 2011 Dataset as a generator.

  Args:
    folder (str): Location of the dataset
    size (tuple): Rescaling size of the images
    pad_mode (str): Defines the padding mode (see `utils.resize_and_pad`)
    summary (Summary): Summary that contains processing information
  '''
  # load the metadata
  folder = cub200_download()
  cub_folder = os.path.join(folder, "CUB_200_2011")
  train_test = pd.read_csv(os.path.join(cub_folder, "train_test_split.txt"), sep=' ', header=None, names=['id', 'is_train'])
  images = pd.read_csv(os.path.join(cub_folder, "images.txt"), sep=' ', header=None, names=['id', 'path'])
  cls_label = pd.read_csv(os.path.join(cub_folder, "image_class_labels.txt"), sep=' ', header=None, names=['id', 'class_id'])
  cls_names = pd.read_csv(os.path.join(cub_folder, "classes.txt"), sep=' ', header=None, names=['class_id', 'class_name'])

  # merge the data
  min_cid = np.min(cls_names['class_id'])
  cls_label['class_id'] = cls_label['class_id'] - min_cid
  cls_names['class_id'] = cls_names['class_id'] - min_cid
  df_images = pd.merge(train_test, images, on='id')
  df_images = pd.merge(df_images, cls_label, on='id')
  num_classes = len(cls_names.index)
  cid = np.array(cls_names['class_id'])

  # write to summary
  summary.add("data loading", "loaded cub200 (generator) dataset", {"folder": folder, "size": size, "pad_mode": pad_mode})

  # shuffle the dataset
  df_images = df_images.sample(frac=1, random_state=seed)

  # create final bunch
  return Bunch({
    'name': 'cub200',
    "train": lambda: _cub200_gen(folder, df_images[df_images['is_train'] == 1], size, pad_mode, num_classes, one_hot),
    "test": lambda: _cub200_gen(folder, df_images[df_images['is_train'] == 0], size, pad_mode, num_classes, one_hot),
    "type": "generator",
    "size": (size[0], size[1], 3),
    "num_classes": num_classes,
    "class_names": list(cls_names['class_name']),
    "unique_classes": np.eye(num_classes)[cid] if one_hot == True else cid
  }), summary
